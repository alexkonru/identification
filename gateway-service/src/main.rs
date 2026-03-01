use glob::glob;
use pgvector::Vector;
use shared::biometry::gatekeeper_server::{Gatekeeper, GatekeeperServer};
use shared::biometry::{
    AccessCheckResponse, AccessCheckResponseV2, AddDeviceRequest, AddRoomRequest, AddZoneRequest,
    CheckAccessRequest, CheckAccessRequestV2, ControlDoorRequest, ControlServiceRequest, Device,
    Empty, GetLogsRequest, GetLogsResponse, GetUserAccessResponse, GrantAccessRequest, IdRequest,
    IdResponse, ImageFrame, ListDevicesRequest, ListDevicesResponse, ListRoomsRequest,
    ListRoomsResponse, ListUsersRequest, ListUsersResponse, ListZonesRequest, ListZonesResponse,
    LogEntry, RegisterUserRequest, Room, RuntimeModeRequest, RuntimeModeResponse,
    ScanHardwareResponse, ServiceStatus, SetAccessRulesRequest, StatusResponse,
    SystemStatusResponse, User, Zone, audio_client::AudioClient, vision_client::VisionClient,
};
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres, Row};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::OpenOptions;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep, timeout};
use tokio::io::AsyncWriteExt;
use tonic::{
    Request, Response, Status,
    transport::{Channel, Server},
};

// Struct to hold full processing details
struct FaceProcessResult {
    embedding: Vec<f32>,
    liveness_score: f32,
    is_live: bool,
    provider: String,
}

struct ClipFrameAssessment {
    index: usize,
    detected: bool,
    is_live: bool,
    liveness_score: f32,
}

struct ClipSelectionSummary {
    best_idx: usize,
    flags: Vec<String>,
    median_liveness: f32,
    live_ratio: f32,
    detected_ratio: f32,
}

struct HardwareProfile {
    cpu_cores: usize,
    cpu_threads: usize,
    gpu_available: bool,
}

struct RuntimePlan {
    mode: String,
    vision_threads: usize,
    audio_threads: usize,
    force_cpu: bool,
    use_cuda: bool,
}

#[derive(Clone)]
struct RoomInfraState {
    room_id: i32,
    zone_name: String,
    room_name: String,
    cameras: i64,
    locks: i64,
    microphones: i64,
    ready: bool,
    reason: String,
}

pub struct GatewayService {
    pool: Pool<Postgres>,
    audio_channel: Channel,
    vision_channel: Channel,
    face_similarity_threshold: f64,
    voice_similarity_threshold: f64,
    test_device_id: i32,
    vision_rpc_timeout: Duration,
    audio_rpc_timeout: Duration,
    shutdown_tx: mpsc::Sender<()>,
    started_at_ms: i64,
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

fn env_parse_f32_clamped(name: &str, default: f32, min: f32, max: f32) -> f32 {
    env_parse::<f32>(name, default).clamp(min, max)
}

impl GatewayService {
    fn pipeline_run_flag_path() -> String {
        std::env::var("SYSTEM_RUN_FLAG_PATH")
            .unwrap_or_else(|_| "/workspace/identification/.system_run".to_string())
    }

    async fn set_pipeline_running(enabled: bool) -> Result<(), Status> {
        let path = Self::pipeline_run_flag_path();
        if let Some(parent) = std::path::Path::new(&path).parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| Status::internal(format!("create run-flag dir failed: {}", e)))?;
        }
        tokio::fs::write(&path, if enabled { "1\n" } else { "0\n" })
            .await
            .map_err(|e| Status::internal(format!("write run-flag failed: {}", e)))
    }

    async fn room_infra_states(&self) -> Result<Vec<RoomInfraState>, Status> {
        let rows = sqlx::query(
            r#"
            SELECT
                r.id AS room_id,
                z.name AS zone_name,
                r.name AS room_name,
                COUNT(*) FILTER (WHERE dt.name = 'camera') AS cameras,
                COUNT(*) FILTER (WHERE dt.name = 'lock') AS locks,
                COUNT(*) FILTER (WHERE dt.name = 'microphone') AS microphones
            FROM rooms r
            JOIN zones z ON z.id = r.zone_id
            LEFT JOIN devices d ON d.room_id = r.id
            LEFT JOIN device_types dt ON dt.id = d.device_type_id
            GROUP BY r.id, z.name, r.name
            ORDER BY z.name, r.name
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("Infra validation query failed: {}", e)))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let cameras: i64 = row.get::<Option<i64>, _>("cameras").unwrap_or(0);
            let locks: i64 = row.get::<Option<i64>, _>("locks").unwrap_or(0);
            let microphones: i64 = row.get::<Option<i64>, _>("microphones").unwrap_or(0);
            let mut reasons = Vec::new();
            if cameras != 1 {
                reasons.push(format!("camera_count={}", cameras));
            }
            if locks != 1 {
                reasons.push(format!("lock_count={}", locks));
            }
            if microphones > 1 {
                reasons.push(format!("microphone_count={}", microphones));
            }
            let ready = reasons.is_empty();
            out.push(RoomInfraState {
                room_id: row.get("room_id"),
                zone_name: row.get("zone_name"),
                room_name: row.get("room_name"),
                cameras,
                locks,
                microphones,
                ready,
                reason: if ready {
                    "ok".to_string()
                } else {
                    reasons.join(",")
                },
            });
        }
        Ok(out)
    }

    async fn room_ready_reason(&self, room_id: i32) -> Result<Option<String>, Status> {
        let states = self.room_infra_states().await?;
        for s in states {
            if s.room_id == room_id && !s.ready {
                return Ok(Some(format!(
                    "room '{} / {}' is disabled: {}",
                    s.zone_name, s.room_name, s.reason
                )));
            }
        }
        Ok(None)
    }

    fn normalize_embedding(mut emb: Vec<f32>) -> Vec<f32> {
        if emb.is_empty() {
            return emb;
        }
        let norm_sq: f32 = emb.iter().map(|v| v * v).sum();
        if norm_sq <= 1e-12 {
            return emb;
        }
        let inv = norm_sq.sqrt().recip();
        for v in &mut emb {
            *v *= inv;
        }
        emb
    }

    fn mean_normalized_embedding(samples: Vec<Vec<f32>>) -> Option<Vec<f32>> {
        if samples.is_empty() {
            return None;
        }
        let dim = samples[0].len();
        if dim == 0 {
            return None;
        }
        let mut acc = vec![0.0f32; dim];
        let mut count = 0usize;
        for s in samples {
            if s.len() != dim {
                continue;
            }
            for (i, v) in s.iter().enumerate() {
                acc[i] += *v;
            }
            count += 1;
        }
        if count == 0 {
            return None;
        }
        let inv = 1.0f32 / count as f32;
        for v in &mut acc {
            *v *= inv;
        }
        Some(Self::normalize_embedding(acc))
    }

    pub fn new(
        pool: Pool<Postgres>,
        audio_channel: Channel,
        vision_channel: Channel,
        face_similarity_threshold: f64,
        voice_similarity_threshold: f64,
        test_device_id: i32,
        vision_rpc_timeout: Duration,
        audio_rpc_timeout: Duration,
        shutdown_tx: mpsc::Sender<()>,
        started_at_ms: i64,
    ) -> Self {
        Self {
            pool,
            audio_channel,
            vision_channel,
            face_similarity_threshold,
            voice_similarity_threshold,
            test_device_id,
            vision_rpc_timeout,
            audio_rpc_timeout,
            shutdown_tx,
            started_at_ms,
        }
    }

    // Updated to return full details
    async fn process_face_full(&self, image: &[u8]) -> Result<Option<FaceProcessResult>, Status> {
        let mut client = VisionClient::new(self.vision_channel.clone());
        let request = Request::new(ImageFrame {
            content: image.to_vec(),
        });

        match timeout(self.vision_rpc_timeout, client.process_face(request)).await {
            Err(_) => {
                // one lightweight retry to avoid startup race false negatives
                let mut retry_client = VisionClient::new(self.vision_channel.clone());
                let retry_request = Request::new(ImageFrame {
                    content: image.to_vec(),
                });
                match timeout(
                    self.vision_rpc_timeout,
                    retry_client.process_face(retry_request),
                )
                .await
                {
                    Err(_) => Err(Status::deadline_exceeded(format!(
                        "Vision service timeout after {} ms",
                        self.vision_rpc_timeout.as_millis()
                    ))),
                    Ok(result) => match result {
                        Ok(response) => {
                            let res = response.into_inner();
                            if res.detected && !res.embedding.is_empty() {
                                Ok(Some(FaceProcessResult {
                                    embedding: Self::normalize_embedding(res.embedding),
                                    liveness_score: res.liveness_score,
                                    is_live: res.is_live,
                                    provider: res.execution_provider,
                                }))
                            } else {
                                Ok(None)
                            }
                        }
                        Err(e) => Err(Status::internal(format!("Vision service error: {}", e))),
                    },
                }
            }
            Ok(result) => match result {
                Ok(response) => {
                    let res = response.into_inner();
                    if res.detected && !res.embedding.is_empty() {
                        Ok(Some(FaceProcessResult {
                            embedding: Self::normalize_embedding(res.embedding),
                            liveness_score: res.liveness_score,
                            is_live: res.is_live,
                            provider: res.execution_provider,
                        }))
                    } else {
                        Ok(None)
                    }
                }
                Err(e) => Err(Status::internal(format!("Vision service error: {}", e))),
            },
        }
    }

    async fn process_voice_embedding(&self, voice: &[u8]) -> Result<Option<Vec<f32>>, Status> {
        let mut client = AudioClient::new(self.audio_channel.clone());
        let request = Request::new(shared::biometry::AudioChunk {
            content: voice.to_vec(),
            sample_rate: 16000,
        });

        match timeout(self.audio_rpc_timeout, client.process_voice(request)).await {
            Err(_) => Err(Status::deadline_exceeded(format!(
                "Audio service timeout after {} ms",
                self.audio_rpc_timeout.as_millis()
            ))),
            Ok(result) => match result {
                Ok(response) => {
                    let bio_result = response.into_inner();
                    if bio_result.detected && !bio_result.embedding.is_empty() {
                        Ok(Some(Self::normalize_embedding(bio_result.embedding)))
                    } else {
                        Ok(None)
                    }
                }
                Err(e) => Err(Status::internal(format!("Audio service error: {}", e))),
            },
        }
    }

    async fn log_access(&self, user_id: Option<i32>, device_id: i32, granted: bool, details: &str) {
        let _ = sqlx::query(
            "INSERT INTO access_logs (user_id, device_id, granted, details) VALUES ($1, $2, $3, $4)",
        )
        .bind(user_id)
        .bind(device_id)
        .bind(granted)
        .bind(details)
        .execute(&self.pool)
        .await
        .map_err(|e| tracing::error!("Failed to write access log: {}", e));
    }

    fn door_controller_audit_path() -> String {
        std::env::var("DOOR_CONTROLLER_LOG_PATH")
            .unwrap_or_else(|_| "/tmp/door_controller.log".to_string())
    }

    fn door_open_ms() -> u64 {
        env_parse::<u64>("DOOR_OPEN_MS", 5000).max(200)
    }

    async fn append_door_controller_line(path: String, line: String) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
        {
            let _ = file.write_all(line.as_bytes()).await;
        }
    }

    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_millis(0))
            .as_millis() as i64
    }

    async fn room_lock_device(&self, room_id: i32) -> Result<Option<(i32, String)>, Status> {
        let row = sqlx::query(
            r#"
            SELECT d.id, d.connection_string
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            WHERE d.room_id = $1 AND dt.name = 'lock'
            LIMIT 1
            "#,
        )
        .bind(room_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("Failed to load room lock: {}", e)))?;
        Ok(row.map(|r| (r.get("id"), r.get("connection_string"))))
    }

    fn emit_lock_sequence(lock_path: String, lock_device_id: i32, command: String) {
        let audit_path = Self::door_controller_audit_path();
        let open_ms = Self::door_open_ms();
        tokio::spawn(async move {
            if let Some(parent) = std::path::Path::new(&lock_path).parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let start = Self::now_ms();
            let _ = tokio::fs::write(&lock_path, "OPEN\n").await;
            let open_line =
                format!("{start};lock_device={lock_device_id};path={lock_path};command={command};state=OPEN\n");
            Self::append_door_controller_line(audit_path.clone(), open_line).await;
            sleep(Duration::from_millis(open_ms)).await;
            let _ = tokio::fs::write(&lock_path, "CLOSE\n").await;
            let end = Self::now_ms();
            let close_line =
                format!("{end};lock_device={lock_device_id};path={lock_path};state=CLOSE\n");
            Self::append_door_controller_line(audit_path, close_line).await;
        });
    }

    fn should_auto_open_lock(&self) -> bool {
        let enabled = env_parse::<i32>("DOOR_AUTO_OPEN_ON_GRANT", 1) != 0;
        if !enabled {
            return false;
        }
        let warmup_ms = env_parse::<i64>("DOOR_OPEN_STARTUP_WARMUP_MS", 15000).max(0);
        let up_ms = Self::now_ms() - self.started_at_ms;
        up_ms >= warmup_ms
    }

    async fn get_default_test_device(&self) -> Device {
        Device {
            id: self.test_device_id,
            room_id: 0,
            name: "Default Camera".to_string(),
            device_type: "camera".to_string(),
            connection_string: "0".to_string(), // Webcam 0
            is_active: true,
        }
    }

    fn map_stage_to_v2(stage: &str, granted: bool) -> i32 {
        if granted {
            return 7; // ACCESS_STAGE_GRANTED
        }
        match stage {
            "presence" | "face_detection" => 1,
            "liveness" => 3,
            "face_match" => 4,
            "voice_detection" | "voice_match" | "audio_warning" | "audio_error" => 5,
            "access_rules" | "device_config" => 6,
            _ => 8,
        }
    }

    fn clip_motion_score(frames: &[Vec<u8>]) -> f32 {
        if frames.len() < 2 {
            return 1.0;
        }
        let mut total = 0.0f32;
        let mut count = 0usize;
        for pair in frames.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            let min_len = a.len().min(b.len());
            if min_len == 0 {
                continue;
            }
            let mut diff_sum: u64 = 0;
            for i in 0..min_len {
                diff_sum += (a[i] as i32 - b[i] as i32).unsigned_abs() as u64;
            }
            let norm = diff_sum as f32 / (min_len as f32 * 255.0);
            total += norm;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    async fn pick_best_clip_frame(&self, frames: &[Vec<u8>]) -> ClipSelectionSummary {
        let mut flags = Vec::new();
        let mut best: Option<ClipFrameAssessment> = None;
        let mut detected_count = 0usize;
        let mut live_count = 0usize;
        let mut live_scores: Vec<f32> = Vec::new();

        for (index, frame) in frames.iter().enumerate() {
            match self.process_face_full(frame).await {
                Ok(Some(face)) => {
                    detected_count += 1;
                    if face.is_live {
                        live_count += 1;
                    }
                    live_scores.push(face.liveness_score);
                    let candidate = ClipFrameAssessment {
                        index,
                        detected: true,
                        is_live: face.is_live,
                        liveness_score: face.liveness_score,
                    };
                    best = match best {
                        None => Some(candidate),
                        Some(current_best) => {
                            let current_score = (if current_best.detected { 1.0 } else { 0.0 })
                                + (if current_best.is_live { 2.0 } else { 0.0 })
                                + current_best.liveness_score;
                            let next_score = (if candidate.detected { 1.0 } else { 0.0 })
                                + (if candidate.is_live { 2.0 } else { 0.0 })
                                + candidate.liveness_score;
                            if next_score > current_score {
                                Some(candidate)
                            } else {
                                Some(current_best)
                            }
                        }
                    };
                }
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!("clip frame {} face pre-check failed: {}", index, err);
                    flags.push("frame_precheck_error".to_string());
                }
            }
        }

        live_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_liveness = if live_scores.is_empty() {
            0.0
        } else {
            live_scores[live_scores.len() / 2]
        };
        let detected_ratio = detected_count as f32 / frames.len().max(1) as f32;
        let live_ratio = live_count as f32 / frames.len().max(1) as f32;

        let has_best = best.is_some();
        let best_idx = best.as_ref().map(|f| f.index).unwrap_or(frames.len() / 2);
        if !has_best {
            flags.push("clip_no_face_precheck_fallback_middle_frame".to_string());
        } else if frames.len() > 1 {
            flags.push("clip_mode_best_frame_selected".to_string());
        }

        ClipSelectionSummary {
            best_idx,
            flags,
            median_liveness,
            live_ratio,
            detected_ratio,
        }
    }

    fn detect_hardware_profile() -> HardwareProfile {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let cpu_cores = (cpu_threads / 2).max(1);
        let gpu_available = PathBuf::from("/dev/nvidia0").exists();
        HardwareProfile {
            cpu_cores,
            cpu_threads,
            gpu_available,
        }
    }

    fn env_threads_or_default(var: &str, default_value: usize) -> usize {
        env_parse::<usize>(var, default_value).max(1)
    }

    fn build_runtime_plan(mode: &str, hw: &HardwareProfile) -> RuntimePlan {
        let auto_gpu = mode == "auto" && hw.gpu_available;

        let default_vision = ((hw.cpu_cores + 1) / 2).max(1);
        let default_audio = (hw.cpu_cores.saturating_sub(default_vision)).max(1);
        let vision_threads = Self::env_threads_or_default("VISION_INTRA_THREADS", default_vision);
        let audio_threads = Self::env_threads_or_default("AUDIO_INTRA_THREADS", default_audio);

        if (mode == "gpu" || auto_gpu) && hw.gpu_available {
            RuntimePlan {
                mode: "gpu".to_string(),
                vision_threads,
                audio_threads,
                force_cpu: false,
                use_cuda: true,
            }
        } else {
            RuntimePlan {
                mode: "cpu".to_string(),
                vision_threads,
                audio_threads,
                force_cpu: true,
                use_cuda: false,
            }
        }
    }

    async fn persist_runtime_mode(plan: &RuntimePlan) -> Result<(), Status> {
        let root = std::env::var("RUNTIME_ROOT_DIR").unwrap_or_else(|_| ".".to_string());
        let env_path = PathBuf::from(root).join(".server_runtime.env");
        let content = format!(
            "# Saved by gateway ApplyRuntimeMode\nRUNTIME_MODE={}\nVISION_FORCE_CPU={}\nAUDIO_FORCE_CPU={}\nAUDIO_USE_CUDA={}\nVISION_CUDA_MEM_LIMIT_MB=1024\nAUDIO_CUDA_MEM_LIMIT_MB=256\nVISION_INTRA_THREADS={}\nAUDIO_INTRA_THREADS={}\nVISION_INTER_THREADS={}\nAUDIO_INTER_THREADS={}\nOPENBLAS_NUM_THREADS={}\n",
            plan.mode,
            if plan.force_cpu { 1 } else { 0 },
            if plan.force_cpu { 1 } else { 0 },
            if plan.use_cuda { 1 } else { 0 },
            plan.vision_threads,
            plan.audio_threads,
            std::env::var("VISION_INTER_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1),
            std::env::var("AUDIO_INTER_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1),
            std::env::var("OPENBLAS_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1),
        );
        tokio::fs::write(env_path, content)
            .await
            .map_err(|e| Status::internal(format!("Failed to persist runtime mode: {}", e)))
    }

    async fn restart_stack(&self) -> Result<(), Status> {
        let root = std::env::var("RUNTIME_ROOT_DIR").unwrap_or_else(|_| ".".to_string());
        let cmd = "if [ -x ./stop_docker.sh ] && [ -x ./start_docker.sh ]; then ./stop_docker.sh && ./start_docker.sh; else exit 1; fi";
        let output = Command::new("bash")
            .arg("-lc")
            .arg(cmd)
            .current_dir(root)
            .output()
            .await
            .map_err(|e| Status::internal(format!("Failed to execute restart scripts: {}", e)))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(Status::internal(format!(
                "Restart failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )))
        }
    }
}

#[tonic::async_trait]
impl Gatekeeper for GatewayService {
    // --- Пользователи ---
    async fn register_user(
        &self,
        request: Request<RegisterUserRequest>,
    ) -> Result<Response<IdResponse>, Status> {
        let req = request.into_inner();
        let req_images_count = req.images.len();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        // 1. Create User
        let row = sqlx::query("INSERT INTO users (name) VALUES ($1) RETURNING id")
            .bind(&req.name)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let user_id: i32 = row
            .try_get("id")
            .map_err(|e| Status::internal(e.to_string()))?;

        // 2. Process Images (require at least one valid face embedding)
        let mut face_samples: Vec<Vec<f32>> = Vec::new();
        for image_bytes in req.images {
            if let Ok(Some(embedding)) = self.process_face_full(&image_bytes).await {
                face_samples.push(embedding.embedding);
            }
        }
        let Some(face_mean) = Self::mean_normalized_embedding(face_samples) else {
            return Err(Status::invalid_argument(format!(
                "No valid face embedding extracted from {} image(s). Move closer to camera, improve light, and retry.",
                req_images_count
            )));
        };
        sqlx::query(
            "INSERT INTO face_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
        )
        .bind(user_id)
        .bind(Vector::from(face_mean))
        .execute(&mut *tx)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        // 3. Process Voices (optional, store averaged embedding when present)
        let mut voice_samples: Vec<Vec<f32>> = Vec::new();
        for voice_bytes in req.voices {
            if let Ok(Some(embedding)) = self.process_voice_embedding(&voice_bytes).await {
                voice_samples.push(embedding);
            }
        }
        if let Some(voice_mean) = Self::mean_normalized_embedding(voice_samples) {
            sqlx::query(
                "INSERT INTO voice_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
            )
            .bind(user_id)
            .bind(Vector::from(voice_mean))
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(IdResponse { id: user_id }))
    }

    async fn add_biometry(
        &self,
        request: Request<shared::biometry::AddBiometryRequest>,
    ) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let user_id = req.user_id;
        let req_images_count = req.images.len();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        // Process Images (if provided, require at least one valid face embedding)
        let mut face_samples: Vec<Vec<f32>> = Vec::new();
        for image_bytes in req.images {
            if let Ok(Some(embedding)) = self.process_face_full(&image_bytes).await {
                face_samples.push(embedding.embedding);
            }
        }
        if req_images_count > 0 {
            let Some(face_mean) = Self::mean_normalized_embedding(face_samples) else {
                return Err(Status::invalid_argument(format!(
                    "No valid face embedding extracted from {} image(s).",
                    req_images_count
                )));
            };
            sqlx::query(
                "INSERT INTO face_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
            )
            .bind(user_id)
            .bind(Vector::from(face_mean))
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        }

        // Process Voices (optional)
        let mut voice_samples: Vec<Vec<f32>> = Vec::new();
        for voice_bytes in req.voices {
            if let Ok(Some(embedding)) = self.process_voice_embedding(&voice_bytes).await {
                voice_samples.push(embedding);
            }
        }
        if let Some(voice_mean) = Self::mean_normalized_embedding(voice_samples) {
            sqlx::query(
                "INSERT INTO voice_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
            )
            .bind(user_id)
            .bind(Vector::from(voice_mean))
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(Empty {}))
    }

    async fn remove_user(&self, request: Request<IdRequest>) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        sqlx::query("DELETE FROM users WHERE id = $1")
            .bind(req.id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM users")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        if remaining == 0 {
            sqlx::query("SELECT setval(pg_get_serial_sequence('users', 'id'), 1, false)")
                .execute(&self.pool)
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
        }

        Ok(Response::new(Empty {}))
    }

    async fn list_users(
        &self,
        _request: Request<ListUsersRequest>,
    ) -> Result<Response<ListUsersResponse>, Status> {
        let rows = sqlx::query("SELECT id, name FROM users")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let users = rows
            .into_iter()
            .map(|r| User {
                id: r.get("id"),
                name: r.get("name"),
            })
            .collect();

        Ok(Response::new(ListUsersResponse { users }))
    }

    // --- Инфраструктура ---
    async fn add_zone(
        &self,
        request: Request<AddZoneRequest>,
    ) -> Result<Response<IdResponse>, Status> {
        let req = request.into_inner();
        let row = sqlx::query("INSERT INTO zones (name) VALUES ($1) RETURNING id")
            .bind(req.name)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let id: i32 = row
            .try_get("id")
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(IdResponse { id }))
    }

    async fn list_zones(
        &self,
        _request: Request<ListZonesRequest>,
    ) -> Result<Response<ListZonesResponse>, Status> {
        let rows = sqlx::query("SELECT id, name FROM zones")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let zones = rows
            .into_iter()
            .map(|r| Zone {
                id: r.get("id"),
                name: r.get("name"),
            })
            .collect();

        Ok(Response::new(ListZonesResponse { zones }))
    }

    async fn add_room(
        &self,
        request: Request<AddRoomRequest>,
    ) -> Result<Response<IdResponse>, Status> {
        let req = request.into_inner();
        let row = sqlx::query("INSERT INTO rooms (name, zone_id) VALUES ($1, $2) RETURNING id")
            .bind(req.name)
            .bind(req.zone_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let id: i32 = row
            .try_get("id")
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(IdResponse { id }))
    }

    async fn list_rooms(
        &self,
        _request: Request<ListRoomsRequest>,
    ) -> Result<Response<ListRoomsResponse>, Status> {
        let rows = sqlx::query("SELECT id, zone_id, name FROM rooms")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let rooms = rows
            .into_iter()
            .map(|r| Room {
                id: r.get("id"),
                zone_id: r.get("zone_id"),
                name: r.get("name"),
            })
            .collect();

        Ok(Response::new(ListRoomsResponse { rooms }))
    }

    async fn add_device(
        &self,
        request: Request<AddDeviceRequest>,
    ) -> Result<Response<IdResponse>, Status> {
        let req = request.into_inner();
        if req.device_type.trim().is_empty() {
            return Err(Status::invalid_argument("device_type must not be empty"));
        }
        if req.connection_string.trim().is_empty() {
            return Err(Status::invalid_argument(
                "connection_string must not be empty",
            ));
        }
        if req.device_type != "camera" && req.device_type != "lock" && req.device_type != "microphone" {
            return Err(Status::invalid_argument(
                "device_type must be one of: camera, lock, microphone",
            ));
        }
        let existing_same_type: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            WHERE d.room_id = $1 AND dt.name = $2
            "#,
        )
        .bind(req.room_id)
        .bind(&req.device_type)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("Failed checking room device limits: {}", e)))?;

        if existing_same_type >= 1 {
            return Err(Status::failed_precondition(format!(
                "Room {} already has device type '{}'; only one is allowed",
                req.room_id, req.device_type
            )));
        }

        let row = sqlx::query(
            r#"
            INSERT INTO devices (name, room_id, device_type_id, connection_string)
            VALUES ($1, $2, (SELECT id FROM device_types WHERE name = $3), $4)
            RETURNING id
            "#,
        )
        .bind(req.name)
        .bind(req.room_id)
        .bind(&req.device_type)
        .bind(&req.connection_string)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| {
            if e.to_string()
                .contains("null value in column \"device_type_id\"")
            {
                Status::invalid_argument("unknown device_type")
            } else {
                Status::internal(e.to_string())
            }
        })?;

        let id: i32 = row
            .try_get("id")
            .map_err(|e| Status::internal(e.to_string()))?;
        if req.device_type == "lock" {
            let lock_path = req.connection_string.clone();
            if let Some(parent) = std::path::Path::new(&lock_path).parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let _ = tokio::fs::write(lock_path, "CLOSE\n").await;
        }
        Ok(Response::new(IdResponse { id }))
    }

    async fn list_devices(
        &self,
        _request: Request<ListDevicesRequest>,
    ) -> Result<Response<ListDevicesResponse>, Status> {
        let rows = sqlx::query(
            r#"
            SELECT d.id, d.room_id, d.name, dt.name AS device_type, d.connection_string
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        let mut devices: Vec<Device> = rows
            .into_iter()
            .map(|r| {
                let id: i32 = r.get("id");
                let room_id: i32 = r.get("room_id");
                let name: String = r.get("name");
                let device_type: String = r.get("device_type");
                let connection_string: String = r.get("connection_string");
                Device {
                    id,
                    room_id,
                    name,
                    device_type,
                    connection_string,
                    is_active: true,
                }
            })
            .collect();

        if devices.is_empty() {
            devices.push(self.get_default_test_device().await);
        }

        Ok(Response::new(ListDevicesResponse { devices }))
    }

    async fn remove_device(&self, request: Request<IdRequest>) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let lock_row = sqlx::query(
            r#"
            SELECT d.connection_string, dt.name as device_type
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            WHERE d.id = $1
            LIMIT 1
            "#,
        )
        .bind(req.id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("remove_device lookup failed: {}", e)))?;
        sqlx::query("DELETE FROM devices WHERE id = $1")
            .bind(req.id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        if let Some(row) = lock_row {
            let dtype: String = row.get("device_type");
            if dtype == "lock" {
                let p: String = row.get("connection_string");
                let _ = tokio::fs::remove_file(p).await;
            }
        }
        Ok(Response::new(Empty {}))
    }

    // --- Контроль Доступа ---

    async fn grant_access(
        &self,
        request: Request<GrantAccessRequest>,
    ) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        sqlx::query(
            "INSERT INTO access_rules_zones (user_id, zone_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        )
        .bind(req.user_id)
        .bind(req.zone_id)
        .execute(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(Empty {}))
    }

    async fn set_access_rules(
        &self,
        request: Request<SetAccessRulesRequest>,
    ) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        sqlx::query("DELETE FROM access_rules_zones WHERE user_id = $1")
            .bind(req.user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        sqlx::query("DELETE FROM access_rules_rooms WHERE user_id = $1")
            .bind(req.user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        for zone_id in req.allowed_zone_ids {
            sqlx::query(
                "INSERT INTO access_rules_zones (user_id, zone_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            )
            .bind(req.user_id)
            .bind(zone_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        }

        for room_id in req.allowed_room_ids {
            sqlx::query(
                "INSERT INTO access_rules_rooms (user_id, room_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            )
            .bind(req.user_id)
            .bind(room_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(Empty {}))
    }

    async fn get_user_access(
        &self,
        request: Request<IdRequest>,
    ) -> Result<Response<GetUserAccessResponse>, Status> {
        let req = request.into_inner();
        let room_rows = sqlx::query("SELECT room_id FROM access_rules_rooms WHERE user_id = $1")
            .bind(req.id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let zone_rows = sqlx::query("SELECT zone_id FROM access_rules_zones WHERE user_id = $1")
            .bind(req.id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let room_ids = room_rows.into_iter().map(|r| r.get("room_id")).collect();
        let zone_ids = zone_rows.into_iter().map(|r| r.get("zone_id")).collect();

        Ok(Response::new(GetUserAccessResponse {
            allowed_room_ids: room_ids,
            allowed_zone_ids: zone_ids,
        }))
    }

    // --- MAIN IDENTIFICATION LOGIC ---
    async fn check_access(
        &self,
        request: Request<CheckAccessRequest>,
    ) -> Result<Response<AccessCheckResponse>, Status> {
        let req = request.into_inner();
        let device_id = req.device_id;

        let mut response = AccessCheckResponse {
            granted: false,
            user_name: "Unknown".into(),
            message: String::new(),
            face_detected: false,
            face_live: false,
            face_liveness_score: 0.0,
            face_distance: 1.0,
            face_match: false,
            voice_provided: !req.audio.is_empty(),
            voice_detected: false,
            voice_distance: 1.0,
            voice_match: false,
            final_confidence: 0.0,
            decision_stage: "start".into(),
        };

        // 1) Face + liveness
        let face_res_opt = match self.process_face_full(&req.image).await {
            Ok(v) => v,
            Err(e) => {
                response.message = format!("Vision backend error: {}", e.message());
                response.decision_stage = "vision_error".into();
                self.log_access(
                    None,
                    device_id,
                    false,
                    &format!("DENIED [VisionError] {}", response.message),
                )
                .await;
                return Ok(Response::new(response));
            }
        };
        let face_info = match face_res_opt {
            Some(info) => info,
            None => {
                response.message = "No face detected".into();
                response.decision_stage = "face_detection".into();
                self.log_access(None, device_id, false, "DENIED [Face] No face detected")
                    .await;
                return Ok(Response::new(response));
            }
        };

        response.face_detected = true;
        response.face_live = face_info.is_live;
        response.face_liveness_score = face_info.liveness_score;

        let min_face_liveness = env_parse_f32_clamped("ACCESS_MIN_FACE_LIVENESS", 0.90, 0.0, 1.0);
        if !response.face_live || response.face_liveness_score < min_face_liveness {
            response.user_name = "Unknown".into();
            response.message = format!(
                "Liveness failed: {:.2}% (< {:.2}) ({})",
                response.face_liveness_score * 100.0,
                min_face_liveness * 100.0,
                face_info.provider
            );
            response.decision_stage = "liveness".into();
            self.log_access(
                None,
                device_id,
                false,
                &format!("DENIED [Spoofing] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }

        // 2) Face identification
        let user_matches = sqlx::query(
            r#"
            SELECT u.id, u.name, (e.embedding <=> $1) as distance
            FROM face_embeddings e
            JOIN users u ON e.user_id = u.id
            ORDER BY distance ASC
            LIMIT 2
            "#,
        )
        .bind(Vector::from(face_info.embedding))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("DB Error: {}", e)))?;
        if user_matches.is_empty() {
            response.user_name = "Unknown".into();
            response.message = "No enrolled face embeddings in database".into();
            response.decision_stage = "face_match".into();
            self.log_access(None, device_id, false, "DENIED [Face] No enrolled face embeddings")
                .await;
            return Ok(Response::new(response));
        }

        let (user_id, user_name, face_dist) = if let Some(best) = user_matches.first() {
            let d: f64 = best.get::<Option<f64>, _>("distance").unwrap_or(1.0);
            let uid: i32 = best.get("id");
            let uname: String = best.get("name");
            (uid, uname, d)
        } else {
            (0, "Unknown".to_string(), 1.0)
        };

        let has_second = user_matches.len() > 1;
        let second_face_dist = user_matches
            .get(1)
            .and_then(|r| r.get::<Option<f64>, _>("distance"))
            .unwrap_or(1.0);
        let face_margin = second_face_dist - face_dist;
        let face_min_margin = env_parse::<f64>("FACE_MATCH_MIN_MARGIN", 0.18);

        response.face_distance = face_dist as f32;
        let strict_face_distance = env_parse::<f64>("ACCESS_STRICT_FACE_DISTANCE", 0.36);
        let single_strict_distance = env_parse::<f64>("FACE_SINGLE_STRICT_DISTANCE", 0.28);
        let effective_strict = if has_second {
            strict_face_distance
        } else {
            strict_face_distance.min(single_strict_distance)
        };
        let effective_threshold = if has_second {
            self.face_similarity_threshold
        } else {
            self.face_similarity_threshold.min(effective_strict)
        };
        let margin_ok = if has_second {
            face_margin >= face_min_margin
        } else {
            true
        };
        response.face_match = face_dist < effective_threshold
            && margin_ok
            && face_dist <= effective_strict;
        response.user_name = if response.face_match {
            user_name.clone()
        } else {
            "Unknown".to_string()
        };

        if !response.face_match {
            response.user_name = "Unknown".into();
            response.message = format!(
                "Face not matched (distance {:.3}, margin {:.3}, threshold {:.3})",
                face_dist, face_margin, self.face_similarity_threshold
            );
            response.decision_stage = "face_match".into();
            self.log_access(
                None,
                device_id,
                false,
                &format!("DENIED [Face] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }

        let face_conf = (1.0 - response.face_distance).clamp(0.0, 1.0);
        let mut voice_boost = 0.0f32;
        let mut voice_note: Option<String> = None;
        if response.voice_provided {
            match self.process_voice_embedding(&req.audio).await {
                Ok(Some(voice_emb)) => {
                    response.voice_detected = true;
                    let voice_match = sqlx::query(
                        "SELECT (embedding <=> $1) as distance FROM voice_embeddings WHERE user_id = $2 LIMIT 1",
                    )
                    .bind(Vector::from(voice_emb))
                    .bind(user_id)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| Status::internal(format!("DB voice error: {}", e)))?;

                    let voice_dist = voice_match
                        .and_then(|r| r.get::<Option<f64>, _>("distance"))
                        .unwrap_or(1.0);
                    response.voice_distance = voice_dist as f32;
                    response.voice_match = voice_dist < self.voice_similarity_threshold;
                    if response.voice_match {
                        voice_boost = 0.10;
                        voice_note = Some(format!("voice=match:{:.3}", voice_dist));
                    } else {
                        voice_note = Some(format!("voice=mismatch:{:.3}", voice_dist));
                    }
                }
                Ok(None) => {
                    response.voice_detected = false;
                    voice_note = Some("voice=not_detected".to_string());
                }
                Err(e) => {
                    response.voice_detected = false;
                    voice_note = Some(format!("voice=error:{}", e.message()));
                }
            }
        }
        response.final_confidence =
            (0.5 * response.face_liveness_score + 0.5 * face_conf + voice_boost).clamp(0.0, 1.0);
        if let Some(note) = voice_note {
            if response.message.is_empty() {
                response.message = note;
            } else {
                response.message = format!("{}; {}", response.message, note);
            }
        }

        let min_confidence = env_parse_f32_clamped("ACCESS_MIN_CONFIDENCE", 0.60, 0.0, 1.0);
        if response.final_confidence < min_confidence {
            response.message = format!(
                "Confidence too low ({:.2} < {:.2})",
                response.final_confidence, min_confidence
            );
            response.decision_stage = "confidence_gate".into();
            self.log_access(
                Some(user_id),
                device_id,
                false,
                &format!("DENIED [Confidence] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }

        // 4) Access policy check
        if device_id == self.test_device_id {
            response.granted = true;
            response.message = format!(
                "Welcome {} (test device, conf {:.2})",
                user_name, response.final_confidence
            );
            response.decision_stage = "granted".into();
            self.log_access(
                Some(user_id),
                device_id,
                true,
                "GRANTED [Test Device] multimodal",
            )
            .await;
            return Ok(Response::new(response));
        }

        let zone_rec = sqlx::query(
            r#"
            SELECT r.id as room_id, r.zone_id, z.name as zone_name, r.name as room_name
            FROM devices d
            JOIN rooms r ON d.room_id = r.id
            JOIN zones z ON r.zone_id = z.id
            WHERE d.id = $1
            "#,
        )
        .bind(device_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("DB Error finding zone: {}", e)))?;

        let (room_id, zone_id, zone_name, room_name) = match zone_rec {
            Some(z) => (
                z.get::<i32, _>("room_id"),
                z.get::<i32, _>("zone_id"),
                z.get::<String, _>("zone_name"),
                z.get::<String, _>("room_name"),
            ),
            None => {
                response.message = "Device not found/assigned".into();
                response.decision_stage = "device_config".into();
                self.log_access(
                    Some(user_id),
                    device_id,
                    false,
                    "DENIED [Config] device not assigned",
                )
                .await;
                return Ok(Response::new(response));
            }
        };

        if let Some(reason) = self.room_ready_reason(room_id).await? {
            response.message = reason;
            response.decision_stage = "device_config".into();
            self.log_access(
                Some(user_id),
                device_id,
                false,
                &format!("DENIED [Infra] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }
        let room_lock = self.room_lock_device(room_id).await?;

        let zone_rule_exists = sqlx::query(
            "SELECT 1 as exists FROM access_rules_zones WHERE user_id = $1 AND zone_id = $2",
        )
        .bind(user_id)
        .bind(zone_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        let room_rule_exists = sqlx::query(
            "SELECT 1 as exists FROM access_rules_rooms WHERE user_id = $1 AND room_id = $2",
        )
        .bind(user_id)
        .bind(room_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        if zone_rule_exists.is_some() || room_rule_exists.is_some() {
            response.granted = true;
            response.message = format!(
                "Welcome {} to {} / {} (conf {:.2})",
                user_name, zone_name, room_name, response.final_confidence
            );
            response.decision_stage = "granted".into();
            if self.should_auto_open_lock() {
                if let Some((lock_device_id, lock_path)) = room_lock {
                    Self::emit_lock_sequence(lock_path, lock_device_id, "open_once".to_string());
                }
            }
            self.log_access(
                Some(user_id),
                device_id,
                true,
                &format!(
                    "GRANTED [Zone: {}, Room: {}] face_dist={:.3} voice_dist={:.3} conf={:.2}",
                    zone_name,
                    room_name,
                    response.face_distance,
                    response.voice_distance,
                    response.final_confidence
                ),
            )
            .await;
        } else {
            response.granted = false;
            response.message = format!("Access denied to {} / {}", zone_name, room_name);
            response.decision_stage = "access_rules".into();
            self.log_access(
                Some(user_id),
                device_id,
                false,
                &format!("DENIED [Rules] Zone: {}, Room: {}", zone_name, room_name),
            )
            .await;
        }

        Ok(Response::new(response))
    }

    async fn get_logs(
        &self,
        request: Request<GetLogsRequest>,
    ) -> Result<Response<GetLogsResponse>, Status> {
        let req = request.into_inner();
        let limit = if req.limit <= 0 { 50 } else { req.limit } as i64;
        let offset = if req.offset < 0 { 0 } else { req.offset } as i64;

        let rows = sqlx::query(
            r#"
            SELECT 
                al.id, 
                al.created_at, 
                al.granted, 
                al.details,
                u.name as user_name,
                d.name as device_name,
                r.name as room_name,
                z.name as zone_name
            FROM access_logs al
            LEFT JOIN users u ON al.user_id = u.id
            LEFT JOIN devices d ON al.device_id = d.id
            LEFT JOIN rooms r ON d.room_id = r.id
            LEFT JOIN zones z ON r.zone_id = z.id
            ORDER BY al.created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        let logs = rows
            .into_iter()
            .map(|row| {
                let id: i32 = row.get("id");
                let timestamp: String = row
                    .get::<Option<chrono::DateTime<chrono::Utc>>, _>("created_at")
                    .map(|t| t.to_rfc3339())
                    .unwrap_or_default();
                let user_name: String = row
                    .get::<Option<String>, _>("user_name")
                    .unwrap_or_else(|| "Unknown".to_string());
                let room_name: String = row
                    .get::<Option<String>, _>("room_name")
                    .unwrap_or_default();
                let zone_name: String = row
                    .get::<Option<String>, _>("zone_name")
                    .unwrap_or_default();
                let access_granted: bool = row.get("granted");
                let details: String = row.get::<Option<String>, _>("details").unwrap_or_default();
                LogEntry {
                    id,
                    timestamp,
                    user_name,
                    room_name,
                    zone_name,
                    access_granted,
                    details,
                }
            })
            .collect();

        Ok(Response::new(GetLogsResponse { logs }))
    }

    async fn check_access_v2(
        &self,
        request: Request<CheckAccessRequestV2>,
    ) -> Result<Response<AccessCheckResponseV2>, Status> {
        let req = request.into_inner();
        if req.frames.is_empty() {
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 1,
                reason: "No frames provided".to_string(),
                confidence: 0.0,
                user_id: 0,
                user_name: "Unknown".to_string(),
                face_detected: false,
                face_quality: 0.0,
                face_live_score: 0.0,
                face_live: false,
                face_distance: 1.0,
                face_match: false,
                voice_provided: !req.audio.is_empty(),
                speech_present: !req.audio.is_empty(),
                voice_live_score: 0.0,
                voice_live: false,
                voice_distance: 1.0,
                voice_match: false,
                suspicious_audio: false,
                flags: vec!["no_frames".to_string()],
                session_id: req.session_id,
            }));
        }

        let mut flags = Vec::new();
        if !req.frame_timestamps_ms.is_empty() && req.frame_timestamps_ms.len() != req.frames.len()
        {
            flags.push("frame_timestamps_mismatch".to_string());
        }

        let clip = self.pick_best_clip_frame(&req.frames).await;
        let mut clip_flags = clip.flags;
        flags.append(&mut clip_flags);

        let motion_score = Self::clip_motion_score(&req.frames);
        let min_motion = env_parse::<f32>("ACCESS_CLIP_MIN_MOTION", 0.020);
        if req.frames.len() >= 2 && motion_score < min_motion {
            flags.push(format!("low_clip_motion:{:.4}", motion_score));
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 3,
                reason: format!(
                    "Clip motion too low ({:.4} < {:.4}), possible photo replay",
                    motion_score, min_motion
                ),
                confidence: 0.0,
                user_id: 0,
                user_name: "Unknown".to_string(),
                face_detected: clip.detected_ratio > 0.0,
                face_quality: clip.detected_ratio,
                face_live_score: clip.median_liveness,
                face_live: false,
                face_distance: 1.0,
                face_match: false,
                voice_provided: !req.audio.is_empty(),
                speech_present: !req.audio.is_empty(),
                voice_live_score: 0.0,
                voice_live: false,
                voice_distance: 1.0,
                voice_match: false,
                suspicious_audio: false,
                flags,
                session_id: req.session_id,
            }));
        }

        let min_clip_liveness =
            env_parse_f32_clamped("ACCESS_CLIP_MIN_MEDIAN_LIVENESS", 0.75, 0.0, 1.0);
        let min_clip_live_ratio =
            env_parse_f32_clamped("ACCESS_CLIP_MIN_LIVE_RATIO", 0.67, 0.0, 1.0);
        let min_clip_detected_ratio =
            env_parse_f32_clamped("ACCESS_CLIP_MIN_DETECTED_RATIO", 0.67, 0.0, 1.0);

        if clip.detected_ratio < min_clip_detected_ratio
            || clip.live_ratio < min_clip_live_ratio
            || clip.median_liveness < min_clip_liveness
        {
            flags.push(format!(
                "clip_liveness_gate:det={:.2},live={:.2},med={:.2}",
                clip.detected_ratio, clip.live_ratio, clip.median_liveness
            ));
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 3,
                reason: format!(
                    "Clip liveness gate failed (det {:.2}/{:.2}, live {:.2}/{:.2}, median {:.2}/{:.2})",
                    clip.detected_ratio,
                    min_clip_detected_ratio,
                    clip.live_ratio,
                    min_clip_live_ratio,
                    clip.median_liveness,
                    min_clip_liveness
                ),
                confidence: 0.0,
                user_id: 0,
                user_name: "Unknown".to_string(),
                face_detected: clip.detected_ratio > 0.0,
                face_quality: clip.detected_ratio,
                face_live_score: clip.median_liveness,
                face_live: false,
                face_distance: 1.0,
                face_match: false,
                voice_provided: !req.audio.is_empty(),
                speech_present: !req.audio.is_empty(),
                voice_live_score: 0.0,
                voice_live: false,
                voice_distance: 1.0,
                voice_match: false,
                suspicious_audio: false,
                flags,
                session_id: req.session_id,
            }));
        }

        let legacy_req = CheckAccessRequest {
            device_id: req.device_id,
            image: req.frames[clip.best_idx].clone(),
            audio: req.audio.clone(),
            audio_sample_rate: req.audio_sample_rate,
        };
        let legacy_resp = self
            .check_access(Request::new(legacy_req))
            .await?
            .into_inner();

        Ok(Response::new(AccessCheckResponseV2 {
            granted: legacy_resp.granted,
            stage: Self::map_stage_to_v2(&legacy_resp.decision_stage, legacy_resp.granted),
            reason: legacy_resp.message,
            confidence: legacy_resp.final_confidence,
            user_id: 0,
            user_name: legacy_resp.user_name,
            face_detected: legacy_resp.face_detected,
            face_quality: if legacy_resp.face_detected { 1.0 } else { 0.0 },
            face_live_score: legacy_resp.face_liveness_score,
            face_live: legacy_resp.face_live,
            face_distance: legacy_resp.face_distance,
            face_match: legacy_resp.face_match,
            voice_provided: legacy_resp.voice_provided,
            speech_present: legacy_resp.voice_provided,
            voice_live_score: if legacy_resp.voice_detected { 1.0 } else { 0.0 },
            voice_live: legacy_resp.voice_detected,
            voice_distance: legacy_resp.voice_distance,
            voice_match: legacy_resp.voice_match,
            suspicious_audio: false,
            flags,
            session_id: req.session_id,
        }))
    }

    async fn get_system_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<SystemStatusResponse>, Status> {
        // Query Audio Worker
        let mut audio_client = AudioClient::new(self.audio_channel.clone());
        let audio_status = match audio_client.get_status(Empty {}).await {
            Ok(resp) => resp.into_inner(),
            Err(e) => ServiceStatus {
                online: false,
                device: "None".to_string(),
                message: format!("Error: {}", e.message()),
            },
        };

        // Query Vision Worker
        let mut vision_client = VisionClient::new(self.vision_channel.clone());
        let vision_status = match vision_client.get_status(Empty {}).await {
            Ok(resp) => resp.into_inner(),
            Err(e) => ServiceStatus {
                online: false,
                device: "Unknown".to_string(),
                message: format!("Error: {}", e.message()),
            },
        };

        let infra_states = self.room_infra_states().await.unwrap_or_default();
        let total_points = infra_states.len();
        let ready_points = infra_states.iter().filter(|s| s.ready).count();
        let mut infra_message = format!("points_ready={}/{}", ready_points, total_points);
        let mut bad_points = Vec::new();
        for s in infra_states.iter().filter(|s| !s.ready).take(5) {
            bad_points.push(format!(
                "{} / {}: {} (cam={}, lock={}, mic={})",
                s.zone_name, s.room_name, s.reason, s.cameras, s.locks, s.microphones
            ));
        }
        if !bad_points.is_empty() {
            infra_message = format!("{}; disabled={}", infra_message, bad_points.join(" | "));
        }

        Ok(Response::new(SystemStatusResponse {
            vision: Some(vision_status),
            audio: Some(audio_status),
            database: Some(ServiceStatus {
                online: true,
                device: "Postgres".to_string(),
                message: "Connected".to_string(),
            }),
            gateway: Some(ServiceStatus {
                online: true,
                device: "Local".to_string(),
                message: infra_message,
            }),
        }))
    }

    async fn shutdown(&self, _request: Request<Empty>) -> Result<Response<Empty>, Status> {
        tracing::info!("Shutdown signal received for gateway");
        // It's okay to ignore the error if the receiver is already dropped.
        let _ = self.shutdown_tx.send(()).await;
        Ok(Response::new(Empty {}))
    }

    async fn control_service(
        &self,
        request: Request<ControlServiceRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let req = request.into_inner();
        let service_name = req.service_name.to_lowercase();
        let action = req.action.to_lowercase();

        if service_name == "pipeline" || service_name == "system" {
            return match action.as_str() {
                "start" => {
                    Self::set_pipeline_running(true).await?;
                    Ok(Response::new(StatusResponse {
                        success: true,
                        message: "Pipeline started".to_string(),
                    }))
                }
                "stop" => {
                    Self::set_pipeline_running(false).await?;
                    Ok(Response::new(StatusResponse {
                        success: true,
                        message: "Pipeline stopped".to_string(),
                    }))
                }
                _ => Ok(Response::new(StatusResponse {
                    success: false,
                    message: "Use action=start|stop for pipeline".to_string(),
                })),
            };
        }

        if action != "shutdown" {
            return Ok(Response::new(StatusResponse {
                success: false,
                message: format!("Action '{}' not supported", action),
            }));
        }

        let (success, message) = match service_name.as_str() {
            "audio" => {
                let mut client = AudioClient::new(self.audio_channel.clone());
                if let Err(e) = client.shutdown(Request::new(Empty {})).await {
                    (false, format!("Failed to shutdown audio worker: {}", e))
                } else {
                    (true, "Audio worker shutdown signal sent".to_string())
                }
            }
            "vision" => {
                let mut client = VisionClient::new(self.vision_channel.clone());
                if let Err(e) = client.shutdown(Request::new(Empty {})).await {
                    (false, format!("Failed to shutdown vision worker: {}", e))
                } else {
                    (true, "Vision worker shutdown signal sent".to_string())
                }
            }
            "gateway" => {
                // To avoid deadlocking, we send the shutdown signal in a separate task
                let tx = self.shutdown_tx.clone();
                tokio::spawn(async move {
                    let _ = tx.send(()).await;
                });
                (true, "Gateway shutdown initiated".to_string())
            }
            _ => (false, format!("Service '{}' not recognized", service_name)),
        };

        Ok(Response::new(StatusResponse { success, message }))
    }

    async fn apply_runtime_mode(
        &self,
        request: Request<RuntimeModeRequest>,
    ) -> Result<Response<RuntimeModeResponse>, Status> {
        let req = request.into_inner();
        let requested_mode = req.mode.to_lowercase();
        if requested_mode != "cpu" && requested_mode != "gpu" && requested_mode != "auto" {
            return Err(Status::invalid_argument("mode must be auto, cpu or gpu"));
        }

        let hw = Self::detect_hardware_profile();
        let plan = Self::build_runtime_plan(&requested_mode, &hw);
        Self::persist_runtime_mode(&plan).await?;

        let provider = if plan.mode == "gpu" { "CUDA" } else { "CPU" };
        let mut message = format!(
            "Saved runtime mode '{}' using {} (vision_threads={}, audio_threads={}, cpu_cores={}, cpu_threads={}, gpu_available={})",
            plan.mode,
            provider,
            plan.vision_threads,
            plan.audio_threads,
            hw.cpu_cores,
            hw.cpu_threads,
            hw.gpu_available
        );

        if req.restart_services {
            match self.restart_stack().await {
                Ok(_) => message.push_str(", services restarted"),
                Err(e) => message.push_str(&format!(", restart skipped: {}", e.message())),
            }
        }

        Ok(Response::new(RuntimeModeResponse {
            success: true,
            message,
            saved_mode: plan.mode,
            cpu_cores: hw.cpu_cores as i32,
            cpu_threads: hw.cpu_threads as i32,
            gpu_available: hw.gpu_available,
            vision_threads: plan.vision_threads as i32,
            audio_threads: plan.audio_threads as i32,
        }))
    }

    async fn control_door(
        &self,
        request: Request<ControlDoorRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let req = request.into_inner();
        if req.device_id <= 0 {
            return Ok(Response::new(StatusResponse {
                success: false,
                message: "invalid device_id".to_string(),
            }));
        }
        let command = if req.command.trim().is_empty() {
            "open_once".to_string()
        } else {
            req.command
        };
        let row = sqlx::query(
            r#"
            SELECT d.id, d.connection_string, dt.name AS device_type
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            WHERE d.id = $1
            LIMIT 1
            "#,
        )
        .bind(req.device_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("control_door query failed: {}", e)))?;
        let Some(lock_row) = row else {
            return Ok(Response::new(StatusResponse {
                success: false,
                message: format!("device {} not found", req.device_id),
            }));
        };
        let device_type: String = lock_row.get("device_type");
        if device_type != "lock" {
            return Ok(Response::new(StatusResponse {
                success: false,
                message: format!("device {} is not lock (type={})", req.device_id, device_type),
            }));
        }
        let lock_path: String = lock_row.get("connection_string");
        Self::emit_lock_sequence(lock_path, req.device_id, command.clone());
        Ok(Response::new(StatusResponse {
            success: true,
            message: format!(
                "Lock controller emulated for device {}: {}",
                req.device_id, command
            ),
        }))
    }

    async fn scan_hardware(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ScanHardwareResponse>, Status> {
        let mut found_devices: Vec<Device> = Vec::new();
        let mut diagnostics: Vec<String> = Vec::new();

        // 1. Get existing devices from DB
        let db_devices: Vec<Device> = sqlx::query_as(
            r#"
            SELECT d.id, d.room_id, d.name, dt.name AS device_type, d.connection_string
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .into_iter()
        .map(
            |(id, room_id, name, device_type, connection_string)| Device {
                id,
                room_id,
                name,
                device_type,
                connection_string,
                is_active: true,
            },
        )
        .collect();

        found_devices.extend(db_devices.clone());

        // 2. Scan server cameras (/dev/video*)
        if let Ok(paths) = glob("/dev/video*") {
            let mut cam_count = 0usize;
            for entry in paths {
                if let Ok(path) = entry {
                    cam_count += 1;
                    let path_str = path.to_string_lossy();
                    let conn_str = path_str.to_string();
                    if !db_devices
                        .iter()
                        .any(|d| d.device_type == "camera" && d.connection_string == conn_str)
                    {
                        found_devices.push(Device {
                            id: -1,
                            room_id: 0,
                            name: format!("Detected Camera {}", conn_str),
                            device_type: "camera".to_string(),
                            connection_string: conn_str,
                            is_active: true,
                        });
                    }
                }
            }
            if cam_count == 0 {
                diagnostics.push("no /dev/video* visible in container".to_string());
            }
        } else {
            diagnostics.push("glob /dev/video* failed".to_string());
        }

        // 3. Scan server microphones via arecord -L
        if let Ok(out) = Command::new("bash").arg("-lc").arg("arecord -L").output().await {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                let mut mic_count = 0usize;
                for (idx, line) in text.lines().enumerate() {
                    let conn = line.trim();
                    if conn.is_empty() || line.starts_with(' ') || conn.starts_with('#') {
                        continue;
                    }
                    if !db_devices
                        .iter()
                        .any(|d| d.device_type == "microphone" && d.connection_string == conn)
                    {
                        mic_count += 1;
                        found_devices.push(Device {
                            id: -1,
                            room_id: 0,
                            name: format!("Detected Mic {}", idx + 1),
                            device_type: "microphone".to_string(),
                            connection_string: conn.to_string(),
                            is_active: true,
                        });
                    }
                }
                if mic_count == 0 {
                    diagnostics.push("arecord -L: no microphone devices detected".to_string());
                }
            } else {
                diagnostics.push("arecord -L returned non-zero status".to_string());
            }
        } else {
            diagnostics.push("arecord binary not available in gateway container".to_string());
        }

        // 4. Scan lock files directory
        let locks_dir = std::env::var("LOCKS_DIR")
            .unwrap_or_else(|_| "/workspace/identification/locks".to_string());
        let _ = tokio::fs::create_dir_all(&locks_dir).await;
        if let Ok(mut rd) = tokio::fs::read_dir(&locks_dir).await {
            let mut lock_count = 0usize;
            while let Ok(Some(entry)) = rd.next_entry().await {
                let p = entry.path();
                if !p.is_file() {
                    continue;
                }
                let path_str = p.to_string_lossy().to_string();
                if !path_str.ends_with(".lock") {
                    continue;
                }
                lock_count += 1;
                if !db_devices
                    .iter()
                    .any(|d| d.device_type == "lock" && d.connection_string == path_str)
                {
                    let name = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("lock")
                        .to_string();
                    found_devices.push(Device {
                        id: -1,
                        room_id: 0,
                        name: format!("Detected Lock {}", name),
                        device_type: "lock".to_string(),
                        connection_string: path_str,
                        is_active: true,
                    });
                }
            }
            if lock_count == 0 {
                diagnostics.push(format!("no *.lock files in {}", locks_dir));
            }
        } else {
            diagnostics.push(format!("cannot read locks dir {}", locks_dir));
        }

        // Add default test camera if nothing found at all
        if found_devices.is_empty() {
            found_devices.push(self.get_default_test_device().await);
        }
        for (i, msg) in diagnostics.into_iter().enumerate() {
            found_devices.push(Device {
                id: -1000 - i as i32,
                room_id: 0,
                name: format!("Diagnostic {}", i + 1),
                device_type: "diagnostic".to_string(),
                connection_string: msg,
                is_active: true,
            });
        }

        Ok(Response::new(ScanHardwareResponse { found_devices }))
    }
}

async fn auto_configure_runtime_on_start() {
    let hw = GatewayService::detect_hardware_profile();
    let plan = GatewayService::build_runtime_plan("auto", &hw);
    match GatewayService::persist_runtime_mode(&plan).await {
        Ok(_) => {
            let provider = if plan.mode == "gpu" { "CUDA" } else { "CPU" };
            tracing::info!(
                "Runtime auto-configured on startup: mode={}, provider={}, cpu_cores={}, cpu_threads={}, vision_threads={}, audio_threads={}, gpu_available={}",
                plan.mode,
                provider,
                hw.cpu_cores,
                hw.cpu_threads,
                plan.vision_threads,
                plan.audio_threads,
                hw.gpu_available
            );
        }
        Err(e) => {
            tracing::warn!(
                "Failed to auto-configure runtime on startup: {}",
                e.message()
            );
        }
    }
}

async fn connect_db_with_retry(database_url: &str) -> Result<Pool<Postgres>, sqlx::Error> {
    let max_attempts = env_parse::<u32>("DB_CONNECT_MAX_ATTEMPTS", 30);
    let delay_ms = env_parse::<u64>("DB_CONNECT_RETRY_DELAY_MS", 1000);

    let mut last_err: Option<sqlx::Error> = None;
    for attempt in 1..=max_attempts {
        match PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
        {
            Ok(pool) => {
                tracing::info!("Database connected on attempt {}/{}", attempt, max_attempts);
                return Ok(pool);
            }
            Err(e) => {
                tracing::warn!(
                    "Database connection attempt {}/{} failed: {}",
                    attempt,
                    max_attempts,
                    e
                );
                last_err = Some(e);
                if attempt < max_attempts {
                    sleep(Duration::from_millis(delay_ms)).await;
                }
            }
        }
    }

    Err(last_err.expect("last database error should be set"))
}

fn env_duration_ms(name: &str, default_ms: u64) -> Duration {
    Duration::from_millis(env_parse::<u64>(name, default_ms))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    auto_configure_runtime_on_start().await;

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://user:password@localhost:5432/biometry_db".to_string());

    let pool = connect_db_with_retry(&database_url).await.map_err(|e| {
        tracing::error!("Could not connect to database after retries: {}", e);
        e
    })?;

    let addr = "0.0.0.0:50051".parse()?;

    let audio_url =
        std::env::var("AUDIO_URL").unwrap_or_else(|_| "http://127.0.0.1:50053".to_string());
    let vision_url =
        std::env::var("VISION_URL").unwrap_or_else(|_| "http://127.0.0.1:50052".to_string());

    tracing::info!(
        "Using backend endpoints: audio={}, vision={}",
        audio_url,
        vision_url
    );

    let audio_channel = tonic::transport::Endpoint::from_shared(audio_url.clone())?.connect_lazy();
    let vision_channel =
        tonic::transport::Endpoint::from_shared(vision_url.clone())?.connect_lazy();

    let face_similarity_threshold = std::env::var("FACE_SIMILARITY_THRESHOLD")
        .unwrap_or_else(|_| "0.36".to_string())
        .parse::<f64>()
        .expect("FACE_SIMILARITY_THRESHOLD must be a valid f64");

    let voice_similarity_threshold = std::env::var("VOICE_SIMILARITY_THRESHOLD")
        .unwrap_or_else(|_| "0.45".to_string())
        .parse::<f64>()
        .expect("VOICE_SIMILARITY_THRESHOLD must be a valid f64");

    let test_device_id = std::env::var("TEST_DEVICE_ID")
        .unwrap_or_else(|_| "9999".to_string())
        .parse::<i32>()
        .expect("TEST_DEVICE_ID must be a valid i32");

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);
    let vision_rpc_timeout = env_duration_ms("VISION_RPC_TIMEOUT_MS", 1500);
    let audio_rpc_timeout = env_duration_ms("AUDIO_RPC_TIMEOUT_MS", 2000);
    let started_at_ms = GatewayService::now_ms();

    let gateway = GatewayService::new(
        pool,
        audio_channel,
        vision_channel,
        face_similarity_threshold,
        voice_similarity_threshold,
        test_device_id,
        vision_rpc_timeout,
        audio_rpc_timeout,
        shutdown_tx,
        started_at_ms,
    );

    tracing::info!("Gateway Service listening on {}", addr);

    Server::builder()
        .add_service(GatekeeperServer::new(gateway))
        .serve_with_shutdown(addr, async {
            shutdown_rx.recv().await;
            tracing::info!("Gracefully shutting down Gateway Service");
        })
        .await?;

    Ok(())
}
