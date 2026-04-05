use glob::glob;
use image::{imageops::FilterType, GrayImage};
use pgvector::Vector;
use shared::biometry::door_agent_client::DoorAgentClient;
use shared::biometry::gatekeeper_server::{Gatekeeper, GatekeeperServer};
use shared::biometry::{
    AccessCheckResponse, AccessCheckResponseV2, AddDeviceRequest, AddRoomRequest, AddZoneRequest,
    BioResult, CheckAccessRequest, CheckAccessRequestV2, ControlDoorRequest, ControlServiceRequest, Device,
    DeviceMediaChunk, DeviceMediaRequest, Empty, GetLogsRequest, GetLogsResponse,
    GetUserAccessResponse, GrantAccessRequest, IdRequest, IdResponse, ImageFrame,
    ListDevicesRequest, ListDevicesResponse, ListRoomsRequest, ListRoomsResponse,
    ListUsersRequest, ListUsersResponse, ListZonesRequest, ListZonesResponse, LogEntry,
    RegisterUserRequest, Room, RuntimeModeRequest, RuntimeModeResponse, ScanHardwareResponse,
    ServiceStatus, SetAccessRulesRequest, StatusResponse, SystemStatusResponse, User, Zone,
    audio_client::AudioClient, vision_client::VisionClient,
};
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres, Row};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep, timeout};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    Request, Response, Status,
    transport::{Channel, Server},
};

// Детали распознавания лица и проверки живости.
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

struct ClipMotionAnalysis {
    global_motion: f32,
    spatial_std: f32,
    active_cell_ratio: f32,
    eye_motion: f32,
    mouth_motion: f32,
    motion_jitter: f32,
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
struct RuntimeProbeSummary {
    vision: ServiceStatus,
    audio: ServiceStatus,
}

struct CameraMjpegStream {
    child: Child,
    stdout: ChildStdout,
    stderr: ChildStderr,
    buffer: Vec<u8>,
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
    door_agent_channel: Channel,
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

fn env_parse_optional<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok().and_then(|v| v.parse::<T>().ok())
}

fn env_parse_f32_clamped(name: &str, default: f32, min: f32, max: f32) -> f32 {
    env_parse::<f32>(name, default).clamp(min, max)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(default)
}

fn normalize_person_name(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn validate_person_name(raw: &str) -> Result<String, Status> {
    let normalized = normalize_person_name(raw);
    if normalized.is_empty() {
        return Err(Status::invalid_argument("Введите ФИО"));
    }
    let parts: Vec<&str> = normalized.split(' ').collect();
    if parts.len() != 3 {
        return Err(Status::invalid_argument(
            "Введите ФИО полностью: Фамилия Имя Отчество",
        ));
    }
    for part in &parts {
        if part.starts_with('-') || part.ends_with('-') || part.contains("--") {
            return Err(Status::invalid_argument("ФИО содержит некорректные дефисы"));
        }
        let letters_count = part.chars().filter(|ch| ch.is_alphabetic()).count();
        if letters_count < 2 {
            return Err(Status::invalid_argument(
                "Каждая часть ФИО должна содержать минимум 2 буквы",
            ));
        }
        if !part.chars().all(|ch| ch.is_alphabetic() || ch == '-') {
            return Err(Status::invalid_argument(
                "ФИО может содержать только буквы, пробелы и дефис",
            ));
        }
    }
    Ok(parts
        .into_iter()
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    let mut out = String::new();
                    out.extend(first.to_uppercase());
                    out.push_str(&chars.as_str().to_lowercase());
                    out
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" "))
}

impl GatewayService {
    fn normalize_audio_pcm(raw: &[u8]) -> Vec<u8> {
        if raw.len() < 2 {
            return raw.to_vec();
        }
        let sample_count = raw.len() / 2;
        if sample_count == 0 {
            return raw.to_vec();
        }
        let mut pcm = Vec::with_capacity(sample_count);
        for chunk in raw.chunks_exact(2) {
            pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32);
        }
        if pcm.is_empty() {
            return raw.to_vec();
        }

        let mean = pcm.iter().sum::<f32>() / pcm.len() as f32;
        for s in &mut pcm {
            *s -= mean;
        }

        let rms = (pcm.iter().map(|v| v * v).sum::<f32>() / pcm.len().max(1) as f32).sqrt() / 32768.0;
        if rms < 0.004 {
            return raw[..sample_count * 2].to_vec();
        }

        let peak = pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if peak < 1.0 {
            return raw[..sample_count * 2].to_vec();
        }

        let target_peak = 24000.0f32;
        let gain = (target_peak / peak).min(8.0);
        let mut out = Vec::with_capacity(sample_count * 2);
        for s in pcm {
            let v = (s * gain).clamp(-32768.0, 32767.0) as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn runtime_plan_from_actual_status(status: &RuntimeProbeSummary) -> RuntimePlan {
        let vision_device = status.vision.device.to_uppercase();
        let audio_device = status.audio.device.to_uppercase();
        let actual_gpu = vision_device.contains("CUDA") || audio_device.contains("CUDA");
        let hw = Self::detect_hardware_profile();
        let requested_mode = if actual_gpu { "gpu" } else { "cpu" };
        Self::build_runtime_plan(requested_mode, &hw)
    }

    fn runtime_mode_from_env() -> String {
        std::env::var("RUNTIME_MODE")
            .unwrap_or_else(|_| "auto".to_string())
            .trim()
            .to_lowercase()
    }

    fn detect_gpu_runtime() -> bool {
        if PathBuf::from("/proc/driver/nvidia/version").exists()
            || PathBuf::from("/dev/nvidiactl").exists()
        {
            return true;
        }
        match std::process::Command::new("bash")
            .arg("-lc")
            .arg("command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1")
            .status()
        {
            Ok(status) => status.success(),
            Err(_) => false,
        }
    }

    fn physical_cores_from_lscpu() -> Option<usize> {
        use std::process::Command as StdCommand;
        let out = StdCommand::new("bash")
            .arg("-lc")
            .arg("lscpu | awk -F: '/^Core\\(s\\) per socket:/{gsub(/ /,\"\",$2); c=$2} /^Socket\\(s\\):/{gsub(/ /,\"\",$2); s=$2} END{if(c ~ /^[0-9]+$/ && s ~ /^[0-9]+$/) print c*s}'")
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        text.trim().parse::<usize>().ok().filter(|v| *v > 0)
    }

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

    async fn is_pipeline_running() -> bool {
        let path = Self::pipeline_run_flag_path();
        tokio::fs::read_to_string(path)
            .await
            .map(|s| s.trim() == "1")
            .unwrap_or(false)
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
        door_agent_channel: Channel,
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
            door_agent_channel,
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

    async fn process_face_bio(&self, image: &[u8]) -> Result<BioResult, Status> {
        let mut client = VisionClient::new(self.vision_channel.clone());
        let request = Request::new(ImageFrame {
            content: image.to_vec(),
        });

        match timeout(self.vision_rpc_timeout, client.process_face(request)).await {
            Err(_) => {
                // Легкий повтор, чтобы уменьшить ложные сбои на старте сервисов.
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
                    Ok(result) => result
                        .map(|response| response.into_inner())
                        .map_err(|e| Status::internal(format!("Vision service error: {}", e))),
                }
            }
            Ok(result) => result
                .map(|response| response.into_inner())
                .map_err(|e| Status::internal(format!("Vision service error: {}", e))),
        }
    }

    async fn process_face_full(&self, image: &[u8]) -> Result<Option<FaceProcessResult>, Status> {
        let res = self.process_face_bio(image).await?;
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

    async fn process_voice_embedding(&self, voice: &[u8]) -> Result<Option<Vec<f32>>, Status> {
        let mut client = AudioClient::new(self.audio_channel.clone());
        let normalized_voice = Self::normalize_audio_pcm(voice);
        let request = Request::new(shared::biometry::AudioChunk {
            content: normalized_voice,
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

    async fn log_v2_precheck_denied(&self, device_id: i32, reason: &str, _flags: &[String]) {
        self.log_access(
            None,
            device_id,
            false,
            &format!("ОТКАЗ [Предпроверка V2] {}", reason),
        )
        .await;
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
            connection_string: "0".to_string(), // Камера по умолчанию: /dev/video0
            is_active: true,
        }
    }

    fn camera_input_from_conn(conn: &str) -> String {
        if conn.starts_with("/dev/video")
            || conn.starts_with("rtsp://")
            || conn.starts_with("http://")
            || conn.starts_with("https://")
        {
            return conn.to_string();
        }
        if conn.chars().all(|c| c.is_ascii_digit()) {
            return format!("/dev/video{}", conn);
        }
        conn.to_string()
    }

    fn camera_lock_path(conn: &str) -> String {
        let normalized = Self::camera_input_from_conn(conn);
        let sanitized: String = normalized
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let compact = sanitized.trim_matches('_');
        let id = if compact.is_empty() { "camera" } else { compact };
        let lock_dir = std::env::var("CAMERA_LOCK_DIR")
            .unwrap_or_else(|_| "/workspace/identification/locks".to_string());
        format!("{}/biometry_cam_{}.lock", lock_dir.trim_end_matches('/'), id)
    }

    async fn open_camera_stream(conn: &str, target_fps: u32) -> Result<CameraMjpegStream, String> {
        let input = Self::camera_input_from_conn(conn);
        let mut cmd = if input.starts_with("/dev/video") {
            let mut command = Command::new("flock");
            command
                .arg("-w")
                .arg("2")
                .arg("-E")
                .arg("75")
                .arg(Self::camera_lock_path(conn))
                .arg("ffmpeg")
                .arg("-loglevel")
                .arg("error")
                .arg("-nostdin")
                .arg("-fflags")
                .arg("nobuffer")
                .arg("-flags")
                .arg("low_delay")
                .arg("-f")
                .arg("v4l2")
                .arg("-framerate")
                .arg(target_fps.to_string())
                .arg("-i")
                .arg(&input)
                .arg("-vf")
                .arg(format!("fps={target_fps}"))
                .arg("-f")
                .arg("image2pipe")
                .arg("-vcodec")
                .arg("mjpeg")
                .arg("-q:v")
                .arg("5")
                .arg("-");
            command
        } else {
            let mut command = Command::new("ffmpeg");
            command
                .arg("-loglevel")
                .arg("error")
                .arg("-nostdin")
                .arg("-fflags")
                .arg("nobuffer")
                .arg("-flags")
                .arg("low_delay")
                .arg("-threads")
                .arg("1");
            if input.starts_with("rtsp://") {
                command.arg("-rtsp_transport").arg("tcp");
            }
            command
                .arg("-i")
                .arg(&input)
                .arg("-vf")
                .arg(format!("fps={target_fps}"))
                .arg("-f")
                .arg("image2pipe")
                .arg("-vcodec")
                .arg("mjpeg")
                .arg("-q:v")
                .arg("5")
                .arg("-");
            command
        };
        let mut child = cmd
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg spawn failed: {}", e))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "ffmpeg stdout unavailable".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "ffmpeg stderr unavailable".to_string())?;
        Ok(CameraMjpegStream {
            child,
            stdout,
            stderr,
            buffer: Vec::with_capacity(256 * 1024),
        })
    }

    async fn read_camera_stderr(stderr: &mut ChildStderr) -> String {
        let mut buf = Vec::new();
        match timeout(Duration::from_millis(150), stderr.read_to_end(&mut buf)).await {
            Ok(Ok(_)) => String::from_utf8_lossy(&buf).trim().to_string(),
            _ => String::new(),
        }
    }

    async fn read_next_jpeg_frame(
        stream: &mut CameraMjpegStream,
        frame_timeout: Duration,
    ) -> Result<Vec<u8>, String> {
        let mut tmp = vec![0u8; 16 * 1024];
        loop {
            let start = stream.buffer.windows(2).position(|w| w == [0xFF, 0xD8]);
            if let Some(start_idx) = start {
                if let Some(rel_end) = stream.buffer[start_idx + 2..]
                    .windows(2)
                    .position(|w| w == [0xFF, 0xD9])
                {
                    let end_idx = start_idx + 2 + rel_end + 2;
                    let frame = stream.buffer[start_idx..end_idx].to_vec();
                    stream.buffer.drain(..end_idx);
                    return Ok(frame);
                }
                if start_idx > 0 {
                    stream.buffer.drain(..start_idx);
                }
            } else if stream.buffer.len() > 512 * 1024 {
                let keep = 64 * 1024;
                let drain_to = stream.buffer.len().saturating_sub(keep);
                if drain_to > 0 {
                    stream.buffer.drain(..drain_to);
                }
            }

            let n = match timeout(frame_timeout, stream.stdout.read(&mut tmp)).await {
                Ok(Ok(n)) => n,
                Ok(Err(e)) => return Err(format!("ffmpeg stream read failed: {}", e)),
                Err(_) => return Err("ffmpeg stream read timeout".to_string()),
            };
            if n == 0 {
                return match stream.child.try_wait() {
                    Ok(Some(status)) => {
                        if status.code() == Some(75) {
                            Err("camera device busy".to_string())
                        } else {
                            let stderr = Self::read_camera_stderr(&mut stream.stderr).await;
                            if stderr.is_empty() {
                                Err(format!("ffmpeg exited with status {}", status))
                            } else {
                                Err(format!("ffmpeg exited with status {}: {}", status, stderr))
                            }
                        }
                    }
                    Ok(None) => Err("ffmpeg stream closed unexpectedly".to_string()),
                    Err(e) => Err(format!("ffmpeg wait check failed: {}", e)),
                };
            }
            stream.buffer.extend_from_slice(&tmp[..n]);
        }
    }

    async fn capture_mic_raw(conn: &str, duration_s: f32, sample_rate: u32) -> Result<Vec<u8>, String> {
        let seconds = duration_s.clamp(0.15, 1.2);
        let sr = sample_rate.clamp(8000, 48000);
        let byte_count = ((sr as f32 * seconds * 2.0) as usize).max((sr as usize / 8) * 2);
        let output = Command::new("bash")
            .arg("-lc")
            .arg(format!(
                "arecord -q -D '{}' -r {} -c 1 -f S16_LE -t raw - | head -c {}",
                conn.replace('\'', ""),
                sr,
                byte_count
            ))
            .output()
            .await
            .map_err(|e| format!("arecord execution failed: {}", e))?;
        if !output.status.success() || output.stdout.is_empty() {
            let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(if err.is_empty() {
                "arecord returned empty chunk".to_string()
            } else {
                err
            });
        }
        Ok(output.stdout)
    }

    async fn resolve_stream_point(
        &self,
        device_id: i32,
    ) -> Result<(String, Option<String>), Status> {
        let cam_row = sqlx::query(
            r#"
            SELECT d.room_id, d.connection_string, dt.name AS device_type
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            WHERE d.id = $1
            LIMIT 1
            "#,
        )
        .bind(device_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("stream lookup failed: {}", e)))?;
        let Some(row) = cam_row else {
            return Err(Status::not_found(format!("device {} not found", device_id)));
        };
        let device_type: String = row.get("device_type");
        if device_type != "camera" {
            return Err(Status::failed_precondition(format!(
                "device {} is not camera (type={})",
                device_id, device_type
            )));
        }
        let room_id: i32 = row.get("room_id");
        let camera_conn: String = row.get("connection_string");

        let mic_row = sqlx::query(
            r#"
            SELECT d.connection_string
            FROM devices d
            JOIN device_types dt ON dt.id = d.device_type_id
            WHERE d.room_id = $1 AND dt.name = 'microphone'
            ORDER BY d.id ASC
            LIMIT 1
            "#,
        )
        .bind(room_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("stream microphone lookup failed: {}", e)))?;
        let mic_conn = mic_row.map(|r| r.get("connection_string"));
        Ok((camera_conn, mic_conn))
    }

    fn map_stage_to_v2(stage: &str, granted: bool) -> i32 {
        if granted {
            return 7;
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

    fn decode_clip_gray(frame: &[u8]) -> Option<GrayImage> {
        let img = image::load_from_memory(frame).ok()?.to_luma8();
        let (w, h) = img.dimensions();
        if w < 24 || h < 24 {
            return None;
        }
        let target_w = 160u32.min(w.max(64));
        let target_h = (((target_w as f32) * (h as f32 / w as f32)).round() as u32)
            .clamp(48, 180);
        Some(image::imageops::resize(
            &img,
            target_w,
            target_h,
            FilterType::Triangle,
        ))
    }

    fn region_mean_diff(
        a: &GrayImage,
        b: &GrayImage,
        x0f: f32,
        x1f: f32,
        y0f: f32,
        y1f: f32,
    ) -> f32 {
        let (w, h) = a.dimensions();
        if w == 0 || h == 0 || a.dimensions() != b.dimensions() {
            return 0.0;
        }
        let x0 = ((w as f32) * x0f).floor().max(0.0) as u32;
        let x1 = ((w as f32) * x1f).ceil().min(w as f32) as u32;
        let y0 = ((h as f32) * y0f).floor().max(0.0) as u32;
        let y1 = ((h as f32) * y1f).ceil().min(h as f32) as u32;
        if x1 <= x0 || y1 <= y0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for y in y0..y1 {
            for x in x0..x1 {
                let da = a.get_pixel(x, y).0[0] as f32 / 255.0;
                let db = b.get_pixel(x, y).0[0] as f32 / 255.0;
                sum += (da - db).abs();
                cnt += 1;
            }
        }
        if cnt == 0 { 0.0 } else { sum / cnt as f32 }
    }

    fn clip_motion_analysis(frames: &[Vec<u8>]) -> Option<ClipMotionAnalysis> {
        if frames.len() < 2 {
            return None;
        }
        let decoded: Vec<_> = frames
            .iter()
            .filter_map(|f| Self::decode_clip_gray(f))
            .collect();
        if decoded.len() < 2 {
            return None;
        }

        let mut pair_globals = Vec::new();
        let mut pair_spatial_std = Vec::new();
        let mut pair_active_ratio = Vec::new();
        let mut pair_eye_motion = Vec::new();
        let mut pair_mouth_motion = Vec::new();

        for pair in decoded.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if a.dimensions() != b.dimensions() {
                continue;
            }
            let (w, h) = a.dimensions();
            let gw = 6u32;
            let gh = 6u32;
            let cw = (w / gw).max(1);
            let ch = (h / gh).max(1);
            let mut cell_means = Vec::with_capacity((gw * gh) as usize);

            for gy in 0..gh {
                for gx in 0..gw {
                    let x0 = gx * cw;
                    let y0 = gy * ch;
                    let x1 = if gx + 1 >= gw { w } else { ((gx + 1) * cw).min(w) };
                    let y1 = if gy + 1 >= gh { h } else { ((gy + 1) * ch).min(h) };
                    if x1 <= x0 || y1 <= y0 {
                        continue;
                    }
                    let mut sum = 0.0f32;
                    let mut cnt = 0usize;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let da = a.get_pixel(x, y).0[0] as f32 / 255.0;
                            let db = b.get_pixel(x, y).0[0] as f32 / 255.0;
                            sum += (da - db).abs();
                            cnt += 1;
                        }
                    }
                    if cnt > 0 {
                        cell_means.push(sum / cnt as f32);
                    }
                }
            }
            if cell_means.is_empty() {
                continue;
            }
            let global = cell_means.iter().sum::<f32>() / cell_means.len() as f32;
            let var = cell_means
                .iter()
                .map(|v| {
                    let d = *v - global;
                    d * d
                })
                .sum::<f32>()
                / cell_means.len() as f32;
            let spatial_std = var.sqrt();
            let active_ratio = cell_means
                .iter()
                .filter(|v| **v > (global * 1.2).max(0.006))
                .count() as f32
                / cell_means.len() as f32;

            let eye_motion = Self::region_mean_diff(a, b, 0.20, 0.80, 0.18, 0.46);
            let mouth_motion = Self::region_mean_diff(a, b, 0.24, 0.76, 0.54, 0.84);

            pair_globals.push(global);
            pair_spatial_std.push(spatial_std);
            pair_active_ratio.push(active_ratio);
            pair_eye_motion.push(eye_motion);
            pair_mouth_motion.push(mouth_motion);
        }

        if pair_globals.is_empty() {
            return None;
        }
        let mean_global = pair_globals.iter().sum::<f32>() / pair_globals.len() as f32;
        let mean_spatial = pair_spatial_std.iter().sum::<f32>() / pair_spatial_std.len() as f32;
        let mean_active = pair_active_ratio.iter().sum::<f32>() / pair_active_ratio.len() as f32;
        let mean_eye = pair_eye_motion.iter().sum::<f32>() / pair_eye_motion.len() as f32;
        let mean_mouth = pair_mouth_motion.iter().sum::<f32>() / pair_mouth_motion.len() as f32;

        let jitter_var = pair_globals
            .iter()
            .map(|v| {
                let d = *v - mean_global;
                d * d
            })
            .sum::<f32>()
            / pair_globals.len() as f32;
        let motion_jitter = jitter_var.sqrt();

        Some(ClipMotionAnalysis {
            global_motion: mean_global,
            spatial_std: mean_spatial,
            active_cell_ratio: mean_active,
            eye_motion: mean_eye,
            mouth_motion: mean_mouth,
            motion_jitter,
        })
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
        let cpu_threads = env_parse_optional::<usize>("HOST_CPU_THREADS")
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        let cpu_cores = env_parse_optional::<usize>("HOST_PHYSICAL_CORES")
            .or_else(Self::physical_cores_from_lscpu)
            .unwrap_or_else(|| (cpu_threads / 2).max(1))
            .min(cpu_threads)
            .max(1);
        let gpu_available = Self::detect_gpu_runtime();
        HardwareProfile {
            cpu_cores,
            cpu_threads,
            gpu_available,
        }
    }

    fn env_threads_or_default(var: &str, default_value: usize) -> usize {
        env_parse::<usize>(var, default_value).max(1)
    }

    fn compute_worker_threads(capacity: usize) -> (usize, usize) {
        // Сначала резерв под gateway/db/door-agent, остальное делим между worker'ами.
        let reserve_system = if capacity <= 8 { 1 } else { 2 };
        let mut worker_budget = capacity.saturating_sub(reserve_system);
        if worker_budget < 2 {
            worker_budget = 2;
        }
        // Vision обычно тяжелее: отдаем ему примерно 75% физического бюджета.
        let mut vision = ((worker_budget * 3) + 1) / 4;
        if vision < 1 {
            vision = 1;
        }
        let mut audio = worker_budget.saturating_sub(vision);
        if audio < 1 {
            audio = 1;
        }
        (vision, audio)
    }

    fn build_runtime_plan(mode: &str, hw: &HardwareProfile) -> RuntimePlan {
        let auto_gpu = mode == "auto" && hw.gpu_available;
        let cpu_capacity = if mode == "cpu" {
            hw.cpu_threads.max(hw.cpu_cores)
        } else {
            hw.cpu_cores
        };
        let (default_vision, default_audio) = Self::compute_worker_threads(cpu_capacity);
        let worker_budget = (default_vision + default_audio).max(2);
        let (mut vision_threads, mut audio_threads) = if mode == "auto" {
            (default_vision.max(1), default_audio.max(1))
        } else {
            (
                Self::env_threads_or_default("VISION_INTRA_THREADS", default_vision).max(1),
                Self::env_threads_or_default("AUDIO_INTRA_THREADS", default_audio).max(1),
            )
        };
        if vision_threads + audio_threads > worker_budget {
            let overflow = vision_threads + audio_threads - worker_budget;
            if vision_threads >= audio_threads {
                vision_threads = vision_threads.saturating_sub(overflow).max(1);
            } else {
                audio_threads = audio_threads.saturating_sub(overflow).max(1);
            }
        }
        if vision_threads + audio_threads > worker_budget {
            audio_threads = worker_budget.saturating_sub(vision_threads).max(1);
        }
        if vision_threads + audio_threads > worker_budget {
            vision_threads = worker_budget.saturating_sub(audio_threads).max(1);
        }

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
        let managed_prefixes = [
            "RUNTIME_MODE=",
            "VISION_FORCE_CPU=",
            "AUDIO_FORCE_CPU=",
            "AUDIO_USE_CUDA=",
            "VISION_CUDA_MEM_LIMIT_MB=",
            "AUDIO_CUDA_MEM_LIMIT_MB=",
            "VISION_INTRA_THREADS=",
            "AUDIO_INTRA_THREADS=",
            "VISION_INTER_THREADS=",
            "AUDIO_INTER_THREADS=",
            "OPENBLAS_NUM_THREADS=",
            "VISION_CPUSET=",
            "AUDIO_CPUSET=",
            "GATEWAY_CPUSET=",
            "DOOR_AGENT_CPUSET=",
            "VISION_CPU_LIMIT=",
            "AUDIO_CPU_LIMIT=",
            "GATEWAY_CPU_LIMIT=",
            "DOOR_AGENT_CPU_LIMIT=",
        ];
        let preserved = tokio::fs::read_to_string(&env_path)
            .await
            .ok()
            .map(|content| {
                content
                    .lines()
                    .filter(|line| {
                        let trimmed = line.trim();
                        !managed_prefixes.iter().any(|prefix| trimmed.starts_with(prefix))
                            && trimmed != "# Saved by gateway ApplyRuntimeMode"
                    })
                    .map(|line| line.to_string())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        let mut content = format!(
            "# Saved by gateway ApplyRuntimeMode\nRUNTIME_MODE={}\nVISION_FORCE_CPU={}\nAUDIO_FORCE_CPU={}\nAUDIO_USE_CUDA={}\nVISION_CUDA_MEM_LIMIT_MB={}\nAUDIO_CUDA_MEM_LIMIT_MB={}\nVISION_INTRA_THREADS={}\nAUDIO_INTRA_THREADS={}\nVISION_INTER_THREADS={}\nAUDIO_INTER_THREADS={}\nOPENBLAS_NUM_THREADS={}\nVISION_CPUSET={}\nAUDIO_CPUSET={}\nGATEWAY_CPUSET={}\nDOOR_AGENT_CPUSET={}\nVISION_CPU_LIMIT={}\nAUDIO_CPU_LIMIT={}\nGATEWAY_CPU_LIMIT={}\nDOOR_AGENT_CPU_LIMIT={}\n",
            plan.mode,
            if plan.force_cpu { 1 } else { 0 },
            if plan.force_cpu { 1 } else { 0 },
            if plan.use_cuda { 1 } else { 0 },
            std::env::var("VISION_CUDA_MEM_LIMIT_MB").unwrap_or_else(|_| "1024".to_string()),
            std::env::var("AUDIO_CUDA_MEM_LIMIT_MB").unwrap_or_else(|_| "256".to_string()),
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
            std::env::var("VISION_CPUSET").unwrap_or_else(|_| "".to_string()),
            std::env::var("AUDIO_CPUSET").unwrap_or_else(|_| "".to_string()),
            std::env::var("GATEWAY_CPUSET").unwrap_or_else(|_| "".to_string()),
            std::env::var("DOOR_AGENT_CPUSET").unwrap_or_else(|_| "".to_string()),
            std::env::var("VISION_CPU_LIMIT").unwrap_or_else(|_| "".to_string()),
            std::env::var("AUDIO_CPU_LIMIT").unwrap_or_else(|_| "".to_string()),
            std::env::var("GATEWAY_CPU_LIMIT").unwrap_or_else(|_| "".to_string()),
            std::env::var("DOOR_AGENT_CPU_LIMIT").unwrap_or_else(|_| "".to_string()),
        );
        if !preserved.is_empty() {
            content.push_str(&preserved.join("\n"));
            content.push('\n');
        }
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

    async fn fetch_runtime_probe(
        audio_channel: Channel,
        vision_channel: Channel,
    ) -> Result<RuntimeProbeSummary, Status> {
        let mut audio_client = AudioClient::new(audio_channel);
        let audio = audio_client
            .get_status(Empty {})
            .await
            .map(|resp| resp.into_inner())
            .map_err(|e| Status::unavailable(format!("audio status unavailable: {}", e.message())))?;

        let mut vision_client = VisionClient::new(vision_channel);
        let vision = vision_client
            .get_status(Empty {})
            .await
            .map(|resp| resp.into_inner())
            .map_err(|e| Status::unavailable(format!("vision status unavailable: {}", e.message())))?;

        Ok(RuntimeProbeSummary { vision, audio })
    }

    async fn fetch_door_agent_preview_frame(
        door_agent_channel: Channel,
        device_id: i32,
    ) -> Result<DeviceMediaChunk, Status> {
        let mut client = DoorAgentClient::new(door_agent_channel);
        client
            .get_latest_preview_frame(IdRequest { id: device_id })
            .await
            .map(|resp| resp.into_inner())
            .map_err(|e| {
                Status::unavailable(format!(
                    "door-agent preview unavailable: {}",
                    e.message()
                ))
            })
    }
}

#[tonic::async_trait]
impl Gatekeeper for GatewayService {
    type StreamDeviceMediaStream = ReceiverStream<Result<DeviceMediaChunk, Status>>;

    // --- Пользователи ---
    async fn register_user(
        &self,
        request: Request<RegisterUserRequest>,
    ) -> Result<Response<IdResponse>, Status> {
        let req = request.into_inner();
        let req_images_count = req.images.len();
        let normalized_name = validate_person_name(&req.name)?;

        let existing_user_id = sqlx::query_scalar::<_, i32>(
            "SELECT id
             FROM users
             WHERE lower(regexp_replace(trim(name), '\\s+', ' ', 'g')) = lower($1)
             LIMIT 1",
        )
        .bind(&normalized_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        if let Some(user_id) = existing_user_id {
            return Err(Status::already_exists(format!(
                "Пользователь с таким ФИО уже существует (id={})",
                user_id
            )));
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        // 1) Создаем пользователя.
        let row = sqlx::query("INSERT INTO users (name) VALUES ($1) RETURNING id")
            .bind(&normalized_name)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let user_id: i32 = row
            .try_get("id")
            .map_err(|e| Status::internal(e.to_string()))?;

        // 2) Обрабатываем фото (нужен минимум один валидный вектор лица).
        let mut face_samples: Vec<Vec<f32>> = Vec::new();
        let mut rejected_reasons: Vec<String> = Vec::new();
        for image_bytes in req.images {
            match self.process_face_bio(&image_bytes).await {
                Ok(res) => {
                    if res.detected && !res.embedding.is_empty() {
                        face_samples.push(Self::normalize_embedding(res.embedding));
                    } else if !res.error_msg.trim().is_empty() {
                        rejected_reasons.push(res.error_msg.trim().to_string());
                    } else {
                        rejected_reasons.push("face_not_detected".to_string());
                    }
                }
                Err(e) => rejected_reasons.push(format!("vision_error: {}", e)),
            }
        }
        let Some(face_mean) = Self::mean_normalized_embedding(face_samples) else {
            let details = rejected_reasons
                .into_iter()
                .take(3)
                .collect::<Vec<_>>()
                .join(" | ");
            return Err(Status::invalid_argument(format!(
                "No valid face embedding extracted from {} image(s). {}",
                req_images_count,
                if details.is_empty() {
                    "Move closer to camera, improve light, and retry.".to_string()
                } else {
                    format!("Reasons: {}", details)
                }
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

        // 3) Обрабатываем голос (опционально, сохраняем усредненный вектор).
        let mut voice_samples: Vec<Vec<f32>> = Vec::new();
        let voice_input_count = req.voices.len();
        for voice_bytes in req.voices {
            if let Ok(Some(embedding)) = self.process_voice_embedding(&voice_bytes).await {
                voice_samples.push(embedding);
            }
        }
        if voice_input_count > 0 && voice_samples.is_empty() {
            return Err(Status::invalid_argument(
                "Не удалось извлечь валидный голосовой эмбеддинг. Проверьте микрофон, речь и повторите запись.",
            ));
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

        // Обрабатываем фото (если переданы, нужен минимум один валидный вектор).
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

        // Обрабатываем голос (опционально).
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
        let room_ids: Vec<i32> = sqlx::query_scalar(
            "SELECT id FROM rooms WHERE zone_id = $1",
        )
        .bind(req.zone_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        for room_id in room_ids {
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
        sqlx::query("DELETE FROM access_rules_rooms WHERE user_id = $1")
            .bind(req.user_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

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

        let room_ids = room_rows.into_iter().map(|r| r.get("room_id")).collect();

        Ok(Response::new(GetUserAccessResponse {
            allowed_room_ids: room_ids,
        }))
    }

    // --- Основная логика идентификации ---
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

        // 1) Лицо + проверка живости
        let face_res_opt = match self.process_face_full(&req.image).await {
            Ok(v) => v,
            Err(e) => {
                response.message = format!("Ошибка vision backend: {}", e.message());
                response.decision_stage = "vision_error".into();
                self.log_access(
                    None,
                    device_id,
                    false,
                    &format!("ОТКАЗ [Ошибка vision] {}", response.message),
                )
                .await;
                return Ok(Response::new(response));
            }
        };
        let face_info = match face_res_opt {
            Some(info) => info,
            None => {
                response.message = "Лицо не обнаружено".into();
                response.decision_stage = "face_detection".into();
                self.log_access(None, device_id, false, "ОТКАЗ [Лицо] Лицо не обнаружено")
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
                "Проверка живости не пройдена: {:.2}% (< {:.2}) ({})",
                response.face_liveness_score * 100.0,
                min_face_liveness * 100.0,
                face_info.provider
            );
            response.decision_stage = "liveness".into();
            self.log_access(
                None,
                device_id,
                false,
                &format!("ОТКАЗ [Спуфинг] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }

        // 2) Идентификация по лицу
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
            response.message = "В базе нет зарегистрированных векторов лица".into();
            response.decision_stage = "face_match".into();
            self.log_access(
                None,
                device_id,
                false,
                "ОТКАЗ [Лицо] В базе нет зарегистрированных векторов лица",
            )
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
                "Лицо не совпало (дистанция {:.3}, отрыв {:.3}, порог {:.3})",
                face_dist, face_margin, self.face_similarity_threshold
            );
            response.decision_stage = "face_match".into();
            self.log_access(
                None,
                device_id,
                false,
                &format!("ОТКАЗ [Лицо] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }

        let face_conf = (1.0 - response.face_distance).clamp(0.0, 1.0);
        let voice_match_bonus =
            env_parse_f32_clamped("VOICE_MATCH_CONFIDENCE_BONUS", 0.10, 0.0, 0.30);
        let voice_mismatch_penalty =
            env_parse_f32_clamped("VOICE_MISMATCH_CONFIDENCE_PENALTY", 0.06, 0.0, 0.30);
        let voice_capture_penalty =
            env_parse_f32_clamped("VOICE_CAPTURE_CONFIDENCE_PENALTY", 0.04, 0.0, 0.20);
        let mut w_live = env_parse_f32_clamped("ACCESS_FACE_LIVE_WEIGHT", 0.40, 0.0, 1.0);
        let mut w_face = env_parse_f32_clamped("ACCESS_FACE_MATCH_WEIGHT", 0.35, 0.0, 1.0);
        let mut w_voice = env_parse_f32_clamped("ACCESS_VOICE_WEIGHT", 0.25, 0.0, 1.0);
        let w_sum = (w_live + w_face + w_voice).max(1e-6);
        w_live /= w_sum;
        w_face /= w_sum;
        w_voice /= w_sum;
        let mismatch_voice_factor =
            env_parse_f32_clamped("VOICE_MISMATCH_CONFIDENCE_FACTOR", 0.35, 0.0, 1.0);
        let mut voice_adjustment = 0.0f32;
        let mut voice_note: Option<String> = None;
        let mut voice_score = 0.0f32;
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
                    let voice_distance_display = voice_dist.clamp(0.0, 1.0) as f32;
                    response.voice_distance = voice_distance_display;
                    response.voice_match = voice_dist < self.voice_similarity_threshold;
                    let raw_voice_score = (1.0 - voice_distance_display).clamp(0.0, 1.0);
                    if response.voice_match {
                        voice_score = raw_voice_score;
                        voice_adjustment = voice_match_bonus;
                        voice_note = Some(format!("voice=match:{:.3}", voice_dist));
                    } else {
                        voice_score = raw_voice_score * mismatch_voice_factor;
                        voice_adjustment = -voice_mismatch_penalty;
                        voice_note = Some(format!("voice=mismatch:{:.3}", voice_dist));
                    }
                }
                Ok(None) => {
                    response.voice_detected = false;
                    voice_adjustment = -voice_capture_penalty;
                    voice_note = Some("voice=not_detected".to_string());
                }
                Err(e) => {
                    response.voice_detected = false;
                    voice_adjustment = -voice_capture_penalty;
                    voice_note = Some(format!("voice=error:{}", e.message()));
                }
            }
        }
        response.final_confidence = (w_live * response.face_liveness_score
            + w_face * face_conf
            + w_voice * voice_score
            + voice_adjustment)
            .clamp(0.0, 1.0);
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
                "Итоговая уверенность слишком низкая ({:.2} < {:.2})",
                response.final_confidence, min_confidence
            );
            response.decision_stage = "confidence_gate".into();
            self.log_access(
                Some(user_id),
                device_id,
                false,
                &format!("ОТКАЗ [Уверенность] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }

        // 4) Проверка правил доступа
        if device_id == self.test_device_id {
            response.granted = true;
            response.message = format!(
                "Доступ разрешен: {} (тестовое устройство, уверенность {:.2})",
                user_name, response.final_confidence
            );
            response.decision_stage = "granted".into();
            self.log_access(
                Some(user_id),
                device_id,
                true,
                "ДОСТУП РАЗРЕШЕН [Тестовое устройство] мультимодальная проверка",
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

        let (room_id, _zone_id, zone_name, room_name) = match zone_rec {
            Some(z) => (
                z.get::<i32, _>("room_id"),
                z.get::<i32, _>("zone_id"),
                z.get::<String, _>("zone_name"),
                z.get::<String, _>("room_name"),
            ),
            None => {
                response.message = "Устройство не найдено или не привязано к комнате".into();
                response.decision_stage = "device_config".into();
                self.log_access(
                    Some(user_id),
                    device_id,
                    false,
                    "ОТКАЗ [Конфигурация] Устройство не привязано к комнате",
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
                &format!("ОТКАЗ [Инфраструктура] {}", response.message),
            )
            .await;
            return Ok(Response::new(response));
        }
        let room_lock = self.room_lock_device(room_id).await?;

        let room_rule_exists = sqlx::query(
            "SELECT 1 as exists FROM access_rules_rooms WHERE user_id = $1 AND room_id = $2",
        )
        .bind(user_id)
        .bind(room_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        if room_rule_exists.is_some() {
            response.granted = true;
            response.message = format!(
                "Доступ разрешен: {} -> зона \"{}\", комната \"{}\" (уверенность {:.2})",
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
                    "ДОСТУП РАЗРЕШЕН | зона: {} | комната: {} | дистанция лица: {:.3} | дистанция голоса: {} | уверенность: {:.2}",
                    zone_name,
                    room_name,
                    response.face_distance,
                    if response.voice_provided && response.voice_detected {
                        format!("{:.3}", response.voice_distance)
                    } else if response.voice_provided {
                        "н/д".to_string()
                    } else {
                        "не использован".to_string()
                    },
                    response.final_confidence
                ),
            )
            .await;
        } else {
            response.granted = false;
            response.message = format!(
                "Доступ запрещен: нет прав для зоны \"{}\" / комнаты \"{}\"",
                zone_name, room_name
            );
            response.decision_stage = "access_rules".into();
            self.log_access(
                Some(user_id),
                device_id,
                false,
                &format!(
                    "ОТКАЗ [Права доступа] зона: {} | комната: {}",
                    zone_name, room_name
                ),
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
            self.log_v2_precheck_denied(req.device_id, "No frames provided", &["no_frames".to_string()])
                .await;
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

        let motion = Self::clip_motion_analysis(&req.frames);
        let min_global_motion = env_parse_f32_clamped("ACCESS_CLIP_MIN_GLOBAL_MOTION", 0.008, 0.0, 1.0);
        let min_spatial_std = env_parse_f32_clamped("ACCESS_CLIP_MIN_SPATIAL_STD", 0.003, 0.0, 1.0);
        let min_active_cells = env_parse_f32_clamped("ACCESS_CLIP_MIN_ACTIVE_CELL_RATIO", 0.18, 0.0, 1.0);
        let min_eye_motion = env_parse_f32_clamped("ACCESS_CLIP_MIN_EYE_MOTION", 0.004, 0.0, 1.0);
        let min_mouth_motion = env_parse_f32_clamped("ACCESS_CLIP_MIN_MOUTH_MOTION", 0.004, 0.0, 1.0);
        let min_motion_jitter = env_parse_f32_clamped("ACCESS_CLIP_MIN_JITTER", 0.0012, 0.0, 1.0);
        let motion_eps = env_parse_f32_clamped("ACCESS_CLIP_MOTION_EPS", 0.00015, 0.0, 0.01);
        let rigid_requires_low_micro = env_bool("ACCESS_CLIP_RIGID_REQUIRE_LOW_MICRO", true);

        let Some(motion) = motion else {
            flags.push("clip_decode_failed".to_string());
            self.log_v2_precheck_denied(
                req.device_id,
                "Unable to decode clip frames for temporal liveness checks",
                &flags,
            )
            .await;
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 3,
                reason: "Unable to decode clip frames for temporal liveness checks".to_string(),
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
        };

        flags.push(format!(
            "clip_motion:global={:.4},spatial={:.4},active={:.2},eye={:.4},mouth={:.4},jitter={:.4}",
            motion.global_motion,
            motion.spatial_std,
            motion.active_cell_ratio,
            motion.eye_motion,
            motion.mouth_motion,
            motion.motion_jitter
        ));

        if motion.global_motion + motion_eps < min_global_motion {
            flags.push(format!(
                "low_global_motion:{:.4}<{:.4}",
                motion.global_motion, min_global_motion
            ));
            let reason = format!(
                "Temporal liveness failed: global motion too low ({:.4} < {:.4})",
                motion.global_motion, min_global_motion
            );
            self.log_v2_precheck_denied(req.device_id, &reason, &flags).await;
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 3,
                reason,
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

        let low_micro_motion =
            motion.eye_motion + motion_eps < min_eye_motion && motion.mouth_motion + motion_eps < min_mouth_motion;
        if motion.spatial_std + motion_eps < min_spatial_std
            && motion.active_cell_ratio + motion_eps < min_active_cells
            && (!rigid_requires_low_micro || low_micro_motion)
        {
            flags.push(format!(
                "low_nonrigid_motion:spatial={:.4}<{:.4},active={:.2}<{:.2}",
                motion.spatial_std, min_spatial_std, motion.active_cell_ratio, min_active_cells
            ));
            let reason = "Temporal liveness failed: motion pattern too rigid across the face".to_string();
            self.log_v2_precheck_denied(req.device_id, &reason, &flags).await;
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 3,
                reason,
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

        if motion.eye_motion < min_eye_motion
            && motion.mouth_motion < min_mouth_motion
            && motion.motion_jitter < min_motion_jitter
        {
            flags.push(format!(
                "low_micro_motion:eye={:.4}<{:.4},mouth={:.4}<{:.4},jitter={:.4}<{:.4}",
                motion.eye_motion,
                min_eye_motion,
                motion.mouth_motion,
                min_mouth_motion,
                motion.motion_jitter,
                min_motion_jitter
            ));
            let reason =
                "Temporal liveness failed: not enough micro-movements in eye/mouth regions".to_string();
            self.log_v2_precheck_denied(req.device_id, &reason, &flags).await;
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 3,
                reason,
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
            let reason = format!(
                "Clip liveness gate failed (det {:.2}/{:.2}, live {:.2}/{:.2}, median {:.2}/{:.2})",
                clip.detected_ratio,
                min_clip_detected_ratio,
                clip.live_ratio,
                min_clip_live_ratio,
                clip.median_liveness,
                min_clip_liveness
            );
            self.log_v2_precheck_denied(req.device_id, &reason, &flags).await;
            return Ok(Response::new(AccessCheckResponseV2 {
                granted: false,
                stage: 3,
                reason,
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

    async fn stream_device_media(
        &self,
        request: Request<DeviceMediaRequest>,
    ) -> Result<Response<Self::StreamDeviceMediaStream>, Status> {
        let req = request.into_inner();
        if req.device_id <= 0 {
            return Err(Status::invalid_argument("device_id must be > 0"));
        }

        let (camera_conn, microphone_conn) = self.resolve_stream_point(req.device_id).await?;
        let target_fps = req.target_fps.clamp(1, 20) as u32;
        let frame_interval_ms = (1000u64 / target_fps as u64).max(50);
        let include_audio = req.include_audio;
        let sample_rate = if req.audio_sample_rate == 0 {
            16000
        } else {
            req.audio_sample_rate.clamp(8000, 48000)
        };
        let audio_chunk_ms = req.audio_chunk_ms.clamp(150, 1200);
        let audio_duration_s = audio_chunk_ms as f32 / 1000.0;
        let audio_capture_interval_ms = (audio_chunk_ms as i64 * 3).clamp(600, 2400);
        let door_agent_channel = self.door_agent_channel.clone();
        let prefer_door_preview = env_bool("PREFER_DOOR_AGENT_PREVIEW", true);

        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let stop_flag = Arc::new(AtomicBool::new(false));
            let shared_audio = Arc::new(Mutex::new(Vec::new()));
            let shared_message = Arc::new(Mutex::new(String::new()));

            if include_audio {
                match microphone_conn {
                    Some(conn) => {
                        let audio_stop_flag = Arc::clone(&stop_flag);
                        let audio_buffer = Arc::clone(&shared_audio);
                        let audio_message = Arc::clone(&shared_message);
                        tokio::spawn(async move {
                            let mut audio_warn_sent = false;
                            while !audio_stop_flag.load(Ordering::Relaxed) {
                                match Self::capture_mic_raw(&conn, audio_duration_s, sample_rate).await {
                                    Ok(data) => {
                                        if let Ok(mut chunk) = audio_buffer.lock() {
                                            *chunk = data;
                                        }
                                    }
                                    Err(e) => {
                                        if let Ok(mut chunk) = audio_buffer.lock() {
                                            chunk.clear();
                                        }
                                        if !audio_warn_sent {
                                            if let Ok(mut message) = audio_message.lock() {
                                                *message = format!("audio_capture_error:{e}");
                                            }
                                            audio_warn_sent = true;
                                        }
                                    }
                                }
                                sleep(Duration::from_millis(audio_capture_interval_ms as u64)).await;
                            }
                        });
                    }
                    None => {
                        if let Ok(mut message) = shared_message.lock() {
                            *message = "audio_capture_error:microphone_not_configured".to_string();
                        }
                    }
                }
            }

            let mut camera_stream: Option<CameraMjpegStream> = None;
            loop {
                if !Self::is_pipeline_running().await {
                    break;
                }
                if prefer_door_preview {
                    match Self::fetch_door_agent_preview_frame(
                        door_agent_channel.clone(),
                        req.device_id,
                    )
                    .await
                    {
                        Ok(mut chunk) => {
                            let message = if let Ok(mut pending) = shared_message.lock() {
                                std::mem::take(&mut *pending)
                            } else {
                                String::new()
                            };
                            if !message.is_empty() {
                                chunk.message = message;
                            }
                            if tx.send(Ok(chunk)).await.is_err() {
                                break;
                            }
                            sleep(Duration::from_millis(frame_interval_ms)).await;
                            continue;
                        }
                        Err(e) => {
                            if tx
                                .send(Ok(DeviceMediaChunk {
                                    device_id: req.device_id,
                                    timestamp_ms: Self::now_ms(),
                                    jpeg_frame: Vec::new(),
                                    audio: Vec::new(),
                                    audio_sample_rate: 0,
                                    message: format!("video_capture_error:{}", e.message()),
                                }))
                                .await
                                .is_err()
                            {
                                break;
                            }
                            sleep(Duration::from_millis(frame_interval_ms)).await;
                            continue;
                        }
                    }
                }
                if camera_stream.is_none() {
                    match Self::open_camera_stream(&camera_conn, target_fps).await {
                        Ok(stream) => {
                            camera_stream = Some(stream);
                        }
                        Err(e) => {
                            let reopen_delay_ms = if e.contains("camera device busy") {
                                env_parse::<u64>("CAMERA_BUSY_RETRY_MS", 2000)
                            } else {
                                250
                            };
                            if tx
                                .send(Ok(DeviceMediaChunk {
                                    device_id: req.device_id,
                                    timestamp_ms: Self::now_ms(),
                                    jpeg_frame: Vec::new(),
                                    audio: Vec::new(),
                                    audio_sample_rate: 0,
                                    message: format!("video_capture_error:{e}"),
                                }))
                                .await
                                .is_err()
                            {
                                break;
                            }
                            sleep(Duration::from_millis(reopen_delay_ms)).await;
                            continue;
                        }
                    }
                }

                let min_frame_timeout_ms = env_parse::<u64>("CAMERA_FRAME_TIMEOUT_MS", 5000);
                let frame_timeout =
                    Duration::from_millis((frame_interval_ms * 3).max(min_frame_timeout_ms));
                let frame = match Self::read_next_jpeg_frame(
                    camera_stream.as_mut().expect("camera stream initialized"),
                    frame_timeout,
                )
                .await
                {
                    Ok(data) => data,
                    Err(e) => {
                        camera_stream = None;
                        if tx
                            .send(Ok(DeviceMediaChunk {
                                device_id: req.device_id,
                                timestamp_ms: Self::now_ms(),
                                jpeg_frame: Vec::new(),
                                audio: Vec::new(),
                                audio_sample_rate: 0,
                                message: format!("video_capture_error:{e}"),
                            }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                        sleep(Duration::from_millis(250)).await;
                        continue;
                    }
                };

                let message = if let Ok(mut pending) = shared_message.lock() {
                    std::mem::take(&mut *pending)
                } else {
                    String::new()
                };
                let audio = if include_audio {
                    shared_audio
                        .lock()
                        .map(|chunk| chunk.clone())
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };

                if tx
                    .send(Ok(DeviceMediaChunk {
                        device_id: req.device_id,
                        timestamp_ms: Self::now_ms(),
                        jpeg_frame: frame,
                        audio,
                        audio_sample_rate: if include_audio { sample_rate } else { 0 },
                        message,
                    }))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            stop_flag.store(true, Ordering::Relaxed);
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn get_system_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<SystemStatusResponse>, Status> {
        // Опрос Audio Worker
        let mut audio_client = AudioClient::new(self.audio_channel.clone());
        let audio_status = match audio_client.get_status(Empty {}).await {
            Ok(resp) => resp.into_inner(),
            Err(e) => ServiceStatus {
                online: false,
                device: "None".to_string(),
                message: format!("Error: {}", e.message()),
            },
        };

        // Опрос Vision Worker
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
        // Ошибку можно игнорировать, если receiver уже завершен.
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
                // Чтобы не блокировать текущий обработчик, отправляем сигнал в отдельной задаче.
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

        // 1) Читаем уже зарегистрированные устройства из БД.
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

        // 2) Сканируем камеры сервера (/dev/video*).
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

        // 3) Сканируем микрофоны сервера через `arecord -L`.
        if let Ok(out) = Command::new("bash").arg("-lc").arg("arecord -L").output().await {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                let mut mic_count = 0usize;
                let mut entries: Vec<(String, Vec<String>)> = Vec::new();
                let mut current_conn: Option<String> = None;
                let mut current_desc: Vec<String> = Vec::new();
                for line in text.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }
                    if line.starts_with(' ') || line.starts_with('\t') {
                        if current_conn.is_some() {
                            current_desc.push(trimmed.to_string());
                        }
                        continue;
                    }
                    if let Some(conn) = current_conn.take() {
                        entries.push((conn, std::mem::take(&mut current_desc)));
                    }
                    current_conn = Some(trimmed.to_string());
                }
                if let Some(conn) = current_conn.take() {
                    entries.push((conn, current_desc));
                }

                for (idx, (conn, desc)) in entries.into_iter().enumerate() {
                    if !db_devices
                        .iter()
                        .any(|d| d.device_type == "microphone" && d.connection_string == conn)
                    {
                        mic_count += 1;
                        let display_name = desc
                            .first()
                            .cloned()
                            .unwrap_or_else(|| format!("Detected Mic {}", idx + 1));
                        found_devices.push(Device {
                            id: -1,
                            room_id: 0,
                            name: display_name,
                            device_type: "microphone".to_string(),
                            connection_string: conn,
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

        // 4) Сканируем каталог lock-файлов.
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

        // Если не найдено ни одного устройства, добавляем тестовую камеру.
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
    let requested_mode = GatewayService::runtime_mode_from_env();
    let plan = GatewayService::build_runtime_plan(&requested_mode, &hw);
    match GatewayService::persist_runtime_mode(&plan).await {
        Ok(_) => {
            let provider = if plan.mode == "gpu" { "CUDA" } else { "CPU" };
            tracing::info!(
                "Runtime configured on startup: requested_mode={}, applied_mode={}, provider={}, cpu_cores={}, cpu_threads={}, vision_threads={}, audio_threads={}, gpu_available={}",
                requested_mode,
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

async fn reconcile_runtime_after_worker_start(audio_channel: Channel, vision_channel: Channel) {
    let requested_mode = GatewayService::runtime_mode_from_env();
    let probe_attempts = env_parse::<u32>("RUNTIME_RECONCILE_ATTEMPTS", 20);
    let probe_delay_ms = env_parse::<u64>("RUNTIME_RECONCILE_DELAY_MS", 3000);

    for attempt in 1..=probe_attempts {
        match GatewayService::fetch_runtime_probe(audio_channel.clone(), vision_channel.clone()).await
        {
            Ok(status) if status.audio.online && status.vision.online => {
                let actual_plan = GatewayService::runtime_plan_from_actual_status(&status);
                match GatewayService::persist_runtime_mode(&actual_plan).await {
                    Ok(_) => {
                        if actual_plan.mode != requested_mode {
                            tracing::warn!(
                                "Runtime reconciled after startup self-test: requested_mode={}, actual_mode={}, vision_device={}, audio_device={}, vision_threads={}, audio_threads={}",
                                requested_mode,
                                actual_plan.mode,
                                status.vision.device,
                                status.audio.device,
                                actual_plan.vision_threads,
                                actual_plan.audio_threads
                            );
                        } else {
                            tracing::info!(
                                "Runtime reconcile confirmed requested_mode={} with vision_device={}, audio_device={}, vision_threads={}, audio_threads={}",
                                requested_mode,
                                status.vision.device,
                                status.audio.device,
                                actual_plan.vision_threads,
                                actual_plan.audio_threads
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        "Failed to reconcile runtime after startup self-test: {}",
                        e.message()
                    ),
                }
                return;
            }
            Ok(status) => {
                tracing::info!(
                    "Waiting for worker runtime probe ({}/{}): vision_online={}, audio_online={}",
                    attempt,
                    probe_attempts,
                    status.vision.online,
                    status.audio.online
                );
            }
            Err(err) => {
                tracing::info!(
                    "Worker runtime probe pending ({}/{}): {}",
                    attempt,
                    probe_attempts,
                    err.message()
                );
            }
        }

        sleep(Duration::from_millis(probe_delay_ms)).await;
    }

    tracing::warn!(
        "Runtime reconcile skipped: workers did not become probe-ready after {} attempts",
        probe_attempts
    );
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
    let door_agent_url =
        std::env::var("DOOR_AGENT_URL").unwrap_or_else(|_| "http://door-agent:50054".to_string());

    tracing::info!(
        "Using backend endpoints: audio={}, vision={}, door_agent={}",
        audio_url,
        vision_url,
        door_agent_url
    );

    let door_agent_channel =
        tonic::transport::Endpoint::from_shared(door_agent_url.clone())?.connect_lazy();
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
        door_agent_channel,
        audio_channel.clone(),
        vision_channel.clone(),
        face_similarity_threshold,
        voice_similarity_threshold,
        test_device_id,
        vision_rpc_timeout,
        audio_rpc_timeout,
        shutdown_tx,
        started_at_ms,
    );

    tokio::spawn(reconcile_runtime_after_worker_start(
        audio_channel.clone(),
        vision_channel.clone(),
    ));

    let pipeline_auto_start = env_bool("PIPELINE_AUTO_START", true);
    if let Err(e) = GatewayService::set_pipeline_running(pipeline_auto_start).await {
        tracing::warn!(
            "Не удалось установить флаг pipeline={} при старте: {}",
            pipeline_auto_start,
            e.message()
        );
    }

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
