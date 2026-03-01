use shared::biometry::door_agent_server::{DoorAgent, DoorAgentServer};
use shared::biometry::gatekeeper_client::GatekeeperClient;
use shared::biometry::{
    CheckAccessRequestV2, DoorAgentStatusResponse, DoorObservationRequest, DoorObservationResponse,
    DoorPipelineStage, Empty, ListDevicesRequest,
};
use std::collections::HashMap;
use tokio::process::Command;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tonic::{Request, Response, Status, transport::Server};

#[derive(Default)]
struct DeviceState {
    stage: i32,
    session_id: String,
    frames: Vec<Vec<u8>>,
    frame_ts: Vec<i64>,
    audio: Vec<u8>,
    audio_sample_rate: u32,
    last_seen_ms: i64,
    cooldown_until_ms: i64,
}

#[derive(Clone)]
struct DoorAgentService {
    states: Arc<Mutex<HashMap<i32, DeviceState>>>,
    gateway_addr: String,
    max_clip_frames: usize,
    presence_timeout_ms: i64,
    cooldown_ms: i64,
}

#[derive(Clone)]
struct RoomPointConfig {
    room_id: i32,
    camera_device_id: i32,
    camera_conn: String,
    microphone_conn: Option<String>,
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str, default: bool) -> bool {
    env_parse::<i32>(name, if default { 1 } else { 0 }) != 0
}

impl DoorAgentService {
    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_millis(0))
            .as_millis() as i64
    }

    fn build_pipeline_flags(result: &shared::biometry::AccessCheckResponseV2) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "step_presence:{}",
            if result.face_detected { "ok" } else { "fail" }
        ));
        out.push(format!(
            "step_liveness:{}:{:.2}",
            if result.face_live { "ok" } else { "fail" },
            result.face_live_score
        ));
        out.push(format!(
            "step_face_id:{}:{:.3}",
            if result.face_match { "ok" } else { "fail" },
            result.face_distance
        ));
        if result.voice_provided {
            out.push(format!(
                "step_voice_id:{}:{:.3}",
                if result.voice_match { "ok" } else { "warn" },
                result.voice_distance
            ));
        } else {
            out.push("step_voice_id:skip".to_string());
        }
        out.push(format!(
            "step_policy:{}",
            if result.granted { "ok" } else { "deny" }
        ));
        out.extend(result.flags.clone());
        out
    }

    fn mk_response(
        access_granted: bool,
        stage: i32,
        reason: impl Into<String>,
        confidence: f32,
        user_name: impl Into<String>,
        pending: bool,
        flags: Vec<String>,
    ) -> DoorObservationResponse {
        DoorObservationResponse {
            access_granted,
            stage,
            reason: reason.into(),
            confidence,
            user_name: user_name.into(),
            pending,
            flags,
        }
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_millis(0))
        .as_millis() as i64
}

fn camera_input_from_conn(conn: &str) -> String {
    if conn.starts_with("/dev/video") || conn.starts_with("rtsp://") || conn.starts_with("http://") || conn.starts_with("https://") {
        return conn.to_string();
    }
    if conn.chars().all(|c| c.is_ascii_digit()) {
        return format!("/dev/video{}", conn);
    }
    conn.to_string()
}

async fn capture_single_jpeg(conn: &str) -> Option<Vec<u8>> {
    let input = camera_input_from_conn(conn);
    let source_arg = if input.starts_with("/dev/video") {
        format!("-f v4l2 -i '{}'", input.replace('\'', ""))
    } else {
        format!("-i '{}'", input.replace('\'', ""))
    };
    let output = Command::new("bash")
        .arg("-lc")
        .arg(format!(
            "ffmpeg -loglevel error -y {} -frames:v 1 -f image2pipe -vcodec mjpeg -",
            source_arg
        ))
        .output()
        .await
        .ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    Some(output.stdout)
}

async fn capture_mic_raw(conn: &str, duration_s: f32) -> Option<Vec<u8>> {
    let sec = duration_s.max(0.2);
    let output = Command::new("bash")
        .arg("-lc")
        .arg(format!(
            "arecord -q -D '{}' -r 16000 -c 1 -f S16_LE -d {} -t raw -",
            conn.replace('\'', ""),
            sec.ceil() as i32
        ))
        .output()
        .await
        .ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    Some(output.stdout)
}

async fn discover_room_points(gateway_addr: &str) -> Result<Vec<RoomPointConfig>, Status> {
    let mut client = GatekeeperClient::connect(gateway_addr.to_string())
        .await
        .map_err(|e| Status::unavailable(format!("gateway connect failed: {e}")))?;
    let devices = client
        .list_devices(Request::new(ListDevicesRequest {}))
        .await
        .map_err(|e| Status::internal(format!("list_devices failed: {e}")))?
        .into_inner()
        .devices;

    let mut by_room: HashMap<i32, Vec<shared::biometry::Device>> = HashMap::new();
    for d in devices {
        by_room.entry(d.room_id).or_default().push(d);
    }

    let mut out = Vec::new();
    for (room_id, ds) in by_room {
        let cameras: Vec<_> = ds.iter().filter(|d| d.device_type == "camera").collect();
        let locks: Vec<_> = ds.iter().filter(|d| d.device_type == "lock").collect();
        let microphones: Vec<_> = ds.iter().filter(|d| d.device_type == "microphone").collect();
        if cameras.len() != 1 || locks.len() != 1 || microphones.len() > 1 {
            tracing::warn!(
                "room {} disabled in background mode: camera={}, lock={}, microphone={}",
                room_id,
                cameras.len(),
                locks.len(),
                microphones.len()
            );
            continue;
        }
        out.push(RoomPointConfig {
            room_id,
            camera_device_id: cameras[0].id,
            camera_conn: cameras[0].connection_string.clone(),
            microphone_conn: microphones.first().map(|d| d.connection_string.clone()),
        });
    }
    Ok(out)
}

async fn run_background_live_loop(gateway_addr: String) {
    let enabled = env_flag("DOOR_BACKGROUND_ENABLED", true);
    if !enabled {
        tracing::info!("door-agent background loop disabled by DOOR_BACKGROUND_ENABLED=0");
        return;
    }
    let frames_per_check = env_parse::<usize>("DOOR_BG_FRAMES", 3).max(1);
    let tick_ms = env_parse::<u64>("DOOR_BG_TICK_MS", 500).max(100);
    let cooldown_ms = env_parse::<i64>("DOOR_BG_COOLDOWN_MS", 2500).max(500);
    let use_microphone = env_flag("DOOR_BG_USE_MIC", true);
    let run_flag_path = std::env::var("SYSTEM_RUN_FLAG_PATH")
        .unwrap_or_else(|_| "/workspace/identification/.system_run".to_string());

    let mut next_allowed_ms: HashMap<i32, i64> = HashMap::new();
    loop {
        let running = tokio::fs::read_to_string(&run_flag_path)
            .await
            .map(|s| s.trim() == "1")
            .unwrap_or(false);
        if !running {
            tokio::time::sleep(Duration::from_millis(tick_ms)).await;
            continue;
        }
        match discover_room_points(&gateway_addr).await {
            Ok(points) => {
                for point in points {
                    let now = now_ms();
                    if now < *next_allowed_ms.get(&point.room_id).unwrap_or(&0) {
                        continue;
                    }
                    let mut frames = Vec::new();
                    let mut ts = Vec::new();
                    for _ in 0..frames_per_check {
                        if let Some(frame) = capture_single_jpeg(&point.camera_conn).await {
                            frames.push(frame);
                            ts.push(now_ms());
                        }
                    }
                    if frames.is_empty() {
                        tracing::warn!(
                            "background point room {} skipped: camera capture failed ({})",
                            point.room_id,
                            point.camera_conn
                        );
                        next_allowed_ms.insert(point.room_id, now + cooldown_ms);
                        continue;
                    }
                    let audio = if use_microphone {
                        match &point.microphone_conn {
                            Some(mic) => capture_mic_raw(mic, 1.0).await.unwrap_or_default(),
                            None => Vec::new(),
                        }
                    } else {
                        Vec::new()
                    };
                    let mut client = match GatekeeperClient::connect(gateway_addr.clone()).await {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!("background connect failed: {}", e);
                            break;
                        }
                    };
                    let req = CheckAccessRequestV2 {
                        session_id: format!("bg-room-{}-{}", point.room_id, now),
                        device_id: point.camera_device_id,
                        frames,
                        audio,
                        audio_sample_rate: 16000,
                        frame_timestamps_ms: ts,
                    };
                    match client.check_access_v2(req).await {
                        Ok(resp) => {
                            let r = resp.into_inner();
                            tracing::info!(
                                "bg room={} device={} granted={} stage={} conf={:.2} reason={}",
                                point.room_id,
                                point.camera_device_id,
                                r.granted,
                                r.stage,
                                r.confidence,
                                r.reason
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "bg check_access_v2 failed room={} device={}: {}",
                                point.room_id,
                                point.camera_device_id,
                                e
                            );
                        }
                    }
                    next_allowed_ms.insert(point.room_id, now + cooldown_ms);
                }
            }
            Err(e) => {
                tracing::warn!("background discovery failed: {}", e.message());
            }
        }
        tokio::time::sleep(Duration::from_millis(tick_ms)).await;
    }
}

#[tonic::async_trait]
impl DoorAgent for DoorAgentService {
    async fn submit_observation(
        &self,
        request: Request<DoorObservationRequest>,
    ) -> Result<Response<DoorObservationResponse>, Status> {
        let req = request.into_inner();
        if req.device_id <= 0 {
            return Ok(Response::new(Self::mk_response(
                false,
                DoorPipelineStage::DoorStageIdle as i32,
                "invalid device_id",
                0.0,
                "Unknown",
                false,
                vec!["invalid_device".to_string()],
            )));
        }
        let now = if req.timestamp_ms > 0 {
            req.timestamp_ms
        } else {
            Self::now_ms()
        };

        let mut states = self.states.lock().await;
        let st = states.entry(req.device_id).or_default();

        if now < st.cooldown_until_ms {
            return Ok(Response::new(Self::mk_response(
                false,
                DoorPipelineStage::DoorStageCooldown as i32,
                "cooldown",
                0.0,
                "Unknown",
                true,
                vec!["cooldown".to_string()],
            )));
        }

        // presence by frame (face-first policy) - if no frame, reset after timeout
        if req.frame.is_empty() {
            if now - st.last_seen_ms > self.presence_timeout_ms {
                *st = DeviceState::default();
                st.stage = DoorPipelineStage::DoorStagePresence as i32;
            }
            return Ok(Response::new(Self::mk_response(
                false,
                DoorPipelineStage::DoorStagePresence as i32,
                "awaiting_face_frame",
                0.0,
                "Unknown",
                true,
                vec!["no_frame".to_string()],
            )));
        }

        st.stage = DoorPipelineStage::DoorStageCollectingClip as i32;
        st.last_seen_ms = now;
        if st.session_id.is_empty() {
            st.session_id = if req.session_id.is_empty() {
                format!("door-{}-{}", req.device_id, now)
            } else {
                req.session_id.clone()
            };
        }
        st.frames.push(req.frame);
        st.frame_ts.push(now);
        if !req.audio.is_empty() {
            st.audio.extend_from_slice(&req.audio);
            st.audio_sample_rate = if req.audio_sample_rate == 0 {
                16000
            } else {
                req.audio_sample_rate
            };
        }

        if st.frames.len() < self.max_clip_frames {
            return Ok(Response::new(Self::mk_response(
                false,
                DoorPipelineStage::DoorStageCollectingClip as i32,
                "collecting_clip",
                0.0,
                "Unknown",
                true,
                vec![format!(
                    "frames:{}/{}",
                    st.frames.len(),
                    self.max_clip_frames
                )],
            )));
        }

        st.stage = DoorPipelineStage::DoorStageIdentification as i32;
        let v2_req = CheckAccessRequestV2 {
            session_id: st.session_id.clone(),
            device_id: req.device_id,
            frames: std::mem::take(&mut st.frames),
            audio: std::mem::take(&mut st.audio),
            audio_sample_rate: st.audio_sample_rate,
            frame_timestamps_ms: std::mem::take(&mut st.frame_ts),
        };

        let gateway_addr = self.gateway_addr.clone();
        drop(states);

        let mut client = GatekeeperClient::connect(gateway_addr)
            .await
            .map_err(|e| Status::unavailable(format!("gateway connect failed: {e}")))?;
        let result = client
            .check_access_v2(v2_req)
            .await
            .map_err(|e| Status::internal(format!("gateway check_access_v2 failed: {e}")))?
            .into_inner();

        let mut states = self.states.lock().await;
        let st = states.entry(req.device_id).or_default();
        st.session_id.clear();
        st.stage = DoorPipelineStage::DoorStageCooldown as i32;
        st.cooldown_until_ms = now + self.cooldown_ms;

        let stage = if result.granted {
            DoorPipelineStage::DoorStageIdentification as i32
        } else {
            DoorPipelineStage::DoorStagePresence as i32
        };

        let flags = Self::build_pipeline_flags(&result);
        Ok(Response::new(Self::mk_response(
            result.granted,
            stage,
            result.reason,
            result.confidence,
            result.user_name,
            false,
            flags,
        )))
    }

    async fn get_door_agent_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<DoorAgentStatusResponse>, Status> {
        let now = Self::now_ms();
        let states = self.states.lock().await;
        let active_sessions = states.values().filter(|s| !s.session_id.is_empty()).count() as i32;
        let cooldown_sessions = states
            .values()
            .filter(|s| s.cooldown_until_ms > now)
            .count() as i32;
        Ok(Response::new(DoorAgentStatusResponse {
            active_sessions,
            cooldown_sessions,
        }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let listen_addr =
        std::env::var("DOOR_AGENT_ADDR").unwrap_or_else(|_| "0.0.0.0:50054".to_string());
    let gateway_addr =
        std::env::var("GATEWAY_ADDR").unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());
    let max_clip_frames = env_parse::<usize>("DOOR_MAX_CLIP_FRAMES", 3).max(1);
    let presence_timeout_ms = env_parse::<i64>("DOOR_PRESENCE_TIMEOUT_MS", 1500).max(200);
    let cooldown_ms = env_parse::<i64>("DOOR_COOLDOWN_MS", 2000).max(200);

    let svc = DoorAgentService {
        states: Arc::new(Mutex::new(HashMap::new())),
        gateway_addr: gateway_addr.clone(),
        max_clip_frames,
        presence_timeout_ms,
        cooldown_ms,
    };

    tokio::spawn(run_background_live_loop(gateway_addr));

    tracing::info!("door-agent listening on {}", listen_addr);
    Server::builder()
        .add_service(DoorAgentServer::new(svc))
        .serve(listen_addr.parse()?)
        .await?;

    Ok(())
}
