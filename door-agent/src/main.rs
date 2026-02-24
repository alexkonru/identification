use shared::biometry::door_agent_server::{DoorAgent, DoorAgentServer};
use shared::biometry::gatekeeper_client::GatekeeperClient;
use shared::biometry::{
    CheckAccessRequestV2, DoorAgentStatusResponse, DoorObservationRequest, DoorObservationResponse,
    DoorPipelineStage, Empty,
};
use std::collections::HashMap;
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
    let max_clip_frames = std::env::var("DOOR_MAX_CLIP_FRAMES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);
    let presence_timeout_ms = std::env::var("DOOR_PRESENCE_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(1500)
        .max(200);
    let cooldown_ms = std::env::var("DOOR_COOLDOWN_MS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(2000)
        .max(200);

    let svc = DoorAgentService {
        states: Arc::new(Mutex::new(HashMap::new())),
        gateway_addr,
        max_clip_frames,
        presence_timeout_ms,
        cooldown_ms,
    };

    tracing::info!("door-agent listening on {}", listen_addr);
    Server::builder()
        .add_service(DoorAgentServer::new(svc))
        .serve(listen_addr.parse()?)
        .await?;

    Ok(())
}
