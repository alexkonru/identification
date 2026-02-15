use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use shared::biometry::{
    AudioChunk, BioResult, Empty, ServiceStatus,
    audio_server::{Audio, AudioServer},
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status, transport::Server};
use tracing::{error, info};

const TARGET_AUDIO_SAMPLE_LENGTH: usize = 64000; // 4 seconds of 16kHz audio

// --- Helper Functions ---

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    // Assume 16-bit Little Endian PCM
    let mut samples = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let sample = LittleEndian::read_i16(chunk);
        samples.push(sample as f32 / 32768.0);
    }
    samples
}

fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(input.to_vec());
    }

    // Calculate resampling ratio
    let ratio = to_rate as f64 / from_rate as f64;

    // Chunk size for processing (must be large enough for the resampler window)
    let chunk_size = 1024;

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
        oversampling_factor: 256,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        ratio, ratio, // max_resample_ratio
        params, chunk_size, 1, // channels
    )?;

    let mut output_buffer = Vec::new();
    let mut input_buffer = vec![vec![0.0; chunk_size]; 1]; // 1 channel

    // Pad input to multiple of chunk_size
    let mut padded_input = input.to_vec();
    let remainder = padded_input.len() % chunk_size;
    if remainder != 0 {
        padded_input.extend(std::iter::repeat(0.0).take(chunk_size - remainder));
    }

    for chunk in padded_input.chunks(chunk_size) {
        input_buffer[0].copy_from_slice(chunk);
        let output_frames = resampler.process(&input_buffer, None)?;
        output_buffer.extend_from_slice(&output_frames[0]);
    }

    // Trim potentially excess samples if we padded?
    // Usually simple approximation is fine for bio-metrics, but ideally we calculate exact length.
    // length = input_len * ratio.
    let expected_len = (input.len() as f64 * ratio) as usize;
    if output_buffer.len() > expected_len {
        output_buffer.truncate(expected_len);
    }

    Ok(output_buffer)
}

// --- ModelStore ---

struct ModelStore {
    vad: Mutex<Session>,
    aasist: Mutex<Session>,
    ecapa: Mutex<Session>,
    provider: String,
    init_message: String,
}

impl ModelStore {
    fn new(models_dir: &str) -> Result<Self> {
        let vad_path = Path::new(models_dir).join("silero_vad.onnx");
        let aasist_path = Path::new(models_dir).join("aasist.onnx");
        let ecapa_path = Path::new(models_dir).join("voxceleb_ECAPA512_LM.onnx");

        info!("Loading models from {}", models_dir);
        info!("Trying to initialize audio models with CUDA...");

        let builder_cuda_res = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build()
            ]);

        let (vad, aasist, ecapa, used_provider, init_message) = match builder_cuda_res {
            Ok(builder_cuda) => {
                match Self::try_load(&builder_cuda, &vad_path, &aasist_path, &ecapa_path) {
                    Ok(sessions) => {
                        info!("Audio models initialized with CUDA");
                        (
                            sessions.0,
                            sessions.1,
                            sessions.2,
                            "CUDA".to_string(),
                            "Initialized successfully with CUDA".to_string(),
                        )
                    }
                    Err(e) => {
                        info!(
                            "CUDA initialized, but model loading failed: {}. Falling back to CPU.",
                            e
                        );
                        let (v, a, ecp, provider) =
                            Self::load_cpu(&vad_path, &aasist_path, &ecapa_path)?;
                        (
                            v,
                            a,
                            ecp,
                            provider,
                            format!("CUDA init failed: {}. Fallback to CPU.", e),
                        )
                    }
                }
            }
            Err(e) => {
                info!(
                    "CUDA execution provider is unavailable: {}. Falling back to CPU.",
                    e
                );
                let (v, a, ecp, provider) = Self::load_cpu(&vad_path, &aasist_path, &ecapa_path)?;
                (
                    v,
                    a,
                    ecp,
                    provider,
                    format!("CUDA unavailable: {}. Fallback to CPU.", e),
                )
            }
        };

        Ok(Self {
            vad: Mutex::new(vad),
            aasist: Mutex::new(aasist),
            ecapa: Mutex::new(ecapa),
            provider: used_provider,
            init_message,
        })
    }

    fn load_cpu(
        vad_path: &Path,
        aasist_path: &Path,
        ecapa_path: &Path,
    ) -> Result<(Session, Session, Session, String)> {
        let builder_cpu = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        let sessions = Self::try_load(&builder_cpu, vad_path, aasist_path, ecapa_path)
            .context("Failed to load audio models even on CPU")?;

        Ok((sessions.0, sessions.1, sessions.2, "CPU".to_string()))
    }

    fn try_load(
        builder: &ort::session::builder::SessionBuilder,
        vad_path: &Path,
        aasist_path: &Path,
        ecapa_path: &Path,
    ) -> Result<(Session, Session, Session)> {
        let vad = builder
            .clone()
            .commit_from_file(vad_path)
            .with_context(|| format!("Failed to load VAD from {:?}", vad_path))?;

        let aasist = builder
            .clone()
            .commit_from_file(aasist_path)
            .with_context(|| format!("Failed to load AASIST from {:?}", aasist_path))?;

        let ecapa = builder
            .clone()
            .commit_from_file(ecapa_path)
            .with_context(|| format!("Failed to load ECAPA from {:?}", ecapa_path))?;

        Ok((vad, aasist, ecapa))
    }
}

// --- Service Implementation ---

use tokio::sync::mpsc;

struct AudioService {
    models: Arc<ModelStore>,
    shutdown_tx: mpsc::Sender<()>,
}

#[tonic::async_trait]
impl Audio for AudioService {
    async fn get_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ServiceStatus>, Status> {
        Ok(Response::new(ServiceStatus {
            online: true,
            device: self.models.provider.clone(),
            message: self.models.init_message.clone(),
        }))
    }

    async fn process_voice(
        &self,
        request: Request<AudioChunk>,
    ) -> Result<Response<BioResult>, Status> {
        let chunk = request.into_inner();

        // Step 1: Preprocessing
        if chunk.content.is_empty() {
            return Err(Status::invalid_argument("Empty audio content"));
        }

        let samples = bytes_to_f32(&chunk.content);

        let processed_samples = resample(&samples, chunk.sample_rate, 16000)
            .map_err(|e| Status::internal(format!("Resampling error: {}", e)))?;

        // Ensure processed_samples has the TARGET_AUDIO_SAMPLE_LENGTH
        let mut model_input_samples = processed_samples;
        if model_input_samples.len() < TARGET_AUDIO_SAMPLE_LENGTH {
            model_input_samples.extend(
                std::iter::repeat(0.0).take(TARGET_AUDIO_SAMPLE_LENGTH - model_input_samples.len()),
            );
        } else if model_input_samples.len() > TARGET_AUDIO_SAMPLE_LENGTH {
            model_input_samples.truncate(TARGET_AUDIO_SAMPLE_LENGTH);
        }

        // Prepare tensor for VAD (variable length)
        let vad_input_shape = vec![1, model_input_samples.len() as i64];
        let vad_input_tensor_values = model_input_samples.clone();

        // Prepare tensor for AASIST and ECAPA (fixed length)
        let fixed_input_shape = vec![1, TARGET_AUDIO_SAMPLE_LENGTH as i64];
        let fixed_input_tensor_values = model_input_samples;

        // --- Step 2: VAD ---
        {
            let mut session = self.models.vad.lock().unwrap();
            let input_value =
                Tensor::from_array((vad_input_shape.clone(), vad_input_tensor_values.clone()))
                    .map_err(|e| Status::internal(format!("VAD input creation error: {}", e)))?;

            let outputs = session
                .run(inputs![input_value])
                .map_err(|e| Status::internal(format!("VAD Inference error: {}", e)))?;

            // Output: (1, 2) prob.
            let (_, output_slice) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|_| Status::internal("VAD output extract error"))?;

            // Assuming [1, 2], index 1 is speech probability.
            // Or [1, 1].
            let speech_prob = if output_slice.len() >= 2 {
                output_slice[1]
            } else if output_slice.len() == 1 {
                output_slice[0]
            } else {
                0.0
            };

            if speech_prob < 0.5 {
                return Ok(Response::new(BioResult {
                    detected: false,
                    is_live: false,
                    liveness_score: 0.0,
                    embedding: vec![],
                    error_msg: "No speech detected".to_string(),
                    execution_provider: self.models.provider.clone(),
                }));
            }
        }

        // --- Step 3: Anti-Spoofing (AASIST) ---
        let is_live;
        let liveness_score;
        {
            let mut session = self.models.aasist.lock().unwrap();
            // AASIST usually expects fixed length input (e.g. 64600 samples ~ 4s) repeated or cut.
            // For simplicity, we pass what we have, assuming model handles variable length or we should fix it.
            // If model fails on variable length, we might need to pad/cut.
            // Most ONNX exports of AASIST are fixed size.
            // Let's try passing as is. If it crashes, user needs to handle input size.
            // AASIST repo: "Input: raw waveform (tensor)".

            let input_value =
                Tensor::from_array((fixed_input_shape.clone(), fixed_input_tensor_values.clone()))
                    .map_err(|e| Status::internal(format!("AASIST input creation error: {}", e)))?;

            let outputs = session
                .run(inputs![input_value])
                .map_err(|e| Status::internal(format!("AASIST Inference error: {}", e)))?;

            // Output: [1, 2] usually. Index 1 = Bonafide (Live).
            let (_, output_slice) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|_| Status::internal("AASIST output extract error"))?;

            if output_slice.len() >= 2 {
                // Apply softmax roughly or just check raw scores
                // Softmax: exp(x) / sum(exp)
                let s0 = output_slice[0].exp();
                let s1 = output_slice[1].exp();
                liveness_score = s1 / (s0 + s1);
                is_live = liveness_score > 0.5;
            } else {
                // Fallback
                liveness_score = output_slice.get(0).cloned().unwrap_or(0.0);
                is_live = liveness_score > 0.5;
            }
        }

        // --- Step 4: Recognition (ECAPA-TDNN) ---
        let embedding_vec;
        {
            let mut session = self.models.ecapa.lock().unwrap();
            let input_value = Tensor::from_array((fixed_input_shape, fixed_input_tensor_values))
                .map_err(|e| Status::internal(format!("ECAPA input creation error: {}", e)))?;

            let outputs = session
                .run(inputs![input_value])
                .map_err(|e| Status::internal(format!("ECAPA Inference error: {}", e)))?;

            // Output: [1, 192] embedding.
            let (_, output_slice) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|_| Status::internal("ECAPA output extract error"))?;

            embedding_vec = output_slice.to_vec();
        }

        Ok(Response::new(BioResult {
            detected: true,
            is_live,
            liveness_score,
            embedding: embedding_vec,
            error_msg: String::new(),
            execution_provider: self.models.provider.clone(),
        }))
    }
    async fn shutdown(&self, _request: Request<Empty>) -> Result<Response<Empty>, Status> {
        info!("Shutdown signal received");
        self.shutdown_tx.send(()).await.map_err(|e| {
            error!("Failed to send shutdown signal: {}", e);
            Status::internal("Failed to send shutdown signal")
        })?;
        Ok(Response::new(Empty {}))
    }
}

// --- Main ---

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let possible_paths = ["audio-worker/models", "models"];
    let mut models_dir = "models"; // fallback
    for path in &possible_paths {
        if Path::new(path).exists() && Path::new(path).join("silero_vad.onnx").exists() {
            models_dir = path;
            break;
        }
    }

    // We create the directory if it doesn't exist to allow start (logic requirement)
    // Though in prod it should be populated.
    if !std::fs::metadata(models_dir).is_ok() {
        std::fs::create_dir_all(models_dir)?;
    }

    info!("Initializing Audio Worker...");
    let models = match ModelStore::new(models_dir) {
        Ok(store) => Arc::new(store),
        Err(e) => {
            error!(
                "Failed to load models (check if onnx files exist in ./models): {:?}",
                e
            );
            return Err(e.into());
        }
    };

    let addr = "0.0.0.0:50053".parse()?; // Different port than vision (50052)
    info!("Audio Worker listening on {}", addr);

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);

    let service = AudioService {
        models,
        shutdown_tx,
    };

    Server::builder()
        .add_service(AudioServer::new(service))
        .serve_with_shutdown(addr, async {
            shutdown_rx.recv().await;
            info!("Gracefully shutting down Audio Worker");
        })
        .await?;

    Ok(())
}
