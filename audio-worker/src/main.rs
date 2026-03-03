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

const TARGET_AUDIO_SAMPLE_LENGTH: usize = 64000; // 4 секунды аудио при 16 кГц.

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok()
}

fn build_cuda_audio_builder() -> Result<ort::session::builder::SessionBuilder> {
    // Важно: для аудио по умолчанию используем CPU, так как на слабых GPU
    // выгода обычно ниже, а VRAM лучше оставить под vision.
    let mut cuda = ort::execution_providers::CUDAExecutionProvider::default();

    if let Some(mem_limit_mb) = env_usize("AUDIO_CUDA_MEM_LIMIT_MB") {
        if mem_limit_mb > 0 {
            cuda = cuda.with_memory_limit(mem_limit_mb.saturating_mul(1024 * 1024));
        }
    }

    cuda = cuda
        .with_conv_algorithm_search(ort::execution_providers::cuda::ConvAlgorithmSearch::Heuristic)
        .with_conv_max_workspace(false)
        .with_arena_extend_strategy(ort::execution_providers::ArenaExtendStrategy::SameAsRequested);

    let intra_threads = env_usize("AUDIO_INTRA_THREADS").unwrap_or(2);
    let inter_threads = env_usize("AUDIO_INTER_THREADS").unwrap_or(1);

    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra_threads)?
        .with_inter_threads(inter_threads)?
        .with_parallel_execution(true)?
        .with_config_entry("session.intra_op.allow_spinning", "1")?
        .with_execution_providers([cuda.build().error_on_failure()])
        .context("Failed to create AUDIO CUDA session builder")
}

// --- Вспомогательные функции ---

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    // Ожидаем 16-bit Little Endian PCM.
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

    // Коэффициент ресемплинга.
    let ratio = to_rate as f64 / from_rate as f64;

    // Размер чанка обработки (должен быть достаточно большим для окна ресемплера).
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
    let mut input_buffer = vec![vec![0.0; chunk_size]; 1]; // 1 канал.

    // Дополняем вход до кратности chunk_size.
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

    // Обрезаем лишние сэмплы после дополнения.
    let expected_len = (input.len() as f64 * ratio) as usize;
    if output_buffer.len() > expected_len {
        output_buffer.truncate(expected_len);
    }

    Ok(output_buffer)
}

// --- Хранилище моделей ---

struct ModelStore {
    aasist: Mutex<Session>,
    ecapa: Mutex<Session>,
    provider: String,
    init_message: String,
}

impl ModelStore {
    fn new(models_dir: &str) -> Result<Self> {
        // Для аудио по умолчанию выбираем CPU, чтобы не отбирать VRAM у vision
        // на малых видеокартах (например MX230 2GB).
        // При необходимости можно явно включить GPU: AUDIO_USE_CUDA=1.
        let force_cpu = env_bool("AUDIO_FORCE_CPU", false);
        let use_cuda = env_bool("AUDIO_USE_CUDA", false) && !force_cpu;

        info!("Loading audio models from {}", models_dir);

        let aasist_name =
            std::env::var("AUDIO_MODEL_AASIST").unwrap_or_else(|_| "aasist.onnx".to_string());
        let ecapa_name = std::env::var("AUDIO_MODEL_ECAPA")
            .unwrap_or_else(|_| "voxceleb_ECAPA512_LM.onnx".to_string());

        let aasist_path = Path::new(models_dir).join(aasist_name);
        let ecapa_path = Path::new(models_dir).join(ecapa_name);

        let load_with_builder =
            |builder: &ort::session::builder::SessionBuilder| -> Result<(Session, Session)> {
                let aasist = builder
                    .clone()
                    .commit_from_file(&aasist_path)
                    .with_context(|| format!("Failed to load AASIST from {:?}", aasist_path))?;

                let ecapa = builder
                    .clone()
                    .commit_from_file(&ecapa_path)
                    .with_context(|| format!("Failed to load ECAPA from {:?}", ecapa_path))?;

                Ok((aasist, ecapa))
            };

        if use_cuda {
            match build_cuda_audio_builder() {
                Ok(builder_cuda) => match load_with_builder(&builder_cuda) {
                    Ok((aasist, ecapa)) => {
                        return Ok(Self {
                            aasist: Mutex::new(aasist),
                            ecapa: Mutex::new(ecapa),
                            provider: "CUDA".to_string(),
                            init_message: "Audio initialized on CUDA".to_string(),
                        });
                    }
                    Err(e) => {
                        info!("Audio CUDA load failed (fallback to CPU): {}", e);
                    }
                },
                Err(e) => {
                    info!("Audio CUDA builder failed (fallback to CPU): {}", e);
                }
            }
        }

        let intra_threads = env_usize("AUDIO_INTRA_THREADS").unwrap_or(2);
        let inter_threads = env_usize("AUDIO_INTER_THREADS").unwrap_or(1);
        let builder_cpu = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(intra_threads)?
            .with_inter_threads(inter_threads)?
            .with_parallel_execution(true)?
            .with_config_entry("session.intra_op.allow_spinning", "1")?;
        let (aasist, ecapa) =
            load_with_builder(&builder_cpu).context("Failed to load audio models on CPU")?;

        let init_message = if force_cpu {
            "Audio forced to CPU (AUDIO_FORCE_CPU=1)".to_string()
        } else if use_cuda {
            "Audio fallback to CPU".to_string()
        } else {
            "Audio initialized on CPU".to_string()
        };

        Ok(Self {
            aasist: Mutex::new(aasist),
            ecapa: Mutex::new(ecapa),
            provider: "CPU".to_string(),
            init_message,
        })
    }
}

// --- Реализация сервиса ---

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

        // Шаг 1: предобработка.
        if chunk.content.is_empty() {
            return Err(Status::invalid_argument("Empty audio content"));
        }

        let samples = bytes_to_f32(&chunk.content);

        let processed_samples = resample(&samples, chunk.sample_rate, 16000)
            .map_err(|e| Status::internal(format!("Resampling error: {}", e)))?;

        // Приводим длину к TARGET_AUDIO_SAMPLE_LENGTH.
        let mut model_input_samples = processed_samples;
        if model_input_samples.len() < TARGET_AUDIO_SAMPLE_LENGTH {
            model_input_samples.extend(
                std::iter::repeat(0.0).take(TARGET_AUDIO_SAMPLE_LENGTH - model_input_samples.len()),
            );
        } else if model_input_samples.len() > TARGET_AUDIO_SAMPLE_LENGTH {
            model_input_samples.truncate(TARGET_AUDIO_SAMPLE_LENGTH);
        }

        // Копия для предварительной проверки речевой энергии.
        let vad_input_tensor_values = model_input_samples.clone();

        // Готовим тензор для AASIST и ECAPA (фиксированная длина).
        let fixed_input_shape = vec![1, TARGET_AUDIO_SAMPLE_LENGTH as i64];
        let fixed_input_tensor_values = model_input_samples;

        // --- Шаг 2: проверка наличия речи ---
        // Текущий экспорт silero_vad в этом окружении нестабилен (shape/rank в recurrent path),
        // поэтому используем легкий RMS-gate до этапов anti-spoofing и эмбеддинга.
        {
            let rms = (vad_input_tensor_values.iter().map(|v| v * v).sum::<f32>()
                / vad_input_tensor_values.len().max(1) as f32)
                .sqrt();
            if rms < 0.005 {
                return Ok(Response::new(BioResult {
                    detected: false,
                    is_live: false,
                    liveness_score: 0.0,
                    embedding: vec![],
                    error_msg: "No speech detected (energy gate)".to_string(),
                    execution_provider: self.models.provider.clone(),
                }));
            }
        }

        // --- Шаг 3: антиспуфинг (AASIST) ---
        let is_live;
        let liveness_score;
        {
            let mut session = self.models.aasist.lock().unwrap();
            // AASIST обычно ожидает фиксированную длину входа; здесь подаем фиксированный тензор.

            let input_value =
                Tensor::from_array((fixed_input_shape.clone(), fixed_input_tensor_values.clone()))
                    .map_err(|e| Status::internal(format!("AASIST input creation error: {}", e)))?;

            let outputs = session
                .run(inputs![input_value])
                .map_err(|e| Status::internal(format!("AASIST Inference error: {}", e)))?;

            // Выход обычно [1, 2], индекс 1 соответствует bonafide (live).
            let (_, output_slice) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|_| Status::internal("AASIST output extract error"))?;

            if output_slice.len() >= 2 {
                // Применяем softmax.
                let s0 = output_slice[0].exp();
                let s1 = output_slice[1].exp();
                liveness_score = s1 / (s0 + s1);
                is_live = liveness_score > 0.5;
            } else {
                // Резервный вариант для нетипичной формы выхода.
                liveness_score = output_slice.get(0).cloned().unwrap_or(0.0);
                is_live = liveness_score > 0.5;
            }
        }

        // --- Шаг 4: распознавание (ECAPA-TDNN) ---
        let embedding_vec;
        {
            let mut session = self.models.ecapa.lock().unwrap();
            let input_value = Tensor::from_array((fixed_input_shape, fixed_input_tensor_values))
                .map_err(|e| Status::internal(format!("ECAPA input creation error: {}", e)))?;

            let outputs = session
                .run(inputs![input_value])
                .map_err(|e| Status::internal(format!("ECAPA Inference error: {}", e)))?;

            // Выход: эмбеддинг [1, 192].
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

// --- Точка входа ---

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let possible_paths = ["audio-worker/models", "models"];
    let mut models_dir = "models"; // Резервный путь.
    for path in &possible_paths {
        if Path::new(path).exists() && Path::new(path).join("aasist.onnx").exists() {
            models_dir = path;
            break;
        }
    }

    // Создаем каталог, если его нет, чтобы сервис мог стартовать.
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

    let addr = "0.0.0.0:50053".parse()?; // Порт отличается от vision (50052).
    info!(
        "Audio Worker listening on {} (Provider: {})",
        addr, models.provider
    );

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
