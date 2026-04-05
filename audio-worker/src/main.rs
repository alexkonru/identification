use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use rustfft::{FftPlanner, num_complex::Complex32};
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
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tonic::{Request, Response, Status, transport::Server};
use tracing::{error, info, warn};

const TARGET_AUDIO_SAMPLE_LENGTH: usize = 64000; // 4 секунды аудио при 16 кГц.
const ECAPA_SAMPLE_RATE: usize = 16000;
const ECAPA_FRAME_LEN: usize = 400;
const ECAPA_HOP_LEN: usize = 160;
const ECAPA_N_FFT: usize = 512;
const ECAPA_MEL_BINS: usize = 80;

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok()
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(default)
}

fn is_cuda_runtime_error(err: &str) -> bool {
    err.contains("CUDNN_FE")
        || err.contains("CUDNN_BACKEND_API_FAILED")
        || err.contains("CUDA error")
        || err.contains("cudaError")
        || err.contains("CUDAExecutionProvider")
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

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn build_mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
        .collect();
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
    let bins: Vec<usize> = hz_points
        .into_iter()
        .map(|hz| (((n_fft + 1) as f32 * hz) / (sample_rate as f32)).floor().max(0.0) as usize)
        .collect();

    let mut filters = vec![vec![0.0f32; n_freqs]; n_mels];
    for m in 0..n_mels {
        let left = bins[m].min(n_freqs - 1);
        let center = bins[m + 1].min(n_freqs - 1);
        let right = bins[m + 2].min(n_freqs - 1);

        if center > left {
            for k in left..center {
                filters[m][k] = (k - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for k in center..right {
                filters[m][k] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }
    filters
}

fn extract_logmel_features(samples: &[f32]) -> Vec<f32> {
    let signal = if samples.is_empty() {
        vec![0.0f32; ECAPA_FRAME_LEN]
    } else {
        samples.to_vec()
    };
    let num_frames = if signal.len() <= ECAPA_FRAME_LEN {
        1
    } else {
        1 + (signal.len() - ECAPA_FRAME_LEN).div_ceil(ECAPA_HOP_LEN)
    };

    let window: Vec<f32> = (0..ECAPA_FRAME_LEN)
        .map(|i| {
            0.54
                - 0.46
                    * ((2.0 * std::f32::consts::PI * i as f32) / (ECAPA_FRAME_LEN as f32 - 1.0))
                        .cos()
        })
        .collect();
    let filters = build_mel_filterbank(
        ECAPA_SAMPLE_RATE,
        ECAPA_N_FFT,
        ECAPA_MEL_BINS,
        20.0,
        7600.0,
    );

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(ECAPA_N_FFT);
    let mut feats = vec![0.0f32; num_frames * ECAPA_MEL_BINS];

    for frame_idx in 0..num_frames {
        let start = frame_idx * ECAPA_HOP_LEN;
        let mut fft_in = vec![Complex32::new(0.0, 0.0); ECAPA_N_FFT];
        for i in 0..ECAPA_FRAME_LEN {
            let sample = signal.get(start + i).copied().unwrap_or(0.0);
            fft_in[i] = Complex32::new(sample * window[i], 0.0);
        }
        fft.process(&mut fft_in);

        let mut power = vec![0.0f32; ECAPA_N_FFT / 2 + 1];
        for (idx, bin) in fft_in.iter().take(ECAPA_N_FFT / 2 + 1).enumerate() {
            power[idx] = (bin.re * bin.re + bin.im * bin.im) / ECAPA_N_FFT as f32;
        }

        for mel_idx in 0..ECAPA_MEL_BINS {
            let energy = filters[mel_idx]
                .iter()
                .zip(power.iter())
                .map(|(w, p)| w * p)
                .sum::<f32>()
                .max(1e-10);
            feats[frame_idx * ECAPA_MEL_BINS + mel_idx] = energy.ln();
        }
    }

    // CMN по времени для каждого mel-канала.
    for mel_idx in 0..ECAPA_MEL_BINS {
        let mut mean = 0.0f32;
        for frame_idx in 0..num_frames {
            mean += feats[frame_idx * ECAPA_MEL_BINS + mel_idx];
        }
        mean /= num_frames as f32;
        for frame_idx in 0..num_frames {
            feats[frame_idx * ECAPA_MEL_BINS + mel_idx] -= mean;
        }
    }

    feats
}

// --- Хранилище моделей ---

struct ModelStore {
    aasist: Mutex<Session>,
    ecapa: Mutex<Session>,
    provider: Mutex<String>,
    init_message: Mutex<String>,
    models_dir: String,
    use_cpu_runtime: AtomicBool,
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
                        let store = Self {
                            aasist: Mutex::new(aasist),
                            ecapa: Mutex::new(ecapa),
                            provider: Mutex::new("CUDA".to_string()),
                            init_message: Mutex::new("Audio initialized on CUDA".to_string()),
                            models_dir: models_dir.to_string(),
                            use_cpu_runtime: AtomicBool::new(false),
                        };
                        if env_bool("AUDIO_STARTUP_SELF_TEST", true) {
                            match store.startup_self_test() {
                                Ok(_) => {
                                    if let Ok(mut msg) = store.init_message.lock() {
                                        *msg =
                                            "Audio initialized on CUDA (startup self-test passed)"
                                                .to_string();
                                    }
                                }
                                Err(e) => {
                                    let details = e.to_string();
                                    warn!(
                                        "Audio CUDA startup self-test failed: {}. Switching to CPU.",
                                        details
                                    );
                                    store.switch_to_cpu(&format!(
                                        "startup self-test failed: {}",
                                        details
                                    ))?;
                                }
                            }
                        }
                        return Ok(store);
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
            provider: Mutex::new("CPU".to_string()),
            init_message: Mutex::new(init_message),
            models_dir: models_dir.to_string(),
            use_cpu_runtime: AtomicBool::new(true),
        })
    }

    fn provider_name(&self) -> String {
        self.provider
            .lock()
            .map(|p| p.clone())
            .unwrap_or_else(|_| "unknown".to_string())
    }

    fn init_message_text(&self) -> String {
        self.init_message
            .lock()
            .map(|m| m.clone())
            .unwrap_or_else(|_| "unknown".to_string())
    }

    fn switch_to_cpu(&self, reason: &str) -> Result<()> {
        if self.use_cpu_runtime.load(Ordering::Relaxed) {
            return Ok(());
        }

        let aasist_path = Path::new(&self.models_dir).join(
            std::env::var("AUDIO_MODEL_AASIST").unwrap_or_else(|_| "aasist.onnx".to_string()),
        );
        let ecapa_path = Path::new(&self.models_dir).join(
            std::env::var("AUDIO_MODEL_ECAPA").unwrap_or_else(|_| "voxceleb_ECAPA512_LM.onnx".to_string()),
        );

        let intra_threads = env_usize("AUDIO_INTRA_THREADS").unwrap_or(2);
        let inter_threads = env_usize("AUDIO_INTER_THREADS").unwrap_or(1);
        let builder_cpu = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(intra_threads)?
            .with_inter_threads(inter_threads)?
            .with_parallel_execution(true)?
            .with_config_entry("session.intra_op.allow_spinning", "1")?;

        let aasist = builder_cpu
            .clone()
            .commit_from_file(&aasist_path)
            .with_context(|| format!("Failed to load AASIST from {:?}", aasist_path))?;
        let ecapa = builder_cpu
            .clone()
            .commit_from_file(&ecapa_path)
            .with_context(|| format!("Failed to load ECAPA from {:?}", ecapa_path))?;

        *self
            .aasist
            .lock()
            .map_err(|_| anyhow::anyhow!("aasist mutex poisoned"))? = aasist;
        *self
            .ecapa
            .lock()
            .map_err(|_| anyhow::anyhow!("ecapa mutex poisoned"))? = ecapa;
        *self
            .provider
            .lock()
            .map_err(|_| anyhow::anyhow!("provider mutex poisoned"))? = "CPU".to_string();
        *self
            .init_message
            .lock()
            .map_err(|_| anyhow::anyhow!("init_message mutex poisoned"))? =
            format!("Audio fallback to CPU after CUDA runtime failure: {}", reason);
        self.use_cpu_runtime.store(true, Ordering::Relaxed);
        warn!("Audio runtime switched to CPU after CUDA failure: {}", reason);
        Ok(())
    }

    fn startup_self_test(&self) -> Result<()> {
        let fixed_input_shape = vec![1, TARGET_AUDIO_SAMPLE_LENGTH as i64];
        let fixed_input_tensor_values = vec![0.0f32; TARGET_AUDIO_SAMPLE_LENGTH];
        let _ = self.run_aasist_scores(fixed_input_shape, fixed_input_tensor_values.clone())?;

        let ecapa_feats = extract_logmel_features(&fixed_input_tensor_values);
        let num_frames = (ecapa_feats.len() / ECAPA_MEL_BINS).max(1);
        let ecapa_input_shape = vec![1, num_frames as i64, ECAPA_MEL_BINS as i64];
        let _ = self.run_ecapa_embedding(ecapa_input_shape, ecapa_feats)?;
        Ok(())
    }

    fn run_aasist_scores(&self, input_shape: Vec<i64>, input_values: Vec<f32>) -> Result<Vec<f32>> {
        let mut session = self
            .aasist
            .lock()
            .map_err(|_| anyhow::anyhow!("aasist mutex poisoned"))?;
        let input_value = Tensor::from_array((input_shape, input_values))
            .context("AASIST input creation error")?;
        let outputs = session
            .run(inputs![input_value])
            .context("AASIST inference failed")?;
        let (_, output_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|_| anyhow::anyhow!("AASIST output extract error"))?;
        Ok(output_slice.to_vec())
    }

    fn run_ecapa_embedding(&self, input_shape: Vec<i64>, input_values: Vec<f32>) -> Result<Vec<f32>> {
        let mut session = self
            .ecapa
            .lock()
            .map_err(|_| anyhow::anyhow!("ecapa mutex poisoned"))?;
        let input_value = Tensor::from_array((input_shape, input_values))
            .context("ECAPA input creation error")?;
        let outputs = session
            .run(inputs![input_value])
            .context("ECAPA inference failed")?;
        let (_, output_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|_| anyhow::anyhow!("ECAPA output extract error"))?;
        Ok(output_slice.to_vec())
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
            device: self.models.provider_name(),
            message: self.models.init_message_text(),
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
            let active_ratio = vad_input_tensor_values
                .iter()
                .filter(|v| v.abs() > 0.02)
                .count() as f32
                / vad_input_tensor_values.len().max(1) as f32;
            let min_rms = env_f32("AUDIO_SPEECH_RMS_MIN", 0.008).clamp(0.0, 0.2);
            let min_active_ratio =
                env_f32("AUDIO_SPEECH_ACTIVE_RATIO_MIN", 0.02).clamp(0.0, 1.0);
            if rms < min_rms || active_ratio < min_active_ratio {
                return Ok(Response::new(BioResult {
                    detected: false,
                    is_live: false,
                    liveness_score: 0.0,
                    embedding: vec![],
                    error_msg: "No speech detected (energy/activity gate)".to_string(),
                    execution_provider: self.models.provider_name(),
                }));
            }
        }

        // --- Шаг 3: антиспуфинг (AASIST) ---
        let is_live;
        let liveness_score;
        {
            let output_slice = match self.models.run_aasist_scores(
                fixed_input_shape.clone(),
                fixed_input_tensor_values.clone(),
            ) {
                Ok(scores) => scores,
                Err(e) => {
                    let err_text = e.to_string();
                    if is_cuda_runtime_error(&err_text) {
                        self.models
                            .switch_to_cpu(&err_text)
                            .map_err(|se| Status::internal(format!("Audio CPU fallback failed: {}", se)))?;
                        self.models
                            .run_aasist_scores(
                                fixed_input_shape.clone(),
                                fixed_input_tensor_values.clone(),
                            )
                            .map_err(|e2| Status::internal(format!("AASIST Inference error: {}", e2)))?
                    } else {
                        return Err(Status::internal(format!("AASIST Inference error: {}", err_text)));
                    }
                }
            };
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
            let ecapa_feats = extract_logmel_features(&fixed_input_tensor_values);
            let num_frames = (ecapa_feats.len() / ECAPA_MEL_BINS).max(1);
            let ecapa_input_shape = vec![1, num_frames as i64, ECAPA_MEL_BINS as i64];
            let output_slice = match self
                .models
                .run_ecapa_embedding(ecapa_input_shape.clone(), ecapa_feats.clone())
            {
                Ok(embedding) => embedding,
                Err(e) => {
                    let err_text = e.to_string();
                    if is_cuda_runtime_error(&err_text) {
                        self.models
                            .switch_to_cpu(&err_text)
                            .map_err(|se| Status::internal(format!("Audio CPU fallback failed: {}", se)))?;
                        self.models
                            .run_ecapa_embedding(ecapa_input_shape, ecapa_feats)
                            .map_err(|e2| Status::internal(format!("ECAPA Inference error: {}", e2)))?
                    } else {
                        return Err(Status::internal(format!("ECAPA Inference error: {}", err_text)));
                    }
                }
            };
            embedding_vec = output_slice;
        }

        Ok(Response::new(BioResult {
            detected: true,
            is_live,
            liveness_score,
            embedding: embedding_vec,
            error_msg: String::new(),
            execution_provider: self.models.provider_name(),
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
        addr,
        models.provider_name()
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
