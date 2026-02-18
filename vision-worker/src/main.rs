use anyhow::{Context, Result};
use image::{ImageBuffer, imageops::FilterType};
use ndarray::Array4;
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel, builder::SessionBuilder},
    value::Tensor,
};
use shared::biometry::{
    BioResult, Empty, ImageFrame, ServiceStatus,
    vision_server::{Vision, VisionServer},
};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tonic::{Request, Response, Status, transport::Server};
use tracing::{debug, error, info, warn};

const YUNET_INPUT_WIDTH: u32 = 640;
const YUNET_INPUT_HEIGHT: u32 = 640;

fn is_cuda_runtime_failure(err: &str) -> bool {
    err.contains("cudaError") || err.contains("CUDA error")
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok()
}

fn env_i32(name: &str) -> Option<i32> {
    std::env::var(name).ok()?.trim().parse::<i32>().ok()
}

fn compact_cuda_error(err: &str) -> String {
    let mut text = err
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    // Самая полезная часть ошибки загрузки CUDA EP обычно идёт после "with error:".
    if let Some(idx) = text.find("with error:") {
        text = text[(idx + "with error:".len())..].trim().to_string();
    } else if let Some(idx) = text.find("FAIL :") {
        text = text[(idx + "FAIL :".len())..].trim().to_string();
    }

    // Урезаем слишком длинные сообщения, чтобы WARN оставался читаемым.
    const MAX_LEN: usize = 220;
    if text.len() > MAX_LEN {
        text.truncate(MAX_LEN);
        text.push('…');
    }

    text
}

fn candidate_cuda_device_ids() -> Vec<i32> {
    // Для гибридных систем (iGPU + dGPU) индекс CUDA-устройства в ONNX Runtime
    // может не совпадать с ожидаемым "0". Поэтому если VISION_CUDA_DEVICE_ID
    // не задан, перебираем небольшой диапазон индексов.
    if let Some(device_id) = env_i32("VISION_CUDA_DEVICE_ID") {
        vec![device_id]
    } else {
        vec![0, 1, 2, 3]
    }
}

fn build_cuda_builder(device_id: i32) -> Result<SessionBuilder> {
    let mut cuda =
        ort::execution_providers::CUDAExecutionProvider::default().with_device_id(device_id);

    if let Some(mem_limit_mb) = env_usize("VISION_CUDA_MEM_LIMIT_MB") {
        if mem_limit_mb > 0 {
            cuda = cuda.with_memory_limit(mem_limit_mb.saturating_mul(1024 * 1024));
        }
    }

    // Снижаем пиковое потребление VRAM при инициализации.
    cuda = cuda
        .with_conv_algorithm_search(ort::execution_providers::cuda::ConvAlgorithmSearch::Heuristic)
        .with_conv_max_workspace(false)
        .with_arena_extend_strategy(ort::execution_providers::ArenaExtendStrategy::SameAsRequested);

    let intra_threads = env_usize("VISION_INTRA_THREADS").unwrap_or(4);

    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra_threads)?
        .with_execution_providers([cuda.build().error_on_failure()])?
        .with_log_level(ort::logging::LogLevel::Warning)
        .context("Failed to create session builder with CUDA execution provider")
}

// --- Helper Structs for Yunet ---
#[derive(Debug, Clone, Copy)]
struct Face {
    bbox: [f32; 4],           // x, y, w, h
    landmarks: [[f32; 2]; 5], // right_eye, left_eye, nose, mouth_right, mouth_left
    score: f32,
}

impl Face {
    fn area(&self) -> f32 {
        self.bbox[2] * self.bbox[3]
    }
}

// --- Helper Functions for Yunet ---
fn decode_yunet_output(
    output_slice: &[f32],
    original_width: u32,
    original_height: u32,
    score_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<Face>> {
    // Yunet output is typically [1, 1, N, 15] or [N, 15]
    // Each row: [x1, y1, x2, y2, score, right_eye_x, right_eye_y, ..., mouth_left_y]
    // The model gives the bbox as x1, y1, x2, y2 (top-left, bottom-right corners)
    // We need to convert it to x, y, w, h (top-left corner, width, height)

    let mut faces: Vec<Face> = Vec::new();

    let num_elements_per_face = 15;
    let num_faces_raw = output_slice.len() / num_elements_per_face;

    if num_faces_raw == 0 {
        return Ok(Vec::new());
    }

    for i in 0..num_faces_raw {
        let offset = i * num_elements_per_face;
        let data = &output_slice[offset..offset + num_elements_per_face];

        let score = data[14]; // Score is the last element
        if score < score_threshold {
            continue;
        }

        let x1 = data[0];
        let y1 = data[1];
        let x2 = data[2];
        let y2 = data[3];

        let bbox = [x1, y1, x2 - x1, y2 - y1]; // x, y, w, h

        let landmarks = [
            [data[4], data[5]],   // right_eye
            [data[6], data[7]],   // left_eye
            [data[8], data[9]],   // nose
            [data[10], data[11]], // mouth_right
            [data[12], data[13]], // mouth_left
        ];

        faces.push(Face {
            bbox,
            landmarks,
            score,
        });
    }

    // Apply Non-Maximum Suppression (NMS) - simple version
    // For a more robust NMS, a dedicated NMS algorithm would be better.
    // This is a basic approach.

    faces.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; faces.len()];
    let mut nms_faces = Vec::new();

    for i in 0..faces.len() {
        if suppressed[i] {
            continue;
        }
        let f1 = &faces[i];
        nms_faces.push(*f1);

        for j in (i + 1)..faces.len() {
            if suppressed[j] {
                continue;
            }
            let f2 = &faces[j];
            let iou = calculate_iou(&f1.bbox, &f2.bbox);
            if iou > nms_threshold {
                suppressed[j] = true;
            }
        }
    }

    // Scale bounding boxes and landmarks to original image size
    let scale_x = original_width as f32;
    let scale_y = original_height as f32;

    for face in &mut nms_faces {
        face.bbox[0] *= scale_x; // x
        face.bbox[1] *= scale_y; // y
        face.bbox[2] *= scale_x; // w
        face.bbox[3] *= scale_y; // h

        for l in &mut face.landmarks {
            l[0] *= scale_x;
            l[1] *= scale_y;
        }
    }

    Ok(nms_faces)
}

fn calculate_iou(bbox1: &[f32; 4], bbox2: &[f32; 4]) -> f32 {
    let x_left = bbox1[0].max(bbox2[0]);
    let y_top = bbox1[1].max(bbox2[1]);
    let x_right = (bbox1[0] + bbox1[2]).min(bbox2[0] + bbox2[2]);
    let y_bottom = (bbox1[1] + bbox1[3]).min(bbox2[1] + bbox2[3]);

    if x_right < x_left || y_bottom < y_top {
        return 0.0;
    }

    let intersection_area = (x_right - x_left) * (y_bottom - y_top);

    let bbox1_area = bbox1[2] * bbox1[3];
    let bbox2_area = bbox2[2] * bbox2[3];

    intersection_area / (bbox1_area + bbox2_area - intersection_area)
}

struct ModelStore {
    yunet: Mutex<Session>,
    arcface: Mutex<Session>,
    liveness: Mutex<Session>,
    provider: Mutex<String>,
    init_message: Mutex<String>,
    models_dir: String,
    use_cpu_runtime: AtomicBool,
}

impl ModelStore {
    fn new(models_dir: &str) -> Result<Self> {
        let arcface_name = std::env::var("VISION_MODEL_ARCFACE")
            .unwrap_or_else(|_| "arcface.onnx".to_string());
        let liveness_name = std::env::var("VISION_MODEL_LIVENESS")
            .unwrap_or_else(|_| "MiniFASNetV2.onnx".to_string());
        let yunet_name = std::env::var("VISION_MODEL_YUNET")
            .unwrap_or_else(|_| "face_detection_yunet_2023mar.onnx".to_string());

        let arcface_path = Path::new(models_dir).join(arcface_name);
        let liveness_path = Path::new(models_dir).join(liveness_name);
        let yunet_path = Path::new(models_dir).join(yunet_name);

        let force_cpu = std::env::var("VISION_FORCE_CPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let device_candidates = candidate_cuda_device_ids();
        info!(
            "Попытка загрузки моделей с CUDA... candidates={:?}, mem_limit_mb={:?}",
            device_candidates,
            env_usize("VISION_CUDA_MEM_LIMIT_MB")
        );

        let (yunet, arcface, liveness, used_provider, init_msg) = if force_cpu {
            warn!("VISION_FORCE_CPU enabled. Starting vision-worker on CPU.");
            let (y, a, l, p) = Self::load_cpu(&yunet_path, &arcface_path, &liveness_path)?;
            (y, a, l, p, "Forced CPU mode".to_string())
        } else {
            let mut cuda_errors_short: Vec<(i32, &'static str, String)> = Vec::new();
            let mut cuda_errors_full = Vec::new();
            let mut loaded = None;

            for device_id in device_candidates {
                match build_cuda_builder(device_id) {
                    Ok(builder_cuda) => {
                        match Self::try_load(
                            &builder_cuda,
                            &yunet_path,
                            &arcface_path,
                            &liveness_path,
                        ) {
                            Ok(sessions) => {
                                info!("Успешно загружено с CUDA (device_id={})", device_id);
                                loaded = Some((
                                    sessions.0,
                                    sessions.1,
                                    sessions.2,
                                    format!("CUDA:{}", device_id),
                                    format!(
                                        "Initialized successfully with CUDA (device_id={})",
                                        device_id
                                    ),
                                ));
                                break;
                            }
                            Err(e) => {
                                let details = format!("{:#}", e);
                                cuda_errors_short.push((
                                    device_id,
                                    "load",
                                    compact_cuda_error(&details),
                                ));
                                cuda_errors_full.push(format!(
                                    "device_id={}: load failed: {}",
                                    device_id, details
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        let details = format!("{:#}", e);
                        cuda_errors_short.push((
                            device_id,
                            "builder",
                            compact_cuda_error(&details),
                        ));
                        cuda_errors_full.push(format!(
                            "device_id={}: builder failed: {}",
                            device_id, details
                        ));
                    }
                }
            }

            if let Some(ok) = loaded {
                ok
            } else {
                let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
                for (device_id, stage, reason) in &cuda_errors_short {
                    grouped
                        .entry(reason.clone())
                        .or_default()
                        .push(format!("{}:{}", device_id, stage));
                }

                let compact_summary = grouped
                    .into_iter()
                    .map(|(reason, sources)| format!("[{}] {}", sources.join(","), reason))
                    .collect::<Vec<_>>()
                    .join(" | ");

                warn!("CUDA недоступна: {}. Переключение на CPU.", compact_summary);
                debug!("CUDA full init errors: {}", cuda_errors_full.join(" || "));
                let (y, a, l, p) = Self::load_cpu(&yunet_path, &arcface_path, &liveness_path)?;
                (
                    y,
                    a,
                    l,
                    p,
                    format!(
                        "CUDA unavailable for all device candidates. Reasons: {}. Fallback to CPU.",
                        compact_summary
                    ),
                )
            }
        };

        Ok(Self {
            yunet: Mutex::new(yunet),
            arcface: Mutex::new(arcface),
            liveness: Mutex::new(liveness),
            provider: Mutex::new(used_provider),
            init_message: Mutex::new(init_msg),
            models_dir: models_dir.to_string(),
            use_cpu_runtime: AtomicBool::new(force_cpu),
        })
    }

    fn load_cpu(
        yunet_path: &Path,
        arc_path: &Path,
        live_path: &Path,
    ) -> Result<(Session, Session, Session, String)> {
        let intra_threads = env_usize("VISION_INTRA_THREADS").unwrap_or(4);
        let builder_cpu = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(intra_threads)?;

        let sessions = Self::try_load(&builder_cpu, yunet_path, arc_path, live_path)
            .context("Не удалось загрузить модели даже на CPU")?;

        Ok((sessions.0, sessions.1, sessions.2, "CPU".to_string()))
    }

    fn switch_to_cpu(&self, reason: &str) -> Result<()> {
        if self.use_cpu_runtime.load(Ordering::Relaxed) {
            return Ok(());
        }

        let arcface_name = std::env::var("VISION_MODEL_ARCFACE")
            .unwrap_or_else(|_| "arcface.onnx".to_string());
        let liveness_name = std::env::var("VISION_MODEL_LIVENESS")
            .unwrap_or_else(|_| "MiniFASNetV2.onnx".to_string());
        let yunet_name = std::env::var("VISION_MODEL_YUNET")
            .unwrap_or_else(|_| "face_detection_yunet_2023mar.onnx".to_string());

        let yunet_path = Path::new(&self.models_dir).join(yunet_name);
        let arcface_path = Path::new(&self.models_dir).join(arcface_name);
        let liveness_path = Path::new(&self.models_dir).join(liveness_name);

        let (y, a, l, _) = Self::load_cpu(&yunet_path, &arcface_path, &liveness_path)?;
        *self
            .yunet
            .lock()
            .map_err(|_| anyhow::anyhow!("yunet mutex poisoned"))? = y;
        *self
            .arcface
            .lock()
            .map_err(|_| anyhow::anyhow!("arcface mutex poisoned"))? = a;
        *self
            .liveness
            .lock()
            .map_err(|_| anyhow::anyhow!("liveness mutex poisoned"))? = l;
        *self
            .provider
            .lock()
            .map_err(|_| anyhow::anyhow!("provider mutex poisoned"))? = "CPU".to_string();
        *self
            .init_message
            .lock()
            .map_err(|_| anyhow::anyhow!("init message mutex poisoned"))? =
            format!("Runtime fallback to CPU: {}", reason);
        self.use_cpu_runtime.store(true, Ordering::Relaxed);
        warn!("Vision runtime switched to CPU fallback: {}", reason);
        Ok(())
    }

    fn try_load(
        builder: &SessionBuilder,
        yunet_path: &Path,
        arc_path: &Path,
        live_path: &Path,
    ) -> Result<(Session, Session, Session)> {
        let yunet = builder
            .clone()
            .commit_from_file(yunet_path)
            .with_context(|| format!("Failed to load Yunet from {:?}", yunet_path))?;

        let arcface = builder
            .clone()
            .commit_from_file(arc_path)
            .with_context(|| format!("Failed to load ArcFace from {:?}", arc_path))?;

        let liveness = builder
            .clone()
            .commit_from_file(live_path)
            .with_context(|| format!("Failed to load Liveness from {:?}", live_path))?;

        Ok((yunet, arcface, liveness))
    }
}

// --- Реализация сервиса ---

use tokio::sync::mpsc;

struct VisionService {
    models: Arc<ModelStore>,
    shutdown_tx: mpsc::Sender<()>,
}

#[tonic::async_trait]
impl Vision for VisionService {
    async fn get_status(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ServiceStatus>, Status> {
        Ok(Response::new(ServiceStatus {
            online: true,
            device: self
                .models
                .provider
                .lock()
                .map(|v| v.clone())
                .unwrap_or_else(|_| "unknown".to_string()),
            message: self
                .models
                .init_message
                .lock()
                .map(|v| v.clone())
                .unwrap_or_else(|_| "status unavailable".to_string()),
        }))
    }

    async fn process_face(
        &self,
        request: Request<ImageFrame>,
    ) -> Result<Response<BioResult>, Status> {
        debug!("Получен запрос на обработку лица");
        let frame = request.into_inner();

        // Шаг 1: Декодирование изображения
        let img = image::load_from_memory(&frame.content).map_err(|e| {
            Status::invalid_argument(format!("Не удалось декодировать изображение: {}", e))
        })?;

        let original_width = img.width();
        let original_height = img.height();
        let rgb_img = img.to_rgb8();

        // --- Шаг 2: Обнаружение лица с Yunet ---
        let resized_yunet = image::imageops::resize(
            &rgb_img,
            YUNET_INPUT_WIDTH,
            YUNET_INPUT_HEIGHT,
            FilterType::Triangle,
        );
        // YuNet in this project expects NCHW: [1, 3, 640, 640]
        let mut input_tensor_yunet = Array4::<f32>::zeros((
            1,
            3,
            YUNET_INPUT_HEIGHT as usize,
            YUNET_INPUT_WIDTH as usize,
        ));

        for (x, y, pixel) in resized_yunet.enumerate_pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            // Normalize to 0-1
            input_tensor_yunet[[0, 0, y as usize, x as usize]] = r / 255.0;
            input_tensor_yunet[[0, 1, y as usize, x as usize]] = g / 255.0;
            input_tensor_yunet[[0, 2, y as usize, x as usize]] = b / 255.0;
        }

        let (data_vec_yunet, _) = input_tensor_yunet.into_raw_vec_and_offset();
        let build_yunet_input = || {
            Tensor::from_array((
                vec![1, 3, YUNET_INPUT_HEIGHT as i64, YUNET_INPUT_WIDTH as i64],
                data_vec_yunet.clone(),
            ))
            .map_err(|e| Status::internal(format!("Ошибка создания тензора Ort (Yunet): {}", e)))
        };

        let first_try = {
            let mut session_yunet = self
                .models
                .yunet
                .lock()
                .map_err(|_| Status::internal("Не удалось захватить мьютекс Yunet"))?;
            let input_value_yunet = build_yunet_input()?;
            match session_yunet.run(inputs![input_value_yunet]) {
                Ok(outputs_yunet) => {
                    let outputs_yunet_value = &outputs_yunet[0];
                    let vec = outputs_yunet_value
                        .try_extract_tensor::<f32>()
                        .map_err(|e| {
                            Status::internal(format!("Ошибка извлечения тензора Yunet: {}", e))
                        })?
                        .1
                        .to_vec();
                    Ok(vec)
                }
                Err(e) => Err(e.to_string()),
            }
        };

        let outputs_yunet_vec = match first_try {
            Ok(vec) => vec,
            Err(err_text) => {
                if is_cuda_runtime_failure(&err_text) {
                    self.models.switch_to_cpu(&err_text).map_err(|se| {
                        Status::internal(format!("Vision CPU fallback failed: {}", se))
                    })?;
                    let mut cpu_session =
                        self.models.yunet.lock().map_err(|_| {
                            Status::internal("Не удалось захватить CPU мьютекс Yunet")
                        })?;
                    let input_value_yunet = build_yunet_input()?;
                    let outputs_yunet =
                        cpu_session.run(inputs![input_value_yunet]).map_err(|e2| {
                            Status::internal(format!(
                                "Ошибка инференса Yunet (CPU fallback): {}",
                                e2
                            ))
                        })?;
                    let outputs_yunet_value = &outputs_yunet[0];
                    outputs_yunet_value
                        .try_extract_tensor::<f32>()
                        .map_err(|e3| {
                            Status::internal(format!("Ошибка извлечения тензора Yunet: {}", e3))
                        })?
                        .1
                        .to_vec()
                } else {
                    return Err(Status::internal(format!(
                        "Ошибка инференса Yunet: {}",
                        err_text
                    )));
                }
            }
        };

        let outputs_yunet_slice = outputs_yunet_vec.as_slice();

        let detected_faces = decode_yunet_output(
            outputs_yunet_slice,
            original_width,
            original_height,
            0.9, // score_threshold (tunable)
            0.3, // nms_threshold (tunable)
        )
        .map_err(|e| Status::internal(format!("Ошибка декодирования вывода Yunet: {}", e)))?;

        let Some(best_face) = detected_faces.into_iter().max_by(|a, b| {
            a.area()
                .partial_cmp(&b.area())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            warn!("Лицо не обнаружено");
            return Ok(Response::new(BioResult {
                detected: false,
                is_live: false,
                liveness_score: 0.0,
                embedding: vec![],
                error_msg: "Лицо не обнаружено".to_string(),
                execution_provider: self
                    .models
                    .provider
                    .lock()
                    .map(|v| v.clone())
                    .unwrap_or_else(|_| "unknown".to_string()),
            }));
        };

        // Crop the face
        let x = best_face.bbox[0].max(0.0).floor() as u32;
        let y = best_face.bbox[1].max(0.0).floor() as u32;
        let w = best_face.bbox[2]
            .min(original_width as f32 - x as f32)
            .floor() as u32;
        let h = best_face.bbox[3]
            .min(original_height as f32 - y as f32)
            .floor() as u32;

        let cropped_face_img =
            ImageBuffer::from_fn(w, h, |px, py| rgb_img.get_pixel(x + px, y + py).to_owned());

        // --- Шаг 3: Распознавание (ArcFace) ---
        let resized_arc =
            image::imageops::resize(&cropped_face_img, 112, 112, FilterType::Triangle);
        let mut input_tensor_arc = Array4::<f32>::zeros((1, 112, 112, 3));

        for (x_i, y_i, pixel) in resized_arc.enumerate_pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            // (x - 127.5) / 128.0
            input_tensor_arc[[0, y_i as usize, x_i as usize, 0]] = (r - 127.5) / 128.0;
            input_tensor_arc[[0, y_i as usize, x_i as usize, 1]] = (g - 127.5) / 128.0;
            input_tensor_arc[[0, y_i as usize, x_i as usize, 2]] = (b - 127.5) / 128.0;
        }

        let (data_vec_arc, _) = input_tensor_arc.into_raw_vec_and_offset();
        let input_value_arc =
            Tensor::from_array((vec![1, 112, 112, 3], data_vec_arc)).map_err(|e| {
                Status::internal(format!("Ошибка создания тензора Ort (ArcFace): {}", e))
            })?;

        let mut session_arc = self
            .models
            .arcface
            .lock()
            .map_err(|_| Status::internal("Не удалось захватить мьютекс ArcFace"))?;

        let outputs_arc = session_arc
            .run(inputs![input_value_arc])
            .map_err(|e| Status::internal(format!("Ошибка инференса ArcFace: {}", e)))?;

        let (_, embedding_slice) = outputs_arc[0].try_extract_tensor::<f32>().map_err(|e| {
            Status::internal(format!("Ошибка извлечения результата ArcFace: {}", e))
        })?;

        let embedding_vec: Vec<f32> = embedding_slice.to_vec();

        // --- Шаг 4: Liveness (MiniFASNetV2) ---
        let resized_live =
            image::imageops::resize(&cropped_face_img, 128, 128, FilterType::Triangle);
        let mut input_tensor_live = Array4::<f32>::zeros((1, 3, 128, 128));

        for (x_i, y_i, pixel) in resized_live.enumerate_pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            // 0-1
            input_tensor_live[[0, 0, y_i as usize, x_i as usize]] = r / 255.0;
            input_tensor_live[[0, 1, y_i as usize, x_i as usize]] = g / 255.0;
            input_tensor_live[[0, 2, y_i as usize, x_i as usize]] = b / 255.0;
        }

        let (data_vec_live, _) = input_tensor_live.into_raw_vec_and_offset();
        let input_value_live =
            Tensor::from_array((vec![1, 3, 128, 128], data_vec_live)).map_err(|e| {
                Status::internal(format!("Ошибка создания тензора Ort (Liveness): {}", e))
            })?;

        let mut session_live = self
            .models
            .liveness
            .lock()
            .map_err(|_| Status::internal("Не удалось захватить мьютекс Liveness"))?;

        let outputs_live = session_live
            .run(inputs![input_value_live])
            .map_err(|e| Status::internal(format!("Ошибка инференса Liveness: {}", e)))?;

        let (_, live_out) = outputs_live[0].try_extract_tensor::<f32>().map_err(|e| {
            Status::internal(format!("Ошибка извлечения результата Liveness: {}", e))
        })?;

        let liveness_score = if live_out.len() >= 2 {
            // Softmax
            let exp_sum: f32 = live_out.iter().map(|x| x.exp()).sum();
            live_out[1].exp() / exp_sum
        } else {
            live_out.get(0).cloned().unwrap_or(0.0)
        };

        let is_live = liveness_score > 0.5;

        let provider = self
            .models
            .provider
            .lock()
            .map(|v| v.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        debug!(
            "Обработка завершена. Liveness: {}, Detected: true, Provider: {}",
            is_live, provider
        );

        // Шаг 5: Возврат BioResult
        Ok(Response::new(BioResult {
            detected: true,
            is_live,
            liveness_score,
            embedding: embedding_vec,
            error_msg: String::new(),
            execution_provider: self
                .models
                .provider
                .lock()
                .map(|v| v.clone())
                .unwrap_or_else(|_| "unknown".to_string()),
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

    // Папка с моделями
    let possible_paths = ["vision-worker/models", "models"];
    let mut models_dir = "models"; // fallback
    for path in &possible_paths {
        if Path::new(path).exists() && Path::new(path).join("MiniFASNetV2.onnx").exists() {
            models_dir = path;
            break;
        }
    }

    // Инициализация хранилища моделей
    info!("Инициализация Vision Worker...");

    let models = match ModelStore::new(models_dir) {
        Ok(store) => Arc::new(store),
        Err(e) => {
            error!("Не удалось загрузить модели: {:?}", e);
            return Err(e.into());
        }
    };

    let addr = "0.0.0.0:50052".parse()?;
    let provider = models
        .provider
        .lock()
        .map(|v| v.clone())
        .unwrap_or_else(|_| "unknown".to_string());
    info!("Vision Worker слушает на {} (Provider: {})", addr, provider);

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);

    let service = VisionService {
        models,
        shutdown_tx,
    };

    Server::builder()
        .add_service(VisionServer::new(service))
        .serve_with_shutdown(addr, async {
            shutdown_rx.recv().await;
            info!("Gracefully shutting down Vision Worker");
        })
        .await?;

    Ok(())
}
