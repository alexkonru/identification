use glob::glob;
use pgvector::Vector;
use shared::biometry::gatekeeper_server::{Gatekeeper, GatekeeperServer};
use shared::biometry::{
    AccessCheckResponse, AddDeviceRequest, AddRoomRequest, AddZoneRequest, CheckAccessRequest,
    ControlDoorRequest, ControlServiceRequest, Device, Empty, GetLogsRequest, GetLogsResponse,
    GetUserAccessResponse, GrantAccessRequest, IdRequest, IdResponse, ImageFrame,
    ListDevicesRequest, ListDevicesResponse, ListRoomsRequest, ListRoomsResponse, ListUsersRequest,
    ListUsersResponse, ListZonesRequest, ListZonesResponse, LogEntry, RegisterUserRequest, Room,
    ScanHardwareResponse, ServiceStatus, SetAccessRulesRequest, StatusResponse,
    SystemStatusResponse, User, Zone, audio_client::AudioClient, vision_client::VisionClient,
};
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres, Row};
use tokio::sync::mpsc;
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

pub struct GatewayService {
    pool: Pool<Postgres>,
    audio_channel: Channel,
    vision_channel: Channel,
    face_similarity_threshold: f64,
    test_device_id: i32,
    shutdown_tx: mpsc::Sender<()>,
}

impl GatewayService {
    pub fn new(
        pool: Pool<Postgres>,
        audio_channel: Channel,
        vision_channel: Channel,
        face_similarity_threshold: f64,
        test_device_id: i32,
        shutdown_tx: mpsc::Sender<()>,
    ) -> Self {
        Self {
            pool,
            audio_channel,
            vision_channel,
            face_similarity_threshold,
            test_device_id,
            shutdown_tx,
        }
    }

    // Updated to return full details
    async fn process_face_full(&self, image: &[u8]) -> Result<Option<FaceProcessResult>, Status> {
        let mut client = VisionClient::new(self.vision_channel.clone());
        let request = Request::new(ImageFrame {
            content: image.to_vec(),
        });

        match client.process_face(request).await {
            Ok(response) => {
                let res = response.into_inner();
                if res.detected && !res.embedding.is_empty() {
                    Ok(Some(FaceProcessResult {
                        embedding: res.embedding,
                        liveness_score: res.liveness_score,
                        is_live: res.is_live,
                        provider: res.execution_provider,
                    }))
                } else {
                    Ok(None)
                }
            }
            Err(e) => Err(Status::internal(format!("Vision service error: {}", e))),
        }
    }

    async fn process_voice_embedding(&self, voice: &[u8]) -> Result<Option<Vec<f32>>, Status> {
        let mut client = AudioClient::new(self.audio_channel.clone());
        let request = Request::new(shared::biometry::AudioChunk {
            content: voice.to_vec(),
            sample_rate: 16000,
        });

        match client.process_voice(request).await {
            Ok(response) => {
                let bio_result = response.into_inner();
                if bio_result.detected && !bio_result.embedding.is_empty() {
                    Ok(Some(bio_result.embedding))
                } else {
                    Ok(None)
                }
            }
            Err(e) => Err(Status::internal(format!("Audio service error: {}", e))),
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
}

#[tonic::async_trait]
impl Gatekeeper for GatewayService {
    // --- Пользователи ---
    async fn register_user(
        &self,
        request: Request<RegisterUserRequest>,
    ) -> Result<Response<IdResponse>, Status> {
        let req = request.into_inner();

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

        // 2. Process Images
        for image_bytes in req.images {
            if let Ok(Some(embedding)) = self.process_face_full(&image_bytes).await {
                sqlx::query(
                    "INSERT INTO face_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                )
                .bind(user_id)
                .bind(Vector::from(embedding.embedding))
                .execute(&mut *tx)
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            }
        }

        // 3. Process Voices
        for voice_bytes in req.voices {
            if let Ok(Some(embedding)) = self.process_voice_embedding(&voice_bytes).await {
                sqlx::query(
                    "INSERT INTO voice_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                )
                .bind(user_id)
                .bind(Vector::from(embedding))
                .execute(&mut *tx)
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            }
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

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        // Process Images
        for image_bytes in req.images {
            if let Ok(Some(embedding)) = self.process_face_full(&image_bytes).await {
                sqlx::query(
                    "INSERT INTO face_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                )
                .bind(user_id)
                .bind(Vector::from(embedding.embedding))
                .execute(&mut *tx)
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            }
        }

        // Process Voices
        for voice_bytes in req.voices {
            if let Ok(Some(embedding)) = self.process_voice_embedding(&voice_bytes).await {
                sqlx::query(
                    "INSERT INTO voice_embeddings (user_id, embedding) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                )
                .bind(user_id)
                .bind(Vector::from(embedding))
                .execute(&mut *tx)
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            }
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

        let row = sqlx::query(
            r#"
            INSERT INTO devices (name, room_id, device_type_id, connection_string)
            VALUES ($1, $2, (SELECT id FROM device_types WHERE name = $3), $4)
            RETURNING id
            "#,
        )
        .bind(req.name)
        .bind(req.room_id)
        .bind(req.device_type)
        .bind(req.connection_string)
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
        sqlx::query("DELETE FROM devices WHERE id = $1")
            .bind(req.id)
            .execute(&self.pool)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
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

        // 1. Process Face
        let face_res_opt = self.process_face_full(&req.image).await?;

        let face_info = match face_res_opt {
            Some(info) => info,
            None => {
                let log = format!("[Vision] No face detected. (Device: {})", device_id);
                self.log_access(None, device_id, false, &log).await;
                return Ok(Response::new(AccessCheckResponse {
                    granted: false,
                    user_name: "Unknown".into(),
                    message: "No face detected".into(),
                }));
            }
        };

        // Liveness check log part
        let liveness_log = format!(
            "Liveness: {:.2}% ({}) on {}",
            face_info.liveness_score * 100.0,
            if face_info.is_live { "Real" } else { "Fake" },
            face_info.provider
        );

        // 2. Find User (Vector Search)
        // distance < 0.6 is typical threshold
        let user_match = sqlx::query(
            r#"
            SELECT u.id, u.name, (e.embedding <-> $1) as distance
            FROM face_embeddings e
            JOIN users u ON e.user_id = u.id
            ORDER BY distance ASC
            LIMIT 1
            "#,
        )
        .bind(Vector::from(face_info.embedding))
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Status::internal(format!("DB Error: {}", e)))?;

        let (user_id, user_name, dist) = match user_match {
            Some(rec) => {
                let d: f64 = rec.get::<Option<f64>, _>("distance").unwrap_or(1.0);
                let uid: i32 = rec.get("id");
                let uname: String = rec.get("name");
                (uid, uname, d)
            }
            None => (0, "Unknown".to_string(), 1.0),
        };

        // Decision Logic
        let is_match = dist < self.face_similarity_threshold;
        let is_live = face_info.is_live;

        // Debug string for UI
        let debug_info = format!(
            "{} | Match: {} (Dist: {:.3})",
            liveness_log,
            if is_match { &user_name } else { "None" },
            dist
        );

        if !is_live {
            let final_log = format!("DENIED [Spoofing] {}", debug_info);
            self.log_access(
                if is_match { Some(user_id) } else { None },
                device_id,
                false,
                &final_log,
            )
            .await;
            return Ok(Response::new(AccessCheckResponse {
                granted: false,
                user_name: "Spoof Attempt".into(),
                message: format!("Liveness Failed. {}", debug_info),
            }));
        }

        if !is_match {
            let final_log = format!("DENIED [Unknown User] {}", debug_info);
            self.log_access(None, device_id, false, &final_log).await;
            return Ok(Response::new(AccessCheckResponse {
                granted: false,
                user_name: "Unknown".into(),
                message: format!("Unknown User. {}", debug_info),
            }));
        }

        // 3. Check Permissions (Zone)
        if device_id == self.test_device_id {
            // Test Device
            let final_log = format!("GRANTED [Test Device] {}", debug_info);
            self.log_access(Some(user_id), device_id, true, &final_log)
                .await;
            return Ok(Response::new(AccessCheckResponse {
                granted: true,
                user_name,
                message: format!("Welcome (Test). {}", debug_info),
            }));
        }

        // Find Zone and Room
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
                let final_log = format!(
                    "DENIED [Config Error] Device {} not in zone. {}",
                    device_id, debug_info
                );
                self.log_access(Some(user_id), device_id, false, &final_log)
                    .await;
                return Err(Status::not_found("Device not found/assigned"));
            }
        };

        // Check Rules: user can be granted by zone OR by room
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
            let final_log = format!(
                "GRANTED [Zone: {}, Room: {}] {}",
                zone_name, room_name, debug_info
            );
            self.log_access(Some(user_id), device_id, true, &final_log)
                .await;
            Ok(Response::new(AccessCheckResponse {
                granted: true,
                user_name,
                message: format!("Welcome to {} / {}. {}", zone_name, room_name, debug_info),
            }))
        } else {
            let final_log = format!(
                "DENIED [Zone: {}, Room: {}] No rights. {}",
                zone_name, room_name, debug_info
            );
            self.log_access(Some(user_id), device_id, false, &final_log)
                .await;
            Ok(Response::new(AccessCheckResponse {
                granted: false,
                user_name,
                message: format!("Access Denied. {}", debug_info),
            }))
        }
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
                message: "Running".to_string(),
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

    async fn control_door(
        &self,
        _request: Request<ControlDoorRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        Ok(Response::new(StatusResponse {
            success: true,
            message: "Command sent".to_string(),
        }))
    }

    async fn scan_hardware(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ScanHardwareResponse>, Status> {
        let mut found_devices: Vec<Device> = Vec::new();

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

        // 2. Scan for local /dev/video* devices
        if let Ok(paths) = glob("/dev/video*") {
            for entry in paths {
                if let Ok(path) = entry {
                    let path_str = path.to_string_lossy();
                    if let Some(num_str) = path_str.strip_prefix("/dev/video") {
                        if let Ok(num) = num_str.parse::<i32>() {
                            let conn_str = num.to_string();
                            // Check if a device with this connection string already exists
                            if !db_devices.iter().any(|d| d.connection_string == conn_str) {
                                found_devices.push(Device {
                                    id: -1, // Not in DB
                                    room_id: 0,
                                    name: format!("New Camera {}", num),
                                    device_type: "camera".to_string(),
                                    connection_string: conn_str,
                                    is_active: true,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Add default device if nothing is found at all
        if found_devices.is_empty() {
            found_devices.push(self.get_default_test_device().await);
        }

        Ok(Response::new(ScanHardwareResponse { found_devices }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://user:password@localhost:5432/biometry_db".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    let addr = "0.0.0.0:50051".parse()?;

    let audio_url = "http://127.0.0.1:50053";
    tracing::info!("Connecting to Audio Worker at {}", audio_url);
    let audio_channel = tonic::transport::Endpoint::from_static(audio_url).connect_lazy();

    let vision_url = "http://127.0.0.1:50052";
    tracing::info!("Connecting to Vision Worker at {}", vision_url);
    let vision_channel = tonic::transport::Endpoint::from_static(vision_url).connect_lazy();

    let face_similarity_threshold = std::env::var("FACE_SIMILARITY_THRESHOLD")
        .unwrap_or_else(|_| "0.6".to_string())
        .parse::<f64>()
        .expect("FACE_SIMILARITY_THRESHOLD must be a valid f64");

    let test_device_id = std::env::var("TEST_DEVICE_ID")
        .unwrap_or_else(|_| "9999".to_string())
        .parse::<i32>()
        .expect("TEST_DEVICE_ID must be a valid i32");

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);

    let gateway = GatewayService::new(
        pool,
        audio_channel,
        vision_channel,
        face_similarity_threshold,
        test_device_id,
        shutdown_tx,
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
