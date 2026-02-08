use chrono::Local;
use dotenv::dotenv;
use opencv::{
    core::Vector,
    imgcodecs,
    prelude::*,
    videoio,
};
use std::env;
use std::time::Duration;
use tokio::time;

pub mod biometry {
    tonic::include_proto!("biometry");
}

use biometry::gatekeeper_client::GatekeeperClient;
use biometry::{CheckAccessRequest, ListDevicesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // 1. Connect to Gateway
    let gateway_url = env::var("GATEWAY_URL").unwrap_or_else(|_| "http://0.0.0.0:50051".to_string());
    println!("Connecting to Gateway at {}", gateway_url);

    let mut client = GatekeeperClient::connect(gateway_url.clone()).await?;

    // 2. List Devices
    let response = client
        .list_devices(ListDevicesRequest {})
        .await?
        .into_inner();

    let cameras: Vec<_> = response
        .devices
        .into_iter()
        .filter(|d| d.device_type == "camera")
        .collect();

    if cameras.is_empty() {
        println!("No cameras found.");
        return Ok(())
    }

    println!("Found {} cameras. Starting workers...", cameras.len());

    // 3. Spawn Tasks
    let mut handles = vec![];

    for camera in cameras {
        // Better to clone the channel/client if possible, but creating new one is safe. 
        // Tonic clients are cheap to clone.
        let client_clone = client.clone();
        
        let handle = tokio::spawn(async move {
            camera_worker(camera.id, camera.connection_string, client_clone).await;
        });
        handles.push(handle);
    }

    // Keep main alive
    for handle in handles {
        let _ = handle.await;
    }

    Ok(())
}

async fn camera_worker(device_id: i32, connection_string: String, mut client: GatekeeperClient<tonic::transport::Channel>) {
    println!("Starting worker for camera {} ({})", device_id, connection_string);

    loop {
        // Attempt to open camera
        let mut cam = if let Ok(index) = connection_string.parse::<i32>() {
            videoio::VideoCapture::new(index, videoio::CAP_ANY).unwrap() // Handle error properly in real app
        } else {
            videoio::VideoCapture::from_file(&connection_string, videoio::CAP_ANY).unwrap()
        };

        if !cam.is_opened().unwrap_or(false) {
            eprintln!("Failed to open camera {}. Retrying in 5s...", device_id);
            time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        loop {
            let mut frame = Mat::default();
            match cam.read(&mut frame) {
                Ok(true) => {
                    // Encode frame
                    let mut buf = Vector::new();
                    let params = Vector::new(); // Assuming params is also a Vector, adjust if it's different
                    if let Ok(_) = imgcodecs::imencode(".jpg", &frame, &mut buf, &params) {
                        let bytes = buf.to_vec();

                        let request = CheckAccessRequest {
                            device_id,
                            image: bytes,
                        };

                        match client.check_access(request).await {
                            Ok(response) => {
                                let resp = response.into_inner();
                                if resp.granted {
                                    let user = resp.user_name;
                                    println!("[OPEN] Access GRANTED for {} at {}", user, device_id);
                                    log_access(device_id, &user).await;
                                } else {
                                    println!("[DENIED] Access denied at {}", device_id);
                                }
                            }
                            Err(e) => {
                                eprintln!("gRPC error for camera {}: {}", device_id, e);
                            }
                        }
                    }
                }
                Ok(false) => {
                    eprintln!("Camera {} stream ended/failed. Reconnecting...", device_id);
                    break; // Break inner loop to reconnect
                }
                Err(e) => {
                    eprintln!("Error reading camera {}: {}. Reconnecting...", device_id, e);
                    break;
                }
            }

            // Limit FPS (e.g., 2 FPS = 500ms)
            time::sleep(Duration::from_millis(500)).await;
        }
        
        // Clean up before retrying
        cam.release().ok();
        time::sleep(Duration::from_secs(5)).await;
    }
}

async fn log_access(device_id: i32, user: &str) {
    let now = Local::now();
    let log_line = format!("{} - Access granted for {} at device {}\n", now.format("%Y-%m-%d %H:%M:%S"), user, device_id); // Corrected: escaped backslash for newline
    
    // Async file append? Standard fs is blocking. 
    // For simplicity/robustness in this snippet, using std::fs::OpenOptions in a blocking way 
    // or tokio::fs. Let's use tokio::fs for correctness in async context.
    
    use tokio::fs::OpenOptions;
    use tokio::io::AsyncWriteExt;

    let result = OpenOptions::new()
        .create(true)
        .append(true)
        .open("door_access.log")
        .await;

    if let Ok(mut file) = result {
        let _ = file.write_all(log_line.as_bytes()).await;
    }
}
