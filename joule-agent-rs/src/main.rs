use axum::{routing::{get, post}, Json, Router};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc, time::{Duration, Instant}};
use sysinfo::System;
use chrono::Utc;

// NVML (GPU power, optional)
fn gpu_power_w(nvml: &Option<nvml_wrapper::Nvml>) -> f64 {
    if let Some(n) = nvml {
        if let Ok(dev) = n.device_by_index(0) {
            if let Ok(mw) = dev.power_usage() { return (mw as f64) / 1000.0; }
        }
    }
    0.0
}

#[derive(Clone)]
struct Cfg { cpu_tdp_w: f64, smoothing_alpha: f64, hz: f64, idle_learn_w: f64 }

#[derive(Default, Clone)]
struct Last {
    ts: f64,
    gpu_w: f64,
    cpu_w: f64,
    idle_gpu_w: f64,
    idle_cpu_w: f64,
    net_w: f64,
    bucket_j: f64,
}
// keep last sample in shared Arc so it can be sent across tasks

#[derive(Clone)]
struct State {
    cfg: Cfg,
    bucket_j: Arc<Mutex<f64>>,
    idle_gpu_w: Arc<Mutex<f64>>,
    idle_cpu_w: Arc<Mutex<f64>>,
    last: Arc<Mutex<Last>>,
}

#[derive(Deserialize)] struct TakeReq { joules: f64 }
#[derive(Serialize)]   struct TakeResp { ok: bool, remaining_j: f64 }

#[tokio::main]
async fn main() {
    let cfg = Cfg {
        cpu_tdp_w: env_f("JOULE_CPU_TDP_W", 65.0),
        smoothing_alpha: env_f("JOULE_SMOOTHING", 0.2),
        hz: env_f("JOULE_HZ", 1.0),
        idle_learn_w: env_f("JOULE_IDLE_LEARN_W", 5.0),
    };
    let st = State {
        cfg: cfg.clone(),
        bucket_j: Arc::new(Mutex::new(0.0)),
        idle_gpu_w: Arc::new(Mutex::new(20.0)),
        idle_cpu_w: Arc::new(Mutex::new(15.0)),
        last: Arc::new(Mutex::new(Last::default())),
    };

    // Try NVML
    let nvml = nvml_wrapper::Nvml::init().ok();

    // Sampler loop
    let st_loop = st.clone();
    tokio::spawn(async move {
        let mut sys = System::new();
        let period = Duration::from_secs_f64(1.0 / st_loop.cfg.hz.max(0.1));
        loop {
            let loop_start = Instant::now();
            sys.refresh_cpu();
            let cpu_usage = avg_cpu_usage(&sys); // 0..100
            let cpu_w = (cpu_usage as f64 / 100.0) * st_loop.cfg.cpu_tdp_w;
            let gpu_w = gpu_power_w(&nvml);

            // read current idles and update EMA baseline in a tight scope so guards are dropped
            {
                let mut idle_g = st_loop.idle_gpu_w.lock();
                let mut idle_c = st_loop.idle_cpu_w.lock();
                let net_w_raw = (gpu_w - *idle_g).max(0.0) + (cpu_w - *idle_c).max(0.0);
                // update EMA baseline **only** when net power ~ idle
                if net_w_raw < st_loop.cfg.idle_learn_w {
                    *idle_g = st_loop.cfg.smoothing_alpha * gpu_w + (1.0 - st_loop.cfg.smoothing_alpha) * *idle_g;
                    *idle_c = st_loop.cfg.smoothing_alpha * cpu_w + (1.0 - st_loop.cfg.smoothing_alpha) * *idle_c;
                }
            }

            let idle_g_now = *st_loop.idle_gpu_w.lock();
            let idle_c_now = *st_loop.idle_cpu_w.lock();
            let net_w = (gpu_w - idle_g_now).max(0.0) + (cpu_w - idle_c_now).max(0.0);

            // integrate Joules (use sampling period, not loop elapsed time)
            let dt = period.as_secs_f64();
            {
                let mut b = st_loop.bucket_j.lock();
                *b += net_w * dt;
            }

            // publish last sample
            {
                let mut s = st_loop.last.lock();
                s.ts = Utc::now().timestamp_millis() as f64 / 1000.0;
                s.gpu_w = gpu_w; s.cpu_w = cpu_w;
                s.idle_gpu_w = idle_g_now; s.idle_cpu_w = idle_c_now;
                s.net_w = net_w; s.bucket_j = *st_loop.bucket_j.lock();
            }

            // cadence
            let slip = loop_start.elapsed();
            let wait = if period > slip { period - slip } else { Duration::from_millis(0) };
            tokio::time::sleep(wait).await;
        }
    });

    // HTTP API
    let app = Router::new()
        .route("/v1/sample", get({
            let st = st.clone();
            move || async move {
                let s = st.last.lock();
                Json(serde_json::json!({
                    "ts": s.ts,
                    "gpu_w": s.gpu_w,
                    "cpu_w": s.cpu_w,
                    "idle_gpu_w": s.idle_gpu_w,
                    "idle_cpu_w": s.idle_cpu_w,
                    "net_w": s.net_w,
                    "bucket_j": s.bucket_j,
                    "hash": blake3::hash(format!("{}:{}", s.ts, s.bucket_j).as_bytes()).to_hex().to_string()
                }))
            }
        }))
        .route("/v1/take", post({
            let st = st.clone();
            move |Json(req): Json<TakeReq>| async move {
                let mut b = st.bucket_j.lock();
                if *b >= req.joules { *b -= req.joules; Json(TakeResp { ok: true, remaining_j: *b }) }
                else { Json(TakeResp { ok: false, remaining_j: *b }) }
            }
        }));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8787));
    println!("[JouleAgent] listening on http://{}", addr);
    // bind a TcpListener and serve via axum::serve for compatibility
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn env_f(key: &str, def: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}
fn avg_cpu_usage(sys: &System) -> f32 {
    let cpus = sys.cpus(); if cpus.is_empty() { return 20.0; }
    let mut s = 0.0; for c in cpus { s += c.cpu_usage(); } s / (cpus.len() as f32)
}
