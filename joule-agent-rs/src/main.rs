use axum::{routing::{get, post}, Json, Router};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{fs, net::SocketAddr, path::Path, sync::Arc, time::{Duration, Instant}};
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

#[derive(Clone, Deserialize)]
struct JouleAgentConfig {
    #[serde(default = "default_hz")]
    hz: f64,
    #[serde(default = "default_cpu_tdp_w")]
    cpu_tdp_w: f64,
    #[serde(default = "default_smoothing_alpha")]
    smoothing_alpha: f64,
    #[serde(default = "default_idle_learn_w")]
    idle_learn_w: f64,
    #[serde(default = "default_bind_addr")]
    bind_addr: String,
}

#[derive(Deserialize)]
struct RootConfig {
    #[serde(default)]
    joule_agent: JouleAgentConfig,
}

fn default_hz() -> f64 { 2.0 }
fn default_cpu_tdp_w() -> f64 { 65.0 }
fn default_smoothing_alpha() -> f64 { 0.2 }
fn default_idle_learn_w() -> f64 { 5.0 }
fn default_bind_addr() -> String { "127.0.0.1:8787".to_string() }

impl Default for JouleAgentConfig {
    fn default() -> Self {
        Self {
            hz: default_hz(),
            cpu_tdp_w: default_cpu_tdp_w(),
            smoothing_alpha: default_smoothing_alpha(),
            idle_learn_w: default_idle_learn_w(),
            bind_addr: default_bind_addr(),
        }
    }
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

fn load_config() -> JouleAgentConfig {
    // Try to load config.toml from repo root (parent directory)
    let config_paths = vec![
        Path::new("../config.toml"),
        Path::new("config.toml"),
        Path::new("../../config.toml"), // in case running from deep nested dir
    ];

    let mut base_config = JouleAgentConfig::default();

    for path in config_paths {
        if path.exists() {
            if let Ok(contents) = fs::read_to_string(path) {
                if let Ok(root) = toml::from_str::<RootConfig>(&contents) {
                    base_config = root.joule_agent;
                    println!("[JouleAgent] Loaded config from {}", path.display());
                    break;
                }
            }
        }
    }

    // Environment variables override config.toml
    if let Ok(v) = std::env::var("JOULE_HZ") {
        if let Ok(val) = v.parse::<f64>() {
            base_config.hz = val;
            println!("[JouleAgent] Override hz={} from JOULE_HZ env", val);
        }
    }
    if let Ok(v) = std::env::var("JOULE_CPU_TDP_W") {
        if let Ok(val) = v.parse::<f64>() {
            base_config.cpu_tdp_w = val;
            println!("[JouleAgent] Override cpu_tdp_w={} from JOULE_CPU_TDP_W env", val);
        }
    }
    if let Ok(v) = std::env::var("JOULE_SMOOTHING") {
        if let Ok(val) = v.parse::<f64>() {
            base_config.smoothing_alpha = val;
            println!("[JouleAgent] Override smoothing_alpha={} from JOULE_SMOOTHING env", val);
        }
    }
    if let Ok(v) = std::env::var("JOULE_IDLE_LEARN_W") {
        if let Ok(val) = v.parse::<f64>() {
            base_config.idle_learn_w = val;
            println!("[JouleAgent] Override idle_learn_w={} from JOULE_IDLE_LEARN_W env", val);
        }
    }
    if let Ok(v) = std::env::var("JOULE_BIND_ADDR") {
        base_config.bind_addr = v;
        println!("[JouleAgent] Override bind_addr={} from JOULE_BIND_ADDR env", base_config.bind_addr);
    }

    base_config
}

#[tokio::main]
async fn main() {
    let agent_cfg = load_config();

    let cfg = Cfg {
        cpu_tdp_w: agent_cfg.cpu_tdp_w,
        smoothing_alpha: agent_cfg.smoothing_alpha,
        hz: agent_cfg.hz,
        idle_learn_w: agent_cfg.idle_learn_w,
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

    // Parse bind address from config
    let addr: SocketAddr = agent_cfg.bind_addr.parse()
        .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], 8787)));
    println!("[JouleAgent] listening on http://{}", addr);
    // bind a TcpListener and serve via axum::serve for compatibility
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn avg_cpu_usage(sys: &System) -> f32 {
    let cpus = sys.cpus(); if cpus.is_empty() { return 20.0; }
    let mut s = 0.0; for c in cpus { s += c.cpu_usage(); } s / (cpus.len() as f32)
}
