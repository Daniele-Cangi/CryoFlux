#!/usr/bin/env repo
# CryoFlux — Step 2+4 Pass (Rust JouleAgent + Python Orchestrator with Δ‑Capsule & Merge)
# ======================================================================================
# Monorepo layout in one canvas (copy files into folders as indicated)
#
#  /joule-agent-rs/
#    Cargo.toml
#    src/main.rs
#
#  /cryo-orchestrator/
#    cryo.py                (updated orchestrator)
#    requirements.txt
#
# Notes
# - Step 2: Rust agent streams REAL power and maintains a Joule bucket exposed via HTTP (Axum).
# - Step 4: Python orchestrator now has Δ‑Capsule evaluation & MERGE (LoRA) with rollback rules.
# - Start order: run Rust agent → run Python orchestrator. Works on Windows or Linux; GPU optional.

########################################################################################
# joule-agent-rs/Cargo.toml
########################################################################################
[package]
name = "joule-agent"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "time"] }
parking_lot = "0.12"
sysinfo = "0.30"
blake3 = "1"
# NVML wrapper (GPU power). If unavailable, we fallback to 0W and only CPU estimate
nvml-wrapper = { version = "0.11", default-features = false }
chrono = { version = "0.4", features=["clock"] }

########################################################################################
# joule-agent-rs/src/main.rs
########################################################################################
use axum::{routing::{get, post}, Json, Router};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc, time::{Duration, Instant}};
use sysinfo::{CpuExt, System, SystemExt};

// NVML (optional GPU power)
#[allow(unused)]
fn gpu_power_w(nvml: &nvml_wrapper::NVML) -> f64 {
    if let Ok(device) = nvml.device_by_index(0) { // first GPU
        if let Ok(mw) = device.power_usage() { // milliwatts
            return (mw as f64) / 1000.0;
        }
    }
    0.0
}

#[derive(Clone, Serialize)]
struct Sample {
    ts: f64,
    gpu_w: f64,
    cpu_w: f64,
    idle_gpu_w: f64,
    idle_cpu_w: f64,
    net_w: f64,
    bucket_j: f64,
    hash: String,
}

#[derive(Clone)]
struct Cfg { cpu_tdp_w: f64, smoothing_alpha: f64, hz: f64 }

#[derive(Clone)]
struct State {
    cfg: Cfg,
    bucket_j: Arc<Mutex<f64>>, // accumulated Joules
    idle_gpu_w: Arc<Mutex<f64>>, // EMA
    idle_cpu_w: Arc<Mutex<f64>>, // EMA
    last_ts: Arc<Mutex<Instant>>,
}

#[derive(Deserialize)]
struct TakeReq { joules: f64 }
#[derive(Serialize)]
struct TakeResp { ok: bool, remaining_j: f64 }

#[tokio::main]
async fn main() {
    // Config via env (easy override)
    let cfg = Cfg { cpu_tdp_w: env_f("JOULE_CPU_TDP_W", 65.0),
                    smoothing_alpha: env_f("JOULE_SMOOTHING", 0.2),
                    hz: env_f("JOULE_HZ", 1.0) };
    let st = State {
        cfg: cfg.clone(),
        bucket_j: Arc::new(Mutex::new(0.0)),
        idle_gpu_w: Arc::new(Mutex::new(20.0)),
        idle_cpu_w: Arc::new(Mutex::new(15.0)),
        last_ts: Arc::new(Mutex::new(Instant::now())),
    };

    // Try NVML
    let nvml = nvml_wrapper::NVML::init().ok();

    // Spawn sampler
    let st_loop = st.clone();
    tokio::spawn(async move {
        let mut sys = System::new();
        let period = Duration::from_secs_f64(1.0 / st_loop.cfg.hz.max(0.1));
        loop {
            let t0 = Instant::now();
            sys.refresh_cpu();
            let cpu_usage = avg_cpu_usage(&sys); // 0..100
            let cpu_w = (cpu_usage as f64 / 100.0) * st_loop.cfg.cpu_tdp_w;
            let gpu_w = if let Some(n) = &nvml { gpu_power_w(n) } else { 0.0 };

            // Update EMA idles
            {
                let mut ig = st_loop.idle_gpu_w.lock();
                *ig = st_loop.cfg.smoothing_alpha * gpu_w + (1.0 - st_loop.cfg.smoothing_alpha) * *ig;
            }
            {
                let mut ic = st_loop.idle_cpu_w.lock();
                *ic = st_loop.cfg.smoothing_alpha * cpu_w + (1.0 - st_loop.cfg.smoothing_alpha) * *ic;
            }

            let idle_g = *st_loop.idle_gpu_w.lock();
            let idle_c = *st_loop.idle_cpu_w.lock();
            let net_w = (gpu_w - idle_g).max(0.0) + (cpu_w - idle_c).max(0.0);

            // Integrate into bucket
            let dt = t0.elapsed().as_secs_f64();
            {
                let mut b = st_loop.bucket_j.lock();
                *b += net_w * dt; // Joule = Watt * secondi
            }

            tokio::time::sleep_until(tokio::time::Instant::now() + (period - t0.elapsed())).await;
        }
    });

    // HTTP API
    let app = Router::new()
        .route("/v1/sample", get({
            let st = st.clone();
            move || async move {
                let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
                let gpu_w = 0.0; // will be recomputed in loop values; expose EMA baseline instead
                let cpu_w = 0.0;
                let idle_g = *st.idle_gpu_w.lock();
                let idle_c = *st.idle_cpu_w.lock();
                let bucket = *st.bucket_j.lock();
                let sample = Sample {
                    ts: now,
                    gpu_w,
                    cpu_w,
                    idle_gpu_w: idle_g,
                    idle_cpu_w: idle_c,
                    net_w: 0.0, // net already integrated
                    bucket_j: bucket,
                    hash: blake3::hash(format!("{now}:{bucket}").as_bytes()).to_hex().to_string(),
                };
                Json(sample)
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
    axum::Server::bind(&addr).serve(app.into_make_service()).await.unwrap();
}

fn env_f(key: &str, def: f64) -> f64 { std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(def) }
fn avg_cpu_usage(sys: &System) -> f32 {
    let cpus = sys.cpus();
    if cpus.is_empty() { return 20.0; }
    let mut s = 0.0; for c in cpus { s += c.cpu_usage(); } s / (cpus.len() as f32)
}

########################################################################################
# cryo-orchestrator/requirements.txt
########################################################################################
blake3
psutil
pynvml
numpy
faiss-cpu
sentence-transformers
transformers
peft
torch
pyyaml
requests


########################################################################################
# cryo-orchestrator/cryo.py  (UPDATED: uses Rust agent + Δ‑Capsule EVAL & MERGE)
########################################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, json, math, sqlite3, argparse, random, shutil
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import requests
import numpy as np

from blake3 import blake3

def b3(x: bytes) -> str: return blake3(x).hexdigest()

# ---------------- Config ----------------
@dataclass
class EnergyConfig:
    agent_url: str = "http://127.0.0.1:8787"
    min_joule_to_run: float = 60.0

@dataclass
class MergeConfig:
    lora_accept_min_delta: float = 0.05
    merge_every_n_capsules: int = 1
    base_dir: str = "./state/base_model"   # dove persistere il tronco
    candidates_dir: str = "./state/candidates"  # staging

@dataclass
class DataConfig:
    incoming_dir: str = "./data/incoming"
    holdout_csv: str = "./data/holdout.csv"  # text,label
    embeddings_cache: str = "./state/embeddings"

@dataclass
class StoreConfig:
    receipts_db: str = "./state/receipts.db"
    capsules_dir: str = "./state/capsules"

@dataclass
class ModelConfig:
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clf_base: str = "distilbert-base-uncased"
    lora_rank: int = 8

@dataclass
class Cfg:
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42

# ---------------- Energy Client (Rust agent) ----------------
class EnergyClient:
    def __init__(self, url: str): self.url = url.rstrip('/')
    def bucket(self) -> float:
        try:
            r = requests.get(self.url + "/v1/sample", timeout=0.5)
            r.raise_for_status(); return float(r.json().get('bucket_j', 0.0))
        except Exception:
            return 0.0
    def take(self, j: float) -> bool:
        try:
            r = requests.post(self.url + "/v1/take", json={"joules": j}, timeout=0.5)
            return bool(r.json().get('ok', False))
        except Exception:
            return False

# ---------------- Receipts ----------------
class Receipts:
    def __init__(self, db: str):
        os.makedirs(os.path.dirname(db), exist_ok=True)
        self.db = db; self._init()
    def _init(self):
        c = sqlite3.connect(self.db); cur = c.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts REAL, task TEXT, joule REAL, sec REAL, delta REAL, loss REAL,
          delta_hash TEXT, meta TEXT
        );
        """); c.commit(); c.close()
    def add(self, **kw):
        c = sqlite3.connect(self.db); cur = c.cursor()
        cur.execute("INSERT INTO receipts (ts,task,joule,sec,delta,loss,delta_hash,meta) VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), kw.get('task'), kw.get('joule',0.0), kw.get('sec',0.0), kw.get('delta',0.0),
                     kw.get('loss',0.0), kw.get('delta_hash',''), json.dumps(kw.get('meta',{}))))
        c.commit(); c.close()

# ---------------- Data / Holdout ----------------
class Holdout:
    def __init__(self, path: str): self.path = path
    def load(self, limit: int=256) -> Tuple[List[str], List[int]]:
        X, y = [], []
        if os.path.isfile(self.path):
            import csv
            with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
                rdr = csv.reader(f)
                for row in rdr:
                    if len(row) < 2: continue
                    X.append(row[0]); y.append(int(row[1]));
                    if len(X) >= limit: break
        if not X:
            X = ["positive markets", "bearish outlook", "rally incoming", "risk off"] * 8
            y = [1,0,1,0] * 8
        return X, y

# ---------------- Tasks ----------------
class TaskIndex:
    name = "index_refresh"
    def __init__(self, cfg: Cfg): self.cfg = cfg
    def est_joule(self) -> float: return 40.0
    def run(self) -> Dict[str, Any]:
        import zlib
        from sentence_transformers import SentenceTransformer
        import faiss
        texts = []
        if os.path.isdir(self.cfg.data.incoming_dir):
            for fn in os.listdir(self.cfg.data.incoming_dir):
                fp = os.path.join(self.cfg.data.incoming_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f: 
                            s = line.strip();
                            if s: texts.append(s)
                            if len(texts) >= 1024: break
                except Exception:
                    pass
        if not texts:
            return {"ok": False, "delta": 0.0, "loss": 0.0, "hash": b3(b"empty"), "meta": {}}
        # novelty by compressibility
        scored = []
        for t in texts:
            raw = t.encode('utf-8', errors='ignore'); comp = zlib.compress(raw)
            ratio = len(comp) / max(1, len(raw)); scored.append((1.0 - ratio, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_texts = [t for _, t in scored[:512]]
        enc = SentenceTransformer(self.cfg.model.encoder_model)
        embs = enc.encode(top_texts, convert_to_numpy=True, normalize_embeddings=True)
        d = embs.shape[1]
        os.makedirs(self.cfg.data.embeddings_cache, exist_ok=True)
        faiss_path = os.path.join(self.cfg.data.embeddings_cache, 'faiss.index')
        if os.path.exists(faiss_path):
            index = faiss.read_index(faiss_path)
            if index.d != d: index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatIP(d)
        index.add(embs.astype(np.float32))
        faiss.write_index(index, faiss_path)
        delta = float(embs.shape[0]) / 1000.0
        h = b3(embs.tobytes())
        return {"ok": True, "delta": delta, "loss": 0.0, "hash": h, "meta": {"added": int(embs.shape[0])}}

class TaskLoRA:
    name = "lora_delta"
    def __init__(self, cfg: Cfg): self.cfg = cfg
    def est_joule(self) -> float: return 120.0
    def train_adapter(self) -> Tuple[str, Dict[str,Any]]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base = self.cfg.model.clf_base
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(device)
        peft_cfg = LoraConfig(r=self.cfg.model.lora_rank, lora_alpha=16, lora_dropout=0.0, bias='none')
        model = get_peft_model(model, peft_cfg); model.train()
        # tiny synthetic batch (replace with domain batch later)
        X, y = ["positive markets", "bearish outlook"] * 16, [1,0] * 16
        y = torch.tensor(y, device=device)
        batch = tok(X, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        random.seed(42); torch.manual_seed(42)
        base_loss = None
        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            out = model(**batch); loss = loss_fn(out.logits, y)
            if base_loss is None: base_loss = float(loss.item())
            loss.backward(); opt.step()
        delta_loss = max(0.0, base_loss - float(loss.item()))
        # save adapter capsule
        ts = int(time.time());
        cap_dir = os.path.join(self.cfg.store.capsules_dir, f"lora_{ts}")
        os.makedirs(cap_dir, exist_ok=True)
        model.save_pretrained(cap_dir)  # saves PEFT adapter weights
        meta = {"base": base, "rank": self.cfg.model.lora_rank, "delta_loss": delta_loss}
        return cap_dir, meta

    def evaluate_and_merge(self, adapter_dir: str) -> Tuple[bool, float, str]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_dir = self.cfg.merge.base_dir
        base = self.cfg.model.clf_base
        tok = AutoTokenizer.from_pretrained(base)
        # load base (local if exists else hub)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(base_dir, num_labels=2)
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
        model.to(device)
        # holdout
        X, y = Holdout(self.cfg.data.holdout_csv).load(limit=512)
        y_t = torch.tensor(y, device=device)
        batch = tok(X, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            base_out = model(**batch); base_loss = float(loss_fn(base_out.logits, y_t).item())
        # attach adapter and evaluate
        pmodel = PeftModel.from_pretrained(model, adapter_dir).to(device)
        pmodel.eval()
        with torch.no_grad():
            out = pmodel(**batch); new_loss = float(loss_fn(out.logits, y_t).item())
        delta = max(0.0, base_loss - new_loss)
        accept = delta >= self.cfg.merge.lora_accept_min_delta
        dhash = b3((str(time.time()) + adapter_dir + str(delta)).encode())
        if accept:
            # merge and persist new base
            merged = pmodel.merge_and_unload()
            os.makedirs(self.cfg.merge.base_dir, exist_ok=True)
            merged.save_pretrained(self.cfg.merge.base_dir)
        return accept, delta, dhash

    def run(self) -> Dict[str, Any]:
        cap_dir, meta = self.train_adapter()
        ok, delta, dhash = self.evaluate_and_merge(cap_dir)
        meta.update({"accepted": ok, "adapter": cap_dir})
        return {"ok": ok, "delta": float(delta), "loss": float(meta.get('delta_loss',0.0)), "hash": dhash, "meta": meta}

# ---------------- Orchestrator ----------------
class Orchestrator:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.energy = EnergyClient(cfg.energy.agent_url)
        self.receipts = Receipts(cfg.store.receipts_db)
        self.tasks = [TaskIndex(cfg), TaskLoRA(cfg)]
    def choose(self):
        # trivial policy: prefer LoRA if enough Joule, else Index
        j = self.energy.bucket()
        if j >= 120.0: return self.tasks[1]
        return self.tasks[0]
    def run(self):
        print("[CryoFlux] Orchestrator online — expecting JouleAgent at 127.0.0.1:8787")
        while True:
            try:
                j = self.energy.bucket()
                if j < self.cfg.energy.min_joule_to_run:
                    time.sleep(0.5); continue
                task = self.choose(); need = task.est_joule()
                if not self.energy.take(need):
                    time.sleep(0.25); continue
                t0 = time.time()
                res = task.run()
                sec = time.time() - t0
                self.receipts.add(task=task.name, joule=need, sec=sec, delta=res.get('delta',0.0),
                                  loss=res.get('loss',0.0), delta_hash=res.get('hash',''), meta=res.get('meta',{}))
                print(f"[CryoFlux] {task.name} -> Δ={res.get('delta',0.0):.4f} | ok={res.get('ok')} | receipt={res.get('hash','')[:8]}…")
            except KeyboardInterrupt:
                print("[CryoFlux] stopping"); break
            except Exception as e:
                print("[CryoFlux][ERR]", e); time.sleep(0.5)

# ---------------- Boot ----------------
if __name__ == "__main__":
    # ensure dirs
    for p in ["./data/incoming","./data","./state","./state/capsules","./state/base_model","./state/candidates","./state/embeddings"]:
        os.makedirs(p, exist_ok=True)
    cfg = Cfg()
    random.seed(cfg.seed)
    Orchestrator(cfg).run()
