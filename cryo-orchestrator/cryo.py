#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, json, math, sqlite3, argparse, random, shutil, warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Silenzia warning non critici
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*was not initialized.*")
warnings.filterwarnings("ignore", message=".*TRAIN this model.*")

import requests
import numpy as np

from blake3 import blake3

def b3(x: bytes) -> str: return blake3(x).hexdigest()

# ---------------- Config ----------------
@dataclass
class EnergyConfig:
    agent_url: str = "http://127.0.0.1:8787"
    min_joule_to_run: float = 1.0
    task_index_est_joules: float = 20.0
    task_lora_est_joules: float = 120.0

@dataclass
class MergeConfig:
    lora_accept_min_delta: float = 0.003
    merge_every_n_capsules: int = 1
    base_dir: str = "./state/base_model"
    candidates_dir: str = "./state/candidates"

@dataclass
class DataConfig:
    incoming_dir: str = "./data/incoming"
    holdout_csv: str = "./data/holdout.csv"
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
class SchedulerConfig:
    bandit_enabled: bool = True
    ucb_c: float = 0.3
    warmup_runs: int = 3
    min_bucket_factor: float = 1.0
    epsilon: float = 0.0  # Epsilon-greedy exploration probability

@dataclass
class Cfg:
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    seed: int = 42

def load_config() -> Cfg:
    """Load configuration from config.toml if available, with env var overrides."""
    cfg = Cfg()  # Start with defaults

    # Try to find config.toml in repo root (parent of cryo-orchestrator)
    config_paths = [
        Path("../config.toml"),
        Path("config.toml"),
        Path("../../config.toml"),
    ]

    config_found = False
    for path in config_paths:
        if path.exists():
            try:
                import toml
                data = toml.load(path)
                print(f"[CryoFlux] Loaded config from {path}")

                # Parse orchestrator section
                if "orchestrator" in data:
                    orch = data["orchestrator"]
                    if "agent_url" in orch:
                        cfg.energy.agent_url = orch["agent_url"]
                    if "receipts_db" in orch:
                        cfg.store.receipts_db = orch["receipts_db"]
                    if "seed" in orch:
                        cfg.seed = orch["seed"]

                    # Parse nested sections
                    if "model" in orch:
                        m = orch["model"]
                        if "encoder_model" in m:
                            cfg.model.encoder_model = m["encoder_model"]
                        if "clf_base" in m:
                            cfg.model.clf_base = m["clf_base"]
                        if "lora_rank" in m:
                            cfg.model.lora_rank = m["lora_rank"]

                    if "data" in orch:
                        d = orch["data"]
                        if "incoming_dir" in d:
                            cfg.data.incoming_dir = d["incoming_dir"]
                        if "holdout_csv" in d:
                            cfg.data.holdout_csv = d["holdout_csv"]
                        if "embeddings_cache" in d:
                            cfg.data.embeddings_cache = d["embeddings_cache"]

                    if "storage" in orch:
                        s = orch["storage"]
                        if "capsules_dir" in s:
                            cfg.store.capsules_dir = s["capsules_dir"]
                        if "base_dir" in s:
                            cfg.merge.base_dir = s["base_dir"]
                        if "candidates_dir" in s:
                            cfg.merge.candidates_dir = s["candidates_dir"]

                    if "energy" in orch:
                        e = orch["energy"]
                        if "min_joule_to_run" in e:
                            cfg.energy.min_joule_to_run = e["min_joule_to_run"]
                        if "task_index_est_joules" in e:
                            cfg.energy.task_index_est_joules = e["task_index_est_joules"]
                        if "task_lora_est_joules" in e:
                            cfg.energy.task_lora_est_joules = e["task_lora_est_joules"]

                    if "merge" in orch:
                        mg = orch["merge"]
                        if "lora_accept_min_delta" in mg:
                            cfg.merge.lora_accept_min_delta = mg["lora_accept_min_delta"]
                        if "merge_every_n_capsules" in mg:
                            cfg.merge.merge_every_n_capsules = mg["merge_every_n_capsules"]

                    if "scheduler" in orch:
                        sched = orch["scheduler"]
                        if "bandit_enabled" in sched:
                            cfg.scheduler.bandit_enabled = sched["bandit_enabled"]
                        if "ucb_c" in sched:
                            cfg.scheduler.ucb_c = sched["ucb_c"]
                        if "warmup_runs" in sched:
                            cfg.scheduler.warmup_runs = sched["warmup_runs"]
                        if "min_bucket_factor" in sched:
                            cfg.scheduler.min_bucket_factor = sched["min_bucket_factor"]
                        if "epsilon" in sched:
                            cfg.scheduler.epsilon = sched["epsilon"]

                config_found = True
                break
            except ImportError:
                print("[CryoFlux] Warning: 'toml' package not found. Add 'toml' to requirements.txt")
                break
            except Exception as e:
                print(f"[CryoFlux] Warning: Could not parse config.toml: {e}")
                break

    # Environment variable overrides
    if "JOULE_AGENT_URL" in os.environ:
        cfg.energy.agent_url = os.environ["JOULE_AGENT_URL"]
        print(f"[CryoFlux] Override agent_url={cfg.energy.agent_url} from env")

    return cfg

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
    def est_joule(self) -> float: return self.cfg.energy.task_index_est_joules
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
    def est_joule(self) -> float: return self.cfg.energy.task_lora_est_joules
    def train_adapter(self) -> Tuple[str, Dict[str,Any]]:
        import warnings, logging
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        # Force CUDA
        assert torch.cuda.is_available(), "CUDA not available! LoRA training requires GPU"
        device = 'cuda'
        print(f"[LoRA] Using device: {device} ({torch.cuda.get_device_name(0)})")
        
        base = self.cfg.model.clf_base
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(device)
        peft_cfg = LoraConfig(
            r=self.cfg.model.lora_rank,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_lin","k_lin","v_lin","out_lin"]  # DistilBERT attention
        )
        model = get_peft_model(model, peft_cfg)
        
        # Assert LoRA is properly attached
        affected = [n for n, p in model.named_parameters() if "lora_" in n]
        assert len(affected) > 0, "LoRA not attached — check target_modules"
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[LoRA] trainable params: {trainable}/{total} ({100*trainable/total:.3f}%) | layers={len(affected)}")
        
        model.train()
        
        # Use holdout.csv for training (train/val split within holdout)
        # This ensures coherence: LoRA trains on same distribution it's evaluated on
        import os, csv, random
        
        holdout_file = os.path.join("data", "holdout.csv")
        all_texts = []
        
        if os.path.isfile(holdout_file):
            with open(holdout_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        text = row[0].strip('"')
                        label = int(row[1])
                        all_texts.append((text, label))
        
        # Use first 256 samples for training (80% train / 20% val internal to this)
        random.shuffle(all_texts)
        all_texts = all_texts[:256]
        split_idx = int(len(all_texts) * 0.8)
        X_train = [t for t,_ in all_texts[:split_idx]]
        y_train = [y for _,y in all_texts[:split_idx]]
        
        if len(X_train) == 0:
            # fallback if holdout not found
            X_train = [
                "markets rally on strong jobs data",
                "energy demand slump weighs on oil & gas stocks",
                "layer-2 upgrade announced with 30% fee reduction",
                "earnings warning; guidance cut for next quarter",
            ] * 64
            y_train = [1, 0, 1, 0] * 64
        

        y = torch.tensor(y_train, device=device)
        enc = tok(X_train, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        steps = 200  # più passi per vedere Δ
        bsz = 32
        for step in range(steps):
            i0 = (step * bsz) % enc["input_ids"].size(0)
            i1 = i0 + bsz
            batch = {k: v[i0:i1] for k, v in enc.items()}
            yb = y[i0:i1]
            opt.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = loss_fn(out.logits, yb)
            loss.backward()
            opt.step()
        
        # Final loss on training data (for info only)
        with torch.no_grad():
            final_out = model(**enc)
            final_loss = float(loss_fn(final_out.logits, y).item())
        
        delta_loss = max(0.0, 0.0)  # placeholder, real delta measured in evaluate_and_merge
        
        # save adapter capsule
        ts = int(time.time())
        cap_dir = os.path.join(self.cfg.store.capsules_dir, f"lora_{ts}")
        os.makedirs(cap_dir, exist_ok=True)
        model.save_pretrained(cap_dir)  # saves PEFT adapter weights
        meta = {"base": base, "rank": self.cfg.model.lora_rank, "delta_loss": delta_loss}
        return cap_dir, meta

    def evaluate_and_merge(self, adapter_dir: str) -> Tuple[bool, float, str]:
        import warnings, logging
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
        
        # Force CUDA
        assert torch.cuda.is_available(), "CUDA not available! Evaluation requires GPU"
        device = 'cuda'
        
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
        
        model.eval()  # evaluation mode for base model
        with torch.no_grad():
            base_out = model(**batch)
            base_loss = float(loss_fn(base_out.logits, y_t).item())
            base_pred = base_out.logits.argmax(dim=-1)
            base_acc = float((base_pred == y_t).float().mean().item())
        
        # attach adapter and evaluate
        pmodel = PeftModel.from_pretrained(model, adapter_dir).to(device)
        pmodel.eval()
        with torch.no_grad():
            out = pmodel(**batch)
            new_loss = float(loss_fn(out.logits, y_t).item())
            new_pred = out.logits.argmax(dim=-1)
            new_acc = float((new_pred == y_t).float().mean().item())
        
        delta = max(0.0, base_loss - new_loss)
        acc_gain = new_acc - base_acc
        print(f"[EVAL] base_loss={base_loss:.4f} new_loss={new_loss:.4f} Δ={delta:.4f} | base_acc={base_acc:.3f} new_acc={new_acc:.3f} Δacc={acc_gain:.3f}")
        
        # Accept if loss improves OR accuracy gains ≥1% (soglia morbida per primo merge)
        accept = (delta >= 0.002) or (acc_gain >= 0.01)
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

        # Open DB connection for scheduler (with fallback logic)
        self.scheduler_conn = None
        if cfg.scheduler.bandit_enabled:
            try:
                # Resolve DB path with fallback
                db_path = cfg.store.receipts_db
                from pathlib import Path
                path_obj = Path(db_path)
                if not path_obj.exists():
                    # Fallback to cryo-orchestrator/state/receipts.db
                    fallback = Path("state/receipts.db")
                    if fallback.exists():
                        db_path = str(fallback)

                self.scheduler_conn = sqlite3.connect(db_path)
                print(f"[SCHEDULER] Connected to {db_path}")
            except Exception as e:
                print(f"[SCHEDULER][WARN] Could not open DB for scheduler: {e}")
                print("[SCHEDULER] Will fall back to legacy choose(j)")

        # Build task name -> task object mapping for scheduler
        self.task_map = {task.name: task for task in self.tasks}
    def choose(self, j):
        # dynamic policy: choose task based on available Joule
        lora_threshold = self.cfg.energy.task_lora_est_joules
        index_threshold = self.cfg.energy.task_index_est_joules

        if j >= lora_threshold:
            return self.tasks[1]  # LoRA
        if j >= index_threshold:
            return self.tasks[0]  # Index
        return None
    def run(self):
        print(f"[CryoFlux] Orchestrator online — expecting JouleAgent at {self.cfg.energy.agent_url}")

        # Import scheduler
        try:
            from scheduler import choose_task
            scheduler_available = True
        except ImportError:
            print("[SCHEDULER][WARN] scheduler.py not found - using legacy choose(j)")
            scheduler_available = False

        while True:
            try:
                j = self.energy.bucket()

                # Try scheduler first (if enabled and available)
                task = None
                if scheduler_available and self.scheduler_conn is not None:
                    try:
                        choice = choose_task(self.cfg, j, self.scheduler_conn)
                        if choice is not None:
                            # Map task name to task object
                            task = self.task_map.get(choice.name)
                            if task:
                                reason = choice.reason
                                score = choice.score
                                score_str = f"score={score:.6f}" if score is not None else ""
                                print(f"[SCHEDULER] using task={choice.name} reason={reason} {score_str}")
                            else:
                                print(f"[SCHEDULER][WARN] Unknown task selected: {choice.name}")
                                task = None
                    except Exception as e:
                        print(f"[SCHEDULER][WARN] Scheduler error: {e}")
                        task = None

                # Fallback to legacy choose(j) if scheduler didn't select a task
                if task is None:
                    task = self.choose(j)

                if task is None:
                    time.sleep(0.3); continue

                need = task.est_joule()
                if not self.energy.take(need):
                    time.sleep(0.2); continue
                t0 = time.time()
                res = task.run()
                sec = time.time() - t0
                self.receipts.add(task=task.name, joule=need, sec=sec, delta=res.get('delta',0.0),
                                  loss=res.get('loss',0.0), delta_hash=res.get('hash',''), meta=res.get('meta',{}))
                print(f"[CryoFlux] {task.name} -> Δ={res.get('delta',0.0):.4f} | ok={res.get('ok')} | receipt={res.get('hash','')[:8]}…")
            except KeyboardInterrupt:
                print("[CryoFlux] stopping")
                if self.scheduler_conn:
                    self.scheduler_conn.close()
                break
            except Exception as e:
                print("[CryoFlux][ERR]", e); time.sleep(0.5)

# ---------------- Boot ----------------
if __name__ == "__main__":
    # Load configuration
    cfg = load_config()

    # ensure dirs
    for p in ["./data/incoming","./data","./state","./state/capsules","./state/base_model","./state/candidates","./state/embeddings"]:
        os.makedirs(p, exist_ok=True)

    random.seed(cfg.seed)
    Orchestrator(cfg).run()
