# stress_mix.py
import time, argparse, requests, sys
import numpy as np

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_TORCH = False
    HAS_CUDA = False

def sample(agent):
    try:
        r = requests.get(agent + "/v1/sample", timeout=0.3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def stress_cpu(seconds: int, size: int):
    # ~heavy BLAS on CPU
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    t0 = time.time(); n=0
    while time.time() - t0 < seconds:
        # matmul + small nonlinearity
        c = a @ b
        a = np.tanh(c)
        n += 1
    return n

def stress_gpu(seconds: int, size: int, dtype: str = "fp16"):
    if not (HAS_TORCH and HAS_CUDA):
        print("CUDA non disponibile: salto GPU.")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    dt = torch.float16 if dtype == "fp16" else torch.bfloat16
    a = torch.randn(size, size, dtype=dt, device=device)
    b = torch.randn(size, size, dtype=dt, device=device)
    t0 = time.time(); n=0
    while time.time() - t0 < seconds:
        c = torch.matmul(a, b)
        a = torch.tanh(c)
        n += 1
        torch.cuda.synchronize()
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--cpu", type=int, default=1, help="1=on, 0=off")
    ap.add_argument("--gpu", type=int, default=1, help="1=on, 0=off")
    ap.add_argument("--size_cpu", type=int, default=3000)
    ap.add_argument("--size_gpu", type=int, default=4096)
    ap.add_argument("--agent", type=str, default="http://127.0.0.1:8787")
    args = ap.parse_args()

    print(f"[stress] start {args.seconds}s  cpu={args.cpu} gpu={args.gpu}  (agent={args.agent})")
    t0 = time.time(); last = t0
    cpu_iters = gpu_iters = 0
    while time.time() - t0 < args.seconds:
        # interleave stress and telemetry every ~2s
        chunk = min(2, args.seconds - int(time.time() - t0))
        if args.cpu: cpu_iters += stress_cpu(chunk, args.size_cpu)
        if args.gpu: gpu_iters += stress_gpu(chunk, args.size_gpu)
        s = sample(args.agent)
        bj = s.get("bucket_j", None)
        net = s.get("net_w", None)
        if bj is not None:
            print(f"[stress] bucket={bj:.2f} J  net_w={net:.2f} W  cpu_it={cpu_iters} gpu_it={gpu_iters}")
    print("[stress] done")
