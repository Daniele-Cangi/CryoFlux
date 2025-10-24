# stress_gpu_only.py
import time, argparse, requests
import torch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16"])
    ap.add_argument("--agent", type=str, default="http://127.0.0.1:8787")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA non disponibile"
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    a = torch.randn(args.size, args.size, dtype=dt, device=device)
    b = torch.randn(args.size, args.size, dtype=dt, device=device)

    t0 = time.time()
    print(f"[gpu] start {args.seconds}s, size={args.size}, dtype={args.dtype}")
    while time.time() - t0 < args.seconds:
        c = torch.matmul(a, b)
        a = torch.tanh(c)
        torch.cuda.synchronize()
        # telemetria
        try:
            r = requests.get(args.agent + "/v1/sample", timeout=0.3).json()
            print(f"[gpu] bucket={r.get('bucket_j',0):.2f}J  net_w={r.get('net_w',0):.1f}W")
        except Exception:
            pass
    print("[gpu] done")
