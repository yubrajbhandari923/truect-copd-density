"""Peak training VRAM / speed of candidate D1 sizes (one fwd+bwd, bf16), to pick what fits a 24 GB A5000."""
import time, torch
from ddpm3d import build
for ch, b, P in [([32, 64, 128, 256], 2, 96), ([32, 64, 128, 256], 4, 96), ([48, 96, 192, 256], 2, 96), ([64, 128, 256, 256], 1, 96), ([64, 128, 256, 256], 4, 64)]:
    try:
        m = build(dict(channels=ch, res_blocks=2)).cuda(); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(b, 2, P, P, P, device="cuda"); t = torch.randint(0, 1000, (b,), device="cuda")
        for i in range(3):
            if i == 1: torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", torch.bfloat16):
                y = m(x, timesteps=t)
            y.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        print(ch, "batch", b, "patch", P, "params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6),
              "peak %.1f GB" % (torch.cuda.max_memory_allocated() / 2**30), "%.2f s/it" % ((time.time() - t0) / 2), flush=True)
    except torch.OutOfMemoryError:
        print(ch, "batch", b, "patch", P, "OOM", flush=True)
    del m; torch.cuda.empty_cache()
