"""Backbone feasibility: load DINOv2 on this machine, report latency + memory.

Usage: python scripts/bench_encoder.py [--model dinov2_vits14] [--size 224]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def rss_mb() -> float:
    """Resident set size in MB via ctypes on Windows, /proc on Linux."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError()
        return pmc.WorkingSetSize / 2**20
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    m0 = rss_mb()
    t0 = time.perf_counter()
    model = torch.hub.load("facebookresearch/dinov2", args.model).eval()
    print(f"load: {time.perf_counter()-t0:.1f}s, model RSS delta: {rss_mb()-m0:.0f} MB")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    x1 = torch.randn(1, 3, args.size, args.size)
    xb = torch.randn(args.batch, 3, args.size, args.size)
    with torch.inference_mode():
        for _ in range(3):
            model(x1)
        single = []
        for _ in range(args.iters):
            t = time.perf_counter(); out = model(x1); single.append(time.perf_counter() - t)
        batch = []
        for _ in range(max(3, args.iters // 4)):
            t = time.perf_counter(); model(xb); batch.append(time.perf_counter() - t)
    print(f"embedding dim: {out.shape[-1]}")
    print(f"single crop @{args.size}: median {statistics.median(single)*1000:.0f} ms, "
          f"p90 {sorted(single)[int(0.9*len(single))-1]*1000:.0f} ms")
    print(f"batch {args.batch}: median {statistics.median(batch)*1000:.0f} ms total, "
          f"{statistics.median(batch)/args.batch*1000:.0f} ms/crop")
    print(f"peak RSS: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
