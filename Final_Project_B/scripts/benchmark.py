"""Measure throughput before committing to a long training run.

Two numbers matter for choosing ``batch_size`` and ``num_workers`` on a small GPU:

*   GPU compute time per step (forward + backward, AMP) and peak VRAM;
*   data-loading time per batch (resize + crop + flip + normalise in workers).

    python scripts/benchmark.py --batch-size 32 --workers 4

The script never trains anything: it runs on random tensors and, for the data
part, on the frozen train split.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

import config
import data as data_lib
import models as models_lib


def gpu_benchmark(batch_size: int, iters: int, warmup: int, image_size: int = 224) -> dict:
    device = torch.device("cuda")
    model = models_lib.ImageOnlyNet(pretrained=False).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    labels = torch.randint(0, 7, (batch_size,), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            loss = criterion(model(images, None), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "ms_per_step": elapsed / iters * 1000,
        "samples_per_s": batch_size * iters / elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "total_vram_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def data_benchmark(batch_size: int, workers: int, batches: int) -> dict:
    cfg = {
        "data": {
            "metadata_mode": "all_metadata",
            "image_size": 224,
            "resize_to": 232,
            "batch_size": batch_size,
            "num_workers": workers,
            "pin_memory": True,
        }
    }
    frame = data_lib.read_split("train")
    feature_columns = data_lib.resolve_feature_columns("all_metadata")
    loader = data_lib.make_loader(frame, feature_columns, cfg, train=True)

    iterator = iter(loader)
    for _ in range(min(3, batches)):  # warm the workers and the page cache
        next(iterator)

    started = time.perf_counter()
    seen = 0
    for _ in range(batches):
        try:
            next(iterator)
        except StopIteration:
            break
        seen += 1
    elapsed = time.perf_counter() - started
    return {"ms_per_batch": elapsed / max(seen, 1) * 1000, "batches": seen}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-batches", type=int, default=25)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is not available; benchmark needs a GPU")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM total: {props.total_memory / 1024**2:.0f} MiB\n")

    gpu = gpu_benchmark(args.batch_size, args.iters, warmup=3)
    print(f"compute (batch {args.batch_size}, AMP): "
          f"{gpu['ms_per_step']:.0f} ms/step | {gpu['samples_per_s']:.0f} img/s | "
          f"peak alloc {gpu['peak_vram_mb']:.0f} MiB / reserved {gpu['total_vram_mb']:.0f} MiB")

    timing = data_benchmark(args.batch_size, args.workers, args.data_batches)
    print(f"data loading ({args.workers} workers): {timing['ms_per_batch']:.0f} ms/batch "
          f"over {timing['batches']} batches")

    step_s = gpu["ms_per_step"] / 1000
    load_s = timing["ms_per_batch"] / 1000
    batches_per_epoch = 7050 // args.batch_size
    print(f"\nrough epoch estimate: {batches_per_epoch} train batches x "
          f"max(compute {step_s:.2f}s, data {load_s:.2f}s) = "
          f"{batches_per_epoch * max(step_s, load_s) / 60:.1f} min/epoch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
