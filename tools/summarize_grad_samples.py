#!/usr/bin/env python3
"""Summarize bounded raw gradient samples emitted for each rank and batch."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sample_dir",
        type=Path,
        help="Directory containing grad_sample_step*_rank*.pt files",
    )
    args = parser.parse_args()
    paths = sorted(args.sample_dir.glob("grad_sample_step*_rank*.pt"))
    if not paths:
        raise SystemExit(f"No gradient samples found in {args.sample_dir}")

    by_step = defaultdict(list)
    metadata = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        step = int(payload["optimizer_step"])
        by_step[step].append(payload["gradients"].to(torch.float32))
        metadata[step] = payload

    print(
        "optimizer_step\titeration\tcomponent\tsampled_values\t"
        "exact_zero_ratio\tabs_p50\tabs_p90\tabs_p99\tabs_max"
    )
    for step in sorted(by_step):
        gradients = torch.cat(by_step[step], dim=0)
        payload = metadata[step]
        offset = 0
        for name, width in zip(
            payload["parameter_names"], payload["parameter_widths"]
        ):
            values = gradients[:, offset : offset + int(width)].reshape(-1)
            absolute = values.abs()
            quantiles = torch.quantile(
                absolute,
                torch.tensor([0.5, 0.9, 0.99]),
            )
            exact_ratio = float((values == 0).to(torch.float64).mean())
            print(
                f"{step}\t{int(payload['iteration'])}\t{name}\t{values.numel()}\t"
                f"{exact_ratio:.6%}\t{float(quantiles[0]):.9g}\t"
                f"{float(quantiles[1]):.9g}\t{float(quantiles[2]):.9g}\t"
                f"{float(absolute.max()):.9g}"
            )
            offset += int(width)


if __name__ == "__main__":
    main()
