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
        by_step[step].append(payload)
        metadata[step] = payload

    print(
        "optimizer_step\titeration\ttile_contribution_mode\tactive_sh_degree\t"
        "active_parameter_width\ttile_group\tcomponent\tsampled_values\t"
        "exact_zero_ratio\tabs_p50\tabs_p90\tabs_p99\tabs_max"
    )
    for step in sorted(by_step):
        gradients = torch.cat(
            [payload["gradients"].to(torch.float32) for payload in by_step[step]],
            dim=0,
        )
        payload = metadata[step]
        active_sh_degree = int(payload.get("active_sh_degree", 3))
        active_parameter_width = int(
            payload.get(
                "active_parameter_width",
                11 + 3 * (active_sh_degree + 1) ** 2,
            )
        )
        tile_mode = str(payload.get("tile_contribution_mode", "legacy"))
        tile_keep = None
        if all("tile_keep" in item for item in by_step[step]):
            tile_keep = torch.cat(
                [item["tile_keep"].bool() for item in by_step[step]], dim=0
            )
        groups = [("all", torch.ones(gradients.shape[0], dtype=torch.bool))]
        if tile_keep is not None:
            groups = [("keep", tile_keep.bool()), ("drop", ~tile_keep.bool())]
        for group_name, group_mask in groups:
            if not bool(group_mask.any()):
                continue
            offset = 0
            for name, width in zip(
                payload["parameter_names"], payload["parameter_widths"]
            ):
                values = gradients[group_mask, offset : offset + int(width)].reshape(-1)
                absolute = values.abs()
                quantiles = torch.quantile(
                    absolute,
                    torch.tensor([0.5, 0.9, 0.99]),
                )
                exact_ratio = float((values == 0).to(torch.float64).mean())
                print(
                    f"{step}\t{int(payload['iteration'])}\t{tile_mode}\t"
                    f"{active_sh_degree}\t{active_parameter_width}\t"
                    f"{group_name}\t{name}\t"
                    f"{values.numel()}\t{exact_ratio:.6%}\t"
                    f"{float(quantiles[0]):.9g}\t{float(quantiles[1]):.9g}\t"
                    f"{float(quantiles[2]):.9g}\t{float(absolute.max()):.9g}"
                )
                offset += int(width)


if __name__ == "__main__":
    main()
