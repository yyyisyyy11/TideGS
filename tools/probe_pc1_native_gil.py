"""GIL evidence probe for real PC1 CUDA entry points (never a speed benchmark).

Run in a disposable process, with an external timeout:
    timeout 180s python tools/probe_pc1_native_gil.py --include-backward --output /tmp/pc1-gil.json
    timeout 240s nsys profile --trace=cuda,nvtx,osrt -o /tmp/pc1-gil \
        python tools/probe_pc1_native_gil.py --include-backward --output /tmp/pc1-gil.json

Inspect the packed native range in Nsight: its count.item() CUDA synchronization
must coexist with Python/NVTX markers and CUDA submissions from the other CPU
thread. Host timestamps below are supporting evidence, not kernel overlap proof.
The loss probes cover the actual fusedssim/fusedssim_backward bindings directly;
their calls can be too short to observe, so absence of a marker is inconclusive.
``--include-backward`` additionally exercises actual packed projection, SH and
raster backward bindings, plus SH/intersection/raster forward bindings. Each
probe retains genuine forward intermediates and has its own NVTX native range.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def nvtx_range(torch, name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def probe_call(torch, name, function, *, compute, delay_cycles=0, repeats=1):
    """Keep the peer ready before entering a real native call on this thread."""
    ready = threading.Event()
    gate = threading.Event()
    stop = threading.Event()
    peer_error = []
    markers = []
    peer_stream = torch.cuda.Stream(device=compute.device)
    marker_tensor = torch.zeros((256,), device=compute.device)
    torch.cuda.synchronize(compute.device)

    def peer():
        try:
            torch.cuda.set_device(compute.device)
            with torch.no_grad(), torch.cuda.stream(peer_stream):
                marker_tensor.add_(0)  # initialize this thread before the probe
                peer_stream.synchronize()
                ready.set()
                if not gate.wait(10):
                    raise TimeoutError("main thread did not enter probe")
                index = 0
                while not stop.is_set():
                    begin = time.perf_counter_ns()
                    label = f"PC1 GIL peer {name} tid={threading.get_native_id()} step={index}"
                    with nvtx_range(torch, label):
                        torch.cuda.nvtx.mark(label)
                        marker_tensor.add_(1)
                    end = time.perf_counter_ns()
                    markers.append((begin, end))
                    index += 1
                    # Yield the GIL so the caller can finish/stop this probe.
                    stop.wait(0.0002)
        except BaseException as error:
            peer_error.append(repr(error))
            ready.set()

    worker = threading.Thread(target=peer, name=f"pc1-gil-peer-{name}", daemon=True)
    worker.start()
    if not ready.wait(10):
        raise TimeoutError("peer CUDA initialization timed out")
    if peer_error:
        raise RuntimeError(peer_error[0])
    intervals = []
    previous_interval = sys.getswitchinterval()
    try:
        # Prevent an ordinary interpreter timeslice immediately after gate.set
        # from being mistaken for progress enabled by the native GIL guard.
        sys.setswitchinterval(1.0)
        with torch.no_grad(), torch.cuda.stream(compute):
            if delay_cycles:
                torch.cuda._sleep(delay_cycles)
            label = f"PC1 GIL native {name} tid={threading.get_native_id()}"
            with nvtx_range(torch, label):
                gate.set()
                for _ in range(repeats):
                    begin = time.perf_counter_ns()
                    result = function()
                    end = time.perf_counter_ns()
                    intervals.append((begin, end))
                    # Make outputs live through their kernels without retaining
                    # one output per repetition or adding a device synchronize.
                    del result
    finally:
        stop.set()
        gate.set()
        sys.setswitchinterval(previous_interval)
        worker.join(timeout=10)
    if worker.is_alive():
        raise TimeoutError("peer did not exit; terminate this disposable process")
    if peer_error:
        raise RuntimeError(peer_error[0])
    torch.cuda.synchronize(compute.device)
    contained = [marker for marker in markers
                 if any(begin < marker[0] < marker[1] < end for begin, end in intervals)]
    durations = [(end - begin) / 1e6 for begin, end in intervals]
    return {
        "name": name, "main_native_tid": threading.get_native_id(),
        "peer_native_tid": worker.native_id, "call_count": len(intervals),
        "max_native_call_ms": max(durations),
        "peer_cuda_submissions_inside_call": len(contained),
        "host_interval_evidence": bool(contained),
        "native_call_intervals_ns": intervals, "peer_submission_intervals_ns": markers,
        "nsight_required": "verify peer CUDA runtime submissions during packed count.item() wait",
    }


def make_backward_probes(torch, native, *, means, quats, scales, viewmats,
                         intrinsics, packed_outputs):
    """Build direct real native calls using the exact modern binding ABI.

    These entry points implement the wrapper autograd backward, but calling
    them directly isolates their GIL interval from Python autograd scheduling.
    All tensors are regular CUDA tensors, never Python-dispatched subclasses.
    """
    _, batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = packed_outputs
    grad_means2d = torch.ones_like(means2d)
    grad_depths = torch.ones_like(depths)
    grad_conics = torch.ones_like(conics)
    projection_backward = lambda: native.projection_ewa_3dgs_packed_bwd(
        means, None, quats, scales, viewmats, intrinsics, 512, 512, 0.3,
        native.CameraModelType.PINHOLE,
        batch_ids, camera_ids, gaussian_ids, conics, None,
        grad_means2d, grad_depths, grad_conics, None, False, False,
    )
    dirs = torch.nn.functional.normalize(means, dim=-1).contiguous()
    coefficients = torch.randn((means.shape[0], 16, 3), device=means.device) * 0.1
    grad_colors = torch.ones_like(dirs)
    sh_forward = lambda: native.spherical_harmonics_fwd(3, dirs, coefficients, None)
    sh_backward = lambda: native.spherical_harmonics_bwd(
        16, 3, dirs, coefficients, None, grad_colors, True,
    )
    colors = (sh_forward()[gaussian_ids] + 0.5).clamp_min(0).contiguous()
    opacities = torch.full((means2d.shape[0],), 0.5, device=means.device)
    intersect = lambda: native.intersect_tile(
        means2d, radii, depths, camera_ids, gaussian_ids,
        1, 16, 32, 32, True, False,
    )
    _, intersection_ids, flatten_ids = intersect()
    offset = lambda: native.intersect_offset(intersection_ids, 1, 32, 32)
    tile_offsets = offset()
    raster_forward = lambda: native.rasterize_to_pixels_3dgs_fwd(
        means2d, conics, colors, opacities, None, None,
        512, 512, 16, tile_offsets, flatten_ids,
    )
    rendered, alphas, last_ids = raster_forward()
    grad_rendered = torch.ones_like(rendered)
    grad_alphas = torch.ones_like(alphas)
    raster_backward = lambda: native.rasterize_to_pixels_3dgs_bwd(
        means2d, conics, colors, opacities, None, None,
        512, 512, 16, tile_offsets, flatten_ids,
        alphas, last_ids, grad_rendered, grad_alphas, False,
    )
    probes = [
        ("projection_ewa_3dgs_packed_bwd", projection_backward),
        ("spherical_harmonics_fwd", sh_forward),
        ("spherical_harmonics_bwd", sh_backward),
        ("intersect_tile", intersect), ("intersect_offset", offset),
        ("rasterize_to_pixels_3dgs_fwd", raster_forward),
        ("rasterize_to_pixels_3dgs_bwd", raster_backward),
    ]
    for _, function in probes:
        function()  # warm every audited native entry point before measurement
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gaussians", type=int, default=65536)
    parser.add_argument("--delay-cycles", type=int, default=300_000_000)
    parser.add_argument("--loss-repeats", type=int, default=256)
    parser.add_argument("--include-backward", action="store_true",
                        help="probe all audited gsplat forward/backward operations")
    parser.add_argument("--backward-repeats", type=int, default=32)
    args = parser.parse_args()
    if min(args.gaussians, args.delay_cycles, args.loss_repeats, args.backward_repeats) <= 0:
        parser.error("gaussians, delay-cycles and repeat counts must be positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA device is required; this probe has no CPU substitute")
    torch.cuda.set_device(args.device)
    from utils.pc1_native_capabilities import check_pc1_native_capabilities
    capabilities = check_pc1_native_capabilities(require_tile_mask=True)
    native = importlib.import_module("gsplat.cuda._backend")._C
    clm = importlib.import_module("clm_kernels._C")
    device = torch.device("cuda", args.device)
    compute = torch.cuda.Stream(device=device)
    with torch.no_grad(), torch.cuda.stream(compute):
        means = torch.randn((args.gaussians, 3), device=device) * 0.15
        means[:, 2] = means[:, 2].abs() + 2
        quats = torch.zeros((args.gaussians, 4), device=device)
        quats[:, 0] = 1
        scales = torch.full_like(means, 0.03)
        viewmats = torch.eye(4, device=device).unsqueeze(0)
        intrinsics = torch.tensor([[500., 0., 256.], [0., 500., 256.], [0., 0., 1.]],
                                  device=device).unsqueeze(0)
        image = torch.rand((1, 3, 512, 512), device=device)
        target = torch.rand_like(image)
        grad_map = torch.ones_like(image)
        packed = lambda: native.projection_ewa_3dgs_packed_fwd(
            means, None, quats, scales, None, viewmats, intrinsics,
            512, 512, 0.3, 0.01, 1e10, 0., False, native.CameraModelType.PINHOLE,
        )
        loss_forward = lambda: clm.fusedssim(0.01 ** 2, 0.03 ** 2, image, target, True)
        # Warm up all three functions and retain the real forward intermediates.
        packed_outputs = packed()
        forward = loss_forward()
        loss_backward = lambda: clm.fusedssim_backward(
            0.01 ** 2, 0.03 ** 2, image, target, grad_map, *forward[1:],
        )
        loss_backward()
        backward_probes = make_backward_probes(
            torch, native, means=means, quats=quats, scales=scales,
            viewmats=viewmats, intrinsics=intrinsics, packed_outputs=packed_outputs,
        ) if args.include_backward else []
    torch.cuda.synchronize(device)
    results = [probe_call(torch, "packed_projection_fwd", packed, compute=compute,
                          delay_cycles=args.delay_cycles)]
    results.append(probe_call(torch, "fusedssim", loss_forward, compute=compute,
                              repeats=args.loss_repeats))
    results.append(probe_call(torch, "fusedssim_backward", loss_backward, compute=compute,
                              repeats=args.loss_repeats))
    for name, function in backward_probes:
        results.append(probe_call(torch, name, function, compute=compute,
                                  repeats=args.backward_repeats))
    report = {"capabilities": capabilities, "probes": results,
              "scope": "GIL evidence only; not training speed or GPU overlap evidence",
              "includes_gsplat_backward": args.include_backward}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text, flush=True)
    packed_result = results[0]
    if packed_result["max_native_call_ms"] < 20:
        raise RuntimeError("packed GPU wait was too short to observe; increase --delay-cycles")
    if not packed_result["host_interval_evidence"]:
        raise RuntimeError("no peer CUDA submission inside packed call; inspect Nsight/GIL behavior")


if __name__ == "__main__":
    main()
