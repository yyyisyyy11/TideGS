"""
GPU-based block frustum culler.

Holds a persistent GPU mirror of block centers and bounding-sphere radii
(~4 MiB for N=272,279 blocks).  Batched 6-plane frustum tests run as a
single GPU kernel launch per camera chunk, replacing the per-camera CPU loop.
"""

from __future__ import annotations

import threading
import time
from typing import List, Tuple

import numpy as np
import torch


# Reuse the same plane-extraction math as the CPU culler.
from .gaussian_block import extract_frustum_planes


class GpuBlockCuller:
    """GPU-resident block frustum culler.

    Only the *geometry summary* lives on GPU — block centers and radii.
    Full Gaussian parameters stay on SSD / in the CPU RAM cache.
    """

    def __init__(
        self,
        block_bounds: np.ndarray,
        block_size: int = 4096,
        camera_chunk: int = 8,
        verbose: bool = True,
    ):
        block_bounds = np.asarray(block_bounds, dtype=np.float32).copy()
        if block_bounds.ndim != 2 or block_bounds.shape[1] != 6:
            raise ValueError(
                f"Expected block_bounds shape (N, 6), got {block_bounds.shape}"
            )

        self.num_blocks = int(block_bounds.shape[0])
        self.block_size = int(block_size)
        self.camera_chunk = max(1, int(camera_chunk))
        self.verbose = bool(verbose)

        # --- compute centers & radii (same formulas as FrustumCuller CPU) ---
        centers = (block_bounds[:, :3] + block_bounds[:, 3:]) / 2.0  # (N, 3)
        diagonals = np.linalg.norm(block_bounds[:, 3:] - block_bounds[:, :3], axis=1)
        radii = (diagonals / 2.0) * 1.15  # per-block radius with 15 % margin

        # Upload persistent GPU tensors.
        device = torch.device("cuda", torch.cuda.current_device())
        # We use torch.as_tensor to avoid an extra copy when the numpy array is
        # already contiguous and we control its lifetime.  But `from_numpy`
        # shares memory with the CPU buffer — unsafe if numpy later reuses it.
        # Safer: copy explicitly.
        self._centers = torch.from_numpy(np.ascontiguousarray(centers)).to(device)
        self._radii = torch.from_numpy(np.ascontiguousarray(radii)).to(device)

        # Homogeneous centers built lazily after dirty.
        self._centers_homo: torch.Tensor | None = None
        self._centers_dirty = True
        self.last_cull_timing = {
            "kernel_ms": 0.0,
            "d2h_ms": 0.0,
        }

        self._lock = threading.Lock()

        self._log(
            f"[GpuBlockCuller] GPU mirror ready: "
            f"{self.num_blocks} blocks, "
            f"{self._centers.element_size() * self._centers.numel() / (1024 ** 2):.2f} MiB centers, "
            f"{self._radii.element_size() * self._radii.numel() / 1024:.2f} KiB radii, "
            f"camera_chunk={self.camera_chunk}"
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # GPU mirror maintenance
    # ------------------------------------------------------------------

    def _ensure_centers_homo(self) -> None:
        """(Re)build ``_centers_homo`` GPU tensor if dirty."""
        if not self._centers_dirty and self._centers_homo is not None:
            return
        with self._lock:
            if not self._centers_dirty and self._centers_homo is not None:
                return  # double-check under lock
            ones = torch.ones(
                (self.num_blocks, 1),
                dtype=torch.float32,
                device=self._centers.device,
            )
            self._centers_homo = torch.cat([self._centers, ones], dim=1)
            self._centers_dirty = False

    def update_block_bounds(
        self,
        block_ids: np.ndarray,
        bounds: np.ndarray,
    ) -> int:
        """Update GPU mirror for *block_ids* to match new *bounds*.

        Called from ``TideStorageAdapter.update_block_bounds()`` after every
        bounds refresh.  Thread-safe.
        """
        ids = np.asarray(block_ids, dtype=np.int64)
        if ids.size == 0:
            return 0

        bounds = np.asarray(bounds, dtype=np.float32)
        if bounds.shape != (ids.size, 6):
            raise ValueError(
                f"Expected block_bounds shape {(ids.size, 6)}, got {bounds.shape}"
            )

        mins = bounds[:, :3]
        maxs = bounds[:, 3:]
        centers_np = (mins + maxs) / 2.0
        diagonals = np.linalg.norm(maxs - mins, axis=1)
        radii_np = (diagonals / 2.0) * 1.15

        centers_t = torch.from_numpy(np.ascontiguousarray(centers_np)).to(
            device=self._centers.device, dtype=torch.float32
        )
        radii_t = torch.from_numpy(np.ascontiguousarray(radii_np)).to(
            device=self._radii.device, dtype=torch.float32
        )
        ids_t = torch.from_numpy(ids).to(device=self._centers.device, dtype=torch.int64)

        with self._lock:
            self._centers[ids_t] = centers_t
            self._radii[ids_t] = radii_t
            self._centers_dirty = True

        return int(ids.size)

    # ------------------------------------------------------------------
    # Batched culling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def cull_batch(
        self,
        view_matrices: np.ndarray,
        proj_matrices: np.ndarray,
    ) -> List[List[int]]:
        """Return per-camera sorted visible block IDs.

        Parameters
        ----------
        view_matrices : (C, 4, 4) float32
            OpenGL-convention view matrices.
        proj_matrices : (C, 4, 4) float32
            OpenGL-convention projection matrices.

        Returns
        -------
        visible : list[list[int]]
            One sorted list of block IDs per camera.
        """
        view_matrices = np.asarray(view_matrices, dtype=np.float32)
        proj_matrices = np.asarray(proj_matrices, dtype=np.float32)

        if view_matrices.ndim != 3 or view_matrices.shape[1:] != (4, 4):
            raise ValueError(
                f"view_matrices must be (C, 4, 4), got {view_matrices.shape}"
            )
        if proj_matrices.shape != view_matrices.shape:
            raise ValueError(
                f"view / proj shape mismatch: {view_matrices.shape} vs {proj_matrices.shape}"
            )

        num_cameras = view_matrices.shape[0]
        if num_cameras == 0:
            self.last_cull_timing = {"kernel_ms": 0.0, "d2h_ms": 0.0}
            return []

        # Ensure homogeneous centers are fresh.
        self._ensure_centers_homo()

        device = self._centers.device
        all_visible: List[List[int]] = []
        kernel_ms = 0.0
        d2h_ms = 0.0

        chunk = self.camera_chunk
        for chunk_start in range(0, num_cameras, chunk):
            chunk_end = min(chunk_start + chunk, num_cameras)
            chunk_views = view_matrices[chunk_start:chunk_end]  # (c, 4, 4)
            chunk_projs = proj_matrices[chunk_start:chunk_end]   # (c, 4, 4)
            c = chunk_views.shape[0]

            # Extract 6 frustum planes per camera on CPU (Gribb-Hartmann).
            planes_list = []
            for i in range(c):
                planes_list.append(
                    extract_frustum_planes(
                        chunk_views[i], chunk_projs[i],
                        coordinate_system="opengl",
                        validate=False,  # skip warnings in hot path
                    )
                )
            planes_np = np.stack(planes_list, axis=0)  # (c, 6, 4)
            planes_t = torch.from_numpy(np.ascontiguousarray(planes_np)).to(device)

            # Batched plane-vs-sphere test.
            #   signed_distances[cam, plane, block] = dot(plane, center_homo)
            #   shape: (c, 6, N)
            with self._lock:
                centers_h = self._centers_homo  # (N, 4)
                radii = self._radii            # (N,)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            signed_distances = torch.matmul(
                planes_t,          # (c, 6, 4)
                centers_h.T,       # (4, N)
            )  # → (c, 6, N)

            # Block is visible to camera i iff ALL 6 plane distances >= -radius[block].
            visible_mask = torch.all(
                signed_distances >= -radii[None, None, :],  # broadcast (1, 1, N)
                dim=1,  # over planes
            )  # → (c, N) bool

            # Compact: torch.nonzero per camera, ascending by construction.
            indices_per_camera = []
            for i in range(c):
                indices_per_camera.append(torch.nonzero(visible_mask[i]).squeeze(-1))
            end_event.record()
            end_event.synchronize()
            kernel_ms += float(start_event.elapsed_time(end_event))

            d2h_start = time.perf_counter()
            all_visible.extend(indices.cpu().tolist() for indices in indices_per_camera)
            d2h_ms += (time.perf_counter() - d2h_start) * 1000.0

        self.last_cull_timing = {
            "kernel_ms": kernel_ms,
            "d2h_ms": d2h_ms,
        }
        return all_visible
