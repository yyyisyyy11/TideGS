import math
import os
from typing import Any, Dict, Tuple

import torch


class GPUStatelessSWAN:
    """Hybrid bounded-SWAN updates over touched rows in the GPU working set.

    SWAN is restricted to the Euclidean ``xyz`` and RGB ``features_dc`` groups.
    The opacity, exponential scaling, quaternion rotation, and higher-order SH
    groups retain the established element-wise normalized-SGD update.
    """

    COMPONENT_SPECS = (
        ('xyz', '_xyz', 0),
        ('opacity', '_opacity', 1),
        ('scaling', '_scaling', 2),
        ('rotation', '_rotation', 3),
        ('features_dc', '_features_dc', 4),
        ('features_rest', '_features_rest', 5),
    )
    NEWTON_SCHULZ_STEPS = 10
    NEWTON_SCHULZ_BETA = 0.8
    UPDATE_ABS_CAP = 1.0
    SWAN_COMPONENTS = frozenset(('xyz', 'features_dc'))

    def __init__(self, batch_size: int = 1, device: str = 'cuda'):
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {self.batch_size}')
        self.device = torch.device(device)
        self.state_mode = 'none'
        self.validate_finite = os.environ.get(
            'TIDEGS_SWAN_VALIDATE_FINITE', '0'
        ) == '1'
        self.log_update_stats = os.environ.get(
            'TIDEGS_SWAN_LOG_UPDATE_STATS', '0'
        ) == '1'
        self._stats = {
            'state_mode': 'none',
            'persistent_state_bytes': 0,
            'optimizer_rows_touched_total': 0,
            'swan_whitened_components_total': 0,
            'swan_gradnorm_fallback_components_total': 0,
            'normalized_sgd_components_total': 0,
            'swan_update_abs_cap': self.UPDATE_ABS_CAP,
        }

    def _normalize_columns_lr(self, columns_lr) -> Dict[str, float]:
        if columns_lr is None:
            raise RuntimeError('[GPUStatelessSWAN] optimizer.columns_lr is required')
        if torch.is_tensor(columns_lr):
            cols = columns_lr.detach()
            if cols.device.type != 'cpu':
                cols = cols.cpu()
            cols = cols.to(dtype=torch.float32)
        else:
            cols = torch.tensor(columns_lr, dtype=torch.float32)
        cols = cols.flatten().contiguous()
        if cols.numel() == 6:
            grouped = cols
        elif cols.numel() == 59:
            grouped = torch.tensor([
                float(cols[0].item()),
                float(cols[3].item()),
                float(cols[4].item()),
                float(cols[7].item()),
                float(cols[11].item()),
                float(cols[14].item()),
            ], dtype=torch.float32)
        else:
            raise RuntimeError(
                '[GPUStatelessSWAN] Unexpected columns_lr '
                f'width={cols.numel()}; expected 6 grouped or 59 expanded entries'
            )
        return {
            name: float(grouped[group_idx].item())
            for name, _, group_idx in self.COMPONENT_SPECS
        }

    def _require_finite(self, tensor: torch.Tensor, context: str) -> None:
        finite = torch.isfinite(tensor).all()
        message = f'[GPUStatelessSWAN] Non-finite {context}'
        if self.validate_finite:
            # Diagnostics only: reading this CUDA scalar synchronizes so the
            # raised Python error identifies the exact component and stage.
            if not bool(finite):
                raise FloatingPointError(message)
        elif tensor.is_cuda:
            # Keep the Paper SSD optimizer stream asynchronous. The following
            # stage-level CUDA synchronize surfaces this assertion if it fails.
            torch._assert_async(finite, message)
        elif not bool(finite):
            raise FloatingPointError(message)

    @staticmethod
    def _gradnorm(
        gradient_matrix: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        row_rms = gradient_matrix.square().mean(dim=1, keepdim=True).sqrt()
        return gradient_matrix / row_rms.clamp_min(eps)

    def _swan_direction(
        self,
        mean_grads: torch.Tensor,
        eps: float,
        context: str = '',
    ) -> Tuple[torch.Tensor, bool]:
        """Return a full-SWAN direction and whether GradNorm fallback was used."""
        if mean_grads.ndim != 2:
            raise ValueError(
                '[GPUStatelessSWAN] Expected a two-dimensional component gradient, '
                f'got shape={tuple(mean_grads.shape)}'
            )
        prefix = f'{context} ' if context else ''
        self._require_finite(mean_grads, f'{prefix}raw gradient')

        touched_rows, width = mean_grads.shape
        if touched_rows == 0:
            return mean_grads, False

        # G has one row per parameter component and one column per Gaussian,
        # matching the m <= n layout used by SWAN.
        gradient_matrix = mean_grads.transpose(0, 1).to(dtype=torch.float32)
        normalized = self._gradnorm(gradient_matrix, eps)
        self._require_finite(normalized, f'{prefix}GradNorm output')

        # Whitening is rank-limited when fewer Gaussian rows than component
        # dimensions are touched. GradNorm remains stateless and well-defined.
        if touched_rows < width:
            return normalized.transpose(0, 1), True

        # SWAN-0 (Appendix K) normalizes before the naive Newton-Schulz
        # inverse-square-root iteration, then runs the iteration in FP32.
        input_matrix = normalized / normalized.norm().clamp_min(eps)
        identity = torch.eye(
            width,
            dtype=input_matrix.dtype,
            device=input_matrix.device,
        )
        y = input_matrix @ input_matrix.transpose(0, 1)
        z = identity
        beta = self.NEWTON_SCHULZ_BETA
        for _ in range(self.NEWTON_SCHULZ_STEPS):
            transform = beta * (3.0 * identity - z @ y)
            y, z = y @ transform, transform @ z

        whitened = z @ input_matrix
        self._require_finite(whitened, f'{prefix}Newton-Schulz output')
        # SWAN rescales the whitened matrix to the GradNorm target norm sqrt(mn).
        target_norm = math.sqrt(float(width * touched_rows))
        whitened = whitened * (target_norm / whitened.norm().clamp_min(eps))
        self._require_finite(whitened, f'{prefix}rescaled SWAN update')
        return whitened.transpose(0, 1), False

    def step(
        self,
        iteration: int,
        gaussians,
        sparse_grad_local_ids: torch.Tensor,
        sparse_grad_components: Dict[str, torch.Tensor],
    ) -> Dict[str, int]:
        if sparse_grad_local_ids is None or sparse_grad_components is None:
            return {'touched_rows': 0, 'state_bytes': 0}
        if sparse_grad_local_ids.numel() == 0:
            return {'touched_rows': 0, 'state_bytes': 0}

        local_ids = sparse_grad_local_ids
        if local_ids.device != self.device:
            local_ids = local_ids.to(self.device)
        local_ids = local_ids.to(dtype=torch.long).contiguous()

        param_views = {
            name: getattr(gaussians, attr_name)
            for name, attr_name, _ in self.COMPONENT_SPECS
        }
        component_lrs = self._normalize_columns_lr(
            getattr(gaussians.optimizer, 'columns_lr', None)
        )
        eps = float(gaussians.optimizer.param_groups[0]['eps'])
        if eps <= 0.0:
            raise ValueError(f'eps must be positive, got {eps}')

        touched_rows = int(local_ids.numel())
        whitened_components = 0
        gradnorm_fallback_components = 0
        normalized_sgd_components = 0
        diagnostic_components = []
        with torch.no_grad():
            for name, _, _ in self.COMPONENT_SPECS:
                if name not in sparse_grad_components:
                    raise KeyError(
                        '[GPUStatelessSWAN] Missing '
                        f'sparse_grad_components[{name!r}] at iter={iteration}'
                    )
                grads = sparse_grad_components[name]
                if grads.shape[0] != touched_rows:
                    raise ValueError(
                        '[GPUStatelessSWAN] Gradient/local-id row mismatch for '
                        f'{name}: gradients={grads.shape[0]} local_ids={touched_rows}'
                    )
                if grads.device != self.device:
                    grads = grads.to(self.device)

                # Keep this division before all normalization, exactly once.
                mean_grads = grads.to(dtype=torch.float32) / float(self.batch_size)
                if name in self.SWAN_COMPONENTS:
                    raw_update, used_gradnorm_fallback = self._swan_direction(
                        mean_grads,
                        eps,
                        context=f'iteration={iteration} component={name}',
                    )
                    # Preserve the prior normalized-SGD direction bound before
                    # applying the existing component learning rates.
                    update = raw_update.clamp(
                        min=-self.UPDATE_ABS_CAP,
                        max=self.UPDATE_ABS_CAP,
                    )
                    component_mode = 'bounded_swan'
                    if used_gradnorm_fallback:
                        gradnorm_fallback_components += 1
                    else:
                        whitened_components += 1
                else:
                    raw_update = mean_grads / (mean_grads.abs() + eps)
                    update = raw_update
                    component_mode = 'normalized_sgd'
                    normalized_sgd_components += 1
                param = param_views[name]
                param.data.index_add_(
                    0,
                    local_ids,
                    update.to(dtype=param.dtype),
                    alpha=-component_lrs[name],
                )
                if self.validate_finite or self.log_update_stats:
                    updated_parameters = param.data.index_select(0, local_ids)
                    self._require_finite(
                        updated_parameters,
                        f'iteration={iteration} component={name} '
                        'post-update parameter',
                    )
                    if self.log_update_stats:
                        diagnostic_components.append({
                            'name': name,
                            'mode': component_mode,
                            'pre_bound_update_abs_max': float(
                                raw_update.abs().amax()
                            ),
                            'applied_update_abs_max': float(update.abs().amax()),
                            'clipped_value_fraction': float(
                                (raw_update.abs() > self.UPDATE_ABS_CAP).float().mean()
                                if component_mode == 'bounded_swan'
                                else 0.0
                            ),
                            'parameter_abs_max': float(
                                updated_parameters.abs().amax()
                            ),
                        })
                    del updated_parameters

        self._stats['optimizer_rows_touched_total'] += touched_rows
        self._stats['swan_whitened_components_total'] += whitened_components
        self._stats['swan_gradnorm_fallback_components_total'] += gradnorm_fallback_components
        self._stats['normalized_sgd_components_total'] += normalized_sgd_components
        step_stats = {
            'touched_rows': touched_rows,
            'state_bytes': 0,
            'swan_whitened_components': whitened_components,
            'swan_gradnorm_fallback_components': gradnorm_fallback_components,
            'normalized_sgd_components': normalized_sgd_components,
        }
        if self.log_update_stats:
            step_stats['swan_diagnostic_components'] = diagnostic_components
        return step_stats

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
