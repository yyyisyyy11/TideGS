from typing import Any, Dict

import torch


class GPUStatelessNormalizedSGD:
    """Stateless normalized SGD over touched rows in the GPU working set."""

    COMPONENT_SPECS = (
        ('xyz', '_xyz', 0),
        ('opacity', '_opacity', 1),
        ('scaling', '_scaling', 2),
        ('rotation', '_rotation', 3),
        ('features_dc', '_features_dc', 4),
        ('features_rest', '_features_rest', 5),
    )

    def __init__(self, batch_size: int = 1, device: str = 'cuda'):
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {self.batch_size}')
        self.device = torch.device(device)
        self.state_mode = 'none'
        self._stats = {
            'state_mode': 'none',
            'persistent_state_bytes': 0,
            'optimizer_rows_touched_total': 0,
        }

    def _normalize_columns_lr(self, columns_lr) -> Dict[str, float]:
        if columns_lr is None:
            raise RuntimeError(
                '[GPUStatelessNormalizedSGD] optimizer.columns_lr is required'
            )
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
                '[GPUStatelessNormalizedSGD] Unexpected columns_lr '
                f'width={cols.numel()}; expected 6 grouped or 59 expanded entries'
            )
        return {
            name: float(grouped[group_idx].item())
            for name, _, group_idx in self.COMPONENT_SPECS
        }

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
        with torch.no_grad():
            for name, _, _ in self.COMPONENT_SPECS:
                if name not in sparse_grad_components:
                    raise KeyError(
                        '[GPUStatelessNormalizedSGD] Missing '
                        f'sparse_grad_components[{name!r}] at iter={iteration}'
                    )
                grads = sparse_grad_components[name]
                if grads.device != self.device:
                    grads = grads.to(self.device)

                mean_grads = grads / float(self.batch_size)
                normalized_update = mean_grads / (mean_grads.abs() + eps)
                param_views[name].data.index_add_(
                    0,
                    local_ids,
                    normalized_update,
                    alpha=-component_lrs[name],
                )

        self._stats['optimizer_rows_touched_total'] += touched_rows
        return {'touched_rows': touched_rows, 'state_bytes': 0}

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
