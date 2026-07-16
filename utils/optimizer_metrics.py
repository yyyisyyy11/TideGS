from pathlib import Path
from typing import Any, Dict, Optional, TextIO


OPTIMIZER_TIMING_FIELDS = (
    'batch_idx',
    'iteration',
    'iter_end',
    'bsz',
    'update_rule',
    'updates_enabled',
    'omega_wait_ms',
    'optimizer_submit_ms',
    'optimizer_cuda_ms',
    'touched_rows',
    'total_gaussians',
    'session_row_updates_total',
    'session_mean_updates_per_gaussian',
)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.6f}'
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


def write_optimizer_timing_metrics(
    *,
    model_path: str,
    iteration: int,
    batch_size: int,
    update_rule: str,
    updates_enabled: bool,
    omega_wait_ms: float,
    optimizer_submit_ms: float,
    optimizer_cuda_ms: float,
    touched_rows: int,
    total_gaussians: int,
    session_row_updates_total: int,
    log_file: Optional[TextIO] = None,
) -> Dict[str, Any]:
    total_gaussians = int(total_gaussians)
    session_row_updates_total = int(session_row_updates_total)
    mean_updates = (
        float(session_row_updates_total) / float(total_gaussians)
        if total_gaussians > 0
        else 0.0
    )
    batch_size = max(1, int(batch_size))
    row = {
        'batch_idx': (int(iteration) - 1) // batch_size,
        'iteration': int(iteration),
        'iter_end': int(iteration) + batch_size,
        'bsz': batch_size,
        'update_rule': str(update_rule),
        'updates_enabled': bool(updates_enabled),
        'omega_wait_ms': float(omega_wait_ms),
        'optimizer_submit_ms': float(optimizer_submit_ms),
        'optimizer_cuda_ms': float(optimizer_cuda_ms),
        'touched_rows': int(touched_rows),
        'total_gaussians': total_gaussians,
        'session_row_updates_total': session_row_updates_total,
        'session_mean_updates_per_gaussian': mean_updates,
    }

    if not model_path:
        return row

    try:
        output_path = Path(model_path) / 'optimizer_timing.tsv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not output_path.exists() or output_path.stat().st_size == 0
        with output_path.open('a', encoding='utf-8') as handle:
            if needs_header:
                handle.write('\t'.join(OPTIMIZER_TIMING_FIELDS) + '\n')
            handle.write(
                '\t'.join(_format_value(row[field]) for field in OPTIMIZER_TIMING_FIELDS)
                + '\n'
            )
    except OSError as exc:
        if log_file is not None:
            log_file.write(
                f'[OPTIMIZER TIMING] Warning: failed to write optimizer_timing.tsv: {exc}\n'
            )
    return row
