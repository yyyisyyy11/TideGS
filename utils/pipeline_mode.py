"""Torch-free canonical mode names for TideGS Micro Pipeline."""


def normalize_pipeline_mode(value="off") -> str:
    """Accept the legacy CLI/checkpoint spelling and return off or micro."""
    mode = str(value).strip().lower()
    if mode == "pc1":
        return "micro"
    if mode not in {"off", "micro"}:
        raise ValueError("tide_forward_pipeline must be 'off' or 'micro' (legacy 'pc1' is accepted)")
    return mode
