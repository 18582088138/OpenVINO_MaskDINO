import torch
from typing import Any

def get_device():
    """Return the preferred device: CUDA if available, otherwise CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(x: Any, device: torch.device = None):
    """Move tensors (or nested lists/tuples/dicts of tensors) to `device`.

    If device is None, uses `get_device()`.
    """
    if device is None:
        device = get_device()

    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x
