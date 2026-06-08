from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Backend:
    lib: Any
    name: str
    requires_grad: bool = False
    device: any = None


def infer_backend(*arrays):
    try:
        import torch
        devices = {a.device for a in arrays if isinstance(a, torch.Tensor)}
        if len(devices) > 1:
            raise ValueError("All tensors must be on the same device")
        device = devices.pop()
        if any(isinstance(a, torch.Tensor) for a in arrays):
            if any(a.requires_grad for a in arrays if isinstance(a, torch.Tensor)):
                # check if all tensors are on the same device
                if len(devices) > 1:
                    raise ValueError("All tensors must be on the same device")
                return Backend(torch, "torch", True, device)
            else:
                return Backend(torch, "torch", False, device)
    except Exception:
        pass

    try:
        import cupy as cp
        if any(isinstance(a, cp.ndarray) for a in arrays):
            return Backend(cp, "cupy")
    except Exception:
        pass

    import numpy as np
    return Backend(np, "numpy")


def asarray(x, backend: Backend):
    if backend.name == "torch":
        return backend.lib.as_tensor(x, device=backend.device).requires_grad_(backend.requires_grad)
    return backend.lib.asarray(x)


def zeros(shape, backend: Backend, dtype=None):
    if backend.name == "torch":
        return backend.lib.zeros(shape, dtype=dtype, device=backend.device).requires_grad_(backend.requires_grad)
    return backend.lib.zeros(shape, dtype=dtype)


def cumsum(x, axis, backend: Backend):
    if backend.name == "torch":
        return backend.lib.cumsum(x, dim=axis)
    return backend.lib.cumsum(x, axis=axis)


def concatenate(xs, axis, backend: Backend):
    if backend.name == "torch":
        return backend.lib.cat(xs, dim=axis)
    return backend.lib.concatenate(xs, axis=axis)