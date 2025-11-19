from __future__ import annotations

import torch


def get_torch_device(device_type: str) -> torch.device:
    """
    Resolve a human-readable device type into a `torch.device` instance.

    Supported values:
      * "auto" — prefer CUDA, then MPS, otherwise CPU;
      * "cpu"
      * "cuda"
      * "mps"

    Raises
    ------
    ValueError
        If the requested device is not available or unsupported.
    """
    if not isinstance(device_type, str):
        raise ValueError(
                f'Parameter "device_type" must be a string, got {type(device_type)!r} instead.'
        )

    normalized = device_type.strip().lower()

    if normalized == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')

        mps_backend = getattr(torch.backends, 'mps', None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device('mps')

        return torch.device('cpu')

    if normalized == 'cpu':
        return torch.device('cpu')

    if normalized == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA device was requested but is not available on this system.')
        return torch.device('cuda')

    if normalized == 'mps':
        mps_backend = getattr(torch.backends, 'mps', None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError('MPS device was requested but is not available on this system.')
        return torch.device('mps')

    raise ValueError(
            f'Unsupported device_type {device_type!r}. Expected "auto", "cpu", "cuda" or "mps".'
    )