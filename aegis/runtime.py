"""
CPU Runtime Configuration.

PyTorch defaults to one intra-op thread per core. On hybrid CPUs (e.g. Intel 13th-gen U-series:
2 performance + 8 efficiency cores) that default was measured 25x slower than 4 threads for
ResNet-18 at 224 px (1,624 vs 64 ms/image), because work is split across slow cores that the fast
ones then wait on. Entry points call configure_torch_threads() before any inference.
"""

import os

import torch

DEFAULT_TORCH_THREADS = 4


def configure_torch_threads(threads: int = 0) -> int:
    """
    Caps PyTorch intra-op threads and returns the count in effect.

    Precedence: explicit argument > AEGIS_TORCH_THREADS > OMP_NUM_THREADS (left to PyTorch)
    > min(DEFAULT_TORCH_THREADS, PyTorch's default).
    """
    if threads <= 0:
        threads = int(os.environ.get("AEGIS_TORCH_THREADS", "0") or 0)
    if threads <= 0 and os.environ.get("OMP_NUM_THREADS"):
        return torch.get_num_threads()
    if threads <= 0:
        threads = min(DEFAULT_TORCH_THREADS, torch.get_num_threads())
    torch.set_num_threads(threads)
    return threads
