"""
Helpers for single-GPU training (MPI removed).
"""

import io
import os
import socket

import blobfile as bf
import torch as th

# Single GPU setup - no distributed training
GPUS_PER_NODE = 1

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    No-op for single-GPU training.
    """
    # Do nothing - no distributed setup needed
    pass


def dev():
    """
    Get the device to use for single-GPU training.
    """
    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file for single-GPU training.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    No-op for single-GPU training.
    """
    # No synchronization needed for single GPU
    pass


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
