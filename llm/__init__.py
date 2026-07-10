from __future__ import annotations

"""
Voxium LLM Module

This module provides the core interface for loading and interacting with GGUF Large Language Models 
using llama-cpp-python. It acts as the backbone for Phase 4 of the Voxium restructuring, decoupling 
LLM reasoning from application logic.

Components:
- model_loader: Centralized model loading with per-path singleton caching.
- inference: The `ChatLLM` class for standard and tool-calling generation.
- streaming: Provides `stream_completion` for token-by-token async iteration.
"""

import os

def _patch_cuda_dlls():
    try:
        import nvidia.cublas.lib
        import nvidia.cuda_runtime.lib
        os.add_dll_directory(os.path.dirname(nvidia.cublas.lib.__file__))
        os.add_dll_directory(os.path.dirname(nvidia.cuda_runtime.lib.__file__))
    except ImportError:
        pass

_patch_cuda_dlls()

from .model_loader import load_model
from .inference import ChatLLM
from .streaming import stream_completion

__all__ = [
    "load_model",
    "ChatLLM",
    "stream_completion",
]
