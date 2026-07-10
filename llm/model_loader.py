from __future__ import annotations

import os
import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except (ImportError, RuntimeError, OSError) as _err:
    _HAS_LLAMA_CPP = False
    Llama = Any  # fallback for type hints
    logger.warning("llama-cpp-python unavailable (native lib not loaded): %s", _err)

# Cache for loaded models
_MODEL_CACHE: Dict[str, Llama] = {}
_CACHE_LOCK = threading.Lock()

def load_model(
    path: str,
    n_ctx: int = 8192,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = -1
) -> Llama:
    """
    Loads a GGUF model from the specified path using llama-cpp-python.
    Caches the model instance per path to avoid reloading.
    
    Args:
        path (str): The absolute path to the GGUF model file.
        n_ctx (int): The context size (default: 8192).
        n_threads (Optional[int]): Number of threads to use for generation.
        n_gpu_layers (int): Number of layers to offload to GPU (default: -1 for all).
        
    Returns:
        Llama: The initialized Llama model instance.
        
    Raises:
        ImportError: If llama-cpp-python is not installed.
        FileNotFoundError: If the model file does not exist.
    """
    if not _HAS_LLAMA_CPP:
        logger.warning("Mocking LLM: llama-cpp-python is not installed.")
        return None
        
    if not os.path.exists(path):
        logger.error(f"Model file not found at {path}")
        raise FileNotFoundError(f"Model file not found at {path}")
        
    with _CACHE_LOCK:
        if path in _MODEL_CACHE:
            logger.info(f"Using cached model for path: {path}")
            return _MODEL_CACHE[path]
            
        logger.info(f"Loading model from {path} (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")
        
        # Instantiate Llama
        kwargs = {
            "model_path": path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False  # suppress overly verbose output
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
            
        model = Llama(**kwargs)
        
        _MODEL_CACHE[path] = model
        logger.info(f"Successfully loaded and cached model from {path}")
        return model
