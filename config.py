import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List

class VoxiumConfig(BaseSettings):
    """Centralized configuration for Voxium loaded from .env

    [M-6] All aliases are aligned to match exact .env variable names.
    """
    
    # Server config — .env uses VOXIUM_HOST, VOXIUM_PORT, VOXIUM_DEBUG
    host: str = Field(default="127.0.0.1", alias="VOXIUM_HOST")
    port: int = Field(default=5000, alias="VOXIUM_PORT")
    debug: bool = Field(default=False, alias="VOXIUM_DEBUG")
    
    # LLM Settings — .env uses LLM_MODEL_PATH, LLM_CONTEXT_LENGTH, LLM_NUM_THREADS
    llm_model_path: str = Field(default="models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf", alias="LLM_MODEL_PATH")
    n_ctx: int = Field(default=4096, alias="LLM_CONTEXT_LENGTH")
    n_threads: int = Field(default=8, alias="LLM_NUM_THREADS")
    n_gpu_layers: int = Field(default=-1, alias="N_GPU_LAYERS")
    
    # Audio & TTS Settings
    piper_voice_model: str = Field(default="en_US-lessac-medium", alias="PIPER_VOICE_MODEL")
    wakeword_sensitivity: float = Field(default=0.5, alias="WAKEWORD_SENSITIVITY")
    
    # Memory Settings — .env uses DATABASE_PATH (not SQLITE_PATH)
    chromadb_path: str = Field(default="data/db/chromadb", alias="CHROMADB_PATH")
    sqlite_path: str = Field(default="data/db/voxium.db", alias="DATABASE_PATH")
    
    # Paths
    export_dir: str = Field(default="data/exports", alias="EXPORT_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        populate_by_name = True
        
# Create a global instance
config = VoxiumConfig()
