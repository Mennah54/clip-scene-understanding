"""
Configuration management using OmegaConf/Hydra.
Provides centralized, reproducible experiment config.
"""

import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from loguru import logger


def load_config(config_path: str = "config.yaml") -> DictConfig:
    """Load and validate project configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_file)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def get_device(cfg: DictConfig) -> str:
    """Resolve device: auto-detect CUDA if set to 'auto'."""
    import torch
    if cfg.model.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.model.device
    logger.info(f"Using device: {device}")
    return device


def ensure_dirs(cfg: DictConfig) -> None:
    """Create all output directories if they don't exist."""
    dirs = [
        cfg.output.figures_dir,
        cfg.output.reports_dir,
        cfg.output.results_dir,
        cfg.output.logs_dir,
        cfg.data.embeddings_root,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("All output directories verified.")