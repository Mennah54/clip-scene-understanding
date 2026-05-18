"""
GPU detection, memory monitoring, and device management utilities.
"""

import torch
from loguru import logger


def get_gpu_info() -> dict:
    """Return detailed GPU information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024**3)
            mem_used = torch.cuda.memory_allocated(i) / (1024**3)
            info["devices"].append({
                "id": i,
                "name": props.name,
                "memory_total_gb": round(mem_total, 2),
                "memory_used_gb": round(mem_used, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        logger.info(f"GPU(s) detected: {[d['name'] for d in info['devices']]}")
    else:
        logger.warning("No GPU detected. Running on CPU — inference will be slower.")
    
    return info


def optimize_memory() -> None:
    """Clear GPU cache for fresh memory state."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared.")


def get_optimal_batch_size(device: str, base_size: int = 64) -> int:
    """Suggest batch size based on available GPU memory."""
    if device == "cpu":
        return min(base_size, 16)
    
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if mem_gb >= 16:
            return base_size * 2       # 128
        elif mem_gb >= 8:
            return base_size           # 64
        elif mem_gb >= 4:
            return base_size // 2      # 32
        else:
            return base_size // 4      # 16
    
    return base_size