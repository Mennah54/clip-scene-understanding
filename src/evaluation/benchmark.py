"""
Performance Benchmarking Module.

Measures:
- Inference throughput (images/second)
- GPU vs CPU comparison
- Batch size impact on speed
- Memory usage
"""

import time
import numpy as np
import torch
from typing import List, Dict
from PIL import Image
from loguru import logger

from src.model.clip_engine import CLIPEngine


class PerformanceBenchmark:
    """Benchmark CLIP inference speed under different configurations."""
    
    def __init__(self, clip_engine: CLIPEngine):
        self.clip = clip_engine
    
    def benchmark_image_encoding(
        self,
        images: List[Image.Image],
        batch_sizes: List[int] = [1, 8, 16, 32, 64],
        num_runs: int = 3,
    ) -> Dict:
        """
        Benchmark image encoding speed across batch sizes.
        
        Returns:
            Dict with batch_size → {throughput, latency, std}
        """
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(images):
                continue
            
            batch = images[:batch_size]
            latencies = []
            
            # Warm-up run
            self.clip.encode_images(batch)
            
            for _ in range(num_runs):
                start = time.perf_counter()
                self.clip.encode_images(batch)
                end = time.perf_counter()
                latencies.append(end - start)
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            throughput = batch_size / avg_latency
            
            results[batch_size] = {
                "avg_latency_ms": round(avg_latency * 1000, 2),
                "std_latency_ms": round(std_latency * 1000, 2),
                "throughput_img_per_sec": round(throughput, 1),
            }
            
            logger.info(
                f"Batch={batch_size:3d}: {throughput:.1f} img/s, "
                f"latency={avg_latency*1000:.1f}ms"
            )
        
        return results
    
    def benchmark_device_comparison(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ) -> Dict:
        """
        Compare CPU vs GPU inference speed.
        (Instantiates a second engine on CPU for comparison)
        """
        results = {}
        
        # Current device
        current_device = self.clip.device
        
        # Benchmark current device
        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            self.clip.encode_images(images[:batch_size])
            end = time.perf_counter()
            latencies.append(end - start)
        
        results[current_device] = {
            "throughput": round(batch_size / np.mean(latencies), 1),
            "latency_ms": round(np.mean(latencies) * 1000, 1),
        }
        
        # If on GPU, also test CPU
        if current_device == "cuda":
            import clip
            cpu_model, cpu_preprocess = clip.load(self.clip.model_name, device="cpu")
            cpu_model.eval()
            
            tensors = torch.stack([cpu_preprocess(img) for img in images[:batch_size]])
            
            latencies_cpu = []
            with torch.no_grad():
                for _ in range(3):
                    start = time.perf_counter()
                    cpu_model.encode_image(tensors)
                    end = time.perf_counter()
                    latencies_cpu.append(end - start)
            
            results["cpu"] = {
                "throughput": round(batch_size / np.mean(latencies_cpu), 1),
                "latency_ms": round(np.mean(latencies_cpu) * 1000, 1),
            }
            
            speedup = results["cuda"]["throughput"] / results["cpu"]["throughput"]
            results["gpu_speedup"] = round(speedup, 2)
            logger.info(f"GPU speedup over CPU: {speedup:.1f}x")
        
        return results