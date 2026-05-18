"""
Core CLIP Engine — the central model wrapper for all CLIP operations.

Design decisions:
- Singleton-style loading to avoid redundant model loads
- Automatic device placement
- Built-in embedding caching for performance
- Supports all standard CLIP variants
"""

import clip
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple
from PIL import Image
from loguru import logger

from src.utils.gpu_utils import optimize_memory


class CLIPEngine:
    """
    Production-grade CLIP wrapper supporting:
    - Image and text encoding
    - Batch processing
    - Embedding caching
    - Multiple model variants
    """
    
    SUPPORTED_MODELS = [
        "ViT-B/32", "ViT-B/16", "ViT-L/14", 
        "ViT-L/14@336px", "RN50", "RN101", "RN50x4"
    ]
    
    def __init__(
        self, 
        model_name: str = "ViT-B/32",
        device: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize CLIP Engine.
        
        Args:
            model_name: CLIP model variant (e.g., 'ViT-B/32')
            device: 'cuda', 'cpu', or 'auto'
            cache_dir: Directory for caching embeddings
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {self.SUPPORTED_MODELS}")
        
        # Device resolution
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Always eval mode for inference
        
        # Embedding dimension
        self.embed_dim = self.model.visual.output_dim
        logger.info(f"CLIP loaded. Embedding dimension: {self.embed_dim}")
    
    @torch.no_grad()
    def encode_images(
        self, 
        images: Union[List[Image.Image], torch.Tensor],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of PIL images into CLIP embeddings.
        
        Args:
            images: List of PIL Images
            batch_size: Processing batch size
            normalize: L2-normalize embeddings (required for cosine similarity)
        
        Returns:
            Numpy array of shape (N, embed_dim)
        """
        if isinstance(images, list):
            # Preprocess PIL images
            tensors = torch.stack([self.preprocess(img) for img in images])
        else:
            tensors = images
        
        all_embeddings = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size].to(self.device)
            embeddings = self.model.encode_image(batch)
            
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        result = np.concatenate(all_embeddings, axis=0)
        logger.debug(f"Encoded {len(result)} images → shape {result.shape}")
        return result
    
    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 256,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text strings into CLIP embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Processing batch size
            normalize: L2-normalize embeddings
        
        Returns:
            Numpy array of shape (N, embed_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
            embeddings = self.model.encode_text(tokens)
            
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        result = np.concatenate(all_embeddings, axis=0)
        logger.debug(f"Encoded {len(result)} texts → shape {result.shape}")
        return result
    
    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        temperature: float = 100.0,
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeddings: (N, D) array of image embeddings
            text_embeddings: (M, D) array of text embeddings
            temperature: Scaling factor (matches CLIP's learned temperature)
        
        Returns:
            Similarity matrix of shape (N, M)
        """
        # Both should be L2-normalized, so dot product = cosine similarity
        similarity = temperature * (image_embeddings @ text_embeddings.T)
        return similarity
    
    def get_model_info(self) -> dict:
        """Return detailed model metadata."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embed_dim": self.embed_dim,
            "total_params_M": round(total_params / 1e6, 1),
            "image_resolution": self.model.visual.input_resolution 
                if hasattr(self.model.visual, 'input_resolution') else "unknown",
        }