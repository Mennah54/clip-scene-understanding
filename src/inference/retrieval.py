"""
Semantic Image Retrieval Module using FAISS.

Builds a vector index over image embeddings and supports:
- Fast nearest-neighbor search (millions of images)
- Text-to-image retrieval
- Image-to-image retrieval
- Hybrid (image+text) query fusion
"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image
from loguru import logger

from src.model.clip_engine import CLIPEngine


class SemanticRetriever:
    """
    FAISS-backed semantic image retrieval system.
    
    Design: Inner Product index with L2-normalized vectors
    = cosine similarity search at FAISS speed.
    """
    
    def __init__(
        self,
        clip_engine: CLIPEngine,
        index_type: str = "FlatIP",
    ):
        self.clip = clip_engine
        self.embed_dim = clip_engine.embed_dim
        self.index_type = index_type
        self.index = None
        self.image_paths = []   # Parallel list to index
        self.embeddings = None  # Store for visualization
        
        logger.info(f"SemanticRetriever initialized (index: {index_type})")
    
    def build_index(
        self,
        images: List[Image.Image],
        image_paths: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> None:
        """
        Build FAISS index from a list of images.
        
        Args:
            images: PIL images to index
            image_paths: Optional paths/IDs for each image
            batch_size: Encoding batch size
        """
        logger.info(f"Building retrieval index over {len(images)} images...")
        
        # Encode all images
        embeddings = self.clip.encode_images(images, batch_size=batch_size, normalize=True)
        self.embeddings = embeddings.astype(np.float32)
        
        # Store paths
        self.image_paths = image_paths if image_paths else [str(i) for i in range(len(images))]
        
        # Build FAISS index
        if self.index_type == "FlatIP":
            # Exact search — inner product (= cosine for normalized vectors)
            self.index = faiss.IndexFlatIP(self.embed_dim)
        elif self.index_type == "IVFFlat":
            # Approximate search — faster for large corpora
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            nlist = min(100, len(images) // 10)
            self.index = faiss.IndexIVFFlat(quantizer, self.embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors")
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve most similar images to a text query.
        
        Returns:
            List of (image_path, similarity_score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Encode text query
        query_embedding = self.clip.encode_texts([query_text], normalize=True)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = [
            (self.image_paths[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0  # FAISS returns -1 for empty slots
        ]
        
        logger.debug(f"Text query '{query_text}' → {len(results)} results")
        return results
    
    def search_by_image(
        self,
        query_image: Image.Image,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Retrieve images similar to a query image."""
        query_embedding = self.clip.encode_images([query_image], normalize=True)
        query_embedding = query_embedding.astype(np.float32)
        
        scores, indices = self.index.search(query_embedding, top_k + 1)  # +1 to exclude self
        
        results = [
            (self.image_paths[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
        
        # Remove the query image itself if it's in the index
        return results[:top_k]
    
    def search_multimodal(
        self,
        query_image: Image.Image,
        query_text: str,
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Hybrid retrieval: fuse image + text query embeddings.
        
        alpha=0.0 → pure text query
        alpha=1.0 → pure image query
        alpha=0.5 → equal fusion
        """
        img_emb = self.clip.encode_images([query_image], normalize=True)
        txt_emb = self.clip.encode_texts([query_text], normalize=True)
        
        # Weighted fusion + re-normalize
        fused = alpha * img_emb + (1 - alpha) * txt_emb
        fused = fused / np.linalg.norm(fused, axis=-1, keepdims=True)
        fused = fused.astype(np.float32)
        
        scores, indices = self.index.search(fused, top_k)
        
        return [
            (self.image_paths[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
    
    def save_index(self, path: str) -> None:
        """Persist FAISS index to disk."""
        faiss.write_index(self.index, f"{path}.faiss")
        np.save(f"{path}_embeddings.npy", self.embeddings)
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str, image_paths: List[str]) -> None:
        """Load persisted FAISS index from disk."""
        self.index = faiss.read_index(f"{path}.faiss")
        self.embeddings = np.load(f"{path}_embeddings.npy")
        self.image_paths = image_paths
        logger.info(f"Index loaded: {self.index.ntotal} vectors")