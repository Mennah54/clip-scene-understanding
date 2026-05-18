"""
Prompt Ensemble Module.

Key insight from CLIP paper: averaging text embeddings from multiple 
prompt templates for the same class dramatically improves accuracy.
This is the "prompt ensemble" technique.
"""

import numpy as np
from typing import List, Dict
from loguru import logger

from src.prompts.templates import PromptBuilder


class PromptEnsemble:
    """
    Implements prompt ensemble averaging for CLIP zero-shot classification.
    
    Instead of picking one prompt per class, encode ALL templates,
    then average their embeddings. This creates a more robust class 
    representation that captures multiple visual perspectives.
    """
    
    def __init__(self, clip_engine, templates: List[str] = None):
        self.clip = clip_engine
        self.builder = PromptBuilder(strategy="ensemble", custom_templates=templates)
    
    def build_class_embeddings(self, class_names: List[str]) -> np.ndarray:
        """
        Build one embedding per class by averaging across template prompts.
        
        Args:
            class_names: List of class label strings
        
        Returns:
            Class embedding matrix: (num_classes, embed_dim)
        """
        logger.info(f"Building ensemble embeddings for {len(class_names)} classes...")
        prompt_dict = self.builder.build_prompts(class_names)
        class_embeddings = []
        
        for label, prompts in prompt_dict.items():
            # Encode all prompts for this class
            embeddings = self.clip.encode_texts(prompts, normalize=True)
            
            # Average and re-normalize (the ensemble step)
            avg_embedding = embeddings.mean(axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            class_embeddings.append(avg_embedding)
        
        result = np.stack(class_embeddings, axis=0)
        logger.info(f"Ensemble embeddings built: {result.shape}")
        return result