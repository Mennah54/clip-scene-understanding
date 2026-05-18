"""
Zero-Shot Classification Module.

This is the core inference pipeline implementing CLIP's zero-shot
classification capability — classifying images without any labeled
training data by comparing against text descriptions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
from scipy.special import softmax
from loguru import logger

from src.model.clip_engine import CLIPEngine
from src.prompts.ensemble import PromptEnsemble
from src.prompts.templates import PromptBuilder


class ZeroShotClassifier:
    """
    Zero-shot image classifier using CLIP.
    
    Usage:
        classifier = ZeroShotClassifier(clip_engine, class_names)
        results = classifier.predict(image)
    """
    
    def __init__(
        self,
        clip_engine: CLIPEngine,
        class_names: List[str],
        use_ensemble: bool = True,
        temperature: float = 100.0,
    ):
        self.clip = clip_engine
        self.class_names = class_names
        self.temperature = temperature
        self.use_ensemble = use_ensemble
        
        # Pre-compute class text embeddings (expensive, do once)
        logger.info("Pre-computing class text embeddings...")
        if use_ensemble:
            ensemble = PromptEnsemble(clip_engine)
            self.class_embeddings = ensemble.build_class_embeddings(class_names)
        else:
            builder = PromptBuilder(strategy="single")
            flat_prompts, _ = builder.get_flat_prompts(class_names)
            self.class_embeddings = clip_engine.encode_texts(flat_prompts, normalize=True)
        
        logger.info(f"ZeroShotClassifier ready: {len(class_names)} classes")
    
    def predict(
        self,
        images: List[Image.Image],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Classify a list of images.
        
        Args:
            images: List of PIL Images
            top_k: Return top-K predictions per image
        
        Returns:
            List of prediction dicts with keys:
                - 'predicted_class': top-1 class name
                - 'confidence': top-1 confidence (0-1)
                - 'top_k': list of (class_name, score) tuples
                - 'raw_similarities': full similarity vector
        """
        # Encode images
        image_embeddings = self.clip.encode_images(images, normalize=True)
        
        # Compute similarity to all classes
        # Shape: (num_images, num_classes)
        similarities = image_embeddings @ self.class_embeddings.T * self.temperature
        
        # Convert to probabilities
        probabilities = softmax(similarities, axis=-1)
        
        results = []
        for i, (sim_row, prob_row) in enumerate(zip(similarities, probabilities)):
            # Top-K indices
            top_k_indices = np.argsort(prob_row)[::-1][:top_k]
            top_k_preds = [
                (self.class_names[idx], float(prob_row[idx]))
                for idx in top_k_indices
            ]
            
            results.append({
                "predicted_class": self.class_names[top_k_indices[0]],
                "confidence": float(prob_row[top_k_indices[0]]),
                "top_k": top_k_preds,
                "raw_similarities": sim_row.tolist(),
            })
        
        return results
    
    def predict_batch_with_labels(
        self,
        images: List[Image.Image],
        true_labels: List[str],
        batch_size: int = 64,
    ) -> Dict:
        """
        Run prediction over a labeled dataset for evaluation.
        
        Returns dict with predictions and ground truth for metrics computation.
        """
        from tqdm import tqdm
        
        all_predictions = []
        all_confidences = []
        all_top5 = []
        
        logger.info(f"Running batch prediction on {len(images)} images...")
        
        for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
            batch = images[i:i + batch_size]
            preds = self.predict(batch, top_k=5)
            
            for pred in preds:
                all_predictions.append(pred["predicted_class"])
                all_confidences.append(pred["confidence"])
                all_top5.append([p[0] for p in pred["top_k"]])
        
        return {
            "predictions": all_predictions,
            "ground_truth": true_labels,
            "confidences": all_confidences,
            "top5_predictions": all_top5,
        }