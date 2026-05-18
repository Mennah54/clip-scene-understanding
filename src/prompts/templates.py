"""
Prompt Engineering Module.

Research insight: CLIP's zero-shot performance is highly sensitive to 
prompt phrasing. This module provides templates, ensemble strategies,
and systematic prompt comparison tools.

Key findings from the original CLIP paper:
- "a photo of a {label}" outperforms just "{label}"
- Ensembling multiple prompt templates significantly boosts accuracy
- Domain-specific prompts (e.g., medical) require domain-adapted templates
"""

from typing import List, Dict
from loguru import logger


# ============================================================
# PROMPT TEMPLATE LIBRARY
# ============================================================

GENERAL_TEMPLATES = [
    "a photo of a {label}",
    "a photograph of a {label}",
    "an image of a {label}",
    "a picture of a {label}",
    "a {label}",
    "a rendering of a {label}",
    "a clear photo of a {label}",
    "a high quality photo of a {label}",
    "a good photo of a {label}",
]

CONTEXT_TEMPLATES = [
    "a photo of a {label} in nature",
    "a {label} in a natural setting",
    "a {label} in the wild",
    "a close-up photo of a {label}",
    "a distant photo of a {label}",
    "a bright photo of a {label}",
    "a dark photo of a {label}",
    "a centered image of a {label}",
]

SCENE_TEMPLATES = [
    "a photo of the {label}",
    "a scene of {label}",
    "an aerial view of {label}",
    "a streetview of {label}",
    "an indoor photo of {label}",
    "an outdoor photo of {label}",
]

ALL_TEMPLATES = GENERAL_TEMPLATES + CONTEXT_TEMPLATES + SCENE_TEMPLATES


# ============================================================
# PROMPT BUILDER
# ============================================================

class PromptBuilder:
    """
    Builds and manages text prompts for CLIP zero-shot inference.
    
    Supports:
    - Single prompt per class
    - Multiple template instantiation
    - Domain-specific custom templates
    """
    
    def __init__(self, strategy: str = "ensemble", custom_templates: List[str] = None):
        """
        Args:
            strategy: 'single' | 'template' | 'ensemble'
            custom_templates: Override with custom template list
        """
        self.strategy = strategy
        self.templates = custom_templates if custom_templates else GENERAL_TEMPLATES
        logger.info(f"PromptBuilder initialized: strategy='{strategy}', {len(self.templates)} templates")
    
    def build_prompts(self, class_names: List[str]) -> Dict[str, List[str]]:
        """
        Generate prompts for each class.
        
        Returns:
            Dict mapping class_name → list of prompt strings
        """
        result = {}
        
        for label in class_names:
            label_clean = label.replace("_", " ").strip()
            
            if self.strategy == "single":
                result[label] = [f"a photo of a {label_clean}"]
            elif self.strategy == "template":
                result[label] = [t.format(label=label_clean) for t in GENERAL_TEMPLATES[:4]]
            elif self.strategy == "ensemble":
                result[label] = [t.format(label=label_clean) for t in self.templates]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        
        total_prompts = sum(len(v) for v in result.values())
        logger.info(f"Generated {total_prompts} prompts for {len(class_names)} classes")
        return result
    
    def get_flat_prompts(self, class_names: List[str]) -> tuple:
        """
        Get flat list of all prompts and a mapping back to class indices.
        
        Returns:
            (flat_prompts: List[str], class_indices: List[int])
        """
        prompt_dict = self.build_prompts(class_names)
        flat_prompts = []
        class_indices = []
        
        for idx, (label, prompts) in enumerate(prompt_dict.items()):
            flat_prompts.extend(prompts)
            class_indices.extend([idx] * len(prompts))
        
        return flat_prompts, class_indices