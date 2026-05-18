"""
Embedding Visualization Module.

Creates publication-quality plots of CLIP embedding spaces using:
- t-SNE (stochastic, nonlinear — best for cluster visualization)
- PCA (linear — fast, good for structure)
- UMAP (optional — best quality, slower)

These visualizations are essential for understanding what CLIP
has learned and are visually impressive for presentations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Optional
from pathlib import Path
from loguru import logger


# Publication-quality plot settings
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class EmbeddingVisualizer:
    """
    Visualize high-dimensional CLIP embeddings in 2D.
    """
    
    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        label_names: Optional[List[str]] = None,
        title: str = "CLIP Embedding Space (t-SNE)",
        filename: str = "tsne_embeddings.png",
        perplexity: int = 30,
        n_iter: int = 1000,
    ) -> plt.Figure:
        """
        Create a t-SNE visualization of embedding space.
        
        Args:
            embeddings: (N, D) array of embeddings
            labels: (N,) list of class indices or names
            label_names: Optional mapping of index → display name
        """
        logger.info(f"Running t-SNE on {len(embeddings)} embeddings...")
        
        # Reduce with PCA first for stability (standard practice)
        n_components_pca = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
        pca = PCA(n_components=n_components_pca)
        reduced = pca.fit_transform(embeddings)
        
        # t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        tsne_result = tsne.fit_transform(reduced)
        
        # Plotting
        unique_labels = sorted(set(labels))
        colors = cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {lbl: col for lbl, col in zip(unique_labels, colors)}
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        for label in unique_labels:
            mask = np.array([l == label for l in labels])
            display_name = label_names[label] if (label_names and isinstance(label, int)) else str(label)
            ax.scatter(
                tsne_result[mask, 0],
                tsne_result[mask, 1],
                c=[label_to_color[label]],
                label=display_name,
                s=20,
                alpha=0.7,
                linewidths=0,
            )
        
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        
        # Legend — handle many classes
        if len(unique_labels) <= 20:
            ax.legend(
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                fontsize=8,
                framealpha=0.9,
                ncol=1 if len(unique_labels) <= 10 else 2,
            )
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"t-SNE plot saved: {save_path}")
        return fig
    
    def plot_pca(
        self,
        embeddings: np.ndarray,
        labels: List,
        label_names: Optional[List[str]] = None,
        title: str = "CLIP Embedding Space (PCA)",
        filename: str = "pca_embeddings.png",
    ) -> plt.Figure:
        """PCA visualization of embeddings (fast, linear)."""
        logger.info("Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(embeddings)
        variance_explained = pca.explained_variance_ratio_
        
        unique_labels = sorted(set(labels))
        colors = cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array([l == label for l in labels])
            display_name = label_names[label] if (label_names and isinstance(label, int)) else str(label)
            ax.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=[color],
                label=display_name,
                s=20, alpha=0.7,
            )
        
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1%} variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1%} variance)", fontsize=12)
        
        if len(unique_labels) <= 20:
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PCA plot saved: {save_path}")
        return fig
    
    def plot_similarity_matrix(
        self,
        similarity_matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "CLIP Similarity Matrix",
        filename: str = "similarity_matrix.png",
    ) -> plt.Figure:
        """Plot a similarity matrix heatmap (great for showing image-text alignment)."""
        fig, ax = plt.subplots(figsize=(max(8, len(col_labels)), max(6, len(row_labels))))
        
        im = ax.imshow(similarity_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = similarity_matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=7, color="black" if abs(val) < 0.7 else "white")
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Similarity matrix saved: {save_path}")
        return fig
    
    def plot_confidence_bars(
        self,
        top_k_predictions: List[tuple],
        image_title: str = "Image",
        filename: str = "confidence_bars.png",
    ) -> plt.Figure:
        """Horizontal bar chart of top-K class predictions with confidence scores."""
        classes = [p[0] for p in reversed(top_k_predictions)]
        scores = [p[1] for p in reversed(top_k_predictions)]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#2ecc71" if i == len(classes)-1 else "#3498db" for i in range(len(classes))]
        bars = ax.barh(classes, scores, color=colors, height=0.6, edgecolor="white")
        
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{score:.1%}", va="center", fontsize=10, fontweight="bold"
            )
        
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Confidence Score", fontsize=11)
        ax.set_title(f"CLIP Zero-Shot Predictions: {image_title}", fontsize=13, fontweight="bold")
        ax.axvline(x=scores[-1], color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig