"""
run_experiments.py
------------------
Real evaluation pipeline for CLIP Zero-Shot Scene Understanding.

Replaces all previously simulated/hardcoded metric values with genuine
evaluation on images loaded from data/test/<class_name>/ folders.

Dataset layout expected:
    data/test/
        cat/
            img1.jpg
            img2.jpg
        dog/
            img1.jpg
        car/
            img1.jpg

Usage:
    python run_experiments.py
"""

import json
import time
from pathlib import Path

from loguru import logger
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from src.utils.config import load_config, get_device, ensure_dirs
from src.utils.logger import setup_logger
from src.model.clip_engine import CLIPEngine
from src.inference.zero_shot import ZeroShotClassifier
from src.evaluation.benchmark import PerformanceBenchmark


# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_test_dataset(test_dir: Path) -> tuple:
    """
    Walk test_dir and collect images with their ground-truth labels.

    Each sub-folder name is treated as a class label:
        test_dir/cat/  → label "cat"
        test_dir/dog/  → label "dog"

    Args:
        test_dir: Path to the root test directory.

    Returns:
        images      : list of PIL.Image objects (RGB)
        true_labels : list of string class labels (ground truth)
        class_names : sorted list of unique class names discovered
    """
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir.resolve()}\n"
            "Create sub-folders named after each class, for example:\n"
            "  data/test/cat/\n"
            "  data/test/dog/\n"
            "  data/test/car/"
        )

    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(
            f"No class sub-folders found in {test_dir.resolve()}. "
            "Each sub-folder name becomes the class label."
        )

    images = []
    true_labels = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            logger.warning(f"No images found in {class_dir} — skipping class '{class_name}'.")
            continue

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                true_labels.append(class_name)
            except Exception as exc:
                logger.warning(f"Could not load {img_path}: {exc}")

    class_names = sorted(set(true_labels))
    logger.info(
        f"Dataset loaded: {len(images)} images | "
        f"{len(class_names)} classes: {class_names}"
    )
    return images, true_labels, class_names


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(true_labels: list, predicted_labels: list) -> dict:
    """
    Compute top-1 accuracy and macro F1-score using sklearn.

    Args:
        true_labels      : ground-truth class names
        predicted_labels : model-predicted class names

    Returns:
        dict with top1_accuracy and macro_f1
    """
    top1_accuracy = round(accuracy_score(true_labels, predicted_labels), 4)
    macro_f1 = round(
        f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
        4,
    )
    return {"top1_accuracy": top1_accuracy, "macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    engine: CLIPEngine,
    images: list,
    true_labels: list,
    class_names: list,
    use_ensemble: bool,
    experiment_name: str,
) -> dict:
    """
    Build a ZeroShotClassifier, call predict_batch_with_labels(),
    and return real computed metrics.

    Args:
        engine          : loaded CLIPEngine instance
        images          : list of PIL.Image objects
        true_labels     : ground-truth label strings
        class_names     : all class names for this experiment
        use_ensemble    : whether to use prompt ensemble averaging
        experiment_name : label used for logging

    Returns:
        dict with top1_accuracy, macro_f1, num_samples, elapsed_seconds
    """
    logger.info(f"  Building classifier for: {experiment_name}")
    start = time.time()

    # Build classifier — text embeddings are pre-computed here
    classifier = ZeroShotClassifier(
        clip_engine=engine,
        class_names=class_names,
        use_ensemble=use_ensemble,
    )

    # Run batch inference and collect predictions + ground truth
    batch_output = classifier.predict_batch_with_labels(
        images=images,
        true_labels=true_labels,
        batch_size=64,
    )

    elapsed = round(time.time() - start, 2)

    # batch_output keys: "predictions", "ground_truth", "confidences", "top5_predictions"
    predicted_labels = batch_output["predictions"]
    ground_truth = batch_output["ground_truth"]

    # Compute real sklearn metrics
    metrics = compute_metrics(ground_truth, predicted_labels)
    metrics["num_samples"] = len(images)
    metrics["elapsed_seconds"] = elapsed

    logger.info(
        f"  {experiment_name} → "
        f"acc={metrics['top1_accuracy']:.4f}  "
        f"f1={metrics['macro_f1']:.4f}  "
        f"({len(images)} samples, {elapsed}s)"
    )
    return metrics


# ---------------------------------------------------------------------------
# Full experiment suite
# ---------------------------------------------------------------------------

def run_all_experiments() -> dict:
    """
    Orchestrate the full real-evaluation experiment suite.

    Experiment 1 — Prompt Strategy Comparison:
        Runs the same dataset through three classifier settings:
        single prompt, 4-template set, and full ensemble.
        Replaces previously hardcoded exp1_results.

    Experiment 2 — Model Size Ablation:
        Loads each CLIP variant separately and evaluates accuracy.
        Replaces previously hardcoded exp2_results.
        Falls back gracefully if a model variant fails to load.

    Experiment 3 — Performance Benchmark:
        Measures real throughput (images/sec) using PerformanceBenchmark.
    """
    # Setup configuration and directories
    cfg = load_config("config.yaml")
    setup_logger(cfg.output.logs_dir)
    ensure_dirs(cfg)

    logger.info("=" * 60)
    logger.info("CLIP SCENE UNDERSTANDING — REAL EVALUATION SUITE")
    logger.info("=" * 60)

    # Load primary model
    device = get_device(cfg)
    engine = CLIPEngine(model_name=cfg.model.name, device=device)

    results_summary = {
        "model": cfg.model.name,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {},
    }

    # Load the real test dataset once; reuse across experiments
    test_dir = Path("data/test")
    logger.info(f"\nLoading test dataset from: {test_dir.resolve()}")
    images, true_labels, class_names = load_test_dataset(test_dir)

    if not images:
        raise RuntimeError(
            "No images were loaded. "
            "Populate data/test/<class_name>/ with image files and retry."
        )

    # ── Experiment 1: Prompt Strategy Comparison ──────────────────────
    logger.info("\n[EXP 1] Prompt Strategy Comparison")

    # 1a — Single generic prompt (use_ensemble=False, single-template mode)
    metrics_single = run_single_experiment(
        engine=engine,
        images=images,
        true_labels=true_labels,
        class_names=class_names,
        use_ensemble=False,
        experiment_name="baseline_single_prompt",
    )

    # 1b — Ensemble of all prompt templates
    metrics_ensemble = run_single_experiment(
        engine=engine,
        images=images,
        true_labels=true_labels,
        class_names=class_names,
        use_ensemble=True,
        experiment_name="ensemble_full",
    )

    results_summary["experiments"]["prompt_comparison"] = {
        "baseline_single_prompt": {
            "top1_accuracy": metrics_single["top1_accuracy"],
            "macro_f1": metrics_single["macro_f1"],
        },
        "ensemble_full": {
            "top1_accuracy": metrics_ensemble["top1_accuracy"],
            "macro_f1": metrics_ensemble["macro_f1"],
        },
    }
    logger.info(
        f"Prompt experiment results:\n"
        f"{json.dumps(results_summary['experiments']['prompt_comparison'], indent=2)}"
    )

    # ── Experiment 2: Model Size Ablation ─────────────────────────────
    logger.info("\n[EXP 2] Model Size Ablation")

    ablation_models = ["RN50", "ViT-B/32", "ViT-B/16"]
    model_ablation_results = {}

    for model_name in ablation_models:
        try:
            logger.info(f"  Loading model: {model_name}")
            ablation_engine = CLIPEngine(model_name=model_name, device=device)
            model_info = ablation_engine.get_model_info()

            # Benchmark speed
            start_speed = time.time()
            ablation_engine.encode_images(images[:32])
            speed_elapsed = time.time() - start_speed
            speed_img_per_sec = round(32 / speed_elapsed, 1)

            # Run real accuracy evaluation
            ablation_metrics = run_single_experiment(
                engine=ablation_engine,
                images=images,
                true_labels=true_labels,
                class_names=class_names,
                use_ensemble=True,
                experiment_name=f"ablation_{model_name.replace('/', '_')}",
            )

            model_ablation_results[model_name] = {
                "top1_accuracy": ablation_metrics["top1_accuracy"],
                "params_M": model_info.get("total_params_M", "N/A"),
                "speed_img_per_sec": speed_img_per_sec,
            }

        except Exception as exc:
            logger.warning(f"  Skipping {model_name}: {exc}")
            model_ablation_results[model_name] = {"error": str(exc)}

    results_summary["experiments"]["model_ablation"] = model_ablation_results

    # ── Experiment 3: Performance Benchmark ───────────────────────────
    logger.info("\n[EXP 3] Performance Benchmark")

    benchmark = PerformanceBenchmark(engine)
    bench_results = benchmark.benchmark_image_encoding(
        images=images,
        batch_sizes=[1, 8, 16, 32, 64],
        num_runs=3,
    )
    results_summary["experiments"]["performance_benchmark"] = bench_results

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    pc = results_summary["experiments"]["prompt_comparison"]
    for variant, m in pc.items():
        logger.info(
            f"  {variant:<35} acc={m['top1_accuracy']:.4f}  f1={m['macro_f1']:.4f}"
        )
    logger.info("  ---")
    for model_name, m in model_ablation_results.items():
        if "error" not in m:
            logger.info(
                f"  {model_name:<15} acc={m['top1_accuracy']:.4f}  "
                f"params={m['params_M']}M  speed={m['speed_img_per_sec']} img/s"
            )

    return results_summary


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, output_path: Path) -> None:
    """Write experiment results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all_experiments()
    output_path = Path("outputs/results/experiment_results.json")
    save_results(results, output_path)
    logger.info("Run `streamlit run src/ui/app.py` to launch the interactive UI.")
