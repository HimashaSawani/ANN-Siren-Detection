"""Step 1 Smoke Test — Project Setup, Configuration, Reproducibility, and Readiness.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure project root is importable when executed as `python scripts/smoke_test.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    get_environment_info,
    load_config,
    resolve_project_paths,
    select_device,
    set_random_seeds,
    setup_logger,
)
from src.evaluate import get_evaluation_spec  # noqa: E402
from src.inference import check_inference_readiness  # noqa: E402
from src.models import get_model_specifications  # noqa: E402
from src.preprocessing import validate_urbansound8k_dataset  # noqa: E402
from src.train import validate_training_readiness  # noqa: E402


def run_smoke_test() -> int:
    """Execute end-to-end Step 1 smoke verification and log results.

    Returns:
        Exit code (0 if Step 1 core foundation checks pass, 1 otherwise).
    """
    logger = setup_logger("smoke_test")
    logger.info("=== Starting Step 1 Smoke Test: ANN-Siren-Detection ===")

    # 1. Configuration Loading
    config = load_config()
    logger.info(
        "1. Configuration loaded: project='%s' (v%s), dataset='%s', task='%s'",
        config["project"]["name"],
        config["project"]["version"],
        config["dataset"]["identifier"],
        config["task"]["description"],
    )

    # 2. Path Handling
    paths = resolve_project_paths(config)
    required_dirs = (
        "raw_data_dir",
        "processed_data_dir",
        "splits_dir",
        "checkpoints_dir",
        "results_dir",
        "metrics_dir",
        "figures_dir",
        "reports_dir",
        "notebooks_dir",
        "demo_dir",
    )
    missing_dirs = [key for key in required_dirs if not paths[key].is_dir()]
    if missing_dirs:
        logger.error("Missing required project directories: %s", missing_dirs)
        return 1
    logger.info("2. Path verification passed: all %d project directories exist.", len(required_dirs))

    # 3. Random Seed Initialization
    configured_seeds = config["reproducibility"]["random_seeds"]
    for seed in configured_seeds:
        seed_state = set_random_seeds(seed=seed)
        logger.info("3. Seed initialized successfully: %s", seed_state)

    # Reset to default seed (42)
    set_random_seeds(seed=config["reproducibility"]["default_seed"])

    # 4. Device Selection
    device_info = select_device(
        preferred=config["hardware"]["preferred_device"],
        allow_gpu=config["hardware"]["allow_gpu"],
        fallback_to_cpu=config["hardware"]["fallback_to_cpu"],
    )
    logger.info(
        "4. Device selection: requested='%s', selected='%s', cuda_available=%s",
        device_info.requested_device,
        device_info.selected_device,
        device_info.cuda_available,
    )

    # 5. Model, Training, Evaluation & Inference Specifications (Step 1 Readiness)
    model_specs = get_model_specifications(config)
    eval_spec = get_evaluation_spec(config)
    train_readiness = validate_training_readiness(config)
    inference_readiness = check_inference_readiness(config)
    logger.info(
        "5. Registered models: %s | Primary metric: %s | Training executed: %s | Checkpoints count: %d",
        list(model_specs.keys()),
        eval_spec.primary_metric,
        train_readiness["training_executed"],
        len(inference_readiness["available_checkpoints"]),
    )

    # 6. Environment & Dependency Status Reporting
    env_report = get_environment_info(config)
    logger.info(
        "6. Runtime environment: Python %s on %s %s (%s) | PyTorch %s | NumPy %s",
        env_report.python_version,
        env_report.platform_system,
        env_report.platform_release,
        env_report.platform_machine,
        env_report.torch_version,
        env_report.numpy_version,
    )

    available_pkgs = [
        f"{name}=={status.version}"
        for name, status in env_report.package_statuses.items()
        if status.available
    ]
    logger.info("   Installed required packages (%d/%d): %s",
                len(available_pkgs),
                len(env_report.package_statuses),
                ", ".join(available_pkgs))

    if env_report.missing_packages:
        logger.warning(
            "   Missing packages in current environment (%d): %s",
            len(env_report.missing_packages),
            ", ".join(env_report.missing_packages),
        )

    # 7. Dataset Preparation Readiness Check (Expected Missing State in Step 1)
    dataset_report = validate_urbansound8k_dataset(config=config, strict=False)
    if dataset_report.is_valid:
        logger.info(
            "7. UrbanSound8K dataset detected and valid (%d rows, %d verified files).",
            dataset_report.total_metadata_rows,
            dataset_report.verified_audio_files,
        )
    else:
        logger.info(
            "7. UrbanSound8K dataset not yet present at '%s' (expected in Step 1). "
            "Validation utility reported %d readiness message(s).",
            dataset_report.dataset_root,
            len(dataset_report.errors),
        )

    # 8. Confirm No Fabricated Real-Data Results or Checkpoints Exist
    allowed_t2_metrics = {
        ".gitkeep",
        "t2_gradient_verification.json",
        "t2_gradient_verification_table.csv",
        "t2_tiny_batch_diagnosis.json",
        "t2_architecture_summary.json",
        "t2_experiment_metadata.json",
        "t3_architecture_comparison.json",
        "t3_architecture_comparison_table.csv",
        "t3_experiment_status.json",
    }
    allowed_t2_figures = {
        ".gitkeep",
        "fig_t2_gradient_verification.png",
        "fig_t2_tiny_batch_overfitting.png",
    }
    if not dataset_report.is_valid:
        unexpected_ckpts = [
            p.name for p in paths["checkpoints_dir"].iterdir() if p.name != ".gitkeep"
        ]
        unexpected_metrics = [
            p.name for p in paths["metrics_dir"].iterdir() if p.name not in allowed_t2_metrics
        ]
        unexpected_figs = [
            p.name for p in paths["figures_dir"].iterdir() if p.name not in allowed_t2_figures
        ]
        if unexpected_ckpts or unexpected_metrics or unexpected_figs:
            logger.error(
                "Unexpected real-data artifacts found while dataset is absent: "
                "ckpts=%s, metrics=%s, figures=%s",
                unexpected_ckpts,
                unexpected_metrics,
                unexpected_figs,
            )
            return 1

    logger.info("8. Verified zero fabricated checkpoints, metrics, or figures exist.")
    logger.info("=== Smoke Test PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())

