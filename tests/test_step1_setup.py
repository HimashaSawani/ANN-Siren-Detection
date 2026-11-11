"""Unit and integration tests for Step 1 — Project Setup and Environment Configuration.

Covers:
1. Configuration loading and validation.
2. Cross-platform path handling via pathlib.
3. Central random seed initialization (Python, NumPy, PyTorch).
4. Hardware and compute device selection.
5. Environment reporting.
6. Expected behavior when the UrbanSound8K dataset is missing or malformed.
7. Package import validation.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import (
    ConfigurationError,
    DatasetNotFoundError,
    DatasetValidationError,
    EnvironmentReport,
    REQUIRED_PACKAGES,
    get_default_config_path,
    get_environment_info,
    get_project_root,
    load_config,
    resolve_project_paths,
    select_device,
    set_random_seeds,
    validate_config,
    validate_package_imports,
)
from src.evaluate import get_evaluation_spec, run_evaluation
from src.inference import check_inference_readiness, predict_audio
from src.models import build_model, get_model_specifications
from src.preprocessing import (
    EXPECTED_FOLDS,
    REQUIRED_METADATA_COLUMNS,
    validate_urbansound8k_dataset,
)
from src.train import run_training, validate_training_readiness


# ==============================================================================
# 1. Configuration Loading Tests
# ==============================================================================


def test_load_default_config_succeeds() -> None:
    """Verify that the default configs/config.yaml loads and passes schema validation."""
    config = load_config()
    assert config["project"]["name"] == "ANN-Siren-Detection"
    assert config["project"]["version"] == "0.1.0"
    assert config["dataset"]["identifier"] == "UrbanSound8K"
    assert config["task"]["type"] == "binary_classification"
    assert config["reproducibility"]["random_seeds"] == [42, 123, 2026]
    assert config["evaluation"]["primary_metric"] == "macro-F1"


def test_load_config_missing_file_raises_clear_error(tmp_path: Path) -> None:
    """Verify that a missing configuration file raises ConfigurationError."""
    missing_file = tmp_path / "nonexistent_config.yaml"
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(missing_file)


def test_load_config_invalid_schema_raises_error() -> None:
    """Verify that tampering with mandatory config fields triggers ConfigurationError."""
    config = copy.deepcopy(load_config())
    config["evaluation"]["primary_metric"] = "accuracy"
    with pytest.raises(ConfigurationError, match="macro-F1"):
        validate_config(config)

    config_bad_seeds = copy.deepcopy(load_config())
    config_bad_seeds["reproducibility"]["random_seeds"] = [1, 2]
    with pytest.raises(ConfigurationError, match="random_seeds"):
        validate_config(config_bad_seeds)


def test_model_and_evaluation_specifications() -> None:
    """Verify candidate models, first-principles NumPy MLP, and Step 1 guards."""
    specs = get_model_specifications()
    assert {"random_forest", "deep_mlp", "cnn_2d", "numpy_mlp"}.issubset(specs.keys())
    assert specs["numpy_mlp"].is_first_principles is True
    assert specs["numpy_mlp"].framework == "numpy"

    eval_spec = get_evaluation_spec()
    assert eval_spec.primary_metric == "macro-F1"

    # Step 1 guards must prevent premature model instantiation, training, evaluation, or inference
    with pytest.raises(NotImplementedError):
        build_model("deep_mlp")
    with pytest.raises(NotImplementedError):
        run_training()
    with pytest.raises(NotImplementedError):
        run_evaluation()
    with pytest.raises(NotImplementedError):
        predict_audio()


# ==============================================================================
# 2. Path Handling Tests
# ==============================================================================


def test_project_paths_resolve_and_exist() -> None:
    """Verify all configured project directories resolve via pathlib and exist on disk."""
    root = get_project_root()
    assert (root / "pyproject.toml").is_file()
    assert get_default_config_path().is_file()

    resolved = resolve_project_paths()
    expected_dirs = [
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
    ]
    for dir_key in expected_dirs:
        dir_path = resolved[dir_key]
        assert isinstance(dir_path, Path)
        assert dir_path.is_dir(), f"Expected directory '{dir_key}' to exist at '{dir_path}'"
        assert root in dir_path.parents


def test_hardcoded_absolute_path_rejected_in_config() -> None:
    """Verify that hardcoded absolute paths in config are rejected for cross-platform safety."""
    config = copy.deepcopy(load_config())
    config["paths"]["raw_data_dir"] = str(Path.cwd().resolve())
    with pytest.raises(ConfigurationError, match="must be relative"):
        resolve_project_paths(config)


# ==============================================================================
# 3. Random Seed Initialization Tests
# ==============================================================================


@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_random_seed_reproducibility(seed: int) -> None:
    """Verify that Python, NumPy, and PyTorch RNGs produce identical sequences per seed."""
    state_a = set_random_seeds(seed)
    py_val_a = random.random()
    np_val_a = np.random.rand(4)
    torch_val_a = torch.rand(4)

    state_b = set_random_seeds(seed)
    py_val_b = random.random()
    np_val_b = np.random.rand(4)
    torch_val_b = torch.rand(4)

    assert state_a["seed"] == seed
    assert state_b["python_hash_seed"] == str(seed)
    assert py_val_a == py_val_b
    assert np.allclose(np_val_a, np_val_b)
    assert torch.allclose(torch_val_a, torch_val_b)


def test_invalid_random_seed_raises_value_error() -> None:
    """Verify that negative or boolean seeds raise ValueError."""
    with pytest.raises(ValueError, match="non-negative integer"):
        set_random_seeds(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        set_random_seeds(True)


# ==============================================================================
# 4. Device Selection Tests
# ==============================================================================


def test_device_selection_cpu_and_auto() -> None:
    """Verify device selection for 'cpu' and 'auto' modes."""
    cpu_info = select_device(preferred="cpu")
    assert cpu_info.selected_device == "cpu"
    assert cpu_info.torch_device == torch.device("cpu")

    auto_info = select_device(preferred="auto")
    assert auto_info.selected_device in {"cpu", "cuda", "mps"}
    if not auto_info.cuda_available and not auto_info.mps_available:
        assert auto_info.selected_device == "cpu"


def test_device_selection_cuda_fallback_and_strict() -> None:
    """Verify graceful fallback to CPU when CUDA is unavailable, and error when strict."""
    info = select_device(preferred="cuda", fallback_to_cpu=True)
    if not info.cuda_available:
        assert info.selected_device == "cpu"
        with pytest.raises(ConfigurationError, match="CUDA device was explicitly requested"):
            select_device(preferred="cuda", fallback_to_cpu=False)


def test_device_selection_invalid_option_raises() -> None:
    """Verify that an invalid device name raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Unsupported device preference"):
        select_device(preferred="tpu_v99")


# ==============================================================================
# 5. Environment Reporting Tests
# ==============================================================================


def test_environment_reporting() -> None:
    """Verify structured environment report generation and training/inference readiness."""
    report = get_environment_info()
    assert isinstance(report, EnvironmentReport)
    assert len(report.python_version) > 0
    assert len(report.torch_version) > 0
    assert len(report.numpy_version) > 0
    assert report.project_root == get_project_root()
    assert "torch" in report.package_statuses
    assert "numpy" in report.package_statuses

    train_ready = validate_training_readiness()
    assert train_ready["training_executed"] is False
    assert train_ready["configured_seeds"] == [42, 123, 2026]

    inf_ready = check_inference_readiness()
    assert inf_ready["checkpoints_dir_exists"] is True
    assert isinstance(inf_ready["available_checkpoints"], list)
    assert inf_ready["ready_for_model_inference"] is (len(inf_ready["available_checkpoints"]) > 0)


# ==============================================================================
# 6. Expected Behavior When Dataset Is Missing or Validated
# ==============================================================================


def test_missing_dataset_behavior(tmp_path: Path) -> None:
    """Verify clear error reporting and exception raising when UrbanSound8K is missing."""
    missing_root = tmp_path / "UrbanSound8K_Missing"

    # Non-strict mode returns a detailed report with is_valid == False
    report = validate_urbansound8k_dataset(dataset_root=missing_root, strict=False)
    assert report.is_valid is False
    assert report.dataset_root_exists is False
    assert report.metadata_file_exists is False
    assert report.audio_dir_exists is False
    assert len(report.errors) >= 1
    assert "UrbanSound8K dataset root directory not found" in report.errors[0]

    # Strict mode raises DatasetNotFoundError
    with pytest.raises(DatasetNotFoundError, match="UrbanSound8K dataset root directory not found"):
        validate_urbansound8k_dataset(dataset_root=missing_root, strict=True)


def test_synthetic_dataset_validation_success_and_missing_audio(tmp_path: Path) -> None:
    """Verify dataset validator checks metadata schema, folds 1..10, classes, and audio files."""
    synth_root = tmp_path / "UrbanSound8K"
    meta_dir = synth_root / "metadata"
    audio_dir = synth_root / "audio"
    meta_dir.mkdir(parents=True)

    for fold in EXPECTED_FOLDS:
        (audio_dir / f"fold{fold}").mkdir(parents=True)

    # Create synthetic non-empty audio files for 2 rows
    siren_file = audio_dir / "fold1" / "1000_8_0_0.wav"
    siren_file.write_bytes(b"RIFF____WAVEfmt ")
    dog_file = audio_dir / "fold2" / "1001_3_0_0.wav"
    dog_file.write_bytes(b"RIFF____WAVEfmt ")

    df = pd.DataFrame(
        [
            {
                "slice_file_name": "1000_8_0_0.wav",
                "fsID": 1000,
                "start": 0.0,
                "end": 4.0,
                "salience": 1,
                "fold": 1,
                "classID": 8,
                "class": "siren",
            },
            {
                "slice_file_name": "1001_3_0_0.wav",
                "fsID": 1001,
                "start": 0.0,
                "end": 4.0,
                "salience": 1,
                "fold": 2,
                "classID": 3,
                "class": "dog_bark",
            },
        ]
    )
    assert list(df.columns) == list(REQUIRED_METADATA_COLUMNS)
    df.to_csv(meta_dir / "UrbanSound8K.csv", index=False)

    valid_report = validate_urbansound8k_dataset(dataset_root=synth_root, strict=True)
    assert valid_report.is_valid is True
    assert valid_report.total_metadata_rows == 2
    assert valid_report.verified_audio_files == 2
    assert "siren" in valid_report.available_classes

    # Remove one audio file and verify strict mode raises DatasetValidationError
    dog_file.unlink()
    with pytest.raises(DatasetValidationError, match="audio file\\(s\\) referenced in metadata are missing"):
        validate_urbansound8k_dataset(dataset_root=synth_root, strict=True)


# ==============================================================================
# 7. Package Import Validation Tests
# ==============================================================================


def test_package_import_validation_handles_present_and_missing_modules() -> None:
    """Verify package import validator accurately reports available and missing packages."""
    custom_check = (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("nonexistent_module_xyz_123", "nonexistent-pkg-xyz"),
    )
    statuses = validate_package_imports(custom_check)
    assert statuses["torch"].available is True
    assert statuses["torch"].version is not None
    assert statuses["numpy"].available is True
    assert statuses["nonexistent-pkg-xyz"].available is False
    assert statuses["nonexistent-pkg-xyz"].error is not None

    # Validate full project package table runs and returns statuses for all 13 packages
    full_statuses = validate_package_imports(REQUIRED_PACKAGES)
    assert len(full_statuses) == len(REQUIRED_PACKAGES)
    for core_pkg in (
        "torch",
        "torchaudio",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "PyYAML",
        "pytest",
        "streamlit",
    ):
        assert full_statuses[core_pkg].available is True, (
            f"Expected core dependency '{core_pkg}' to be importable."
        )

