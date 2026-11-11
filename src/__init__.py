"""Core configuration, reproducibility, logging, and environment utilities for ANN-Siren-Detection.

Module: NB6057CEM — Artificial Neural Networks
Project: Robust Emergency Siren Classification Under Urban Noise:
         A Comparative Study of MLP and CNN Architectures
Step: 1 — Project Setup and Environment Configuration
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


__version__ = "0.1.0"
__project_name__ = "ANN-Siren-Detection"

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("yaml", "PyYAML"),
    ("pytest", "pytest"),
    ("streamlit", "streamlit"),
)

REQUIRED_CONFIG_SECTIONS: tuple[str, ...] = (
    "project",
    "dataset",
    "task",
    "reproducibility",
    "evaluation",
    "models",
    "paths",
    "hardware",
)


class ConfigurationError(ValueError):
    """Raised when the project configuration file is missing, malformed, or invalid."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected UrbanSound8K dataset directory or metadata is missing."""


class DatasetValidationError(ValueError):
    """Raised when the UrbanSound8K dataset structure or metadata fails validation."""


@dataclass(frozen=True)
class PackageStatus:
    """Represents the import and version status of a required Python dependency."""

    import_name: str
    distribution_name: str
    available: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware and PyTorch compute device details."""

    requested_device: str
    selected_device: str
    torch_device: torch.device
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    mps_available: bool


@dataclass(frozen=True)
class EnvironmentReport:
    """Comprehensive snapshot of runtime environment, hardware, and dependencies."""

    python_version: str
    python_implementation: str
    platform_system: str
    platform_release: str
    platform_machine: str
    project_root: Path
    torch_version: str
    numpy_version: str
    device_info: DeviceInfo
    package_statuses: dict[str, PackageStatus] = field(default_factory=dict)

    @property
    def all_required_packages_available(self) -> bool:
        """Return True only if every required package imported cleanly."""
        return all(status.available for status in self.package_statuses.values())

    @property
    def missing_packages(self) -> list[str]:
        """Return list of distribution names that failed to import."""
        return [
            status.distribution_name
            for status in self.package_statuses.values()
            if not status.available
        ]


def setup_logger(
    name: str = "ann_siren_detection",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return a structured logger with consistent formatting.

    Args:
        name: Logger namespace identifier.
        level: Logging verbosity level (e.g., ``logging.INFO``).
        log_file: Optional file path to persist structured log records.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        resolved_log = Path(log_file)
        resolved_log.parent.mkdir(parents=True, exist_ok=True)
        existing_file_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
            and Path(h.baseFilename) == resolved_log.resolve()
        ]
        if not existing_file_handlers:
            file_handler = logging.FileHandler(resolved_log, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_project_root(marker_file: str = "pyproject.toml") -> Path:
    """Locate the project root directory dynamically using pathlib.

    Args:
        marker_file: Filename expected at the root of the repository.

    Returns:
        Resolved ``Path`` pointing to the repository root.

    Raises:
        ConfigurationError: If the project root cannot be determined.
    """
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / marker_file).is_file():
            return parent
    raise ConfigurationError(
        f"Could not locate project root containing '{marker_file}' starting from '{current}'."
    )


def get_default_config_path() -> Path:
    """Return the default path to ``configs/config.yaml`` relative to the project root."""
    return get_project_root() / "configs" / "config.yaml"


def get_default_data_config_path() -> Path:
    """Return the default path to ``configs/data_config.yaml`` relative to the project root."""
    return get_project_root() / "configs" / "data_config.yaml"


def get_default_model_config_path() -> Path:
    """Return the default path to ``configs/model_config.yaml`` relative to the project root."""
    return get_project_root() / "configs" / "model_config.yaml"


def load_model_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the Step 4 competing models configuration file (`configs/model_config.yaml`).

    Args:
        config_path: Optional path to ``model_config.yaml``.

    Returns:
        Validated dictionary of model and comparison settings.

    Raises:
        ConfigurationError: If the file is missing, unreadable, or lacks mandatory sections.
    """
    target_path = (
        Path(config_path) if config_path is not None else get_default_model_config_path()
    )
    if not target_path.is_file():
        raise ConfigurationError(f"Model configuration file not found at '{target_path}'.")

    try:
        with target_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except (yaml.YAMLError, OSError) as exc:
        raise ConfigurationError(
            f"Failed to load model configuration '{target_path}': {exc}"
        ) from exc

    if not isinstance(parsed, dict) or not parsed:
        raise ConfigurationError(
            f"Model configuration '{target_path}' must be a non-empty dictionary."
        )

    required_sections = (
        "comparison_protocol",
        "random_forest",
        "deep_mlp",
        "cnn_2d",
    )
    missing = [sec for sec in required_sections if sec not in parsed]
    if missing:
        raise ConfigurationError(
            f"Model configuration is missing required sections: {missing}"
        )

    if parsed["comparison_protocol"].get("primary_selection_metric") != "macro-F1":
        raise ConfigurationError(
            "comparison_protocol.primary_selection_metric must be 'macro-F1'."
        )
    if parsed["comparison_protocol"].get("allow_test_set_access_during_training_or_tuning") is not False:
        raise ConfigurationError(
            "comparison_protocol.allow_test_set_access_during_training_or_tuning must be False."
        )
    return parsed


def load_data_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the Step 2 data and preprocessing configuration file.

    Args:
        config_path: Optional path to ``data_config.yaml``. Defaults to
            ``<project_root>/configs/data_config.yaml``.

    Returns:
        Parsed and validated data configuration dictionary.

    Raises:
        ConfigurationError: If the file is missing, unreadable, or lacks mandatory sections.
    """
    target_path = (
        Path(config_path) if config_path is not None else get_default_data_config_path()
    )
    if not target_path.is_file():
        raise ConfigurationError(f"Data configuration file not found at '{target_path}'.")

    try:
        with target_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except (yaml.YAMLError, OSError) as exc:
        raise ConfigurationError(
            f"Failed to load data configuration '{target_path}': {exc}"
        ) from exc

    if not isinstance(parsed, dict) or not parsed:
        raise ConfigurationError(
            f"Data configuration '{target_path}' must be a non-empty dictionary."
        )

    required_sections = (
        "dataset_provenance",
        "task_definition",
        "split_protocol",
        "audio_signal",
        "features",
        "augmentation",
        "dataloader",
        "audit_outputs",
    )
    missing = [sec for sec in required_sections if sec not in parsed]
    if missing:
        raise ConfigurationError(
            f"Data configuration is missing required sections: {missing}"
        )
    return parsed


def validate_config(config: dict[str, Any]) -> None:
    """Validate that a parsed configuration dictionary contains all mandatory settings.

    Args:
        config: Parsed YAML dictionary.

    Raises:
        ConfigurationError: If required sections or mandatory values are missing or invalid.
    """
    if not isinstance(config, dict) or not config:
        raise ConfigurationError("Configuration must be a non-empty dictionary.")

    missing_sections = [
        section for section in REQUIRED_CONFIG_SECTIONS if section not in config
    ]
    if missing_sections:
        raise ConfigurationError(
            f"Configuration is missing required top-level sections: {missing_sections}"
        )

    # Validate project metadata
    project_cfg = config["project"]
    if not isinstance(project_cfg, dict) or not project_cfg.get("name") or not project_cfg.get("version"):
        raise ConfigurationError("Section 'project' must define non-empty 'name' and 'version'.")

    # Validate dataset identifier
    dataset_cfg = config["dataset"]
    if dataset_cfg.get("identifier") != "UrbanSound8K":
        raise ConfigurationError(
            f"Expected dataset identifier 'UrbanSound8K', got '{dataset_cfg.get('identifier')}'."
        )

    # Validate task definition
    task_cfg = config["task"]
    if task_cfg.get("type") != "binary_classification":
        raise ConfigurationError(
            f"Expected task.type 'binary_classification', got '{task_cfg.get('type')}'."
        )

    # Validate reproducibility seeds
    repro_cfg = config["reproducibility"]
    seeds = repro_cfg.get("random_seeds")
    expected_seeds = [42, 123, 2026]
    if not isinstance(seeds, list) or seeds != expected_seeds:
        raise ConfigurationError(
            f"Expected reproducibility.random_seeds to be {expected_seeds}, got {seeds}."
        )

    # Validate primary evaluation metric
    eval_cfg = config["evaluation"]
    if eval_cfg.get("primary_metric") != "macro-F1":
        raise ConfigurationError(
            f"Expected evaluation.primary_metric to be 'macro-F1', got '{eval_cfg.get('primary_metric')}'."
        )

    # Validate candidate models & first-principles NumPy MLP
    models_cfg = config["models"]
    candidates = models_cfg.get("candidate_models", [])
    candidate_names = {m.get("name") for m in candidates if isinstance(m, dict)}
    required_candidates = {"Random Forest", "Deep MLP", "2D CNN"}
    if not required_candidates.issubset(candidate_names):
        raise ConfigurationError(
            f"Models configuration must include candidate models {sorted(required_candidates)}; "
            f"found {sorted(candidate_names)}."
        )

    fp_model = models_cfg.get("first_principles_model", {})
    if not isinstance(fp_model, dict) or fp_model.get("framework") != "numpy":
        raise ConfigurationError(
            "Models configuration must define 'first_principles_model' with framework='numpy'."
        )

    # Validate paths
    paths_cfg = config["paths"]
    required_path_keys = (
        "raw_data_dir",
        "processed_data_dir",
        "splits_dir",
        "checkpoints_dir",
        "results_dir",
        "metrics_dir",
        "figures_dir",
    )
    missing_paths = [k for k in required_path_keys if not paths_cfg.get(k)]
    if missing_paths:
        raise ConfigurationError(
            f"Section 'paths' is missing required path entries: {missing_paths}"
        )


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration file.

    Args:
        config_path: Optional path to a YAML configuration file. Defaults to
            ``<project_root>/configs/config.yaml``.

    Returns:
        Validated configuration dictionary.

    Raises:
        ConfigurationError: If the file does not exist, cannot be parsed, or fails validation.
    """
    target_path = Path(config_path) if config_path is not None else get_default_config_path()

    if not target_path.exists():
        raise ConfigurationError(
            f"Configuration file not found at '{target_path}'. "
            "Ensure 'configs/config.yaml' exists before running the project."
        )
    if not target_path.is_file():
        raise ConfigurationError(
            f"Configuration path '{target_path}' is not a regular file."
        )

    try:
        with target_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML configuration file '{target_path}': {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"OS error while reading configuration file '{target_path}': {exc}"
        ) from exc

    if parsed is None:
        raise ConfigurationError(f"Configuration file '{target_path}' is empty.")

    validate_config(parsed)
    return parsed


def resolve_project_paths(
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Path]:
    """Resolve configured relative paths into cross-platform ``pathlib.Path`` objects.

    Args:
        config: Optional loaded configuration dictionary. If omitted, loads default config.
        project_root: Optional explicit root directory override.

    Returns:
        Dictionary mapping path keys to resolved ``Path`` instances.
    """
    cfg = config if config is not None else load_config()
    root = Path(project_root).resolve() if project_root is not None else get_project_root()

    paths_section = cfg.get("paths", {})
    resolved: dict[str, Path] = {"project_root": root}
    for key, rel_value in paths_section.items():
        rel_path = Path(str(rel_value))
        if rel_path.is_absolute():
            raise ConfigurationError(
                f"Path entry '{key}: {rel_value}' must be relative, not a hardcoded absolute path."
            )
        resolved[key] = (root / rel_path).resolve()

    return resolved


def set_random_seeds(seed: int = 42, deterministic_cudnn: bool = True) -> dict[str, Any]:
    """Initialize random number generators across Python, NumPy, and PyTorch.

    Args:
        seed: Non-negative integer seed value (e.g., 42, 123, or 2026).
        deterministic_cudnn: Whether to configure cuDNN for deterministic execution.

    Returns:
        Dictionary summarizing the applied seed state.

    Raises:
        ValueError: If ``seed`` is not a non-negative integer.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Random seed must be a non-negative integer, got {seed!r}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic_cudnn)
        torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cuda_seeded": cuda_available,
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
    }


def select_device(
    preferred: str = "auto",
    allow_gpu: bool = True,
    fallback_to_cpu: bool = True,
) -> DeviceInfo:
    """Detect hardware capabilities and select the appropriate PyTorch compute device.

    Args:
        preferred: Requested device string (``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``).
        allow_gpu: Whether GPU acceleration is permitted when available.
        fallback_to_cpu: Whether to fall back to ``"cpu"`` if the requested accelerator
            is unavailable.

    Returns:
        ``DeviceInfo`` instance containing device selection and hardware metadata.

    Raises:
        ConfigurationError: If an unsupported device string is passed or if a requested
            accelerator is unavailable and ``fallback_to_cpu`` is ``False``.
    """
    normalized = preferred.strip().lower()
    valid_options = {"auto", "cpu", "cuda", "mps"}
    if normalized not in valid_options:
        raise ConfigurationError(
            f"Unsupported device preference '{preferred}'. Valid options: {sorted(valid_options)}."
        )

    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_name = torch.cuda.get_device_name(0) if cuda_available and cuda_count > 0 else None
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if not allow_gpu or normalized == "cpu":
        selected = "cpu"
    elif normalized == "auto":
        if cuda_available:
            selected = "cuda"
        elif mps_available:
            selected = "mps"
        else:
            selected = "cpu"
    elif normalized == "cuda":
        if cuda_available:
            selected = "cuda"
        elif fallback_to_cpu:
            selected = "cpu"
        else:
            raise ConfigurationError("CUDA device was explicitly requested but is not available.")
    elif normalized == "mps":
        if mps_available:
            selected = "mps"
        elif fallback_to_cpu:
            selected = "cpu"
        else:
            raise ConfigurationError("MPS device was explicitly requested but is not available.")
    else:
        selected = "cpu"

    return DeviceInfo(
        requested_device=normalized,
        selected_device=selected,
        torch_device=torch.device(selected),
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_name=cuda_name,
        mps_available=mps_available,
    )


def validate_package_imports(
    packages: tuple[tuple[str, str], ...] = REQUIRED_PACKAGES,
) -> dict[str, PackageStatus]:
    """Attempt to import each required package and record its version or error.

    Args:
        packages: Sequence of ``(import_name, distribution_name)`` pairs.

    Returns:
        Dictionary mapping ``distribution_name`` to its ``PackageStatus``.
    """
    statuses: dict[str, PackageStatus] = {}
    for import_name, dist_name in packages:
        try:
            module = importlib.import_module(import_name)
            try:
                version = importlib.metadata.version(dist_name)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, "__version__", "unknown")
            statuses[dist_name] = PackageStatus(
                import_name=import_name,
                distribution_name=dist_name,
                available=True,
                version=str(version),
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            statuses[dist_name] = PackageStatus(
                import_name=import_name,
                distribution_name=dist_name,
                available=False,
                version=None,
                error=f"{type(exc).__name__}: {exc}",
            )
    return statuses


def get_environment_info(config: dict[str, Any] | None = None) -> EnvironmentReport:
    """Collect a structured report of the current Python, OS, hardware, and package state.

    Args:
        config: Optional loaded project configuration dictionary.

    Returns:
        ``EnvironmentReport`` instance.
    """
    cfg = config if config is not None else load_config()
    hw_cfg = cfg.get("hardware", {})
    device_info = select_device(
        preferred=str(hw_cfg.get("preferred_device", "auto")),
        allow_gpu=bool(hw_cfg.get("allow_gpu", True)),
        fallback_to_cpu=bool(hw_cfg.get("fallback_to_cpu", True)),
    )
    pkg_statuses = validate_package_imports()

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        project_root=get_project_root(),
        torch_version=str(torch.__version__),
        numpy_version=str(np.__version__),
        device_info=device_info,
        package_statuses=pkg_statuses,
    )

