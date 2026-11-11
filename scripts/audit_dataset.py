"""Comprehensive Dataset Audit and Exploratory Figure Generation — T1(b).

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Analyzes:
- Total recordings, binary siren vs. non-siren counts, and class imbalance ratio.
- Original 10-class distribution and 10-fold class stratification.
- Audio durations, native sample rates, and channel counts from WAV headers.
- Missing metadata, missing/corrupt/zero-duration audio files, and invalid labels.
- Exact duplicate rows/filenames and source-recording (fsID) grouping structure.

Generates 4 figures in `results/figures/` and machine-readable JSON/CSV reports in
`results/metrics/` ONLY when a validated dataset is present.

Usage:
    python scripts/audit_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    DatasetNotFoundError,
    DatasetValidationError,
    load_config,
    load_data_config,
    resolve_project_paths,
    setup_logger,
)
from src.audio_features import inspect_audio_file  # noqa: E402
from src.preprocessing import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_FOLDS,
    REQUIRED_METADATA_COLUMNS,
    SIREN_CLASS_ID,
)


def _compute_file_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_audit_figures(
    df: pd.DataFrame,
    sample_rate_counts: dict[str, int],
    figures_dir: Path,
    figure_names: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Generate and save the 4 required T1(b) exploratory figures.

    Every figure includes a descriptive title, labeled axes, and an explicit legend.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = figure_names or {
        "class_distribution": "fig1_original_10class_distribution.png",
        "binary_distribution": "fig2_binary_siren_distribution.png",
        "duration_distribution": "fig3_audio_duration_distribution.png",
        "sample_rate_and_fold_distribution": "fig4_sample_rate_and_fold_distribution.png",
    }

    saved_paths: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Figure 1: Original 10-Class Distribution
    # ------------------------------------------------------------------
    fig1_path = figures_dir / names["class_distribution"]
    class_counts = df["class"].value_counts().sort_index()
    classes_ordered = list(class_counts.index)
    counts_ordered = [int(class_counts[c]) for c in classes_ordered]
    colors_fig1 = ["#d62728" if c == "siren" else "#1f77b4" for c in classes_ordered]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    siren_Indices = [i for i, c in enumerate(classes_ordered) if c == "siren"]
    non_siren_Indices = [i for i, c in enumerate(classes_ordered) if c != "siren"]

    if non_siren_Indices:
        ax.bar(
            non_siren_Indices,
            [counts_ordered[i] for i in non_siren_Indices],
            color="#1f77b4",
            edgecolor="black",
            alpha=0.85,
            label="Non-Siren Urban Classes (Label 0)",
        )
    if siren_Indices:
        ax.bar(
            siren_Indices,
            [counts_ordered[i] for i in siren_Indices],
            color="#d62728",
            edgecolor="black",
            alpha=0.90,
            label="Target Siren Class (Label 1)",
        )

    ax.set_xticks(range(len(classes_ordered)))
    ax.set_xticklabels(classes_ordered, rotation=30, ha="right")
    ax.set_title("UrbanSound8K Original 10-Class Audio Slice Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Urban Sound Class", fontsize=11)
    ax.set_ylabel("Number of Audio Recordings", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(fig1_path)
    plt.close(fig)
    saved_paths["class_distribution"] = fig1_path

    # ------------------------------------------------------------------
    # Figure 2: Binary Class Distribution (Siren vs. Non-Siren)
    # ------------------------------------------------------------------
    fig2_path = figures_dir / names["binary_distribution"]
    siren_count = int((df["binary_label"] == 1).sum())
    non_siren_count = int((df["binary_label"] == 0).sum())
    total_count = max(1, len(df))

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    bar_non = ax.bar(
        [0],
        [non_siren_count],
        color="#2b5c8f",
        edgecolor="black",
        width=0.5,
        label=f"Non-Siren (Label 0): {non_siren_count} ({100.0 * non_siren_count / total_count:.1f}%)",
    )
    bar_sir = ax.bar(
        [1],
        [siren_count],
        color="#d62728",
        edgecolor="black",
        width=0.5,
        label=f"Siren (Label 1): {siren_count} ({100.0 * siren_count / total_count:.1f}%)",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-Siren (Label 0)", "Siren (Label 1)"], fontsize=11)
    ax.set_title("Binary Target Class Distribution (Siren vs. Non-Siren)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Binary Classification Target", fontsize=11)
    ax.set_ylabel("Number of Audio Recordings", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    for rect in list(bar_non) + list(bar_sir):
        height = rect.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(fig2_path)
    plt.close(fig)
    saved_paths["binary_distribution"] = fig2_path

    # ------------------------------------------------------------------
    # Figure 3: Audio Duration Distribution
    # ------------------------------------------------------------------
    fig3_path = figures_dir / names["duration_distribution"]
    dur_col = "actual_duration_sec" if "actual_duration_sec" in df.columns else "metadata_duration_sec"
    siren_durs = df.loc[df["binary_label"] == 1, dur_col].dropna().to_numpy(dtype=float)
    non_siren_durs = df.loc[df["binary_label"] == 0, dur_col].dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=150)
    bins = np.linspace(0.0, max(4.2, float(df[dur_col].max() or 4.0)), 25)
    ax.hist(
        non_siren_durs,
        bins=bins,
        alpha=0.65,
        color="#1f77b4",
        edgecolor="black",
        label="Non-Siren Recordings (Label 0)",
    )
    ax.hist(
        siren_durs,
        bins=bins,
        alpha=0.75,
        color="#d62728",
        edgecolor="black",
        label="Siren Recordings (Label 1)",
    )
    ax.axvline(
        4.0,
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        label="Canonical Target Window (4.0 s)",
    )
    ax.set_title("Audio Recording Duration Distribution by Binary Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Audio Duration (seconds)", fontsize=11)
    ax.set_ylabel("Number of Audio Recordings", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(fig3_path)
    plt.close(fig)
    saved_paths["duration_distribution"] = fig3_path

    # ------------------------------------------------------------------
    # Figure 4: Sample-Rate and Predefined Fold Distribution
    # ------------------------------------------------------------------
    fig4_path = figures_dir / names["sample_rate_and_fold_distribution"]
    fig, (ax_fold, ax_sr) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=150)

    folds_sorted = sorted(df["fold"].astype(int).unique())
    fold_non_siren = [
        int(((df["fold"] == f) & (df["binary_label"] == 0)).sum()) for f in folds_sorted
    ]
    fold_siren = [
        int(((df["fold"] == f) & (df["binary_label"] == 1)).sum()) for f in folds_sorted
    ]
    x_pos = np.arange(len(folds_sorted))
    ax_fold.bar(
        x_pos,
        fold_non_siren,
        color="#1f77b4",
        edgecolor="black",
        label="Non-Siren (Label 0)",
    )
    ax_fold.bar(
        x_pos,
        fold_siren,
        bottom=fold_non_siren,
        color="#d62728",
        edgecolor="black",
        label="Siren (Label 1)",
    )
    ax_fold.set_xticks(x_pos)
    ax_fold.set_xticklabels([f"Fold {f}" for f in folds_sorted], rotation=30, ha="right")
    ax_fold.set_title("Predefined Fold Distribution (Folds 1-10)", fontsize=12, fontweight="bold")
    ax_fold.set_xlabel("Official Predefined Fold", fontsize=11)
    ax_fold.set_ylabel("Number of Recordings", fontsize=11)
    ax_fold.grid(axis="y", linestyle="--", alpha=0.4)
    ax_fold.legend(loc="upper right", frameon=True)

    sr_keys = sorted(sample_rate_counts.keys(), key=lambda x: int(x))
    sr_vals = [sample_rate_counts[k] for k in sr_keys]
    ax_sr.bar(
        range(len(sr_keys)),
        sr_vals,
        color="#9467bd",
        edgecolor="black",
        label="Native WAV Sample Rate Count",
    )
    ax_sr.set_xticks(range(len(sr_keys)))
    ax_sr.set_xticklabels([f"{k} Hz" for k in sr_keys], rotation=30, ha="right")
    ax_sr.set_title("Native Audio Sample Rate Distribution", fontsize=12, fontweight="bold")
    ax_sr.set_xlabel("Sampling Frequency (Hz)", fontsize=11)
    ax_sr.set_ylabel("Number of Audio Recordings", fontsize=11)
    ax_sr.grid(axis="y", linestyle="--", alpha=0.4)
    ax_sr.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "UrbanSound8K Fold Stratification and Native Sampling Rate Profile",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig4_path)
    plt.close(fig)
    saved_paths["sample_rate_and_fold_distribution"] = fig4_path

    return saved_paths


def run_dataset_audit(
    dataset_root: str | Path,
    metrics_dir: str | Path,
    figures_dir: str | Path,
    figure_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Perform a complete T1(b) audit on an existing UrbanSound8K dataset directory.

    Args:
        dataset_root: Path to `UrbanSound8K/` root containing `metadata/UrbanSound8K.csv`
            and `audio/fold1..fold10`.
        metrics_dir: Directory where `dataset_audit.json`, `class_distribution.csv`,
            and `fold_distribution.csv` will be written.
        figures_dir: Directory where the 4 exploratory figures will be saved.
        figure_names: Optional mapping of figure keys to filenames.

    Returns:
        Dictionary containing complete machine-readable audit findings.

    Raises:
        DatasetNotFoundError: If the dataset root or metadata CSV does not exist.
        DatasetValidationError: If required columns are missing.
    """
    root_path = Path(dataset_root).resolve()
    meta_path = root_path / "metadata" / "UrbanSound8K.csv"
    audio_dir = root_path / "audio"

    if not root_path.is_dir() or not meta_path.is_file() or not audio_dir.is_dir():
        raise DatasetNotFoundError(
            f"Real-data audit is blocked: UrbanSound8K dataset not found at '{root_path}'. "
            "Expected 'metadata/UrbanSound8K.csv' and 'audio/fold1..fold10'."
        )

    df = pd.read_csv(meta_path)
    missing_cols = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(
            f"Metadata CSV '{meta_path}' is missing required columns: {missing_cols}"
        )

    # 1. Missing metadata values
    missing_metadata_by_col = {
        col: int(df[col].isna().sum()) for col in REQUIRED_METADATA_COLUMNS
    }
    total_missing_cells = int(sum(missing_metadata_by_col.values()))

    # 2. Label validation and binary label mapping
    invalid_label_rows: list[dict[str, Any]] = []
    binary_labels: list[int] = []
    for idx, row in df.iterrows():
        cid = row["classID"]
        cname = str(row["class"]).strip()
        try:
            cid_int = int(cid)
            if cid_int not in EXPECTED_CLASSES or EXPECTED_CLASSES[cid_int] != cname:
                invalid_label_rows.append(
                    {"row_index": int(idx), "classID": cid, "class": cname}
                )
                binary_labels.append(1 if cid_int == SIREN_CLASS_ID else 0)
            else:
                binary_labels.append(1 if cid_int == SIREN_CLASS_ID else 0)
        except (ValueError, TypeError):
            invalid_label_rows.append(
                {"row_index": int(idx), "classID": cid, "class": cname}
            )
            binary_labels.append(0)

    df = df.copy()
    df["binary_label"] = np.asarray(binary_labels, dtype=np.int64)
    df["metadata_duration_sec"] = df["end"].astype(float) - df["start"].astype(float)

    # 3. Inspect every referenced audio file header
    missing_audio_files: list[str] = []
    corrupt_audio_files: list[dict[str, str]] = []
    zero_or_unusual_duration_files: list[dict[str, Any]] = []
    sample_rate_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}
    actual_durations: list[float | None] = []
    audio_md5_hashes: dict[str, list[str]] = {}

    for row in df.itertuples(index=False):
        fold_val = int(getattr(row, "fold"))
        fname = str(getattr(row, "slice_file_name"))
        rel_path = f"fold{fold_val}/{fname}"
        wav_path = audio_dir / f"fold{fold_val}" / fname

        header = inspect_audio_file(wav_path)
        if not header.exists:
            missing_audio_files.append(rel_path)
            actual_durations.append(None)
        elif header.is_corrupt:
            corrupt_audio_files.append(
                {"file": rel_path, "reason": str(header.error_message)}
            )
            actual_durations.append(header.duration_sec)
            if header.duration_sec is not None and header.duration_sec <= 0.0:
                zero_or_unusual_duration_files.append(
                    {"file": rel_path, "duration_sec": header.duration_sec}
                )
        else:
            sr_key = str(header.sample_rate)
            sample_rate_counts[sr_key] = sample_rate_counts.get(sr_key, 0) + 1
            ch_key = str(header.num_channels)
            channel_counts[ch_key] = channel_counts.get(ch_key, 0) + 1
            dur_val = float(header.duration_sec or 0.0)
            actual_durations.append(dur_val)
            if dur_val <= 0.05 or dur_val > 4.05:
                zero_or_unusual_duration_files.append(
                    {"file": rel_path, "duration_sec": round(dur_val, 4)}
                )
            md5_hex = _compute_file_md5(wav_path)
            audio_md5_hashes.setdefault(md5_hex, []).append(rel_path)

    df["actual_duration_sec"] = actual_durations
    exact_audio_duplicates = [
        paths for paths in audio_md5_hashes.values() if len(paths) > 1
    ]

    # 4. Duplicate metadata & source-recording (fsID) analysis
    exact_row_duplicates = int(df.duplicated(subset=list(REQUIRED_METADATA_COLUMNS)).sum())
    duplicate_slice_names = int(df["slice_file_name"].duplicated().sum())
    possible_interval_duplicates = int(
        df.duplicated(subset=["fsID", "start", "end", "classID"]).sum()
    )

    fsid_group_sizes = df.groupby("fsID").size()
    fsid_fold_nunique = df.groupby("fsID")["fold"].nunique()
    cross_fold_fsids = (
        fsid_fold_nunique[fsid_fold_nunique > 1].index.astype(int).tolist()
    )

    # 5. Class and fold distributions
    total_recordings = int(len(df))
    siren_count = int((df["binary_label"] == 1).sum())
    non_siren_count = int((df["binary_label"] == 0).sum())
    imbalance_ratio = (
        round(float(non_siren_count) / float(siren_count), 4)
        if siren_count > 0
        else None
    )

    class_dist_rows: list[dict[str, Any]] = []
    for cid, cname in sorted(EXPECTED_CLASSES.items()):
        c_count = int((df["classID"] == cid).sum())
        class_dist_rows.append(
            {
                "classID": cid,
                "class": cname,
                "binary_label": 1 if cid == SIREN_CLASS_ID else 0,
                "count": c_count,
                "percentage": round(100.0 * c_count / max(1, total_recordings), 2),
            }
        )

    fold_dist_rows: list[dict[str, Any]] = []
    for f in sorted(df["fold"].astype(int).unique()):
        f_df = df[df["fold"] == f]
        f_total = int(len(f_df))
        f_siren = int((f_df["binary_label"] == 1).sum())
        f_non_siren = int((f_df["binary_label"] == 0).sum())
        fold_dist_rows.append(
            {
                "fold": int(f),
                "total_count": f_total,
                "siren_count": f_siren,
                "non_siren_count": f_non_siren,
                "siren_percentage": round(100.0 * f_siren / max(1, f_total), 2),
                "unique_fsIDs": int(f_df["fsID"].nunique()),
            }
        )

    valid_durs = df["actual_duration_sec"].dropna().to_numpy(dtype=float)
    if valid_durs.size == 0:
        valid_durs = df["metadata_duration_sec"].to_numpy(dtype=float)

    duration_summary = {
        "min_sec": round(float(np.min(valid_durs)), 4) if valid_durs.size else None,
        "max_sec": round(float(np.max(valid_durs)), 4) if valid_durs.size else None,
        "mean_sec": round(float(np.mean(valid_durs)), 4) if valid_durs.size else None,
        "median_sec": round(float(np.median(valid_durs)), 4) if valid_durs.size else None,
        "std_sec": round(float(np.std(valid_durs)), 4) if valid_durs.size else None,
        "clips_under_4s_count": int(np.sum(valid_durs < 3.99)),
        "clips_approx_4s_count": int(np.sum((valid_durs >= 3.99) & (valid_durs <= 4.01))),
        "clips_over_4s_count": int(np.sum(valid_durs > 4.01)),
    }

    out_metrics = Path(metrics_dir)
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_figures = Path(figures_dir)
    saved_figs = generate_audit_figures(
        df=df,
        sample_rate_counts=sample_rate_counts,
        figures_dir=out_figures,
        figure_names=figure_names,
    )

    audit_payload: dict[str, Any] = {
        "dataset_root": str(root_path),
        "total_audio_recordings": total_recordings,
        "binary_distribution": {
            "siren_count_label_1": siren_count,
            "non_siren_count_label_0": non_siren_count,
            "siren_percentage": round(100.0 * siren_count / max(1, total_recordings), 2),
            "non_siren_percentage": round(
                100.0 * non_siren_count / max(1, total_recordings), 2
            ),
            "imbalance_ratio_non_siren_to_siren": imbalance_ratio,
        },
        "original_10_class_distribution": class_dist_rows,
        "fold_distribution": fold_dist_rows,
        "audio_duration_summary": duration_summary,
        "sample_rate_distribution_hz": sample_rate_counts,
        "channel_count_distribution": channel_counts,
        "data_quality": {
            "missing_metadata_cells_total": total_missing_cells,
            "missing_metadata_by_column": missing_metadata_by_col,
            "missing_audio_files_count": len(missing_audio_files),
            "missing_audio_files": missing_audio_files,
            "corrupt_or_unreadable_audio_count": len(corrupt_audio_files),
            "corrupt_or_unreadable_audio": corrupt_audio_files,
            "invalid_labels_count": len(invalid_label_rows),
            "invalid_labels": invalid_label_rows,
            "zero_or_unusual_duration_count": len(zero_or_unusual_duration_files),
            "zero_or_unusual_duration_files": zero_or_unusual_duration_files,
            "exact_duplicate_metadata_rows": exact_row_duplicates,
            "duplicate_slice_file_names": duplicate_slice_names,
            "possible_duplicate_intervals": possible_interval_duplicates,
            "exact_duplicate_audio_content_groups": exact_audio_duplicates,
        },
        "source_recording_grouping": {
            "unique_fsID_count": int(df["fsID"].nunique()),
            "multi_slice_fsID_count": int((fsid_group_sizes > 1).sum()),
            "min_slices_per_fsID": int(fsid_group_sizes.min()),
            "median_slices_per_fsID": float(fsid_group_sizes.median()),
            "mean_slices_per_fsID": round(float(fsid_group_sizes.mean()), 2),
            "max_slices_per_fsID": int(fsid_group_sizes.max()),
            "cross_fold_fsIDs": cross_fold_fsids,
        },
        "generated_figures": {k: str(v) for k, v in saved_figs.items()},
    }

    pd.DataFrame(class_dist_rows).to_csv(
        out_metrics / "class_distribution.csv", index=False
    )
    pd.DataFrame(fold_dist_rows).to_csv(
        out_metrics / "fold_distribution.csv", index=False
    )
    (out_metrics / "dataset_audit.json").write_text(
        json.dumps(audit_payload, indent=2), encoding="utf-8"
    )

    return audit_payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the UrbanSound8K dataset audit."""
    parser = argparse.ArgumentParser(
        description="Run the CW1 T1(b) UrbanSound8K dataset audit and generate exploratory figures."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to UrbanSound8K directory (defaults to configured data/raw/UrbanSound8K).",
    )
    args = parser.parse_args(argv)

    logger = setup_logger("audit_dataset")
    cfg = load_config()
    data_cfg = load_data_config()
    paths = resolve_project_paths(cfg)

    dataset_root = (
        Path(args.dataset_root).resolve()
        if args.dataset_root
        else paths["urbansound8k_root"]
    )
    audit_cfg = data_cfg.get("audit_outputs", {})

    try:
        audit_result = run_dataset_audit(
            dataset_root=dataset_root,
            metrics_dir=paths["metrics_dir"],
            figures_dir=paths["figures_dir"],
            figure_names=audit_cfg.get("figure_filenames"),
        )
    except DatasetNotFoundError as exc:
        logger.warning("%s", exc)
        logger.warning(
            "No audit figures or metrics were fabricated. Acquire the dataset first via "
            "`python scripts/acquire_dataset.py`."
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.error("Dataset audit failed: %s", exc)
        return 1

    logger.info(
        "Dataset audit complete: %d total clips (%d siren, %d non-siren). "
        "Saved JSON/CSV reports to '%s' and 4 figures to '%s'.",
        audit_result["total_audio_recordings"],
        audit_result["binary_distribution"]["siren_count_label_1"],
        audit_result["binary_distribution"]["non_siren_count_label_0"],
        paths["metrics_dir"],
        paths["figures_dir"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

