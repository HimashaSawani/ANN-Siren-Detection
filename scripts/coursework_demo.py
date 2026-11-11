"""Fast, code-first CW1 technical demonstration.

This runner deliberately does not train models. It loads existing experiment
artifacts, executes one real checkpoint inference, and exercises the invalid
input path so the demonstration is safe to run live.

Usage:
    python scripts/coursework_demo.py
    python scripts/coursework_demo.py --audio path\\to\\clip.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_features import CorruptAudioError
from src.inference import SirenInferenceEngine


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _show_project_evidence() -> None:
    split = _read_json(PROJECT_ROOT / "data" / "splits" / "split_summary.json")
    gradient = _read_json(PROJECT_ROOT / "results" / "metrics" / "t2_gradient_verification.json")
    comparison = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "t4_model_comparison.csv")
    ablation = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "t4_ablation_results.csv")
    errors = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "t4_error_analysis.csv")

    _print_header("1. Verified coursework protocol and artifacts")
    print("Dataset: UrbanSound8K binary siren versus non-siren")
    print(
        "Split: "
        f"train folds {split['verification']['train_folds']}, "
        f"validation folds {split['verification']['val_folds']}, "
        f"test folds {split['verification']['test_folds']}"
    )
    print(
        "Counts: "
        f"{split['verification']['train_count']} train / "
        f"{split['verification']['val_count']} validation / "
        f"{split['verification']['test_count']} held-out test"
    )
    print(
        "Gradient check: "
        f"passed={gradient['all_passed']}, "
        f"max relative difference={gradient['max_relative_difference']:.3e}"
    )

    _print_header("2. Existing model comparison (no retraining)")
    for row in comparison.itertuples(index=False):
        print(
            f"{row.model_name:<14} | "
            f"validation Macro-F1={row.val_macro_f1_mean:.4f} | "
            f"test Macro-F1={row.test_macro_f1_mean:.4f} | "
            f"seeds={row.num_seeds}"
        )

    dropout = ablation.groupby("condition", as_index=False)["val_macro_f1"].mean()
    print("\nDesign choice: dropout ablation on the Deep MLP (validation only)")
    for row in dropout.itertuples(index=False):
        print(f"{row.condition:<24} mean validation Macro-F1={row.val_macro_f1:.4f}")

    if not errors.empty:
        error = errors.iloc[0]
        print("\nObserved failure from held-out error analysis:")
        print(
            f"{error['model_id']} false positive on {error['class_name']} "
            f"({error['slice_file_name']}), P(siren)={error['predicted_siren_prob']:.4f}. "
            f"{error['acoustic_failure_mode']}"
        )


def _run_inference(audio_path: Path) -> None:
    _print_header("3. Code-based saved-checkpoint inference")
    print(f"Input: {audio_path}")
    engine = SirenInferenceEngine()
    result = engine.predict(audio_path.read_bytes(), threshold=0.50)
    print(f"Checkpoint: {Path(result['checkpoint_path']).name}")
    print(f"Model: {result['model_name']}")
    print(f"Features: {result['feature_dim']}-D MFCC")
    print(f"Prediction: {result['predicted_class']}")
    print(f"Estimated siren probability: {result['probability']:.4f}")
    print(f"Decision threshold: {result['threshold']:.2f}")
    print(
        f"Canonical audio: {result['sample_rate']} Hz, "
        f"{result['duration_sec']:.1f} seconds, {result['num_samples']} samples"
    )


def _run_invalid_input_check() -> None:
    _print_header("4. Invalid input handling")
    try:
        SirenInferenceEngine().predict(b"not a WAV file")
    except (CorruptAudioError, ValueError, OSError) as exc:
        print(f"Handled invalid audio safely: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("Invalid audio unexpectedly completed inference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_audio = PROJECT_ROOT / "demo" / "sample_audio" / "siren_sample_165166-8-0-0.wav"
    parser.add_argument("--audio", type=Path, default=default_audio)
    args = parser.parse_args()
    audio_path = args.audio if args.audio.is_absolute() else PROJECT_ROOT / args.audio
    if not audio_path.is_file():
        raise FileNotFoundError(f"Demo audio file not found: {audio_path}")

    _show_project_evidence()
    _run_inference(audio_path)
    _run_invalid_input_check()
    print("\nDemo complete: existing checkpoints and metrics were used; no training was run.")


if __name__ == "__main__":
    main()
