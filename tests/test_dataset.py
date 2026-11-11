"""Unit tests for metadata loading, binary label mapping, dataset validation, and PyTorch Dataset/DataLoaders.

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Note:
    All audio files generated in these tests are clearly labeled synthetic unit-test
    fixtures created inside pytest's temporary directory (`tmp_path`).
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile as wavfile
import torch

from scripts.acquire_dataset import (
    check_disk_space_guard,
    compute_file_md5,
    download_zenodo_archive_resumable,
    validate_and_extract_archive,
)
from scripts.audit_dataset import run_dataset_audit
from src import DatasetNotFoundError, DatasetValidationError, load_data_config
from src.augmentation import AugmentationPolicyError, TrainingAudioAugmenter
from src.dataset import UrbanSoundSirenDataset, create_dataloaders
from src.preprocessing import (
    EXPECTED_CLASSES,
    EXPECTED_FOLDS,
    REQUIRED_METADATA_COLUMNS,
    load_and_enrich_metadata,
    map_binary_siren_label,
    validate_urbansound8k_dataset,
)


def _write_synthetic_fixture_wav(
    file_path: Path,
    sample_rate: int = 22050,
    duration_sec: float = 1.0,
    freq_hz: float = 800.0,
    stereo: bool = False,
) -> None:
    """Create a clearly labeled synthetic sine-wave WAV fixture for unit testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(round(sample_rate * duration_sec))
    t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    sine = (0.4 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    if stereo:
        sine = np.stack([sine, 0.5 * sine], axis=1)
    pcm16 = np.clip(sine * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(file_path), sample_rate, pcm16)


def _create_synthetic_urbansound_fixture(root_dir: Path) -> Path:
    """Create a complete 10-fold synthetic UrbanSound8K fixture inside `root_dir`."""
    us_root = root_dir / "UrbanSound8K"
    meta_dir = us_root / "metadata"
    audio_dir = us_root / "audio"
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in EXPECTED_FOLDS:
        fold_dir = audio_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Create 1 siren clip (classID=8) and 1 non-siren clip per fold with fold-isolated fsIDs
        siren_fsid = 10000 + fold * 10 + 1
        siren_name = f"{siren_fsid}-8-0-0.wav"
        _write_synthetic_fixture_wav(
            fold_dir / siren_name,
            sample_rate=22050 if fold % 2 == 0 else 16000,
            duration_sec=2.0,
            freq_hz=950.0,
            stereo=(fold == 1),
        )
        rows.append(
            {
                "slice_file_name": siren_name,
                "fsID": siren_fsid,
                "start": 0.0,
                "end": 2.0,
                "salience": 1,
                "fold": fold,
                "classID": 8,
                "class": "siren",
            }
        )

        other_cid = (fold - 1) % 8  # classIDs 0..7
        other_cname = EXPECTED_CLASSES[other_cid]
        other_fsid = 20000 + fold * 10 + 2
        other_name = f"{other_fsid}-{other_cid}-0-0.wav"
        _write_synthetic_fixture_wav(
            fold_dir / other_name,
            sample_rate=22050,
            duration_sec=1.5,
            freq_hz=300.0,
        )
        rows.append(
            {
                "slice_file_name": other_name,
                "fsID": other_fsid,
                "start": 0.0,
                "end": 1.5,
                "salience": 1,
                "fold": fold,
                "classID": other_cid,
                "class": other_cname,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(meta_dir / "UrbanSound8K.csv", index=False)
    return us_root


# ==============================================================================
# 1. Metadata Loading, Binary Label Mapping, Required Columns & Path Validation
# ==============================================================================


def test_binary_label_mapping_all_classes() -> None:
    """Verify binary label mapping: siren (8) -> 1, all other 9 classes -> 0."""
    for cid, cname in EXPECTED_CLASSES.items():
        label = map_binary_siren_label(cid, cname)
        if cid == 8:
            assert cname == "siren"
            assert label == 1
        else:
            assert label == 0

    # Mismatched classID and class name must raise DatasetValidationError
    with pytest.raises(DatasetValidationError, match="Inconsistent metadata label"):
        map_binary_siren_label(8, "dog_bark")
    with pytest.raises(DatasetValidationError, match="Inconsistent metadata label"):
        map_binary_siren_label(3, "siren")
    with pytest.raises(DatasetValidationError, match="Unknown UrbanSound8K classID"):
        map_binary_siren_label(99, "unknown")


def test_metadata_loading_and_required_columns(tmp_path: Path) -> None:
    """Verify metadata loading enriches binary labels and rejects missing required columns."""
    us_root = _create_synthetic_urbansound_fixture(tmp_path)
    meta_csv = us_root / "metadata" / "UrbanSound8K.csv"
    audio_dir = us_root / "audio"

    df = load_and_enrich_metadata(meta_csv, audio_dir=audio_dir)
    assert len(df) == 20
    assert set(REQUIRED_METADATA_COLUMNS).issubset(df.columns)
    assert "binary_label" in df.columns
    assert int((df["binary_label"] == 1).sum()) == 10
    assert int((df["binary_label"] == 0).sum()) == 10

    # Test missing required column handling
    incomplete_csv = tmp_path / "incomplete.csv"
    df.drop(columns=["fsID"]).to_csv(incomplete_csv, index=False)
    with pytest.raises(DatasetValidationError, match="missing required columns"):
        load_and_enrich_metadata(incomplete_csv)


def test_dataset_path_validation_and_missing_files(tmp_path: Path) -> None:
    """Verify path validation detects missing audio files and missing metadata."""
    with pytest.raises(DatasetNotFoundError):
        load_and_enrich_metadata(tmp_path / "does_not_exist.csv")

    us_root = _create_synthetic_urbansound_fixture(tmp_path)
    valid_report = validate_urbansound8k_dataset(dataset_root=us_root, strict=True)
    assert valid_report.is_valid is True
    assert valid_report.verified_audio_files == 20

    # Remove one audio file and verify missing file detection
    first_wav = next((us_root / "audio" / "fold1").glob("*.wav"))
    first_wav.unlink()
    report_missing = validate_urbansound8k_dataset(dataset_root=us_root, strict=False)
    assert report_missing.is_valid is False
    assert len(report_missing.missing_audio_files) == 1


# ==============================================================================
# 2. PyTorch Dataset, DataLoader, and Audit Pipeline Tests on Test Fixtures
# ==============================================================================


def test_pytorch_dataset_and_dataloaders_mfcc_and_mel(tmp_path: Path) -> None:
    """Verify UrbanSoundSirenDataset and DataLoaders for both MFCC and Mel representations."""
    us_root = _create_synthetic_urbansound_fixture(tmp_path)
    df = load_and_enrich_metadata(
        us_root / "metadata" / "UrbanSound8K.csv", audio_dir=us_root / "audio"
    )
    train_df = df[df["fold"].isin([1, 2, 3, 4, 5, 6, 7, 8])].reset_index(drop=True)
    val_df = df[df["fold"] == 9].reset_index(drop=True)
    test_df = df[df["fold"] == 10].reset_index(drop=True)

    # MFCC DataLoaders
    mfcc_loaders = create_dataloaders(
        train_manifest=train_df,
        val_manifest=val_df,
        test_manifest=test_df,
        audio_dir=us_root / "audio",
        representation="mfcc",
        batch_size=4,
        seed=42,
    )
    mfcc_batch_x, mfcc_batch_y = next(iter(mfcc_loaders["train"]))
    assert mfcc_batch_x.shape == (4, 80)
    assert mfcc_batch_x.dtype == torch.float32
    assert mfcc_batch_y.shape == (4,)
    assert set(mfcc_batch_y.tolist()).issubset({0, 1})

    # Mel Spectrogram DataLoaders
    mel_loaders = create_dataloaders(
        train_manifest=train_df,
        val_manifest=val_df,
        test_manifest=test_df,
        audio_dir=us_root / "audio",
        representation="mel",
        batch_size=2,
        seed=42,
    )
    mel_batch_x, mel_batch_y = next(iter(mel_loaders["val"]))
    assert mel_batch_x.shape == (2, 1, 128, 173)
    assert mel_batch_x.dtype == torch.float32
    assert mel_batch_y.shape == (2,)

    # Verify attaching an enabled augmenter to 'val' or 'test' raises AugmentationPolicyError
    enabled_aug = TrainingAudioAugmenter(enabled=True, probability=1.0, seed=42)
    with pytest.raises(AugmentationPolicyError, match="Cannot attach an enabled TrainingAudioAugmenter"):
        UrbanSoundSirenDataset(
            manifest=val_df,
            audio_dir=us_root / "audio",
            split="val",
            representation="mfcc",
            augmenter=enabled_aug,
        )


def test_archive_extraction_and_audit_generation_on_fixture(tmp_path: Path) -> None:
    """Verify archive integrity extraction and full T1(b) audit + 4 figures on synthetic fixture."""
    fixture_dir = tmp_path / "source_build"
    us_root = _create_synthetic_urbansound_fixture(fixture_dir)

    # Package fixture into a .tar.gz archive and test validate_and_extract_archive
    archive_path = tmp_path / "UrbanSound8K_fixture.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(us_root, arcname="UrbanSound8K")

    extract_dest = tmp_path / "extracted_raw"
    extracted_root = validate_and_extract_archive(archive_path, extract_dest)
    assert (extracted_root / "metadata" / "UrbanSound8K.csv").is_file()

    # Also verify flat-layout .zip archive extraction (UrbanSound8K.csv and fold1..10 at zip root)
    import zipfile

    flat_zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(flat_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(us_root / "metadata" / "UrbanSound8K.csv", arcname="UrbanSound8K.csv")
        for wav_p in (us_root / "audio").rglob("*.wav"):
            rel_wav = wav_p.relative_to(us_root / "audio").as_posix()
            zf.write(wav_p, arcname=rel_wav)

    flat_extract_dest = tmp_path / "extracted_flat_zip"
    extracted_flat_root = validate_and_extract_archive(flat_zip_path, flat_extract_dest)
    assert (extracted_flat_root / "metadata" / "UrbanSound8K.csv").is_file()
    assert (extracted_flat_root / "audio" / "fold10").is_dir()

    # Run dataset audit on extracted fixture
    metrics_out = tmp_path / "audit_metrics"
    figures_out = tmp_path / "audit_figures"
    data_cfg = load_data_config()
    audit_res = run_dataset_audit(
        dataset_root=extracted_root,
        metrics_dir=metrics_out,
        figures_dir=figures_out,
        figure_names=data_cfg["audit_outputs"]["figure_filenames"],
    )

    assert audit_res["total_audio_recordings"] == 20
    assert audit_res["binary_distribution"]["siren_count_label_1"] == 10
    assert audit_res["binary_distribution"]["non_siren_count_label_0"] == 10
    assert (metrics_out / "dataset_audit.json").is_file()
    assert (metrics_out / "class_distribution.csv").is_file()
    assert (metrics_out / "fold_distribution.csv").is_file()
    assert len(audit_res["generated_figures"]) == 4
    for fig_path_str in audit_res["generated_figures"].values():
        fig_file = Path(fig_path_str)
        assert fig_file.is_file()
        assert fig_file.stat().st_size > 0


def test_zenodo_resumable_download_and_gdrive_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify HTTP 206 Range resume, MD5 verification, and Google Drive cache persistence."""
    import io
    import urllib.request

    fixture_dir = tmp_path / "fixture_src"
    us_root = _create_synthetic_urbansound_fixture(fixture_dir)
    source_tar = tmp_path / "source_UrbanSound8K.tar.gz"
    with tarfile.open(source_tar, "w:gz") as tar:
        tar.add(us_root, arcname="UrbanSound8K")

    full_bytes = source_tar.read_bytes()
    expected_size = len(full_bytes)
    expected_md5 = compute_file_md5(source_tar)

    # Simulate an interrupted partial download (.part file already has first 40% of bytes)
    dl_dir = tmp_path / "colab_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    dest_archive = dl_dir / "UrbanSound8K.tar.gz"
    part_file = dl_dir / "UrbanSound8K.tar.gz.part"
    split_offset = expected_size // 2
    part_file.write_bytes(full_bytes[:split_offset])

    seen_range_headers: list[str | None] = []

    class _MockHTTPResponse:
        def __init__(self, payload: bytes, status: int) -> None:
            self._stream = io.BytesIO(payload)
            self.status = status
            self.headers = {"Accept-Ranges": "bytes"}

        def read(self, size: int = -1) -> bytes:
            return self._stream.read(size)

        def geturl(self) -> str:
            return "https://zenodo.org/api/records/1203745/files/UrbanSound8K.tar.gz/content"

        def __enter__(self) -> "_MockHTTPResponse":
            return self

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
            return None

    def _mock_urlopen(req: urllib.request.Request, timeout: int = 60) -> _MockHTTPResponse:
        range_hdr = req.get_header("Range")
        seen_range_headers.append(range_hdr)
        if range_hdr and range_hdr.startswith("bytes="):
            start_byte = int(range_hdr.split("bytes=")[1].split("-")[0])
            return _MockHTTPResponse(full_bytes[start_byte:], status=206)
        return _MockHTTPResponse(full_bytes, status=200)

    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen)

    gdrive_cache = tmp_path / "gdrive_cache"
    res = download_zenodo_archive_resumable(
        dest_archive_path=dest_archive,
        expected_bytes=expected_size,
        expected_md5=expected_md5,
        gdrive_cache_dir=gdrive_cache,
        forbid_windows_c_drive=False,
    )

    assert seen_range_headers == [f"bytes={split_offset}-"]
    assert res["status"] == "downloaded_and_verified"
    assert res["bytes_downloaded_this_run"] == expected_size - split_offset
    assert dest_archive.is_file()
    assert not part_file.exists()
    assert (gdrive_cache / "UrbanSound8K.tar.gz").is_file()

    # Now remove local dest_archive and verify second call reuses Google Drive cache with 0 network bytes
    dest_archive.unlink()
    res_cached = download_zenodo_archive_resumable(
        dest_archive_path=dest_archive,
        expected_bytes=expected_size,
        expected_md5=expected_md5,
        gdrive_cache_dir=gdrive_cache,
        forbid_windows_c_drive=False,
    )
    assert res_cached["status"] == "reused_gdrive_cached_archive"
    assert res_cached["bytes_downloaded_this_run"] == 0


def test_windows_c_drive_guard() -> None:
    """Verify that check_disk_space_guard blocks 6 GB downloads onto Windows C: drive by default."""
    import platform

    if platform.system().lower() == "windows":
        with pytest.raises(PermissionError, match="Refusing to download 6 GB archive onto Windows C: drive"):
            check_disk_space_guard(Path("C:/Temp/urbansound_test"), required_bytes=1024, forbid_windows_c_drive=True)


def test_colab_notebook_bootstrap_bundle_integrity(tmp_path: Path) -> None:
    """Verify notebooks/colab_t3_training.ipynb exists, has valid JSON, and its Cell 2 bundle unpacks cleanly."""
    import base64
    import hashlib
    import io
    import json
    import zipfile

    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "colab_t3_training.ipynb"
    assert nb_path.is_file()
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 6

    cell2_src = "".join(nb_data["cells"][2]["source"])
    sha_line = [line for line in cell2_src.splitlines() if line.startswith("EXPECTED_BUNDLE_SHA256 = ")][0]
    b64_line = [line for line in cell2_src.splitlines() if line.startswith("EMBEDDED_PROJECT_BUNDLE_B64 = ")][0]
    expected_sha = sha_line.split('"')[1]
    b64_payload = b64_line.split('"')[1]

    raw_zip = base64.b64decode(b64_payload)
    assert hashlib.sha256(raw_zip).hexdigest() == expected_sha

    sim_colab_root = tmp_path / "content" / "ANN-Siren-Detection"
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
        zf.extractall(sim_colab_root)

    for req_rel in (
        "configs/config.yaml",
        "configs/data_config.yaml",
        "configs/model_config.yaml",
        "src/models.py",
        "scripts/acquire_dataset.py",
        "scripts/audit_dataset.py",
        "scripts/prepare_dataset.py",
        "scripts/run_numpy_mlp_t2.py",
        "scripts/run_competing_models_t3.py",
    ):
        assert (sim_colab_root / req_rel).is_file(), f"Missing {req_rel} in unpacked Colab bundle"



