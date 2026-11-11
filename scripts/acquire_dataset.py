"""Official UrbanSound8K dataset acquisition, Zenodo resumable download, and validation CLI.

Supports:
1. Official Zenodo source verification (`https://zenodo.org/records/1203745`)
2. Resumable HTTP Range downloads (`UrbanSound8K.tar.gz`, 6,023,741,708 bytes,
   MD5 `9aa69802bbf37fb986f71ec1483a196e`) with automatic Google Drive persistence
3. Windows C: drive protection guard (prevents accidental 6 GB downloads onto C:)
4. Archive integrity and structure validation (`metadata/UrbanSound8K.csv` and `audio/fold1..10`)
5. Provenance and checksum recording (`data/raw/DATASET_PROVENANCE.json`)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time
from typing import Any
import urllib.error
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.preprocessing import EXPECTED_FOLDS, load_and_enrich_metadata

EXPECTED_TOTAL_RECORDINGS = 8732
load_urbansound_metadata = load_and_enrich_metadata

OFFICIAL_SOURCE_URL = "https://urbansounddataset.weebly.com/urbansound8k.html"
ZENODO_RECORD_ID = "1203745"
ZENODO_RECORD_URL = "https://zenodo.org/records/1203745"
ZENODO_API_URL = "https://zenodo.org/api/records/1203745"
ZENODO_CONTENT_URL = "https://zenodo.org/api/records/1203745/files/UrbanSound8K.tar.gz/content"
ZENODO_WEB_DOWNLOAD_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
URBANSOUND8K_ARCHIVE_NAME = "UrbanSound8K.tar.gz"
URBANSOUND8K_EXPECTED_BYTES = 6023741708
URBANSOUND8K_EXPECTED_MD5 = "9aa69802bbf37fb986f71ec1483a196e"

OFFICIAL_CITATION = (
    "J. Salamon, C. Jacoby and J. P. Bello, 'A Dataset and Taxonomy for Urban Sound Research', "
    "22nd ACM International Conference on Multimedia, Orlando, FL, USA, Nov. 2014."
)
LICENCE_NOTES = "Academic and research use only; subject to UrbanSound8K terms of use."


def detect_runtime_environment() -> dict[str, Any]:
    """Detect whether execution is inside a Google Colab runtime vs. local machine."""
    in_colab_module = "google.colab" in sys.modules
    colab_env_vars = any(
        key in __import__("os").environ
        for key in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_JUPYTER_IP", "GCS_READ_CACHE_BLOCK_SIZE_MB")
    )
    content_dir = Path("/content")
    is_linux = platform.system().lower() == "linux"
    is_colab_runtime = bool(in_colab_module or colab_env_vars or (is_linux and content_dir.is_dir()))
    gdrive_mount = Path("/content/drive/MyDrive")
    return {
        "is_colab_runtime": is_colab_runtime,
        "in_colab_module": in_colab_module,
        "colab_env_vars_present": colab_env_vars,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "content_dir_exists": content_dir.is_dir(),
        "gdrive_mounted": gdrive_mount.is_dir(),
        "gdrive_mydrive_path": str(gdrive_mount) if gdrive_mount.is_dir() else None,
    }


def compute_file_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_md5(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute the MD5 hex digest of a file (matches Zenodo's official MD5 checksum)."""
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_disk_space_guard(
    target_dir: Path,
    required_bytes: int = URBANSOUND8K_EXPECTED_BYTES,
    forbid_windows_c_drive: bool = True,
) -> dict[str, Any]:
    """Ensure target directory is not on Windows C: drive (unless allowed) and has sufficient space."""
    resolved_dir = target_dir.resolve()
    anchor_probe = resolved_dir
    while not anchor_probe.exists() and anchor_probe.parent != anchor_probe:
        anchor_probe = anchor_probe.parent

    drive_letter = resolved_dir.drive.upper()
    is_windows_c = platform.system().lower() == "windows" and drive_letter.startswith("C:")
    if forbid_windows_c_drive and is_windows_c:
        raise PermissionError(
            f"Refusing to download 6 GB archive onto Windows C: drive ({resolved_dir}). "
            "Use Google Colab (/content) with Google Drive persistence or specify a path on D:."
        )

    usage = shutil.disk_usage(anchor_probe)
    # Require at least 1.1x archive size for safe download
    min_free_bytes = int(required_bytes * 1.1)
    has_enough_space = usage.free >= min_free_bytes
    if not has_enough_space:
        raise OSError(
            f"Insufficient free disk space at {anchor_probe}: "
            f"free={usage.free / (1024**3):.2f} GiB, required={min_free_bytes / (1024**3):.2f} GiB."
        )

    return {
        "target_dir": str(resolved_dir),
        "drive": drive_letter or "/",
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "required_bytes": int(required_bytes),
        "has_enough_space": has_enough_space,
    }


def check_zenodo_source(
    api_url: str = ZENODO_API_URL,
    download_url: str = ZENODO_CONTENT_URL,
    timeout: int = 15,
) -> dict[str, Any]:
    """Verify connectivity, HTTP status, redirects, and Range support for Zenodo record 1203745."""
    report: dict[str, Any] = {
        "record_url": ZENODO_RECORD_URL,
        "api_url": api_url,
        "download_url": download_url,
        "api_accessible": False,
        "download_accessible": False,
        "supports_range_resume": False,
        "http_status_api": None,
        "http_status_download": None,
        "final_redirect_url": None,
        "expected_filename": URBANSOUND8K_ARCHIVE_NAME,
        "expected_bytes": URBANSOUND8K_EXPECTED_BYTES,
        "expected_md5": URBANSOUND8K_EXPECTED_MD5,
        "error": None,
    }

    headers = {"User-Agent": "ANN-Siren-Detection-Coursework/1.0 (Python-urllib)"}
    try:
        req_api = urllib.request.Request(api_url, headers=headers, method="GET")
        with urllib.request.urlopen(req_api, timeout=timeout) as resp:
            report["http_status_api"] = getattr(resp, "status", 200)
            payload = json.loads(resp.read().decode("utf-8"))
            files = payload.get("files", [])
            for f_entry in files:
                if f_entry.get("key") == URBANSOUND8K_ARCHIVE_NAME:
                    report["api_accessible"] = True
                    report["expected_bytes"] = int(f_entry.get("size", URBANSOUND8K_EXPECTED_BYTES))
                    raw_checksum = str(f_entry.get("checksum", ""))
                    if raw_checksum.startswith("md5:"):
                        report["expected_md5"] = raw_checksum.split("md5:", 1)[1]
                    links = f_entry.get("links", {})
                    if "self" in links:
                        report["zenodo_self_content_url"] = links["self"]
                    break
    except Exception as exc:
        report["error"] = f"Zenodo API check failed: {type(exc).__name__}: {exc}"

    try:
        range_headers = {
            "User-Agent": "ANN-Siren-Detection-Coursework/1.0 (Python-urllib)",
            "Range": "bytes=0-0",
        }
        req_dl = urllib.request.Request(download_url, headers=range_headers, method="GET")
        with urllib.request.urlopen(req_dl, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            report["http_status_download"] = status
            report["final_redirect_url"] = resp.geturl()
            report["download_accessible"] = status in (200, 206)
            accept_ranges = resp.headers.get("Accept-Ranges", "")
            content_range = resp.headers.get("Content-Range", "")
            report["supports_range_resume"] = (status == 206) or ("bytes" in accept_ranges.lower()) or bool(content_range)
    except Exception as exc:
        dl_err = f"Zenodo download probe failed: {type(exc).__name__}: {exc}"
        report["error"] = f"{report['error']}; {dl_err}" if report["error"] else dl_err

    return report


def verify_archive_file_checksum_and_size(
    archive_path: Path,
    expected_bytes: int | None = URBANSOUND8K_EXPECTED_BYTES,
    expected_md5: str | None = URBANSOUND8K_EXPECTED_MD5,
) -> dict[str, Any]:
    """Verify exact byte size and MD5 checksum of a downloaded archive."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive file does not exist: {archive_path}")

    actual_bytes = archive_path.stat().st_size
    size_matches = True if expected_bytes is None else (actual_bytes == expected_bytes)

    actual_md5: str | None = None
    md5_matches = True
    if expected_md5 and size_matches:
        actual_md5 = compute_file_md5(archive_path)
        md5_matches = actual_md5.lower() == expected_md5.lower()

    return {
        "archive_path": str(archive_path),
        "actual_bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "size_matches": size_matches,
        "actual_md5": actual_md5,
        "expected_md5": expected_md5,
        "md5_matches": md5_matches,
        "is_valid": bool(size_matches and md5_matches),
    }


def download_zenodo_archive_resumable(
    dest_archive_path: Path,
    download_url: str = ZENODO_CONTENT_URL,
    fallback_url: str = ZENODO_WEB_DOWNLOAD_URL,
    expected_bytes: int | None = URBANSOUND8K_EXPECTED_BYTES,
    expected_md5: str | None = URBANSOUND8K_EXPECTED_MD5,
    gdrive_cache_dir: Path | None = None,
    forbid_windows_c_drive: bool = True,
    chunk_size: int = 8 * 1024 * 1024,
    max_retries: int = 5,
    timeout: int = 60,
) -> dict[str, Any]:
    """Download UrbanSound8K.tar.gz from Zenodo with HTTP Range resume and optional Google Drive cache.

    Workflow:
    1. If `dest_archive_path` already exists and passes size + MD5 verification, reuse it immediately.
    2. If `gdrive_cache_dir` contains a verified `UrbanSound8K.tar.gz`, reuse or copy it without downloading from Zenodo.
    3. Otherwise, check disk space guard (blocking accidental 6 GB downloads onto Windows C: drive),
       and stream to `<dest_archive_path>.part` using HTTP `Range: bytes=<offset>-` headers.
    4. Verify size and MD5 checksum of the completed `.part` file, atomically rename to `dest_archive_path`,
       and optionally persist a copy to `gdrive_cache_dir` for future Colab sessions.
    """
    dest_archive_path = dest_archive_path.resolve()
    dest_archive_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Check if dest_archive_path is already present and valid
    if dest_archive_path.is_file():
        check = verify_archive_file_checksum_and_size(dest_archive_path, expected_bytes, expected_md5)
        if check["is_valid"]:
            return {
                "status": "reused_local_archive",
                "archive_path": str(dest_archive_path),
                "bytes_downloaded_this_run": 0,
                "verification": check,
                "persisted_to_gdrive": None,
            }

    # 2. Check Google Drive cache if configured
    cached_archive_path: Path | None = None
    if gdrive_cache_dir is not None:
        gdrive_cache_dir = gdrive_cache_dir.resolve()
        candidate_cache = gdrive_cache_dir / dest_archive_path.name
        if candidate_cache.is_file():
            cache_check = verify_archive_file_checksum_and_size(candidate_cache, expected_bytes, expected_md5)
            if cache_check["is_valid"]:
                return {
                    "status": "reused_gdrive_cached_archive",
                    "archive_path": str(candidate_cache),
                    "bytes_downloaded_this_run": 0,
                    "verification": cache_check,
                    "persisted_to_gdrive": str(candidate_cache),
                }

    # 3. Check disk space guard before downloading
    req_bytes = expected_bytes if expected_bytes is not None else URBANSOUND8K_EXPECTED_BYTES
    check_disk_space_guard(
        target_dir=dest_archive_path.parent,
        required_bytes=req_bytes,
        forbid_windows_c_drive=forbid_windows_c_drive,
    )

    part_path = dest_archive_path.with_name(dest_archive_path.name + ".part")
    bytes_downloaded_this_run = 0
    urls_to_try = [download_url]
    if fallback_url and fallback_url != download_url:
        urls_to_try.append(fallback_url)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        active_url = urls_to_try[(attempt - 1) % len(urls_to_try)]
        existing_bytes = part_path.stat().st_size if part_path.is_file() else 0

        if expected_bytes is not None and existing_bytes > expected_bytes:
            # Corrupted oversized partial file; restart cleanly
            part_path.unlink()
            existing_bytes = 0

        if expected_bytes is not None and existing_bytes == expected_bytes:
            break

        headers = {"User-Agent": "ANN-Siren-Detection-Coursework/1.0 (Python-urllib)"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        try:
            req = urllib.request.Request(active_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if existing_bytes > 0 and status == 200:
                    # Server ignored Range header and sent full content from byte 0
                    mode = "wb"
                    existing_bytes = 0
                elif existing_bytes > 0 and status == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                    existing_bytes = 0

                last_report_time = time.monotonic()
                with part_path.open(mode) as out_f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        bytes_downloaded_this_run += len(chunk)
                        existing_bytes += len(chunk)

                        now = time.monotonic()
                        if now - last_report_time >= 5.0:
                            pct = (
                                f"{(existing_bytes / expected_bytes) * 100:.1f}%"
                                if expected_bytes
                                else "unknown%"
                            )
                            print(
                                f"  [Download Progress] {existing_bytes / (1024**2):.1f} MiB "
                                f"({pct}) written to {part_path.name}",
                                flush=True,
                            )
                            last_report_time = now

            if expected_bytes is None or (part_path.is_file() and part_path.stat().st_size >= expected_bytes):
                last_error = None
                break

        except urllib.error.HTTPError as http_err:
            if http_err.code == 416 and expected_bytes is not None and existing_bytes == expected_bytes:
                # Range Not Satisfiable because file is already completely downloaded
                last_error = None
                break
            last_error = http_err
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))

    if last_error is not None:
        raise RuntimeError(
            f"Download failed after {max_retries} attempts. "
            f"Partial download preserved at {part_path} ({part_path.stat().st_size if part_path.is_file() else 0} bytes) "
            f"for automatic resume. Last error: {type(last_error).__name__}: {last_error}"
        )

    # 4. Verify completed .part file before renaming
    verification = verify_archive_file_checksum_and_size(
        part_path,
        expected_bytes=expected_bytes,
        expected_md5=expected_md5,
    )
    if not verification["is_valid"]:
        raise ValueError(
            f"Downloaded archive failed integrity check: "
            f"size_matches={verification['size_matches']} "
            f"(actual={verification['actual_bytes']}, expected={verification['expected_bytes']}), "
            f"md5_matches={verification['md5_matches']} "
            f"(actual={verification['actual_md5']}, expected={verification['expected_md5']})."
        )

    part_path.replace(dest_archive_path)
    verification["archive_path"] = str(dest_archive_path)

    # 5. Persist verified archive to Google Drive cache if requested
    persisted_path: str | None = None
    if gdrive_cache_dir is not None:
        gdrive_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_archive_path = gdrive_cache_dir / dest_archive_path.name
        if cached_archive_path.resolve() != dest_archive_path.resolve():
            shutil.copy2(dest_archive_path, cached_archive_path)
        persisted_path = str(cached_archive_path)

    return {
        "status": "downloaded_and_verified",
        "archive_path": str(dest_archive_path),
        "bytes_downloaded_this_run": bytes_downloaded_this_run,
        "verification": verification,
        "persisted_to_gdrive": persisted_path,
    }


def _classify_archive_members(members: list[str]) -> dict[str, Any]:
    """Identify metadata CSV, fold directories, and internal layout ('nested' vs 'flat')."""
    norm_members = [m.replace("\\", "/").lstrip("/") for m in members if m]
    has_nested_metadata = any(m.endswith("metadata/UrbanSound8K.csv") for m in norm_members)
    has_flat_metadata = any(
        m == "UrbanSound8K.csv"
        or (m.endswith("/UrbanSound8K.csv") and not m.endswith("metadata/UrbanSound8K.csv"))
        for m in norm_members
    )
    has_metadata_csv = has_nested_metadata or has_flat_metadata

    found_folds: list[int] = []
    for fold in EXPECTED_FOLDS:
        if any(
            f"audio/fold{fold}/" in m
            or m.endswith(f"audio/fold{fold}")
            or m.startswith(f"fold{fold}/")
            or f"/fold{fold}/" in m
            or m == f"fold{fold}"
            or m.endswith(f"/fold{fold}")
            for m in norm_members
        ):
            found_folds.append(fold)

    has_nested_folds = any(
        f"audio/fold{fold}/" in m or m.endswith(f"audio/fold{fold}")
        for fold in EXPECTED_FOLDS
        for m in norm_members
    )
    layout = "nested" if (has_nested_metadata and has_nested_folds) else "flat"
    return {
        "has_metadata_csv": has_metadata_csv,
        "found_folds": found_folds,
        "layout": layout,
    }


def _map_flat_member_to_canonical(member_name: str) -> Path | None:
    """Map a flat-layout archive member path into canonical UrbanSound8K/metadata or UrbanSound8K/audio."""
    parts = [p for p in member_name.replace("\\", "/").split("/") if p and p != "."]
    if ".." in parts:
        raise ValueError(f"Unsafe path traversal detected in archive member: {member_name}")
    if not parts:
        return None
    if parts[-1] == "UrbanSound8K.csv":
        return Path("UrbanSound8K") / "metadata" / "UrbanSound8K.csv"
    valid_fold_names = {f"fold{f}" for f in EXPECTED_FOLDS}
    for idx, part in enumerate(parts):
        if part in valid_fold_names:
            return Path("UrbanSound8K") / "audio" / Path(*parts[idx:])
    return Path("UrbanSound8K") / Path(*parts)


def verify_archive_integrity(archive_path: Path, verify_crc: bool = False) -> dict[str, Any]:
    """Verify archive readability and confirm required UrbanSound8K internal paths exist."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive file does not exist: {archive_path}")

    name_lower = archive_path.name.lower()
    members: list[str] = []
    archive_size_bytes = archive_path.stat().st_size

    if name_lower.endswith((".tar.gz", ".tgz", ".tar")):
        if not tarfile.is_tarfile(archive_path):
            raise ValueError(f"Invalid or corrupted tar archive: {archive_path}")
        with tarfile.open(archive_path, "r:*") as tar:
            members = [m.name.replace("\\", "/") for m in tar.getmembers()]
    elif name_lower.endswith(".zip"):
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"Invalid or corrupted zip archive: {archive_path}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            if verify_crc or archive_size_bytes < 100 * 1024 * 1024:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise ValueError(f"Corrupted member inside zip archive: {bad_file}")
            members = [name.replace("\\", "/") for name in zf.namelist()]
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}. Expected .tar.gz, .tgz, .tar, or .zip.")

    classification = _classify_archive_members(members)
    has_metadata_csv = classification["has_metadata_csv"]
    found_folds = classification["found_folds"]
    layout = classification["layout"]

    if not has_metadata_csv:
        raise ValueError(
            f"Archive {archive_path.name} is missing 'metadata/UrbanSound8K.csv' (or root 'UrbanSound8K.csv')."
        )
    if len(found_folds) != len(EXPECTED_FOLDS):
        missing = sorted(set(EXPECTED_FOLDS) - set(found_folds))
        raise ValueError(f"Archive {archive_path.name} is missing audio fold directories: {missing}")

    return {
        "archive_name": archive_path.name,
        "archive_size_bytes": archive_size_bytes,
        "member_count": len(members),
        "has_metadata_csv": has_metadata_csv,
        "found_folds": found_folds,
        "layout": layout,
    }


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path, layout: str = "nested") -> None:
    """Safely extract a tar archive while preventing path traversal outside dest_dir."""
    dest_resolved = dest_dir.resolve()
    members = tar.getmembers()
    if layout == "nested":
        for member in members:
            member_target = (dest_resolved / member.name).resolve()
            if not str(member_target).startswith(str(dest_resolved)):
                raise ValueError(f"Unsafe path traversal detected in tar archive member: {member.name}")
        tar.extractall(path=dest_resolved, filter="data")
    else:
        for member in members:
            rel_target = _map_flat_member_to_canonical(member.name)
            if rel_target is None:
                continue
            member_target = (dest_resolved / rel_target).resolve()
            if not str(member_target).startswith(str(dest_resolved)):
                raise ValueError(f"Unsafe path traversal detected in tar archive member: {member.name}")
            if member.isdir():
                member_target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                member_target.parent.mkdir(parents=True, exist_ok=True)
                extracted_f = tar.extractfile(member)
                if extracted_f is not None:
                    with extracted_f as src_f, member_target.open("wb") as out_f:
                        shutil.copyfileobj(src_f, out_f)


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path, layout: str = "nested") -> None:
    """Safely extract a zip archive (with automatic per-file CRC-32 verification and flat-layout adaptation)."""
    dest_resolved = dest_dir.resolve()
    infos = zf.infolist()
    total_files = sum(1 for info in infos if not info.is_dir())
    extracted_count = 0

    for info in infos:
        rel_target = (
            Path(info.filename)
            if layout == "nested"
            else _map_flat_member_to_canonical(info.filename)
        )
        if rel_target is None:
            continue
        member_target = (dest_resolved / rel_target).resolve()
        if not str(member_target).startswith(str(dest_resolved)):
            raise ValueError(f"Unsafe path traversal detected in zip archive member: {info.filename}")
        if info.is_dir() or info.filename.endswith("/"):
            member_target.mkdir(parents=True, exist_ok=True)
            continue
        member_target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info, "r") as src_f, member_target.open("wb") as out_f:
            shutil.copyfileobj(src_f, out_f, length=1024 * 1024)
        extracted_count += 1
        if extracted_count % 1000 == 0 or extracted_count == total_files:
            print(f"  [Extract Progress] {extracted_count}/{total_files} files extracted...", flush=True)


def extract_dataset_archive(archive_path: Path, raw_dir: Path) -> Path:
    """Validate and extract an UrbanSound8K archive into raw_dir."""
    integrity_info = verify_archive_integrity(archive_path)
    layout = str(integrity_info.get("layout", "nested"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    name_lower = archive_path.name.lower()
    if name_lower.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar:
            _safe_extract_tar(tar, raw_dir, layout=layout)
    elif name_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract_zip(zf, raw_dir, layout=layout)

    candidate = raw_dir / "UrbanSound8K"
    if (candidate / "metadata" / "UrbanSound8K.csv").is_file():
        return candidate
    if (raw_dir / "metadata" / "UrbanSound8K.csv").is_file():
        return raw_dir
    raise FileNotFoundError("Extraction finished, but metadata/UrbanSound8K.csv was not found in expected paths.")


validate_and_extract_archive = extract_dataset_archive


def validate_extracted_dataset(dataset_root: Path, require_complete: bool = True) -> dict[str, Any]:
    """Verify metadata CSV and referenced audio files exist inside dataset_root."""
    metadata_csv = dataset_root / "metadata" / "UrbanSound8K.csv"
    audio_root = dataset_root / "audio"

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing metadata CSV at {metadata_csv}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing audio directory at {audio_root}")

    df = load_urbansound_metadata(metadata_csv)
    present_folds = [fold for fold in EXPECTED_FOLDS if (audio_root / f"fold{fold}").is_dir()]

    missing_files: list[str] = []
    existing_count = 0
    for row in df.itertuples(index=False):
        rel = f"fold{int(row.fold)}/{row.slice_file_name}"
        if (audio_root / rel).is_file():
            existing_count += 1
        else:
            missing_files.append(rel)

    if require_complete and missing_files:
        raise FileNotFoundError(
            f"Dataset incomplete: {len(missing_files)} referenced audio files are missing "
            f"(first missing: {missing_files[0]})."
        )

    return {
        "dataset_root": str(dataset_root.resolve()),
        "metadata_csv": str(metadata_csv.resolve()),
        "metadata_rows": int(len(df)),
        "expected_official_rows": EXPECTED_TOTAL_RECORDINGS,
        "matches_official_row_count": int(len(df)) == EXPECTED_TOTAL_RECORDINGS,
        "present_folds": present_folds,
        "existing_audio_files": existing_count,
        "missing_audio_files": len(missing_files),
    }


def copy_dataset_directory(source_dir: Path, target_dataset_root: Path) -> Path:
    """Copy or link a validated local UrbanSound8K folder into target_dataset_root."""
    if (source_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv").is_file():
        source_dir = source_dir / "UrbanSound8K"
    validate_extracted_dataset(source_dir, require_complete=True)

    if source_dir.resolve() != target_dataset_root.resolve():
        target_dataset_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dataset_root, dirs_exist_ok=True)
    return target_dataset_root


def write_provenance_manifest(
    output_path: Path,
    acquisition_mode: str,
    validation_summary: dict[str, Any],
    archive_info: dict[str, Any] | None = None,
) -> Path:
    """Write dataset provenance, attribution, and validation summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": "UrbanSound8K",
        "official_source_url": OFFICIAL_SOURCE_URL,
        "zenodo_record_url": ZENODO_RECORD_URL,
        "zenodo_api_url": ZENODO_API_URL,
        "citation": OFFICIAL_CITATION,
        "access_conditions": LICENCE_NOTES,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_mode": acquisition_mode,
        "archive_info": archive_info,
        "validation_summary": validation_summary,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire, download (resumable from Zenodo), and validate the official UrbanSound8K dataset."
    )
    parser.add_argument("--archive", type=Path, default=None, help="Path to local UrbanSound8K.tar.gz or .zip archive.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Path to extracted UrbanSound8K directory.")
    parser.add_argument(
        "--target-raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Target raw data directory (default: data/raw).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download UrbanSound8K.tar.gz from official Zenodo record 1203745 with HTTP Range resume.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Directory to store the downloaded UrbanSound8K.tar.gz archive (defaults to target-raw-dir).",
    )
    parser.add_argument(
        "--gdrive-cache-dir",
        type=Path,
        default=None,
        help="Optional Google Drive directory to cache/reuse the verified 6 GB archive across Colab sessions.",
    )
    parser.add_argument(
        "--zenodo-url",
        type=str,
        default=ZENODO_CONTENT_URL,
        help="Zenodo direct content download URL.",
    )
    parser.add_argument(
        "--expected-bytes",
        type=int,
        default=URBANSOUND8K_EXPECTED_BYTES,
        help="Expected archive size in bytes (default: 6023741708).",
    )
    parser.add_argument(
        "--expected-md5",
        type=str,
        default=URBANSOUND8K_EXPECTED_MD5,
        help="Expected MD5 checksum of the archive (default: 9aa69802bbf37fb986f71ec1483a196e).",
    )
    parser.add_argument(
        "--allow-c-drive",
        action="store_true",
        help="Override the safety guard that prevents downloading 6 GB onto the Windows C: drive.",
    )
    parser.add_argument(
        "--check-network",
        action="store_true",
        help="Probe Zenodo record 1203745 connectivity, redirects, and HTTP Range support without downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify whether the dataset is already present in target-raw-dir/UrbanSound8K.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ = load_config()
    target_raw_dir: Path = args.target_raw_dir.resolve()
    target_dataset_root = target_raw_dir / "UrbanSound8K"
    provenance_path = target_raw_dir / "DATASET_PROVENANCE.json"

    if args.check_network:
        env_info = detect_runtime_environment()
        net_report = check_zenodo_source(download_url=args.zenodo_url)
        print(json.dumps({"runtime": env_info, "zenodo_check": net_report}, indent=2))
        return 0 if net_report["download_accessible"] or net_report["api_accessible"] else 2

    try:
        # Fast path: if already extracted and valid in target_dataset_root, verify and return 0
        if (target_dataset_root / "metadata" / "UrbanSound8K.csv").is_file() and not (
            args.archive or args.source_dir
        ):
            try:
                summary = validate_extracted_dataset(target_dataset_root, require_complete=True)
                write_provenance_manifest(provenance_path, "existing_verified_dataset", summary)
                print(
                    f"[OK] UrbanSound8K already extracted and verified at {target_dataset_root} "
                    f"({summary['existing_audio_files']} audio files across {len(summary['present_folds'])} folds)."
                )
                return 0
            except Exception:
                if not args.download:
                    raise

        if args.download:
            download_dir = (args.download_dir or target_raw_dir).resolve()
            dest_archive = download_dir / URBANSOUND8K_ARCHIVE_NAME
            print(f"[INFO] Starting resumable Zenodo acquisition -> {dest_archive}")
            dl_result = download_zenodo_archive_resumable(
                dest_archive_path=dest_archive,
                download_url=args.zenodo_url,
                expected_bytes=args.expected_bytes if args.expected_bytes > 0 else None,
                expected_md5=args.expected_md5 if args.expected_md5 else None,
                gdrive_cache_dir=args.gdrive_cache_dir,
                forbid_windows_c_drive=not args.allow_c_drive,
            )
            archive_to_extract = Path(dl_result["archive_path"])
            archive_info = verify_archive_integrity(archive_to_extract)
            archive_info["md5"] = dl_result["verification"].get("actual_md5") or args.expected_md5
            archive_info["download_status"] = dl_result["status"]
            archive_info["persisted_to_gdrive"] = dl_result["persisted_to_gdrive"]
            print(f"[INFO] Extracting verified archive {archive_to_extract} -> {target_raw_dir} ...")
            dataset_root = extract_dataset_archive(archive_to_extract, target_raw_dir)
            summary = validate_extracted_dataset(dataset_root, require_complete=True)
            write_provenance_manifest(provenance_path, "zenodo_resumable_download", summary, archive_info)
        elif args.archive is not None:
            archive_info = verify_archive_integrity(args.archive)
            archive_info["sha256"] = compute_file_sha256(args.archive)
            dataset_root = extract_dataset_archive(args.archive, target_raw_dir)
            summary = validate_extracted_dataset(dataset_root, require_complete=True)
            write_provenance_manifest(provenance_path, "archive_extraction", summary, archive_info)
        elif args.source_dir is not None:
            dataset_root = copy_dataset_directory(args.source_dir, target_dataset_root)
            summary = validate_extracted_dataset(dataset_root, require_complete=True)
            write_provenance_manifest(provenance_path, "local_directory_copy", summary)
        else:
            if not target_dataset_root.exists():
                print(
                    "[BLOCKED] Official UrbanSound8K dataset not found at "
                    f"{target_dataset_root}.\n"
                    f"Official Zenodo source: {ZENODO_RECORD_URL}\n"
                    "To download directly (recommended in Google Colab with Google Drive persistence):\n"
                    "  python scripts/acquire_dataset.py --download --gdrive-cache-dir /content/drive/MyDrive/ANN-Siren-Detection/cache\n"
                    "Or provide an existing local archive or directory:\n"
                    "  python scripts/acquire_dataset.py --archive <path/to/UrbanSound8K.tar.gz>\n"
                    "  python scripts/acquire_dataset.py --source-dir <path/to/UrbanSound8K>"
                )
                return 2
            summary = validate_extracted_dataset(target_dataset_root, require_complete=True)
            write_provenance_manifest(provenance_path, "verify_existing", summary)

        print(
            f"[OK] UrbanSound8K validated at {summary['dataset_root']} "
            f"({summary['existing_audio_files']} audio files across {len(summary['present_folds'])} folds)."
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Dataset acquisition/validation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
