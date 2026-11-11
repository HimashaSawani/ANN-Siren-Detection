"""Build and update `notebooks/colab_t3_training.ipynb` with embedded project source bootstrap.

Ensures that when `notebooks/colab_t3_training.ipynb` is executed on a remote Google Colab
kernel from VS Code (where local `D:\\ANN-Siren-Detection` files are not mounted on `/content`),
the notebook:
1. Explicitly verifies whether it is running inside a Google Colab runtime before using `/content`.
2. Automatically unpacks the exact project source files (`configs/`, `src/`, `scripts/`,
   `requirements.txt`, `pyproject.toml`) into the Colab workspace if not already present.
3. Downloads `UrbanSound8K.tar.gz` directly from Zenodo (`https://zenodo.org/records/1203745`)
   into the Colab runtime with HTTP Range resume, MD5 verification, and Google Drive persistence.
4. Executes `audit_dataset.py`, `prepare_dataset.py`, `run_numpy_mlp_t2.py`, and
   `run_competing_models_t3.py`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_compressed_project_bundle() -> tuple[str, str, list[str]]:
    """Compress all project source/config files into a base64-encoded ZIP string."""
    rel_paths: list[Path] = []
    for folder in ("configs", "src", "scripts", "reports"):
        folder_path = PROJECT_ROOT / folder
        if folder_path.is_dir():
            for p in sorted(folder_path.rglob("*")):
                if p.is_file() and p.suffix in (".py", ".yaml", ".md") and "__pycache__" not in p.parts:
                    if p.name == "update_colab_notebook.py":
                        continue
                    rel_paths.append(p.relative_to(PROJECT_ROOT))

    for root_file in ("requirements.txt", "pyproject.toml", "README.md"):
        rf = PROJECT_ROOT / root_file
        if rf.is_file():
            rel_paths.append(Path(root_file))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in rel_paths:
            zf.write(PROJECT_ROOT / rel, arcname=rel.as_posix())

    raw_bytes = buf.getvalue()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    return b64_str, sha256_hex, [r.as_posix() for r in rel_paths]


def generate_colab_notebook() -> Path:
    b64_bundle, bundle_sha256, bundled_files = build_compressed_project_bundle()
    notebook_path = PROJECT_ROOT / "notebooks" / "colab_t3_training.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# NB6057CEM — Artificial Neural Networks (CW1 T1–T3 Real-Data Training & Zenodo Acquisition)\n",
                "**Project:** Robust Emergency Siren Classification Under Urban Noise: A Comparative Study of MLP and CNN Architectures\n",
                "\n",
                "This self-contained notebook is engineered for **VS Code with a Google Colab kernel**, browser **Google Colab Pro**, or a local environment:\n",
                "1. **Runtime Verification:** Verifies whether execution is genuinely inside a Google Colab Linux VM before touching `/content` paths, and protects the local Windows `C:` drive from 6 GB downloads.\n",
                "2. **Automatic Cloud Workspace Bootstrap:** Because a VS Code Colab kernel runs on a remote VM without direct access to local `D:\\ANN-Siren-Detection` files, Cell 2 automatically unpacks all verified project modules (`configs/`, `src/`, `scripts/`, `requirements.txt`) into `/content/ANN-Siren-Detection`.\n",
                "3. **Resumable Official Zenodo Download + Google Drive Persistence:** Downloads `UrbanSound8K.tar.gz` (`6,023,741,708` bytes, MD5 `9aa69802bbf37fb986f71ec1483a196e`) from `https://zenodo.org/records/1203745` with HTTP `Range` resume support, persists the verified archive to Google Drive so you never have to re-download 6 GB, and safely extracts it.\n",
                "4. **Full T1–T3 Pipeline Execution:** Runs `audit_dataset.py`, `prepare_dataset.py`, `run_numpy_mlp_t2.py`, and `run_competing_models_t3.py`."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 1: Verify Runtime Environment (Colab Cloud VM vs. Local Machine) & Mount Google Drive\n",
                "import os\n",
                "import platform\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "in_colab_module = \"google.colab\" in sys.modules\n",
                "has_colab_env = any(k in os.environ for k in (\"COLAB_RELEASE_TAG\", \"COLAB_GPU\", \"COLAB_JUPYTER_IP\"))\n",
                "is_linux_content = platform.system().lower() == \"linux\" and Path(\"/content\").is_dir()\n",
                "IS_COLAB_RUNTIME = bool(in_colab_module or has_colab_env or is_linux_content)\n",
                "\n",
                "print(f\"Platform         : {platform.system()} {platform.release()}\")\n",
                "print(f\"Python Executable: {sys.executable}\")\n",
                "print(f\"Is Colab Runtime : {IS_COLAB_RUNTIME}\")\n",
                "\n",
                "GDRIVE_CACHE_DIR = None\n",
                "GDRIVE_ARTIFACTS_DIR = None\n",
                "\n",
                "if IS_COLAB_RUNTIME:\n",
                "    PROJECT_ROOT = Path(os.environ.get(\"ANN_PROJECT_ROOT\", \"/content/ANN-Siren-Detection\")).resolve()\n",
                "    DOWNLOAD_SCRATCH_DIR = Path(\"/content/urbansound8k_dl\").resolve()\n",
                "    # Mount Google Drive for persistent caching of the 6.02 GB archive and trained checkpoints\n",
                "    if os.environ.get(\"MOUNT_GDRIVE\", \"1\") == \"1\":\n",
                "        try:\n",
                "            from google.colab import drive  # type: ignore\n",
                "            mount_point = os.environ.get(\"GDRIVE_MOUNT_POINT\", \"/content/drive\")\n",
                "            if not Path(mount_point, \"MyDrive\").is_dir():\n",
                "                drive.mount(mount_point)\n",
                "            gdrive_root = Path(mount_point) / \"MyDrive\" / \"ANN-Siren-Detection\"\n",
                "            GDRIVE_CACHE_DIR = gdrive_root / \"cache\"\n",
                "            GDRIVE_ARTIFACTS_DIR = gdrive_root / \"artifacts\"\n",
                "            GDRIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)\n",
                "            GDRIVE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)\n",
                "            print(f\"Google Drive Cache Dir    : {GDRIVE_CACHE_DIR}\")\n",
                "            print(f\"Google Drive Artifacts Dir: {GDRIVE_ARTIFACTS_DIR}\")\n",
                "        except Exception as exc:\n",
                "            print(f\"[WARN] Google Drive mount skipped or unavailable ({exc}). Using /content storage only.\")\n",
                "else:\n",
                "    # Running on local machine (e.g. D:\\ANN-Siren-Detection) — NEVER use /content paths\n",
                "    cwd = Path.cwd().resolve()\n",
                "    PROJECT_ROOT = cwd.parent if cwd.name == \"notebooks\" else cwd\n",
                "    DOWNLOAD_SCRATCH_DIR = PROJECT_ROOT / \"data\" / \"raw\"\n",
                "\n",
                "PROJECT_ROOT.mkdir(parents=True, exist_ok=True)\n",
                "os.chdir(PROJECT_ROOT)\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "print(f\"Active Project Root       : {PROJECT_ROOT}\")"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 2: Bootstrap Project Files into Colab Runtime (No Local File Assumption)\n",
                "import base64\n",
                "import hashlib\n",
                "import io\n",
                "import zipfile\n",
                "\n",
                f"EXPECTED_BUNDLE_SHA256 = \"{bundle_sha256}\"\n",
                f"EMBEDDED_PROJECT_BUNDLE_B64 = \"{b64_bundle}\"\n",
                "\n",
                "required_markers = [\n",
                "    PROJECT_ROOT / \"configs\" / \"config.yaml\",\n",
                "    PROJECT_ROOT / \"configs\" / \"data_config.yaml\",\n",
                "    PROJECT_ROOT / \"configs\" / \"model_config.yaml\",\n",
                "    PROJECT_ROOT / \"src\" / \"models.py\",\n",
                "    PROJECT_ROOT / \"scripts\" / \"acquire_dataset.py\",\n",
                "    PROJECT_ROOT / \"scripts\" / \"audit_dataset.py\",\n",
                "    PROJECT_ROOT / \"scripts\" / \"prepare_dataset.py\",\n",
                "    PROJECT_ROOT / \"scripts\" / \"run_numpy_mlp_t2.py\",\n",
                "    PROJECT_ROOT / \"scripts\" / \"run_competing_models_t3.py\",\n",
                "]\n",
                "\n",
                "if IS_COLAB_RUNTIME or not all(p.is_file() for p in required_markers):\n",
                "    raw_zip = base64.b64decode(EMBEDDED_PROJECT_BUNDLE_B64)\n",
                "    actual_sha = hashlib.sha256(raw_zip).hexdigest()\n",
                "    assert actual_sha == EXPECTED_BUNDLE_SHA256, f\"Bundle checksum mismatch: {actual_sha}\"\n",
                "    with zipfile.ZipFile(io.BytesIO(raw_zip), \"r\") as zf:\n",
                "        zf.extractall(PROJECT_ROOT)\n",
                "    print(f\"[OK] Bootstrapped {len(required_markers)}+ project files into {PROJECT_ROOT}\")\n",
                "else:\n",
                "    print(f\"[OK] Local project files verified in {PROJECT_ROOT}\")\n",
                "\n",
                "for rel_dir in [\n",
                "    \"data/raw\",\n",
                "    \"data/processed\",\n",
                "    \"data/splits\",\n",
                "    \"checkpoints\",\n",
                "    \"results/metrics\",\n",
                "    \"results/figures\",\n",
                "    \"reports\",\n",
                "]:\n",
                "    (PROJECT_ROOT / rel_dir).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "for marker in required_markers:\n",
                "    assert marker.is_file(), f\"Missing required project file: {marker}\"\n",
                "print(\"[OK] All required project scripts, configs, and source modules are ready.\")"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 3: Verify Dependencies, Hardware (GPU/CPU), and Zenodo Network Connectivity\n",
                "import subprocess\n",
                "\n",
                "if IS_COLAB_RUNTIME:\n",
                "    # Colab already pre-installs torch, torchaudio, numpy, pandas, scipy, scikit-learn, librosa, matplotlib\n",
                "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"pyyaml\", \"soundfile\", \"seaborn\"])\n",
                "\n",
                "from src import get_environment_info, load_config, select_device\n",
                "from scripts.acquire_dataset import check_zenodo_source\n",
                "\n",
                "cfg = load_config()\n",
                "env_report = get_environment_info(cfg)\n",
                "device_info = select_device(\"auto\")\n",
                "\n",
                "print(f\"Python Version : {env_report.python_version}\")\n",
                "print(f\"PyTorch Version: {env_report.torch_version}\")\n",
                "print(f\"Selected Device: {device_info.selected_device} (CUDA available: {device_info.cuda_available})\")\n",
                "if device_info.cuda_device_name:\n",
                "    print(f\"GPU Device Name: {device_info.cuda_device_name}\")\n",
                "\n",
                "zenodo_status = check_zenodo_source()\n",
                "print(\"\\nZenodo Official Source Check (https://zenodo.org/records/1203745):\")\n",
                "print(f\"  API Accessible       : {zenodo_status['api_accessible']} (HTTP {zenodo_status['http_status_api']})\")\n",
                "print(f\"  Download Accessible  : {zenodo_status['download_accessible']} (HTTP {zenodo_status['http_status_download']})\")\n",
                "print(f\"  Supports Range Resume: {zenodo_status['supports_range_resume']}\")\n",
                "print(f\"  Expected Archive Size: {zenodo_status['expected_bytes']:,} bytes (6.02 GB)\")\n",
                "print(f\"  Expected Archive MD5 : {zenodo_status['expected_md5']}\")"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 4: Resumable Zenodo Download, Integrity Verification, Google Drive Persistence & Safe Extraction\n",
                "from scripts.acquire_dataset import main as acquire_main\n",
                "\n",
                "acquire_cli_args = [\n",
                "    \"--download\",\n",
                "    \"--download-dir\",\n",
                "    str(DOWNLOAD_SCRATCH_DIR),\n",
                "    \"--target-raw-dir\",\n",
                "    str(PROJECT_ROOT / \"data\" / \"raw\"),\n",
                "]\n",
                "if GDRIVE_CACHE_DIR is not None:\n",
                "    acquire_cli_args.extend([\"--gdrive-cache-dir\", str(GDRIVE_CACHE_DIR)])\n",
                "\n",
                "rc_acquire = acquire_main(acquire_cli_args)\n",
                "assert rc_acquire == 0, f\"Dataset acquisition failed with exit code {rc_acquire}\""
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 5: Execute Real-Data Audit (T1b), Split/Scaler Prep (T1c), NumPy MLP (T2), and Competing Models (T3)\n",
                "import subprocess\n",
                "import json\n",
                "\n",
                "pipeline_scripts = [\n",
                "    \"scripts/audit_dataset.py\",\n",
                "    \"scripts/prepare_dataset.py\",\n",
                "    \"scripts/run_numpy_mlp_t2.py\",\n",
                "    \"scripts/run_competing_models_t3.py\",\n",
                "]\n",
                "\n",
                "for script_rel in pipeline_scripts:\n",
                "    print(\"\\n\" + \"=\" * 78)\n",
                "    print(f\"RUNNING: python {script_rel}\")\n",
                "    print(\"=\" * 78, flush=True)\n",
                "    proc = subprocess.run([sys.executable, script_rel], cwd=str(PROJECT_ROOT), check=False)\n",
                "    if proc.returncode != 0:\n",
                "        raise RuntimeError(f\"{script_rel} exited with non-zero code {proc.returncode}\")\n",
                "\n",
                "print(\"\\n\" + \"=\" * 78)\n",
                "print(\"REAL-DATA PIPELINE EXECUTION SUMMARY (T1 - T3)\")\n",
                "print(\"=\" * 78)\n",
                "for metric_file in [\n",
                "    \"results/metrics/dataset_audit.json\",\n",
                "    \"results/metrics/split_summary.json\",\n",
                "    \"results/metrics/t2_real_data_training.json\",\n",
                "    \"results/metrics/t3_experiment_status.json\",\n",
                "    \"results/metrics/t3_rf_tuning_seed42.json\",\n",
                "    \"results/metrics/t3_deep_mlp_val_seed42.json\",\n",
                "    \"results/metrics/t3_cnn_2d_val_seed42.json\",\n",
                "]:\n",
                "    p = PROJECT_ROOT / metric_file\n",
                "    if p.is_file():\n",
                "        print(f\"[VERIFIED] {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size:,} bytes)\")\n",
                "        if p.name in (\"t2_real_data_training.json\", \"t3_experiment_status.json\"):\n",
                "            print(json.dumps(json.loads(p.read_text(encoding='utf-8')), indent=2))\n",
                "    else:\n",
                "        print(f\"[MISSING]  {metric_file}\")"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 6: Persist Generated Results/Checkpoints to Google Drive & Package Artifact ZIP for Local Sync\n",
                "import shutil\n",
                "\n",
                "artifact_dirs = [\n",
                "    \"checkpoints\",\n",
                "    \"results/metrics\",\n",
                "    \"results/figures\",\n",
                "    \"data/splits\",\n",
                "    \"data/processed\",\n",
                "    \"data/raw/DATASET_PROVENANCE.json\",\n",
                "]\n",
                "\n",
                "staging_dir = PROJECT_ROOT / \"_artifact_bundle\"\n",
                "if staging_dir.exists():\n",
                "    shutil.rmtree(staging_dir)\n",
                "\n",
                "for rel_item in artifact_dirs:\n",
                "    src_item = PROJECT_ROOT / rel_item\n",
                "    if src_item.is_dir():\n",
                "        shutil.copytree(src_item, staging_dir / rel_item, dirs_exist_ok=True)\n",
                "        if GDRIVE_ARTIFACTS_DIR is not None:\n",
                "            shutil.copytree(src_item, GDRIVE_ARTIFACTS_DIR / rel_item, dirs_exist_ok=True)\n",
                "    elif src_item.is_file():\n",
                "        dest_file = staging_dir / rel_item\n",
                "        dest_file.parent.mkdir(parents=True, exist_ok=True)\n",
                "        shutil.copy2(src_item, dest_file)\n",
                "        if GDRIVE_ARTIFACTS_DIR is not None:\n",
                "            gd_file = GDRIVE_ARTIFACTS_DIR / rel_item\n",
                "            gd_file.parent.mkdir(parents=True, exist_ok=True)\n",
                "            shutil.copy2(src_item, gd_file)\n",
                "\n",
                "zip_base = PROJECT_ROOT / \"ann_siren_t1_t2_t3_artifacts\"\n",
                "zip_file = shutil.make_archive(str(zip_base), \"zip\", root_dir=staging_dir)\n",
                "shutil.rmtree(staging_dir)\n",
                "print(f\"[OK] Packaged all real-data artifacts into: {zip_file}\")\n",
                "\n",
                "if GDRIVE_ARTIFACTS_DIR is not None:\n",
                "    gd_zip = GDRIVE_ARTIFACTS_DIR / Path(zip_file).name\n",
                "    shutil.copy2(zip_file, gd_zip)\n",
                "    print(f\"[OK] Saved persistent copy of artifact bundle to Google Drive: {gd_zip}\")"
            ],
        },
    ]

    notebook_obj = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path.write_text(json.dumps(notebook_obj, indent=1) + "\n", encoding="utf-8")
    print(
        f"Updated {notebook_path} with {len(bundled_files)} embedded files "
        f"(SHA256={bundle_sha256[:12]}..., base64_len={len(b64_bundle):,})."
    )
    return notebook_path


if __name__ == "__main__":
    generate_colab_notebook()

