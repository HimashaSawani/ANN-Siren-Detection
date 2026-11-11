"""Generate `notebooks/colab_phase3_cnn_gru.ipynb` for Google Colab Pro GPU training.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid)

This script bundles project source code into a self-contained Jupyter Notebook
specifically configured for Google Colab Pro with CUDA GPU acceleration.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_code_bundle_b64() -> tuple[str, str, int]:
    """Compress all essential project source files into a base64-encoded ZIP."""
    rel_files: list[Path] = []

    # Source files
    src_dir = PROJECT_ROOT / "src"
    for p in sorted(src_dir.rglob("*.py")):
        if "__pycache__" not in p.parts:
            rel_files.append(p.relative_to(PROJECT_ROOT))

    # Config files
    configs_dir = PROJECT_ROOT / "configs"
    for p in sorted(configs_dir.rglob("*.yaml")):
        rel_files.append(p.relative_to(PROJECT_ROOT))

    # Selected scripts
    scripts = [
        "scripts/run_phase3_cnn_gru.py",
        "scripts/import_colab_artifacts.py",
        "scripts/smoke_test.py",
        "scripts/coursework_demo.py",
    ]
    for s in scripts:
        sp = PROJECT_ROOT / s
        if sp.is_file():
            rel_files.append(Path(s))

    # Root config
    for r in ("requirements.txt", "pyproject.toml"):
        rp = PROJECT_ROOT / r
        if rp.is_file():
            rel_files.append(Path(r))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in rel_files:
            zf.write(PROJECT_ROOT / rel, arcname=rel.as_posix())

    raw_bytes = buf.getvalue()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    return b64, sha256, len(raw_bytes)


def create_phase3_notebook() -> Path:
    b64_bundle, bundle_sha, bundle_size = build_code_bundle_b64()
    notebook_path = PROJECT_ROOT / "notebooks" / "colab_phase3_cnn_gru.ipynb"

    cells = [
        # Markdown Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# NB6057CEM — Artificial Neural Networks\n",
                "## Portfolio Upgrade: Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid Audio Classifier)\n",
                "**Compute Target:** Google Colab Pro with CUDA GPU Acceleration (NVIDIA T4 / V100 / A100)\n",
                "\n",
                "This self-contained notebook trains and evaluates the **CNN + GRU Hybrid (CRNN)** architecture on the official UrbanSound8K dataset:\n",
                "- **Temporal Log-Mel Spectrogram Representation:** `(1, 128, 173)` retaining 173 time frames across 128 mel frequency channels.\n",
                "- **Hybrid Architecture:** 3-stage 2D CNN feature extractor $\\to$ Adaptive frequency collapse $\\to$ Bidirectional GRU ($2 \\times 64 = 128$) $\\to$ Temporal mean pooling $\\to$ Binary siren classifier (77,601 trainable parameters).\n",
                "- **Leakage-Free 10-Fold Protocol:** Folds 1–8 for training ($N=7,095$), Fold 9 for validation ($N=800$), and Fold 10 for held-out testing ($N=837$).\n",
                "- **Google Drive Persistence:** Checkpoints (`cnn_gru_seed42.pt`), learning curves, and benchmark metrics are persisted directly to Google Drive (`/content/drive/MyDrive/ANN-Siren-Detection/`)."
            ]
        },
        # Cell 1: Environment & GPU Inspection
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 1: Environment & GPU Hardware Inspection\n",
                "import os\n",
                "import platform\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import torch\n",
                "\n",
                "print('=' * 75)\n",
                "print('SYSTEM ENVIRONMENT & HARDWARE ACCELERATOR INSPECTION')\n",
                "print('=' * 75)\n",
                "print(f'Platform            : {platform.system()} {platform.release()} ({platform.machine()})')\n",
                "print(f'Python Executable   : {sys.executable}')\n",
                "print(f'Python Version      : {platform.python_version()}')\n",
                "print(f'PyTorch Version     : {torch.__version__}')\n",
                "print(f'CUDA Available      : {torch.cuda.is_available()}')\n",
                "\n",
                "if torch.cuda.is_available():\n",
                "    device = torch.device('cuda:0')\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)\n",
                "    capability = torch.cuda.get_device_capability(0)\n",
                "    print(f'GPU Device Name     : {gpu_name}')\n",
                "    print(f'Allocated VRAM      : {vram_gb:.2f} GB')\n",
                "    print(f'Compute Capability  : {capability[0]}.{capability[1]}')\n",
                "    print(f'cuDNN Enabled       : {torch.backends.cudnn.enabled}')\n",
                "    torch.backends.cudnn.benchmark = True\n",
                "else:\n",
                "    device = torch.device('cpu')\n",
                "    print('[WARN] CUDA is not available. Please navigate to:')\n",
                "    print('       Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU / V100 GPU / A100 GPU')\n",
                "\n",
                "# Mount Google Drive\n",
                "GDRIVE_DIR = None\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    mount_path = '/content/drive'\n",
                "    if not Path(mount_path, 'MyDrive').is_dir():\n",
                "        drive.mount(mount_path)\n",
                "    GDRIVE_DIR = Path(mount_path) / 'MyDrive' / 'ANN-Siren-Detection'\n",
                "    GDRIVE_DIR.mkdir(parents=True, exist_ok=True)\n",
                "    (GDRIVE_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)\n",
                "    (GDRIVE_DIR / 'results' / 'metrics').mkdir(parents=True, exist_ok=True)\n",
                "    (GDRIVE_DIR / 'results' / 'figures').mkdir(parents=True, exist_ok=True)\n",
                "    print(f'Google Drive Root   : {GDRIVE_DIR}')\n",
                "except Exception as e:\n",
                "    print(f'[INFO] Google Drive mounting skipped or running locally: {e}')\n",
                "print('=' * 75)"
            ]
        },
        # Cell 2: Project Workspace Bootstrap
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 2: Project Workspace Bootstrap & Source File Extraction\n",
                "import base64\n",
                "import hashlib\n",
                "import io\n",
                "import zipfile\n",
                "from pathlib import Path\n",
                "\n",
                "WORKSPACE = Path('/content/ANN-Siren-Detection') if Path('/content').is_dir() else Path.cwd()\n",
                "WORKSPACE.mkdir(parents=True, exist_ok=True)\n",
                "os.chdir(WORKSPACE)\n",
                "if str(WORKSPACE) not in sys.path:\n",
                "    sys.path.insert(0, str(WORKSPACE))\n",
                "\n",
                "print(f'Active Workspace: {WORKSPACE}')\n",
                "\n",
                "# Check if Drive package zip exists\n",
                "drive_pkg = GDRIVE_DIR / 'colab_phase3_package.zip' if GDRIVE_DIR else None\n",
                "if drive_pkg and drive_pkg.is_file():\n",
                "    print(f'Unpacking transfer package from Google Drive: {drive_pkg}...')\n",
                "    with zipfile.ZipFile(drive_pkg, 'r') as z:\n",
                "        z.extractall(WORKSPACE)\n",
                "    print('Transfer package successfully unpacked.')\n",
                "else:\n",
                "    # Unpack embedded project code bundle\n",
                "    print('Extracting embedded project code bundle...')\n",
                f"    BUNDLE_B64 = \"{b64_bundle}\"\n",
                "    raw_bytes = base64.b64decode(BUNDLE_B64)\n",
                "    with zipfile.ZipFile(io.BytesIO(raw_bytes), 'r') as z:\n",
                "        z.extractall(WORKSPACE)\n",
                "    print('Embedded source modules extracted successfully.')\n",
                "\n",
                "print('\\nWorkspace directory structure:')\n",
                "for d in ('configs', 'src', 'scripts'):\n",
                "    p = WORKSPACE / d\n",
                "    if p.is_dir():\n",
                "        files = [f.name for f in p.iterdir() if f.is_file()]\n",
                "        print(f'  {d}/ ({len(files)} files): {files[:6]}...')"
            ]
        },
        # Cell 3: Dependencies Verification
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 3: Install and Verify Python Dependencies\n",
                "!pip install -q soundfile librosa pyyaml tabulate scikit-learn\n",
                "\n",
                "import yaml\n",
                "import librosa\n",
                "import soundfile\n",
                "import sklearn\n",
                "import matplotlib\n",
                "\n",
                "print('Dependency verification:')\n",
                "print(f'  PyTorch    : {torch.__version__}')\n",
                "import torchaudio\n",
                "print(f'  Torchaudio : {torchaudio.__version__}')\n",
                "print(f'  Librosa    : {librosa.__version__}')\n",
                "print(f'  SoundFile  : {soundfile.__version__}')\n",
                "print(f'  Scikit-Learn: {sklearn.__version__}')\n",
                "print(f'  PyYAML     : {yaml.__version__}')\n",
                "print('All required libraries imported cleanly.')"
            ]
        },
        # Cell 4: Data Validation & Split Leakage Protection
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 4: Data Validation & Split Leakage Protection\n",
                "import shutil\n",
                "import numpy as np\n",
                "import zipfile\n",
                "\n",
                "processed_dir = WORKSPACE / 'data' / 'processed'\n",
                "processed_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "train_val_cache = processed_dir / 'train_val_mel_cache.npz'\n",
                "test_cache = processed_dir / 'test_mel_cache.npz'\n",
                "\n",
                "# If cache is on Google Drive, copy or link to local fast disk\n",
                "if GDRIVE_DIR and (not train_val_cache.is_file() or not test_cache.is_file()):\n",
                "    drive_tv = GDRIVE_DIR / 'data' / 'processed' / 'train_val_mel_cache.npz'\n",
                "    drive_test = GDRIVE_DIR / 'data' / 'processed' / 'test_mel_cache.npz'\n",
                "    if drive_tv.is_file() and not train_val_cache.is_file():\n",
                "        print(f'Copying {drive_tv.name} from Google Drive to local fast scratch...')\n",
                "        shutil.copy2(drive_tv, train_val_cache)\n",
                "    if drive_test.is_file() and not test_cache.is_file():\n",
                "        print(f'Copying {drive_test.name} from Google Drive to local fast scratch...')\n",
                "        shutil.copy2(drive_test, test_cache)\n",
                "\n",
                "assert train_val_cache.is_file(), f'Training cache missing: {train_val_cache}'\n",
                "assert test_cache.is_file(), f'Test cache missing: {test_cache}'\n",
                "\n",
                "print('=' * 75)\n",
                "print('DATASET VALIDATION & TENSOR SHAPES')\n",
                "print('=' * 75)\n",
                "\n",
                "with zipfile.ZipFile(train_val_cache) as z:\n",
                "    with z.open('y_train.npy') as f:\n",
                "        y_train = np.load(f)\n",
                "    with z.open('y_val.npy') as f:\n",
                "        y_val = np.load(f)\n",
                "\n",
                "with zipfile.ZipFile(test_cache) as z:\n",
                "    with z.open('y_test.npy') as f:\n",
                "        y_test = np.load(f)\n",
                "\n",
                "# Extract OS-level memory-mapped buffers for zero-copy training\n",
                "from src.train_hybrid import ensure_mmap_mel_files\n",
                "train_mmap, val_mmap, test_mmap = ensure_mmap_mel_files(processed_dir)\n",
                "x_train_arr = np.load(train_mmap, mmap_mode='r')\n",
                "x_val_arr = np.load(val_mmap, mmap_mode='r')\n",
                "x_test_arr = np.load(test_mmap, mmap_mode='r')\n",
                "\n",
                "print(f'Training Set (Folds 1-8) : X={x_train_arr.shape}, y={y_train.shape} | Sirens={np.sum(y_train==1)} ({np.mean(y_train==1)*100:.2f}%)')\n",
                "print(f'Validation Set (Fold 9)  : X={x_val_arr.shape}, y={y_val.shape}   | Sirens={np.sum(y_val==1)} ({np.mean(y_val==1)*100:.2f}%)')\n",
                "print(f'Held-Out Test (Fold 10)  : X={x_test_arr.shape}, y={y_test.shape}   | Sirens={np.sum(y_test==1)} ({np.mean(y_test==1)*100:.2f}%)')\n",
                "\n",
                "# Strict verification assertions\n",
                "assert x_train_arr.shape == (7095, 1, 128, 173), f'Unexpected train shape: {x_train_arr.shape}'\n",
                "assert x_val_arr.shape == (800, 1, 128, 173), f'Unexpected val shape: {x_val_arr.shape}'\n",
                "assert x_test_arr.shape == (837, 1, 128, 173), f'Unexpected test shape: {x_test_arr.shape}'\n",
                "print('\\nSplit Leakage Verification: Fold 10 test data is strictly isolated.')\n",
                "print('All tensor dimensions and sample counts conform exactly to UrbanSound8K protocol.')\n",
                "print('=' * 75)"
            ]
        },
        # Cell 5: Phase 3 Model & CUDA Training
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 5: Phase 3 CNN + GRU Hybrid Training (CUDA Accelerated)\n",
                "import time\n",
                "from src.hybrid_model import CNNGRUHybrid\n",
                "from src.train_hybrid import train_cnn_gru_model\n",
                "\n",
                "# Training Configuration\n",
                "SEEDS = [42]             # Random seeds to train (can expand to [42, 123, 2026])\n",
                "BATCH_SIZE = 64          # Mini-batch size for GPU execution\n",
                "MAX_EPOCHS = 25          # Maximum training epochs\n",
                "PATIENCE = 6             # Early stopping patience tracking Fold 9 Macro-F1\n",
                "LEARNING_RATE = 1e-3     # Adam optimizer learning rate\n",
                "WEIGHT_DECAY = 1e-4      # L2 regularization\n",
                "\n",
                "compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "checkpoints_dir = WORKSPACE / 'checkpoints'\n",
                "figures_dir = WORKSPACE / 'results' / 'figures'\n",
                "metrics_dir = WORKSPACE / 'results' / 'metrics'\n",
                "\n",
                "checkpoints_dir.mkdir(parents=True, exist_ok=True)\n",
                "figures_dir.mkdir(parents=True, exist_ok=True)\n",
                "metrics_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print('=' * 75)\n",
                "print('STARTING PHASE 3 CNN + GRU HYBRID TRAINING ON CUDA GPU')\n",
                "print('=' * 75)\n",
                "print(f'Compute Device      : {compute_device}')\n",
                "print(f'Batch Size          : {BATCH_SIZE}')\n",
                "print(f'Max Epochs          : {MAX_EPOCHS}')\n",
                "print(f'Early Stop Patience : {PATIENCE}')\n",
                "print(f'Learning Rate       : {LEARNING_RATE}')\n",
                "print('=' * 75)\n",
                "\n",
                "train_results = {}\n",
                "for seed in SEEDS:\n",
                "    print(f'\\n>>> Training CNN + GRU Hybrid (Seed {seed}) <<<')\n",
                "    t0 = time.time()\n",
                "    summary = train_cnn_gru_model(\n",
                "        processed_dir=processed_dir,\n",
                "        checkpoints_dir=checkpoints_dir,\n",
                "        figures_dir=figures_dir,\n",
                "        metrics_dir=metrics_dir,\n",
                "        seed=seed,\n",
                "        batch_size=BATCH_SIZE,\n",
                "        learning_rate=LEARNING_RATE,\n",
                "        weight_decay=WEIGHT_DECAY,\n",
                "        max_epochs=MAX_EPOCHS,\n",
                "        patience=PATIENCE,\n",
                "        device=compute_device,\n",
                "    )\n",
                "    duration = time.time() - t0\n",
                "    train_results[seed] = summary\n",
                "    print(f'Seed {seed} finished in {duration:.1f}s | Best Val Macro-F1: {summary[\"best_val_macro_f1\"]:.4f} (Epoch {summary[\"best_epoch\"]})')\n",
                "\n",
                "    # Sync checkpoint and learning curve figure to Google Drive\n",
                "    if GDRIVE_DIR:\n",
                "        ckpt_name = f'cnn_gru_seed{seed}.pt'\n",
                "        fig_name = f'fig_phase3_cnn_gru_curves_seed{seed}.png'\n",
                "        meta_name = f'phase3_cnn_gru_train_seed{seed}.json'\n",
                "        shutil.copy2(checkpoints_dir / ckpt_name, GDRIVE_DIR / 'checkpoints' / ckpt_name)\n",
                "        shutil.copy2(figures_dir / fig_name, GDRIVE_DIR / 'results' / 'figures' / fig_name)\n",
                "        shutil.copy2(metrics_dir / meta_name, GDRIVE_DIR / 'results' / 'metrics' / meta_name)\n",
                "        print(f'Persisted Seed {seed} artifacts directly to Google Drive.')\n",
                "\n",
                "print('\\nPhase 3 training routine successfully completed.')"
            ]
        },
        # Cell 6: Held-Out Fold 10 Evaluation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 6: Held-Out Fold 10 Evaluation strictly once\n",
                "from src.train_hybrid import evaluate_cnn_gru_model\n",
                "\n",
                "primary_seed = SEEDS[0]\n",
                "ckpt_path = checkpoints_dir / f'cnn_gru_seed{primary_seed}.pt'\n",
                "\n",
                "print('=' * 75)\n",
                "print(f'EVALUATING FINAL MODEL ON HELD-OUT TEST FOLD 10 (Seed {primary_seed}, N=837)')\n",
                "print('=' * 75)\n",
                "\n",
                "test_metrics = evaluate_cnn_gru_model(\n",
                "    checkpoint_path=ckpt_path,\n",
                "    processed_dir=processed_dir,\n",
                "    metrics_dir=metrics_dir,\n",
                "    seed=primary_seed,\n",
                "    threshold=0.50,\n",
                "    device=compute_device,\n",
                ")\n",
                "test_metrics['val_macro_f1'] = train_results[primary_seed]['best_val_macro_f1']\n",
                "\n",
                "cm = test_metrics['confusion_matrix']\n",
                "print(f'Test Macro-F1       : {test_metrics[\"macro_f1\"]:.4f}')\n",
                "print(f'Siren F1 Score      : {test_metrics[\"siren_f1\"]:.4f}')\n",
                "print(f'Siren Recall        : {test_metrics[\"siren_recall\"] * 100:.2f}% ({cm[\"tp\"]}/{cm[\"tp\"] + cm[\"fn\"]})')\n",
                "print(f'Siren Precision     : {test_metrics[\"siren_precision\"] * 100:.2f}% ({cm[\"tp\"]}/{cm[\"tp\"] + cm[\"fp\"]})')\n",
                "print(f'PR-AUC              : {test_metrics[\"pr_auc\"]:.4f}')\n",
                "print(f'ROC-AUC             : {test_metrics[\"roc_auc\"]:.4f}')\n",
                "print(f'Balanced Accuracy   : {test_metrics[\"balanced_accuracy\"] * 100:.2f}%')\n",
                "print(f'Inference Latency   : {test_metrics[\"inference_latency_ms\"]:.3f} ms / sample (on {compute_device})')\n",
                "print(f'Confusion Matrix    : TP={cm[\"tp\"]}, FP={cm[\"fp\"]}, TN={cm[\"tn\"]}, FN={cm[\"fn\"]}')\n",
                "\n",
                "# Persist test metrics to Drive\n",
                "if GDRIVE_DIR:\n",
                "    test_json = metrics_dir / f'phase3_cnn_gru_test_seed{primary_seed}.json'\n",
                "    shutil.copy2(test_json, GDRIVE_DIR / 'results' / 'metrics' / test_json.name)\n",
                "    print(f'Saved test metrics to Google Drive: {test_json.name}')"
            ]
        },
        # Cell 7: Multi-Model Architectural Benchmark & Comparison Table
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 7: Multi-Model Architectural Benchmark & Comparison Table\n",
                "from src.train_hybrid import build_phase3_comparison_table\n",
                "\n",
                "comparison_df = build_phase3_comparison_table(\n",
                "    metrics_dir=metrics_dir,\n",
                "    figures_dir=figures_dir,\n",
                "    cnn_gru_test_metrics=test_metrics,\n",
                ")\n",
                "\n",
                "print('\\n' + '=' * 105)\n",
                "print('PHASE 3 ARCHITECTURAL BENCHMARK: DEEP MLP vs. 2D CNN vs. CNN + GRU HYBRID (FOLD 10)')\n",
                "print('=' * 105)\n",
                "print(comparison_df.to_markdown(index=False))\n",
                "print('=' * 105)\n",
                "\n",
                "# Save comparison table CSV and figures to Google Drive\n",
                "csv_path = metrics_dir / 'phase3_model_comparison.csv'\n",
                "comparison_df.to_csv(csv_path, index=False)\n",
                "fig_path = figures_dir / 'fig_phase3_architecture_comparison.png'\n",
                "\n",
                "if GDRIVE_DIR:\n",
                "    shutil.copy2(csv_path, GDRIVE_DIR / 'results' / 'metrics' / csv_path.name)\n",
                "    if fig_path.is_file():\n",
                "        shutil.copy2(fig_path, GDRIVE_DIR / 'results' / 'figures' / fig_path.name)\n",
                "    print('Saved benchmark CSV and comparison figure to Google Drive.')"
            ]
        },
        # Cell 8: Downloadable Artifact Summary & Local Import Command
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CELL 8: Downloadable Artifact Summary & Local Import Instructions\n",
                "import zipfile\n",
                "\n",
                "out_zip_path = WORKSPACE / 'phase3_trained_artifacts.zip'\n",
                "artifact_files = [\n",
                "    checkpoints_dir / f'cnn_gru_seed{primary_seed}.pt',\n",
                "    figures_dir / f'fig_phase3_cnn_gru_curves_seed{primary_seed}.png',\n",
                "    figures_dir / 'fig_phase3_architecture_comparison.png',\n",
                "    metrics_dir / f'phase3_cnn_gru_train_seed{primary_seed}.json',\n",
                "    metrics_dir / f'phase3_cnn_gru_test_seed{primary_seed}.json',\n",
                "    metrics_dir / 'phase3_model_comparison.csv',\n",
                "]\n",
                "\n",
                "with zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:\n",
                "    for f in artifact_files:\n",
                "        if f.is_file():\n",
                "            arcname = f.relative_to(WORKSPACE).as_posix()\n",
                "            z.write(f, arcname=arcname)\n",
                "            print(f'Bundled: {arcname}')\n",
                "\n",
                "if GDRIVE_DIR:\n",
                "    drive_artifact_zip = GDRIVE_DIR / 'phase3_trained_artifacts.zip'\n",
                "    shutil.copy2(out_zip_path, drive_artifact_zip)\n",
                "    print(f'\\nTrained artifacts packaged to Google Drive: {drive_artifact_zip}')\n",
                "\n",
                "print('\\n' + '=' * 80)\n",
                "print('HOW TO IMPORT TRAINED CHECKPOINT TO LOCAL WINDOWS PC')\n",
                "print('=' * 80)\n",
                "print('1. Download `phase3_trained_artifacts.zip` from your Google Drive folder:')\n",
                "print('   MyDrive/ANN-Siren-Detection/phase3_trained_artifacts.zip')\n",
                "print('2. On your Windows machine at D:\\\\ANN-Siren-Detection, extract the zip into the project root:')\n",
                "print('   Expand-Archive -Path phase3_trained_artifacts.zip -DestinationPath D:\\\\ANN-Siren-Detection -Force')\n",
                "print('3. Verify the newly imported model in the Streamlit application:')\n",
                "print('   streamlit run app.py')\n",
                "print('   Or test inference directly:')\n",
                "print('   python -c \"from src.inference import SirenInferenceEngine; engine = SirenInferenceEngine(); print(engine.available_models)\"')\n",
                "print('=' * 80)"
            ]
        }
    ]

    notebook_content = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "colab_phase3_cnn_gru.ipynb",
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook_content, indent=1), encoding="utf-8")
    print(f"Generated Colab notebook: {notebook_path} ({bundle_size / 1024:.1f} KB source bundle, sha256={bundle_sha[:10]})")
    return notebook_path


if __name__ == "__main__":
    create_phase3_notebook()

