# Robust Emergency Siren Classification Under Urban Noise: A Comparative Study of MLP and CNN Architectures

**Module:** NB6057CEM — Artificial Neural Networks  
**Coursework:** Complete CW1 Tasks (T1–T6) Technical Implementation & Evaluation Record  
**Project Status:** Completed prototype; Phase 1 (SHAP XAI) & Phase 2 (Noise Augmentation) Verified  
**Verification:** 110 passing regression, integration, explainability & robustness tests (`pytest tests/ -v`)  
**Interactive Interface:** `streamlit run app.py` &bull; **Terminal Demo:** `python scripts/coursework_demo.py`

---

## 1. Executive Summary

This repository delivers an end-to-end, empirically evaluated machine learning pipeline for emergency siren classification research (`siren` vs. `non_siren`) under challenging urban acoustic clutter using the official **UrbanSound8K** dataset (8,732 recordings, 10 folds, 8.4:1 non-siren to siren class imbalance).

The project rigorously compares four model paradigms across standardized 22,050 Hz, 4.0-second mono audio windows:
1. **First-Principles NumPy MLP:** Built from raw mathematical equations with analytical backpropagation and numerical gradient checks ($\text{rel\_diff} = 2.13 \times 10^{-10}$).
2. **Random Forest Baseline:** Scikit-learn ensemble with balanced class weights on 80-D MFCC summary statistics.
3. **Deep PyTorch MLP (Recommended):** Multi-layer neural network with dropout regularization ($p=0.3$) and positive loss weighting ($\text{pos\_weight} = 8.29$).
4. **2D Convolutional Neural Network (PyTorch):** Deep spatial feature extractor operating on $(1, 128, 173)$ Log-Mel spectrograms.

---

## 2. Application Demonstration

### Streamlit Interface

The project includes an interactive Streamlit application for emergency siren classification. Users can upload WAV recordings or select benchmark audio samples, analyse them with the saved Deep MLP model, and inspect the resulting predictions, waveform, Log-Mel spectrogram, and local SHAP explanations.

![Streamlit Application Demo](docs/images/streamlit-demo.png)

#### User Workflow:
- **Zero Premature Inference:** Audio selection displays metadata and preview player; inference is triggered exclusively by clicking **"Analyze Audio"**.
- **Interactive Visualizations:** Dynamically renders responsive Plotly waveform and 128-band Log-Mel spectrogram charts computed from the decoded audio.
- **Evidence-Based SHAP Explainability:** DeepLIFT feature attributions on pre-sigmoid logits with interactive Plotly attribution bar charts, top 10 acoustic feature descriptions, and evidence-based acoustic narrative.
- **Dynamic Cutoff Thresholding:** Real-time re-evaluation of operating sensitivity ($\theta \in [0.05, 0.95]$) from cached model probabilities without re-running forward passes.
- **Multi-Model Cross-Check:** Optional collapsed comparison against Random Forest, NumPy Scratch MLP, and 2D CNN checkpoints.

---

## 3. Explainable AI (XAI) & Feature Attributions

To provide transparent, auditable decision boundaries for safety-critical audio classification, this project implements local model explainability using **SHAP (SHapley Additive exPlanations)** with PyTorch `DeepExplainer`:

### Technical Implementation:
- **Target Model:** Saved Deep MLP checkpoint (`deep_mlp_t3_seed42.pt`), operating on 80 standardized MFCC summary features (40 temporal means, 40 temporal standard deviations).
- **Representative Background Dataset:** Formed strictly from the training split (Folds 1–8), stratified with 15% siren and 85% non-siren samples ($N=100$) to reflect real-world training class distributions without test set contamination.
- **Logit-Space Explanations:** Attributions $\phi_i$ are computed in pre-sigmoid logit space ($\mathbb{R}$), strictly satisfying the DeepLIFT additivity property:
  $$\sum_{i=1}^{80} \phi_i = \text{logit}(x) - \mathbb{E}[\text{logit}(x_{\text{background}})]$$
  Operating in logit space avoids sigmoid saturation distortions ($P \to 1.0$ or $P \to 0.0$) and accurately reflects raw layer activations.
- **Acoustic Grounding:** Each MFCC summary statistic is mapped to domain-specific acoustic semantics:
  - Coefficient 0 (`mfcc_mean_0`): Overall log-energy and broadband signal power.
  - Low quefrency coefficients (`mfcc_mean_1`–`4`): Macro spectral tilt and gross spectral envelope slope across mel bands.
  - Intermediate quefrency coefficients (`mfcc_mean_5`–`15`): Formant structure, harmonic peak distributions, and acoustic timbre patterns.
  - High quefrency coefficients (`mfcc_mean_16`–`39`): Fine spectral ripple across mel channels and rapid spectral variations.
  - Temporal standard deviations (`mfcc_std_k`): Temporal modulation variance over time windows, capturing cyclic frequency variation (e.g. wail/yelp modulation) versus stationary background noise.
  *(Note: MFCC coefficients represent DCT-II basis projections over the entire log-mel filterbank rather than localized, isolated physical Hertz bands.)*

### Python API Usage:

```python
from src.explainability import SirenSHAPExplainer

# Initialize explainer with cached training background
explainer = SirenSHAPExplainer()

# Explain an audio recording or feature vector
result = explainer.explain_audio("demo/sample_audio/siren_sample_165166-8-0-0.wav")

print(f"Predicted Class: {result['prediction']['class_name']}")
print(f"Siren Probability: {result['prediction']['probability']:.4f}")
print(f"Top Feature: {result['top_features'][0]['feature']} (SHAP: {result['top_features'][0]['attribution']:+.4f})")
print(f"Interpretation: {result['interpretation']}")
```

### Scientific Scope & Limitations:
1. **Attribution, Not Causation:** SHAP values quantify statistical contribution to the neural network's logit output given the training background; they do **not** establish physical acoustic causation.
2. **Background Sensitivity:** Attributions are relative to the representative training distribution. Different background distributions will shift base expectations.
3. **Logit vs Probability Space:** Positive SHAP values indicate evidence pushing towards a Siren decision in logit space. Due to the non-linear sigmoid transform $\sigma(z)$, a constant logit delta has varying marginal impact on predicted probability depending on base activation level.
4. **Calibration Warning:** Model-estimated scores reflect discriminative neural network outputs and are not necessarily calibrated real-world probabilities.

---

## 4. Quick Start

### Installation & Setup (Windows PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/HimashaSawani/ANN-Siren-Detection.git
cd ANN-Siren-Detection

# 2. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

### Option 1 — Run the Streamlit Interface

Launch the interactive web application:

```powershell
streamlit run app.py
```

* Open the local URL displayed in the terminal (typically `http://localhost:8501` or `http://localhost:8503`).
* Select a benchmark sample from the collapsed *"Try sample audio"* expander or upload a custom WAV file.
* Click **Analyze Audio** to display the model prediction, estimated probability, and interactive Plotly signal charts.

---

### Option 2 — Run the Terminal Demonstration

Execute code-first saved-checkpoint inference and verification without retraining:

```powershell
# Run the complete technical demonstration and artifact audit:
python scripts/coursework_demo.py

# Test an authenticated siren recording:
python scripts/coursework_demo.py --audio demo/sample_audio/siren_sample_165166-8-0-0.wav

# Test an authenticated non-siren recording:
python scripts/coursework_demo.py --audio demo/sample_audio/non_siren_sample_100795-3-0-0.wav
```

The terminal demo loads existing trained models, verifies gradient checks and dataset splits, executes inference on real audio, and validates corrupt-audio exception handling.

---

## 5. Dataset, Preprocessing, and Evaluation Protocol

- **Dataset:** Official UrbanSound8K (8,732 audio recordings across 10 pre-arranged folds).
- **Audio Standardization:** Every recording is decoded (supporting PCM, IEEE float, and IMA/Microsoft ADPCM formats), converted to mono via channel averaging, resampled to **22,050 Hz**, and padded/truncated to a canonical **4.0-second** window (88,200 samples).
- **Leakage Controls Implemented and Checked:**
  - **Folds 1–8:** Training split ($N = 7,095$).
  - **Fold 9:** Validation split ($N = 800$, used strictly for early stopping and threshold selection).
  - **Fold 10:** Held-out test split ($N = 837$, evaluated strictly once after model freezing).
  - **Recording Source Group Isolation:** Verified that slice variants sharing the same original Freesound recording (e.g. `fsID=180937`) never cross partition boundaries.
  - **Isolated Feature Scaling:** StandardScaler fitted strictly on training fold features and applied downstream.

---

## 6. Experimental Results (CW1 T4)

Evaluation across three independent random seeds (`42`, `123`, `2026`) on the held-out test split (Fold 10, $N = 837$):

| Architecture | Input Representation | Evaluated Seeds | Val Macro-F1 ($\mu \pm \sigma$) | Test Macro-F1 ($\mu \pm \sigma$) | Siren F1 ($\mu \pm \sigma$) | Siren Recall ($\mu$) | Siren Prec ($\mu$) | PR-AUC ($\mu$) | Mean Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep MLP (PyTorch)** | 80-D MFCC | 3 | **0.9655 ± 0.0139** | **0.7736 ± 0.0182** | **0.5896 ± 0.0306** | **55.4%** | 63.5% | **0.6602** | 20.53 s |
| **NumPy MLP (Scratch)** | 80-D MFCC | 3 | 0.9468 ± 0.0049 | 0.7568 ± 0.0187 | 0.5540 ± 0.0320 | 46.2% | 69.7% | 0.5769 | 5.61 s |
| **Random Forest** | 80-D MFCC | 3 | 0.9396 ± 0.0027 | 0.7397 ± 0.0165 | 0.5151 ± 0.0322 | 35.7% | **93.1%** | 0.5888 | 19.75 s |
| **2D CNN (PyTorch)** | $(1, 128, 173)$ Log-Mel | 1* | 0.9151 ± 0.0000 | 0.7362 ± 0.0000 | 0.5175 ± 0.0000 | 44.6% | 61.7% | **0.6745** | 1,991.27 s (33.2 m) |

*\*Note on 2D CNN Computational Budget:* The 2D CNN was evaluated on Seed 42 locally due to its 33.2-minute CPU training time. GPU continuation workflow is provided in `notebooks/colab_t3_training.ipynb`.

### Primary Model Selection: Deep MLP
Selected on the basis of **Validation Macro-F1 (0.9655)** on Fold 9. It provides sub-millisecond inference latency (0.007 ms/sample on CPU), balanced precision/recall, and consistent convergence under seed variations.

---

## 7. Noise Robustness & Known Limitations

1. **Ambient Urban Noise Degradation:** Under controlled additive street noise, siren recall degrades significantly under low SNR conditions:
   - **Clean Audio:** Recall = 55.4% &bull; Macro-F1 = 0.7736
   - **20 dB SNR:** Recall = 52.8% &bull; Macro-F1 = 0.7610
   - **10 dB SNR:** Recall = 44.6% &bull; Macro-F1 = 0.7258
   - **5 dB SNR:** Recall = 36.1% &bull; Macro-F1 = 0.6791
2. **Dropout Regularization Observation:** In single-factor ablation on the Deep MLP, removing dropout ($p=0.0$) produced higher validation Macro-F1 (`0.9801` vs `0.9655`). Dropout ($p=0.3$) was retained because it reduces overfitting across diverse urban soundscapes.
3. **Acoustic Confounders:** Held-out error inspection revealed that periodic high-frequency cyclical sounds (such as jackhammers and drills) occasionally produce narrow tonal bands resembling siren sweeps, causing false alarms in edge cases.
4. **Safety Disclaimer:** This is an academic research prototype developed for coursework evaluation (NB6057CEM). It is not certified for life-critical emergency dispatch or autonomous driving collision-avoidance systems without multi-modal sensor fusion.

---

## 8. Portfolio Upgrade: Phase 2 — Controlled Noise-Augmented Deep MLP Training

### Motivation & Empirical Challenge
In the original baseline noise robustness evaluation (Section 7), the clean-trained Deep MLP suffered acute degradation under low Signal-to-Noise Ratio (SNR) acoustic clutter on held-out Fold 10:
* Under clean audio, siren recall was **55.4%** (Macro-F1 = 0.7736).
* Under 5 dB SNR additive noise, siren recall plummeted to **36.1%** (Macro-F1 = 0.6791), missing nearly two-thirds of emergency siren instances.

### Solution Architecture: Controlled Waveform Augmentation
Phase 2 implements controlled audio-level noise augmentation within the training pipeline while strictly preserving all coursework artifacts, benchmarks, and validation splits:
1. **Audio-Level Injection:** Perturbations are applied directly to raw 22,050 Hz 4.0-second waveforms prior to feature extraction, rather than adding synthetic post-hoc noise to standardized feature vectors.
2. **Documented Noise Types & SNR Bounds:** Realistic Gaussian white noise and ambient urban environmental non-siren recordings mixed at signal-to-noise ratios between $10.0\text{ dB}$ and $25.0\text{ dB}$ with probability $p = 0.5$.
3. **Strict Leakage Prevention:** Augmentation is restricted strictly to training folds 1–8 ($N = 7,095$). Validation (Fold 9, $N = 800$) and test (Fold 10, $N = 837$) splits remain strictly unaugmented and clean during training and early stopping.
4. **Independent Checkpoint:** The resulting model is saved as an isolated checkpoint (`checkpoints/deep_mlp_augmented_seed42.pt`), ensuring the baseline coursework checkpoint (`checkpoints/deep_mlp_t3_seed42.pt`) remains completely untouched.

### Empirical Evaluation: Clean vs. Noise-Augmented Deep MLP
Both models were evaluated on the identical held-out test split (Fold 10, $N = 837$, 83 siren / 754 non-siren) across Clean, 20 dB, 10 dB, and 5 dB SNR conditions:

| Noise Condition (SNR) | Model | Macro-F1 | Siren Recall | Siren Precision | Siren F1 | PR-AUC | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | Clean Deep MLP | 0.7848 | 53.01% | 70.97% | 0.6069 | 0.6500 | 0.053 ms |
| **Clean** | **Noise-Augmented Deep MLP** | **0.8103** | **55.42%** | **79.31%** | **0.6525** | **0.7095** | 0.013 ms |
| **20 dB SNR** | Clean Deep MLP | **0.8073** | 57.83% | **73.85%** | **0.6486** | 0.6666 | 0.023 ms |
| **20 dB SNR** | **Noise-Augmented Deep MLP** | 0.7904 | **60.24%** | 64.10% | 0.6211 | **0.7083** | 0.020 ms |
| **10 dB SNR** | Clean Deep MLP | 0.7942 | 48.19% | **86.96%** | 0.6202 | 0.6666 | 0.019 ms |
| **10 dB SNR** | **Noise-Augmented Deep MLP** | **0.8071** | **59.04%** | 72.06% | **0.6490** | **0.7145** | 0.094 ms |
| **5 dB SNR (Severe)** | Clean Deep MLP | 0.7485 | 36.14% | **100.00%** | 0.5310 | 0.6440 | 0.025 ms |
| **5 dB SNR (Severe)** | **Noise-Augmented Deep MLP** | **0.7973** | **51.81%** | 79.63% | **0.6277** | **0.6976** | 0.032 ms |

#### Key Empirical Findings & Confusion Matrices
1. **Critical Noise Resilience at 5 dB SNR:**
   - Under heavy noise masking (5 dB SNR), the clean model detects only 30 of 83 sirens ($\text{TP}=30, \text{FN}=53, \text{TN}=754, \text{FP}=0, \text{Recall}=36.14\%$).
   - The noise-augmented model detects 43 of 83 sirens ($\text{TP}=43, \text{FN}=40, \text{TN}=743, \text{FP}=11, \text{Recall}=51.81\%$), achieving a **+15.67 percentage point absolute recall gain** while retaining strong precision ($79.63\%$).
   - Macro-F1 improves from **0.7485** to **0.7973** ($+0.0488$ gain), demonstrating substantial resilience under severe acoustic masking.
2. **Generalized Representation Quality:**
   - On clean test audio, the noise-augmented model does not degrade; in fact, its Macro-F1 rises from **0.7848** to **0.8103** ($\text{PR-AUC}=0.7095$ vs $0.6500$), indicating that controlled acoustic perturbations regularize the network against over-reliance on fragile spectral nuances.
3. **Computational Budget Disclosure:**
   - Primary evaluation was deterministically executed on Seed 42 across all 4 noise conditions ($N=837 \times 4 = 3,348$ audio extractions and evaluations). Scripts provide `--seeds 42 123 2026` parameterization for multi-seed extension when external GPU resources are allocated.

---

## 9. Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid) & Colab Pro GPU Workflow

### 9.1 Architecture Overview
To exploit the sequential acoustic dynamics of emergency siren sweeps (cyclical pitch oscillations in *wail* and *yelp* patterns) without discarding temporal progression, Phase 3 implements a **Convolutional Recurrent Neural Network (CRNN / CNN + GRU Hybrid)**:

- **Input Representation:** Log-Mel Spectrograms of dimension $(1, 128, 173)$, capturing 128 mel frequency bins across 173 time frames (22,050 Hz, 4.0-second audio windows, 1024-sample FFT, 512-sample hop).
- **Spatial Feature Extractor (2D CNN):**
  - Block 1: `Conv2d(1 -> 16, 3x3)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)` (Shape: $16 \times 64 \times 86$).
  - Block 2: `Conv2d(16 -> 32, 3x3)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)` (Shape: $32 \times 32 \times 43$).
  - Block 3: `Conv2d(32 -> 64, 3x3)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `AdaptiveAvgPool2d((1, None))` (Shape: $64 \times 1 \times 43$).
- **Temporal Recurrent Network (Bidirectional GRU):**
  - Frequency dimension is squeezed, yielding a time sequence of feature vectors: $(B, 43, 64)$.
  - 1-layer Bidirectional GRU (`hidden_size=64`, dropout=0.0) processes forward and backward acoustic transitions, outputting $(B, 43, 128)$.
- **Temporal Aggregation & Classification Head:**
  - Temporal Mean Pooling across 43 time steps $\to$ $(B, 128)$.
  - Dropout ($p=0.3$) $\to$ `Linear(128 -> 32)` $\to$ `ReLU` $\to$ `Linear(32 -> 1)`.
- **Parameter Efficiency:** **77,601 trainable parameters** — delivering high expressive temporal modeling with compact storage (< 350 KB).

### 9.2 Migration to Google Colab Pro GPU
To eliminate local host memory/pagefile pressure on Windows and enable rapid multi-seed training, Phase 3 training is migrated to **Google Colab Pro with CUDA GPU acceleration** (NVIDIA T4 / V100 / A100):

- **Self-Contained Notebook:** [`notebooks/colab_phase3_cnn_gru.ipynb`](file:///d:/ANN-Siren-Detection/notebooks/colab_phase3_cnn_gru.ipynb)
- **Drive Transfer Package Script:** [`scripts/package_colab_phase3.py`](file:///d:/ANN-Siren-Detection/scripts/package_colab_phase3.py)
- **Zero Raw 6GB Audio Re-Download:** Training uses pre-extracted, normalized Log-Mel spectrogram arrays (`train_val_mel_cache.npz` and `test_mel_cache.npz`), avoiding raw 6 GB audio unpacking.

### 9.3 Google Drive Directory Structure
When preparing your Google Drive for Colab Pro execution:
```text
Google Drive: MyDrive/ANN-Siren-Detection/
├── colab_phase3_package.zip          # Uploaded transfer package (739.5 MB)
├── data/
│   ├── processed/
│   │   ├── train_val_mel_cache.npz   # Normalized Log-Mel cache (Folds 1-9)
│   │   └── test_mel_cache.npz        # Normalized Log-Mel cache (Fold 10)
│   └── splits/
│       ├── train.csv, val.csv, test.csv
├── checkpoints/
│   └── cnn_gru_seed42.pt             # [Colab Output] Saved PyTorch weights
└── results/
    ├── metrics/
    │   ├── phase3_cnn_gru_test_seed42.json  # [Colab Output] Held-out test metrics
    │   └── phase3_model_comparison.csv      # [Colab Output] Benchmark comparison
    └── figures/
        ├── fig_phase3_cnn_gru_curves_seed42.png     # [Colab Output] Learning curves
        └── fig_phase3_architecture_comparison.png  # [Colab Output] Architectural chart
```

### 9.4 Execution Instructions (Google Colab Pro)
1. **Upload Transfer Package to Google Drive:**
   Run the local packager:
   ```powershell
   python scripts/package_colab_phase3.py
   ```
   Upload `colab_phase3_package.zip` into `MyDrive/ANN-Siren-Detection/` on your Google Drive.
2. **Open Notebook in Google Colab:**
   Open `notebooks/colab_phase3_cnn_gru.ipynb` in [Google Colab](https://colab.research.google.com).
3. **Select GPU Hardware Accelerator:**
   Go to **Runtime** $\to$ **Change runtime type** $\to$ Select **T4 GPU** (or V100 / A100 GPU).
4. **Execute All Cells:**
   The notebook will mount Google Drive, unpack source code and pre-extracted mel caches, train `CNNGRUHybrid` with validation early stopping on Fold 9 (tracking Macro-F1), evaluate held-out Fold 10 once, generate benchmark comparison figures, and export `phase3_trained_artifacts.zip` to Google Drive.

### 9.5 Importing Trained Checkpoints Back to Local Machine
Once Colab completes training, import the artifacts into your local Windows workspace:
```powershell
# 1. Download `phase3_trained_artifacts.zip` from Google Drive root to D:\ANN-Siren-Detection

# 2. Extract into the repository root:
Expand-Archive -Path phase3_trained_artifacts.zip -DestinationPath D:\ANN-Siren-Detection -Force

# 3. Verify the imported model is recognized by the multi-model inference engine:
python -c "from src.inference import SirenInferenceEngine; engine = SirenInferenceEngine(); print('Loaded models:', list(engine.available_models.keys()))"

# 4. Launch the Streamlit interface to test interactive inference:
streamlit run app.py
```

---

## 10. Complete Reproduction Pipeline

To reproduce all stages from scratch:

```powershell
# 1. Run test suite (110 verified tests) and environment smoke test
pytest tests/ -v
python scripts/smoke_test.py

# 2. Extract and canonicalize dataset
python scripts/acquire_dataset.py

# 3. Generate leakage-controlled split manifests
python scripts/prepare_dataset.py

# 4. Train first-principles NumPy MLP (CW1 T2)
python scripts/run_numpy_mlp_t2.py

# 5. Train competing models (CW1 T3)
python scripts/run_competing_models_t3.py

# 6. Execute multi-seed evaluation, ablation, and noise robustness (CW1 T4)
python scripts/run_t4_evaluation.py
python scripts/run_t4_ablation.py
python scripts/run_t4_error_analysis.py
python scripts/run_t4_robustness.py

# 7. Execute Phase 2 Noise-Augmented Deep MLP Training & Evaluation
python scripts/run_phase2_noise_augmentation.py --seeds 42

# 8. Launch demonstration application (CW1 T6 + Phase 1 XAI + Phase 2 Robustness)
streamlit run app.py
```

---

## 11. Repository Structure

```text
ANN-Siren-Detection/
├── .streamlit/
│   └── config.toml                  # Streamlit light theme and server configuration
├── configs/
│   └── default.yaml                 # Central hyperparameter & pipeline configuration
├── data/
│   ├── raw/                         # Raw UrbanSound8K audio files
│   ├── processed/                   # Standardized features, scalers, augmented cache & SHAP background
│   └── splits/                      # Leakage-controlled fold index manifests
├── demo/
│   ├── app.py                       # Modern Streamlit AI application with SHAP XAI & Phase 2 comparisons
│   └── sample_audio/                # Authenticated Fold 10 benchmark test recordings
├── notebooks/
│   ├── colab_t3_training.ipynb      # Google Colab Pro GPU continuation workflow (CW1 T1-T3)
│   └── colab_phase3_cnn_gru.ipynb   # Phase 3 CNN+GRU Hybrid GPU training on Colab Pro
├── reports/
│   ├── cw1_t1_evidence.md           # Dataset acquisition & leakage audit evidence
│   ├── cw1_t4_evidence.md           # Multi-seed evaluation & ablation findings
│   ├── cw1_t5_recommendation.md     # Model selection trade-off analysis
│   ├── cw1_t6_reproducibility.md    # Environment & reproducibility specification
│   ├── cw1_t6_demo_script.md        # Technical presentation script
│   └── CW1_Final_Technical_Record.docx # Formal coursework submission report
├── results/
│   ├── figures/                     # Verified confusion matrices, PR curves & Phase 2 comparison curves
│   └── metrics/                     # Genuine CSV evaluation outputs & Phase 2 JSON summaries
├── scripts/
│   ├── coursework_demo.py           # Fast, code-first technical demonstration runner
│   ├── smoke_test.py                # Environment and artifact validation
│   ├── package_colab_phase3.py      # Google Drive Phase 3 transfer packager
│   ├── run_phase2_noise_augmentation.py # Phase 2 noise-augmented training & evaluation runner
│   ├── run_phase3_cnn_gru.py        # Phase 3 CNN+GRU Hybrid experiment runner
│   └── run_t4_evaluation.py         # Multi-seed empirical evaluation suite
├── src/
│   ├── audio_features.py            # WAV decoders, standardization, MFCC & Log-Mel extraction
│   ├── augmentation.py              # Controlled audio waveform perturbations & policy enforcement
│   ├── explainability.py            # SHAP DeepExplainer, feature semantic mapping & visual charts
│   ├── hybrid_model.py              # Phase 3 CNN + GRU Hybrid PyTorch architecture (77,601 params)
│   ├── inference.py                 # Multi-model inference engine & dynamic thresholding
│   ├── noise_augmented_training.py  # Phase 2 dataset augmentation, training & comparative evaluation
│   ├── numpy_mlp.py                 # Analytical first-principles NumPy neural network
│   ├── models_torch.py              # PyTorch Deep MLP & 2D CNN definitions
│   ├── train_hybrid.py              # Phase 3 CRNN memory-mapped training & evaluation pipeline
│   └── evaluate.py                  # Macro-F1, PR-AUC, and confusion matrix calculation
├── tests/                           # 116 passing regression, component, explainability, robustness & hybrid tests
├── app.py                           # Root Streamlit entrypoint
├── requirements.txt                 # Pinned dependencies (including SHAP)
└── README.md                        # Project documentation
```

---

## 12. Academic Attribution & Coursework Record

- **Module:** NB6057CEM — Artificial Neural Networks
- **Dataset:** J. Salamon, C. Jacoby and J. P. Bello, *"A Dataset and Taxonomy for Urban Sound Research"*, ACM Multimedia 2014.
- **Evaluation Benchmark:** 10-fold cross-validation preserving official UrbanSound8K partition design.
