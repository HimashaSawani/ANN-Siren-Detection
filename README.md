# Robust Emergency Siren Classification Under Urban Noise: A Comparative Study of MLP and CNN Architectures

**Module:** NB6057CEM — Artificial Neural Networks  
**Coursework:** Complete CW1 Tasks (T1–T6) & Technical Reports  
**Current Stage:** **Fully Completed, Tested (75/75 passing tests), and Live Demo Ready**  
**Demo Application:** `streamlit run app.py`

---

## 1. Executive Overview

This repository contains the complete coursework implementation for **NB6057CEM — Artificial Neural Networks**. The project delivers an end-to-end, leak-free, empirically verified machine learning pipeline for safety-critical emergency siren detection (`siren` vs. `non_siren`) under challenging urban background clutter using the official **UrbanSound8K** dataset.

### Delivered Capabilities
1. **CW1 T1 (Data Pipeline):** Official dataset ingestion with bitwise decoders for Microsoft & IMA ADPCM formats; leak-free fold splitting (Folds 1–8 train, Fold 9 val, Fold 10 test) with source-recording (`fsID=180937`) group isolation; and training-only feature scaler fitting.
2. **CW1 T2 (First-Principles ANN):** Complete multi-layer perceptron built in raw NumPy with analytical backpropagation, numerical gradient checks ($\text{rel\_diff} = 2.13 \times 10^{-10}$), tiny-batch memorization, and real-data training.
3. **CW1 T3 (Competing Architectures):** Tuned Random Forest baseline, Deep PyTorch MLP with Dropout, and 2D ConvNet on Log-Mel spectrograms.
4. **CW1 T4 (Controlled Experiments & Evaluation):** Repeated three-seed evaluation (seeds `42`, `123`, `2026`), test confusion matrices, continuous Precision-Recall curves, single-factor dropout ablation, and controlled noise robustness evaluation (Clean to 5 dB SNR).
5. **CW1 T5 (Model Recommendation):** Rigorous operational trade-off justification recommending Deep MLP on 80-D MFCCs.
6. **CW1 T6 (Reproducibility & Interactive Demo):** Interactive Streamlit web application, verified test suite (75 passing tests), and reproducibility documentation.

---

## 2. Interactive Streamlit Demonstration (CW1 T6)

Launch the interactive web application to perform live audio inference, inspect waveforms and Log-Mel spectrograms, adjust detection thresholds, and run multi-model comparisons:

```powershell
# Launch the Streamlit application
streamlit run app.py
```
*Access in browser at:* `http://localhost:8501`

### Application Highlights:
- **Audio Playback & Inspection:** Dual-panel waveform and Log-Mel spectrogram visualization.
- **Genuine Live Inference:** Powered by the frozen Deep MLP checkpoint (`checkpoints/deep_mlp_t3_seed42.pt`).
- **Benchmark Presets:** Includes authenticated held-out test recordings from Fold 10:
  - Siren Sample: `demo/sample_audio/siren_sample_165166-8-0-0.wav` (Predicted: 100% Siren)
  - Non-Siren Sample: `demo/sample_audio/non_siren_sample_100795-3-0-0.wav` (Predicted: 99.97% Non-Siren)
- **Multi-Model Cross-Check:** Simultaneously displays predictions from Deep MLP, Random Forest, NumPy MLP, and 2D CNN.
- **Evidence Dashboard:** Embedded confusion matrices, PR curves, and noise degradation plots loaded from disk.

---

## 3. Experimental Results Summary (CW1 T4)

Evaluation across all three random seeds (`42`, `123`, `2026`) on the held-out test split (Fold 10, N=837):

| Architecture | Input Representation | Evaluated Seeds | Val Macro-F1 ($\mu \pm \sigma$) | Test Macro-F1 ($\mu \pm \sigma$) | Siren F1 ($\mu \pm \sigma$) | Siren Recall ($\mu$) | Siren Prec ($\mu$) | PR-AUC ($\mu$) | Mean Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep MLP** | 80-D MFCC | 3 | **0.9655 ± 0.0139** | **0.7736 ± 0.0182** | **0.5896 ± 0.0306** | **55.4%** | 63.5% | **0.6602** | 20.53 s |
| **NumPy MLP** | 80-D MFCC | 3 | 0.9468 ± 0.0049 | 0.7568 ± 0.0187 | 0.5540 ± 0.0320 | 46.2% | 69.7% | 0.5769 | 5.61 s |
| **Random Forest** | 80-D MFCC | 3 | 0.9396 ± 0.0027 | 0.7397 ± 0.0165 | 0.5151 ± 0.0322 | 35.7% | **93.1%** | 0.5888 | 19.75 s |
| **2D CNN** | $(1, 128, 173)$ Log-Mel | 1* | 0.9151 ± 0.0000 | 0.7362 ± 0.0000 | 0.5175 ± 0.0000 | 44.6% | 61.7% | **0.6745** | 1,991.27 s (33.2 m) |

*\*Note on CNN Computational Budget:* The 2D CNN was evaluated on Seed 42 locally due to its 33.2-minute CPU training time. GPU multi-seed training is supported via `notebooks/colab_t3_training.ipynb`.

---

## 4. End-to-End Reproduction Steps

All steps can be executed sequentially from the repository root:

```powershell
# 1. Verify dependencies and run regression tests (75 passing tests)
pytest -v
python scripts/smoke_test.py

# 2. Extract and canonicalize the UrbanSound8K dataset
python scripts/acquire_dataset.py

# 3. Run exploratory dataset audit
python scripts/audit_dataset.py

# 4. Generate leak-free split manifests and fit training-only scalers
python scripts/prepare_dataset.py

# 5. Train first-principles NumPy MLP (CW1 T2)
python scripts/run_numpy_mlp_t2.py

# 6. Train Random Forest, Deep MLP, and 2D CNN (CW1 T3)
python scripts/run_competing_models_t3.py

# 7. Run three-seed evaluation and export evaluation tables (CW1 T4a & T4b)
python scripts/run_t4_evaluation.py

# 8. Run controlled single-factor dropout ablation (CW1 T4c)
python scripts/run_t4_ablation.py

# 9. Run error analysis and controlled noise robustness experiment (CW1 T4d)
python scripts/run_t4_error_analysis.py
python scripts/run_t4_robustness.py

# 10. Start the interactive demo application
streamlit run app.py
```

---

## 5. Coursework Documentation Index

Detailed evidence reports and outlines are located in `reports/`:

- [`reports/cw1_t1_evidence.md`](file:///D:/ANN-Siren-Detection/reports/cw1_t1_evidence.md): Dataset acquisition, leakage resolution, and audit evidence.
- [`reports/cw1_t4_evidence.md`](file:///D:/ANN-Siren-Detection/reports/cw1_t4_evidence.md): Full experimental evaluation, ablation findings, and error inspections.
- [`reports/cw1_t5_recommendation.md`](file:///D:/ANN-Siren-Detection/reports/cw1_t5_recommendation.md): Formal operational model recommendation and trade-off analysis.
- [`reports/cw1_t6_reproducibility.md`](file:///D:/ANN-Siren-Detection/reports/cw1_t6_reproducibility.md): Reproducibility checklist and environment specification.
- [`reports/cw1_t6_demo_script.md`](file:///D:/ANN-Siren-Detection/reports/cw1_t6_demo_script.md): Complete 5–7 minute demonstration script.
- [`reports/cw1_technical_record_outline.md`](file:///D:/ANN-Siren-Detection/reports/cw1_technical_record_outline.md): 1,500-word CW1 technical report outline.
- [`reports/cw2_theory_report_outline.md`](file:///D:/ANN-Siren-Detection/reports/cw2_theory_report_outline.md): 2,500-word CW2 theory report outline (Q1–Q4).
