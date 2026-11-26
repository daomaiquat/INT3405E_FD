# S6E11 — Version 2 (Archived Label `V2`; Originally Repository `V3`)

## 1. Original Provenance & Source Paths
- **Archive Label:** `V2` *(preserved original repository identity: **`V3`**)*
- **Original Notebook Path:** `notebooks/S6E11_Fraud_Detection_Grandmaster_V3.ipynb`
- **Original Submission Path:** `research_v3/submission_v3.csv` (identical copies at `outputs/submissions/submission_v3.csv` and `archive/root_duplicates/submission_v3_from_root.csv`)
- **Original Summary & Manifest:** `research_v3/final_ensemble_summary.json`, `research_v3/checkpoints/v3_final_artifacts/v3_manifest.json`
- **Archived SHA-256 Checksums:**
  - `notebook.ipynb`: `33647fae2a265a55b9d9a9c4b978dd29c5173a5b3c9a0c766749a74425d06544`
  - `submission.csv`: `2cfae2ca1787437e9739e53cc0d57b7387e85f6eca3d68b78fe3e1e8f8c197bf`
- **Kaggle CLI Metadata Provenance:** Uploaded as `ref=58486891` (`fileName=submission_v3.csv`, `2026-11-18 00:15:47 UTC`, `publicScore=0.95092`; also submitted under `ref=58487106` & `ref=58572009`, all `0.95092`). Note: Kaggle CLI does not expose server-side CSV SHA-256 hashes; local identity is verified by Cell 21 printed SHA-256 (`2cfae2ca1787437e...`), `research_v3/FINAL_RESEARCH_REPORT.md`, and byte-identical copies across all three export locations.

## 2. Leaderboard & Local Cross-Validation Metrics
- **Public Leaderboard ROC AUC:** **`0.95092`**
- **Private Leaderboard ROC AUC:** Not released / unavailable via Kaggle CLI (`privateScore` blank).
- **Verified Local Nested 5-Fold OOF Metrics (`StratifiedKFold(n_splits=5, random_state=42)`):**
  - **Ensemble (`simplex_stdlogit`):** OOF ROC AUC = **`0.9510771`** (`+0.0001782` vs Frozen V1, 1,000-resample paired bootstrap $p=0.0080$)
  - **Calibration (`1D Platt` logit):** `LogLoss = 0.0440757`, `PR-AUC = 0.5337296`, `Brier = 0.010693`, Submission Mean Prob = `0.016599` (matching `0.016700` train prevalence)
  - **Secondary Meta-Split Replication (`seed=7`):** Nested OOF ROC AUC = `0.9510790`
  - Best Single V3 Model (`C2_XGB_GLMBoost_5Seed`): `0.9508547` OOF AUC

## 3. Model Architecture
- **Dual-Spline + Multi-Resolution Independent-Permutation GLM Base (`A3`, `A4`):** Combines 10-knot uniform cubic splines on `[merchant_risk_score, velocity_score]` with 7-knot quantile splines, `float64` decimal rounding, and independent inner-TE permutations.
- **Strictly-Nested 5-Fold GLM Margins (`eta_off=1.0`)** feed residual shallow GBDTs across LightGBM (`B2/B3` standalone, `B5/B6` GLMBoost), CUDA XGBoost (`C0` 3-leaf, `C2` GLMBoost), and GPU Symmetric CatBoost (`D1` standalone, `D2` GLMBoost).
- **10-Model Nested Convex Simplex Ensemble (`simplex_stdlogit`)** with outer Platt scaling.

## 4. Dependencies & Environment
- Self-contained notebook (zero `.py` helper imports). Optional V1 comparisons check `versions/v1_frozen/...exists()` and skip gracefully when omitted.
- Install packages: `pip install -r requirements.txt`

## 5. Data Download & Execution Instructions
1. Download competition CSVs cleanly via Kaggle CLI:
   ```bash
   kaggle competitions download -c playground-series-s6e11
   mkdir -p data && unzip playground-series-s6e11.zip -d data/
   ```
2. Place `notebook.ipynb` at a repo root containing `data/train.csv` (or inside a subfolder; `resolve_project_paths()` auto-checks `cwd`, `cwd.parent`, `cwd.parent.parent`, `/kaggle/input/...`, `/content/data`).
3. Execute `notebook.ipynb` end-to-end (`CONFIG["QUICK_RUN"] = True` for a 2-fold 20k smoke test, or `False` for full scratch retraining in ~45–60 min on CUDA GPU + multi-core CPU). When checkpoint folder `v3_final_artifacts/` is omitted, `try_load_verified_checkpoint()` returns `None` and cleanly retrains all 10 models from raw CSVs.

## 6. Known Limitations
- Because large `.npy`/`.npz` caches (`~161 MB`) are excluded from this compact Drive archive, full reproduction retrains all 10 multi-seed models from scratch (~45–60 min with GPU).
- Cell 19 writes `research_v3/figures/v3_diagnostics.png` and Cell 21 writes `submission.csv`, `outputs/submissions/submission_v3.csv`, and `research_v3/submission_v3.csv` relative to `PROJECT_DIR` (directories auto-created).
