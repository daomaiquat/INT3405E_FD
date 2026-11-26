# S6E11 — Version 1 (V1 Frozen Baseline)

## 1. Original Provenance & Source Paths
- **Archive Label:** `V1`
- **Original Notebook Path:** `versions/v1_frozen/S6E11_Fraud_Detection_Grandmaster_V1.ipynb`
- **Original Submission Path:** `versions/v1_frozen/submission_V1.csv` (duplicate copy at `outputs/submissions/submission.csv`)
- **Original Manifest Path:** `versions/v1_frozen/V1_FREEZE_MANIFEST.json`
- **Archived SHA-256 Checksums:**
  - `notebook.ipynb`: `ba57311e6a35955b48ec10a3e063abe32b287db39cbfc082a3e9eca71db7c373`
  - `submission.csv`: `aa1fa985c92bf8c349950124bee03f6e09bc386fea0639bb5cc8a84c3c789542`
- **Kaggle CLI Metadata Provenance:** Uploaded as `ref=58183027`, `fileName=submission.csv`, `2026-11-10 02:32:30 UTC` (`09:32:30 +07:00`), `publicScore=0.95069`. Note: Kaggle CLI does not expose server-side uploaded CSV SHA-256 hashes; correspondence is verified via local freeze manifest timestamps (`2026-11-10 09:26:23 +07:00`), Cell 26 printed SHA-256 (`aa1fa985...`), and earlier `09:00:08` run JSON in the Kaggle description (`ensemble=0.9508989`).

## 2. Leaderboard & Local Cross-Validation Metrics
- **Public Leaderboard ROC AUC:** **`0.95069`**
- **Private Leaderboard ROC AUC:** Not released / unavailable via Kaggle CLI (`privateScore` blank).
- **Verified Local 5-Fold Pooled OOF ROC AUC:** **`0.9508989`** (`0.9508988577291527`, `StratifiedKFold(n_splits=5, random_state=42)`)
  - Regularized Spline/TE Logistic GLM: `0.9504425` (weight `45%`)
  - 3-Seed Shallow Depth-2 XGBoost (`seeds=(7, 42, 2026)`): `0.9504396` (weight `35%`)
  - Shallow Depth-2 LightGBM: `0.9504212` (weight `20%`)
  - **Submission Distribution (`submission.csv`):** `299,984` rows, normalized rank-percentile scale `[0.00002, 1.00000]`, mean `0.500002`.

## 3. Model Architecture
- Leakage-safe out-of-fold m-smoothed target encoding on multi-resolution keys (`36` key dictionaries), `OneHotEncoder`, 7-knot natural cubic spline on `merchant_risk_score`, and value frequency counts.
- Linear Logistic Regression + Depth-2 LightGBM + 3-Seed Depth-2 CUDA/CPU XGBoost combined via a **Tri-Model Rank-Percentile Blend (`0.45 GLM + 0.35 XGB_3Seed + 0.20 LGB`)**.

## 4. Dependencies & Environment
- Self-contained notebook with zero external `.py` helper imports.
- Install packages: `pip install -r requirements.txt`

## 5. Data Download & Execution Instructions
1. Download the official competition CSVs via Kaggle CLI:
   ```bash
   kaggle competitions download -c playground-series-s6e11
   unzip playground-series-s6e11.zip -d .
   ```
2. Ensure `train.csv`, `test.csv`, and `sample_submission.csv` sit in the same working directory as `notebook.ipynb` (`CONFIG["DATA_DIR"] = Path(".")`), or edit `CONFIG["DATA_DIR"]`.
3. Optional smoke test: set `CONFIG["QUICK_RUN"] = True` in Cell 2 (`2` folds, fewer rounds).
4. Execute `notebook.ipynb` end-to-end (`jupyter nbconvert --to notebook --execute notebook.ipynb`). If `./artifacts/` checkpoints are omitted (as in this compact Drive archive), `USE_CHECKPOINT_CACHE=True` automatically falls back to full scratch training and regenerates `submission.csv`.

## 6. Known Limitations
- Output probabilities in `submission.csv` are uniformly distributed rank percentiles (`mean=0.500002`) optimized purely for ROC AUC rather than log-loss / probability calibration.
- Intermediate 5-fold `.npy` checkpoints (`versions/v1_frozen/artifacts/`) are excluded from the Drive archive to keep size compact and are regenerated automatically if rerun.
