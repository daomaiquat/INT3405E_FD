# Kaggle S6E11 — Independent ML-Only Research Report

**Competition:** `playground-series-s6e11` (Predicting Credit Card Fraud, Evaluation Metric: ROC AUC)  
**Research Directory:** `independent_kaggle_research/`  
**Final Standalone Deliverable:** S6E11_Independent_Grandmaster.ipynb (`S6E11_Independent_Grandmaster.ipynb`)  
**Final Submission File:** submission.csv (`submission.csv`)

---

## 1. Executive Summary

Starting solely from the official competition data (`data/train.csv`: `699,960` rows × 26 columns, `11,689` positives = `1.6700%`; `data/test.csv`: `299,984` rows × 25 columns), first-principles statistical diagnostics, and a static audit of 201 public competition notebooks (`32` inspected in detail), we designed, validated, and locked a strictly ML-only **7-model, 4-family architecture** stitched together via **Residual-Decomposed Non-Negative Logit Stacking (`ResDecomp_NonNeg`, `logit_eps = 1e-10`)**.

- **Standalone Notebook (`SMOKE_TEST=0`, 10-Fold `seed=2026`, 100% Data):**
  - **Final Stack OOF ROC AUC:** **`0.951267`** (`0.951267252810`, 95% DeLong CI = `[0.949405, 0.953129]`, `DeLong SE = 0.000950`, `Fold Std = 0.002050`)
  - **Calibration & Secondary Metrics:** `OOF LogLoss = 0.043960`, `PR-AUC = 0.535517`, `Brier = 0.010668`
  - **Gain vs Best Single Base Model (`03b_OffXGB = 0.950816`):** **`+0.000451`** (paired DeLong **$p = 1.21 \times 10^{-8}$**)
  - **Total Notebook Wall-Clock Runtime:** `4,297.72s` (~71.6 min on Ryzen 7 8845HS + RTX 4060 GPU; zero cached artifacts loaded)
- **Protected Out-of-Sample Confirmation (`EXP09_Confirm10_LockedFinal` vs `EXP08` Dev5 Selection):**
  - Moving from 5-fold development (`EXP08` Dev5 OOF AUC = `0.950793`) to untouched 10-fold confirmation splits (`EXP09` = `0.951265`, standalone notebook = `0.951267`) improved out-of-sample AUC by **`+0.000471` / `+0.000474`**, confirming **zero selection-bias collapse** and positive scaling with additional per-fold training data (`80%` $\rightarrow$ `90%`).
  - Notebook OOF vs `EXP09` OOF Spearman rank correlation = **`0.999994`** (`|ΔAUC| = 2.68e-06`); notebook `submission.csv` vs `EXP09` test prediction Spearman correlation = **`1.000000`**.

---

## 2. Part A Status — Workspace Organization & Coursework Notebooks (E8/E9)

- **Repository Organization (`WORKSPACE_ORGANIZATION_AUDIT.md`):**
  - Moved the 11 downloaded public reference notebooks from the workspace root into `references/kaggle_public_notebooks/` preserving exact bytes and SHA-256 checksums, cataloged in `references/kaggle_public_notebooks/CATALOG.md`.
  - Audited `git status -sb` before and after organization (`fatal: not a git repository` — no `.git` history exists; `.gitignore` and `README.md` updated non-destructively).
  - Preserved all original competition CSVs (`data/`), historical notebooks (`notebooks/01_EDA.ipynb`, `02_Baseline_Pipeline.ipynb`, `S6E11_Fraud_Detection_Grandmaster_V3.ipynb`, `E5–E7`), frozen artifacts (`versions/v1_frozen/`), `archive/`, `outputs/`, `reports/`, `research_v3/`, `independent_research/`, `submission.csv`, and `.venv/`.
- **Coursework Notebooks Status (Strictly Unexecuted as Instructed):**
  - `notebooks/experiments/E8_Robustness_Generalization.ipynb` (`25` cells, `executed_cells = 0`): **`[PENDING EXECUTION IN COLAB]`**
  - `notebooks/experiments/E9_Final_Model_Selection_Submission.ipynb` (`20` cells, `executed_cells = 0`): **`[PENDING EXECUTION IN COLAB]`**

---

## 3. Public Competition Audit & Academic ML-Only Governance

Static inspection of `201` public competition kernels (`32` downloaded and audited cell-by-cell in `public_research/PUBLIC_SOURCE_CATALOG.json`) decomposed the Public Leaderboard into three distinct regimes:

1. **Community OOF Banks & Public Submission Blends (`0.9535–0.9615` Public LB) — STRICTLY EXCLUDED:** Pools `40–145` precomputed participant `.npy`/`.csv` files (`s6e11-oof-bank`, `scored-files`). Topic `757897` explicitly documents public-board optimization moving `+0.00027` public vs `-0.00010` private.
2. **Autoregressive Language Models (GPT-2 / GReaT Row Serialization, `0.9500–0.954+` Public LB) — STRICTLY EXCLUDED:** Fine-tunes autoregressive GPT-2 on target-first serialized rows to estimate sub-numeric token likelihoods. Inspected solely for mathematical motivation (discrete conditional density modeling), never used anywhere.
3. **Independent Conventional ML Pipelines (`0.946–0.9515` Reported) — ELIGIBLE DOMAIN:** Conventional GLMs, shallow GBDTs, digit/exact-value representations, GLM-offset residual boosting, EBM GA2Ms, and tabular neural architectures (`RealMLP`/`TabM`/`DCNv2`) trained from scratch without pretraining.

**Governance Verification:** Our final pipeline uses **zero** LLMs, GPT-2, language embeddings, TabPFN/TFMs, external pretrained weights, external datasets, other participants' OOF/submission files, or legacy `V1–V4` / `research_v3` / `independent_research` / `E5–E9` artifacts.

---

## 4. First-Principles EDA Discoveries

Computed dynamically on `data/train.csv` and `data/test.csv`:

1. **Stationarity & Leakage-Free Competition Structure:**
   - Zero missing cells (`0`), zero feature-only duplicates (`0`), zero conflicting duplicate label groups (`0`), zero exact test-in-train feature matches (`0`).
   - 5-fold LightGBM Adversarial Validation AUC = **`0.500245`**; maximum numeric KS = `0.0038` (`day_of_week`); maximum categorical/boolean TVD = `0.0023` (`auth_method`, `merchant_category`); `id` decile fraud rates strictly bounded in `[0.01587, 0.01734]`. Stratified 5-fold and 10-fold CV accurately represent the test distribution without importance reweighting.
   - Comparison with `uditjain13/credit-card-fraud-detection-2026` (`20,000` rows, `1.695%` fraud) showed strong marginal covariate shift (`amount_usd` KS = `0.3831`, `account_balance_usd` KS = `0.1880`, `merchant_category` TVD = `0.1347`) despite high discrete vocabulary overlap (`99.987%` `merchant_risk_score`, `97.36%` `amount_usd`). Decision: build strictly from the 3 official competition CSVs cleanly inside the standalone pipeline.
2. **Sharp Sub-Spline Discrete Value Risk Spikes (`exact_value_residual_signal.csv`):**
   Adding inductive out-of-fold exact-value Empirical-Bayes residuals onto a smooth 10-knot cubic Spline-GLM baseline (`0.943276` OOF AUC) yielded substantial OOF gains on discrete-spiked columns:
   - `customer_age` (64 unique): **`+0.001294`** $\Delta\text{AUC}$ (`+0.000364` $\Delta\text{LogLoss}$)
   - `merchant_risk_score` (950 unique): **`+0.000554`** $\Delta\text{AUC}$ (`+0.000199` $\Delta\text{LogLoss}$)
   - `velocity_score` (666 unique): **`+0.000529`** $\Delta\text{AUC}$ (exact integers are `15.96%` of rows at `1.10%` fraud vs `1.78%` non-integers; e.g., `48.0` = `22.97%` fraud)
   - `hours_since_last_txn` (3,633 unique): **`+0.000482`** $\Delta\text{AUC}$
   - `account_balance_usd` (`43,317` unique): **`+0.000107`** $\Delta\text{AUC}$; `amount_usd` (`13,387` unique): **`+0.000076`** $\Delta\text{AUC}$
   - **Negative Controls (Smooth Curves):** Exact OHE on `time_of_day_hour` (`-0.000174`), `distance_from_home_km` (`-0.000060`), and `card_age_months` (`-0.000058`) degraded OOF AUC relative to smooth splines.
3. **Statistically Significant Cent & Decimal Digit Artifacts:**
   - `amount_usd` cents (`0..99`): $\chi^2 = 370.68$, **$p = 4.99 \times 10^{-33}$** (`.49` = `2.49%` across 7,274 rows, `.00` int = `2.09%` vs `1.67%` baseline).
   - `account_balance_usd` cents: $\chi^2 = 154.39$, **$p = 3.11 \times 10^{-4}$**.
   - `hours_since_last_txn` last decimal digit: $\chi^2 = 32.17$, **$p = 1.86 \times 10^{-4}$**.
4. **Low-Order Interaction Screen (`276` feature pairs over Spline-GLM):**
   Top residual gains concentrate around `merchant_risk_score` and `velocity_score` paired with risk booleans, `txn_count_last_24h`, `cvv_retry_count`, `prior_disputes`, `auth_method`, and `merchant_risk_score × velocity_score` (`+0.000174` to `+0.000216` AUC). Meanwhile, deep unregularized trees (`31 leaves`) overfit `1.67%` minority labels and lose AUC against `4-leaf` constrained trees.

---

## 5. Complete Experimental Ledger & Hypothesis Progression (`Dev5 S42` $\rightarrow$ `Confirm10`)

All 28 formally logged experiments in `experiments_ledger.jsonl`:

| ID | Model Family | Key Features / Config | CV Design | OOF AUC | LogLoss | Outcome / Scientific Lesson |
|---|---|---|---|---:|---:|---|
| `EXP01a` | Spline GLM | Raw 24 features, 10-knot splines + OHE | Dev5 | `0.943276` | `0.046514` | **RETAIN** — Smooth additive baseline anchor |
| `EXP01b` | LightGBM | Raw 24, `num_leaves=31, min_child=50` | Dev5 | `0.946405` | `0.045830` | **DIAGNOSTIC** — Standard depth overfits rare target |
| `EXP01c` | LightGBM | Raw 24, `num_leaves=4, min_child=100` | Dev5 | `0.947413` | `0.045370` | **RETAIN** — Shallow 4-leaf beats 31-leaf by `+0.001009` |
| `EXP01d` | XGBoost GPU | Raw 24, `max_depth=2` | Dev5 | `0.945640` | `0.045936` | **RETAIN** — Depth-2 pairwise baseline |
| `EXP01e` | CatBoost GPU | Raw 24, `depth=4, SymmetricTree` | Dev5 | `0.946075` | `0.045797` | **RETAIN** — Symmetric ordered TS baseline |
| `CANARY` | Leakage Check | Full FoldInductiveEncoder on shuffled `y` | Dev5 | `0.501145` | — | **PASS** (`GLM=0.503320`, `LGBM=0.501145`) |
| `EXP05` | Generative Bayes | Chow-Liu MST LLR (`24+5` bins) | Dev5 | `0.911784` | `0.060040` | **DIAGNOSTIC** — Double counting & uncalibrated edges |
| `EXP02a` | Sparse GLM | Selective exact OHE + splines (`dim=38,281`) | Dev5 | `0.948680` | `0.044793` | **RETAIN** — `+0.001266` vs `EXP01c` ($p=7.24\times 10^{-6}$) |
| `EXP02b` | Sparse GLM | Negative control: all exact OHE (`dim=42,390`) | Dev5 | `0.948459` | `0.044836` | **DIAGNOSTIC** — Smooth exact keys hurt (`p=0.0101`) |
| `EXP02c` | Sparse GLM | Negative control: + screened raw pair OHE (`dim=39,640`) | Dev5 | `0.948379` | `0.044899` | **DIAGNOSTIC** — Redundant OHE variance hurts (`p=0.0328`) |
| `EXP02d` | Dense TE GLM | Newton-Cholesky on splines + `m=[10,50]` TE | Dev5 | `0.949453` | `0.044605` | **RETAIN** — `+0.000774` vs `EXP02a` (had singular Hessian warnings) |
| `EXP02e` | Sparse GLM | Digit-Trie hierarchy + block scales `(1.5,1.2,0.6)` (`dim=73,277`) | Dev5 | **`0.949109`** | `0.044629` | **RETAIN (Locked)** — `+0.000429` vs `EXP02a` ($p=6.77\times 10^{-4}$) |
| `EXP02f` | Dense TE GLM | Digit-Trie 3-seed nested TE `m=[10,50]` + splines, L-BFGS (`dim=339`) | Dev5 | **`0.949620`** | `0.044578` | **RETAIN (Locked)** — `50/50` blend (`2e+2f`) reaches **`0.950100`** |
| `EXP05b` | Calibrated Bayes | Miller-Madow Chow-Liu + inner logistic cal | Dev5 | `0.935843` | `0.049358` | **DIAGNOSTIC** — Fold-0 dropoff (`0.914369`) from MST BFS bug |
| `EXP03a` | Offset LightGBM | `leaves=4, extra_trees, n=297` over Nested `Blend50` | Dev5 | **`0.950412`** | `0.044270` | **RETAIN (Locked)** — `+0.000313` vs `Blend50` ($p=3.80\times 10^{-9}$) |
| `EXP03b` | Offset XGB GPU | `lossguide 4L, max_bin=1024, n=174` over Nested `Blend50` | Dev5 | **`0.950432`** | `0.044269` | **RETAIN (Locked)** — Best Dev5 single model ($p=2.58\times 10^{-9}$) |
| `EXP03c` | Offset CatBoost GPU | `depth=4 SymmetricTree, n=429` over Nested `Blend50` | Dev5 | **`0.950423`** | `0.044238` | **RETAIN (Locked)** — Lowest single Dev5 LogLoss ($p=4.82\times 10^{-11}$) |
| `EXP03d-LGB` | Standalone LGBM | `leaves=4, max_bin=1024, n=2585`, DigitTE+qpairs | Dev5 | `0.949528` | `0.044659` | **RETAIN** — Pruned in `EXP08` LOFO parsimonious selection |
| `EXP03d-XGB` | Standalone XGB GPU | `depth=2, max_bin=1024, n=2346`, DigitTE+qpairs | Dev5 | **`0.949525`** | `0.044646` | **RETAIN (Locked)** — Orthogonal standalone family anchor |
| `EXP03d-CAT` | Standalone CAT GPU | `depth=4, border=254, n=2935`, DigitTE+qpairs | Dev5 | `0.949042` | `0.044636` | **RETAIN** — Pruned in `EXP08` LOFO parsimonious selection |
| `EXP04` | EBM GA2M | `max_bins=1024, interactions=25, outer_bags=4`, **zero TE** | Dev5 | **`0.948610`** | `0.044878` | **RETAIN (Locked)** — High-res non-TE cyclic GA2M |
| `EXP06a-St` | PyTorch PLE-ResNet | Compact 16-bin PLE + 9 embeddings (`~250` vocab) | Dev5 | `0.946843` | `0.045619` | **RETAIN** — Pruned in `EXP08`; replaced over-parameterized DeepFM prototype |
| `EXP06a-Res` | PyTorch PLE-ResNet | Zero-init residual head over Nested `Blend50` | Dev5 | `0.950116` | `0.044372` | **RETAIN** — Pruned in `EXP08` (`Spearman=0.999988`, `Pearson=0.999990` vs `Blend50`) |
| `EXP06b` | PyTorch TabM | `K=6` BatchEnsemble zero-init residual over `Blend50` | Dev5 | `0.950097` | `0.044374` | **RETAIN** — Pruned in `EXP08` (`Spearman=0.999995`, `Pearson=0.999990` vs `Blend50`) |
| `EXP07` | DCNv2 + AUC Loss | CrossNetV2 (`rank=32`) + `0.25` pairwise AUC loss | Dev5 | `0.950053` | `0.044398` | **RETAIN** — Pruned in `EXP08` (`Spearman=0.997919`, `Pearson=0.999024` vs `Blend50`) |
| `EXP05c` | Calibrated Bayes Fixed | Full-component Chow-Liu BFS + `tr_idx` cardinality | Dev5 | `0.941838` | `0.047109` | **RETAIN** — Fixed Fold 0 (`0.9144` $\rightarrow$ `0.9391`); gated out (`keep_bayes_05c=false`) |
| `EXP08` | **ResDecomp Stacker** | **LOFO-pruned 7-model `ResDecomp_NonNeg` (`eps=1e-10`)** | **Dev5** | **`0.950793`** | **`0.044146`** | **RETAIN (Locked)** — Beats `03b` by `+0.000362` ($p=1.13\times 10^{-6}$); `delta_vs_all14=+0.000029` |
| `EXP09` | **Confirm10 Stacker** | **Locked 7-model retrain on untouched 10 folds** | **Confirm10** | **`0.951265`** | **`0.043961`** | **RETAIN** — `+0.000471` vs Dev5; beats `03b` by `+0.000448` ($p=1.18\times 10^{-8}$) |

*(Additional exploratory development scripts in `independent_kaggle_research/scripts/` — an initial high-cardinality Fourier DeepFM prototype later overwritten by PLE-ResNet in `exp06_deepfm_periodic.py`, CPU `exp06b_realmlp_tabm_benchmark.py`, and slow native CatBoost CTR `exp03e_catboost_native_ctr.py` — were retired before formal ledger promotion.)*

---

## 6. Key Methodological Findings & Stacking Ablations (`EXP08`)

1. **Nested-OOF GLM Offsets Prevent In-Sample Margin Memorization:** Computing strictly inner-CV out-of-fold `Blend50` offsets for `tr_idx` margins focuses shallow 174–429 round GBDTs on genuine residual structure (`0.9504+` Dev5, `0.9508+` Confirm10) without relying on inflated in-sample `tr_idx` fits (tracked alongside `Ab_Nested_vs_InSample_p=4.78e-01` in `EXP03a–c` ledger notes).
2. **Residual Decomposition Solves the Simplex Constraint Trap (`EXP08` 14-Model Outer-Blind Comparison):**
   - `Shallow_Meta_GBDT`: `0.949761` (`p=2.08e-06` worse than `03b_OffXGB`)
   - `Probit_Rank_L2`: `0.950502`
   - `Logit_L2`: `0.950565`
   - `NonNeg_Simplex`: `0.950702`
   - **`ResDecomp_NonNeg` (`eps=1e-10`):** **`0.950764`** (14 models) $\rightarrow$ **`0.950793`** (pruned 7-model, 4-family stack). Parameterizing $\Delta_m = z_m - z_{\text{b50}}$ lets each GLM-offset tree contribute additive residual lift directly rather than fighting GLM weights under `w >= 0`.
3. **Final LOFO Family Significance (`EXP08` Dev5):**
   - Dropping `Offset_GBDTs` (`03a/03b/03c`): `-0.000213` ($p = 2.43 \times 10^{-5}$)
   - Dropping `EBM_GA2M` (`04_EBM`): `-0.000153` ($p = 1.10 \times 10^{-3}$)
   - Dropping `GLM_Pair` (`02e/02f`): `-0.000094` ($p = 1.55 \times 10^{-3}$)
   - Dropping `Standalone_GBDT` (`03d_StXGB`): `-0.000063` ($p = 5.13 \times 10^{-2}$)

---

## 7. Final Standalone Notebook Results & Learned Ensemble (`S6E11_Independent_Grandmaster.ipynb`)

| Component | Role / Config Summary | Standalone 10-Fold OOF AUC | Learned Meta-Weight (`mean +/- std`) |
|---|---|---:|---|
| `02e_SpGLM` | Sparse Digit-Trie OHE + 10-knot Splines (`C=0.03`) | `0.949502` | `0.2816 +/- 0.0140` |
| `02f_DeGLM` | Dense 3-Seed Nested TE (`m=10,50`) + Splines (`C=0.10`) | `0.949881` | `0.2969 +/- 0.0165` |
| `03a_OffLGB` | Nested-OOF `Blend50` Offset LGBM (`4L extra_trees`, `n=297`, 3 seeds) | `0.950802` | $\Delta = 0.4458 \pm 0.0242$ |
| `03b_OffXGB` | Nested-OOF `Blend50` Offset XGB CUDA (`lossguide 4L`, `n=174`, 3 seeds) | `0.950816` | $\Delta = 0.4384 \pm 0.0195$ |
| `03c_OffCAT` | Nested-OOF `Blend50` Offset CatBoost GPU (`depth=4 Symmetric`, `n=429`, 3 seeds) | `0.950742` | $\Delta = 0.6075 \pm 0.0207$ |
| `03d_StXGB` | Standalone High-Bin XGB CUDA (`depth=2, max_bin=1024`, `n=2346`, 3 seeds) | `0.950247` | `0.2085 +/- 0.0203` |
| `04_EBM` | High-Resolution Zero-TE Cyclic EBM GA2M (`max_bins=1024, inter=25`) | `0.948936` | `0.2240 +/- 0.0110` |
| **Final Stack** | **Outer-Fold-Blind 10-Fold `ResDecomp_NonNeg_Logit` (`eps=1e-10`)** | **`0.951267`** | **LogLoss=`0.043960`, FoldStd=`0.002050`** |

- **Per-Fold Runtime Breakdown:** Stage 1 (GLMs + 3-inner-fold nested offsets) = `1,790.29s`; Stage 2 (3-seed GBDTs) = `1,631.80s`; Stage 3 (10-fold EBM) = `648.70s`; Stage 4 (`ResDecomp` inner-CV stacking) = `218.61s`; Total = `4,297.72s`.

---

## 8. Complete 10-Point Verification & Acceptance Matrix (`artifacts/verification_suite_summary.json`)

| # | Acceptance Check | Measured Result / Evidence (Persisted in `verification_suite_summary.json` & Metrics JSONs) | Status |
|---|---|---|---|
| 1 | **Notebook Structure & Python AST** | `nbformat.validate(v4)` + `ast.parse` across all `21` cells (`10` code cells), `kernelspec = python3` | **PASS** |
| 2 | **Standalone & ML-Only Static Scan** | Zero references to LLMs/GPT-2/TabPFN/TabICL/PyTorch, zero `src/` imports, zero `.npz`/cache loads, zero `E5–E9` or `V1–V4` refs | **PASS** |
| 3 | **Fast End-to-End Smoke Test (`SMOKE_TEST=1`)** | Isolated temp copy (`25k/5k`, 2 folds): `AUC=0.931259`, `LogLoss=0.049933`, `4` PNGs, `0` errors, `kernelspec=python3` | **PASS** |
| 4 | **Full Notebook Execution (`SMOKE_TEST=0`)** | Executed `nbconvert --inplace` (`4,297.72s`, exit `0`), `10/10` executed code cells, `0` errors, `4` embedded PNG diagnostics | **PASS** |
| 5 | **Protected Confirm10 Alignment Audit** | Notebook (`0.951267`) vs `EXP09` (`0.951265`): `|diff| = 2.68e-06`, OOF Spearman = `0.999994` | **PASS** |
| 6 | **Shuffled-Label Canary & Row Isolation** | 100k canary `seed=9999`: `02e=0.51812 (SE=0.00714, p=0.0112)`, `02f=0.49990 (p=0.9893)`, `03a=0.51096 (p=0.1293)`, `04_EBM=0.51169 (p=0.1027)` + 3-seed `02e` null check (`0.49079, 0.50957, 0.50224`, `p=0.1951..0.7538`); 200-row Test-Perturbation Row-0 `|diff| = 0.00e+00` | **PASS** |
| 7 | **`submission.csv` Integrity Audit** | Shape `(299984, 2)`, header `id,is_fraud`, exact `id` equality, `0` NaN/Inf, bounds `[0, 1]`, `mean = 0.016607`, `q01/50/99 = (0.000006/0.001379/0.380014)`, Spearman vs `EXP09` = `1.000000` | **PASS** |
| 8 | **Independent `roc_auc_score` Check** | `roc_auc_score(y, final_notebook_oof.npy) = 0.951267252810` (`|diff| = 1.11e-16`) | **PASS** |
| 9 | **Reproducibility & Workspace Cleanliness** | `E8_Robustness_Generalization.ipynb` and `E9_Final_Model_Selection_Submission.ipynb` verified untouched (`0` executed cells); zero `catboost_info/` or scratch clutter in repo | **PASS** |
| 10 | **Gate 3 Adversarial Review** | Independent `research` audit (`1fb07046-90da-46b0-9255-af876619ab3f`): standalone compliance, submission integrity, artifact-backed numbers (`verification_suite_summary.json`), E8/E9 untouched (`executed_cells=0`), workspace cleanliness | **PASS** |

---

## 9. Limitations & Scope Boundaries

1. **Private Test Ceiling vs Target-First LMs:** Public S6E11 pipelines reaching `0.9525–0.955+` combine GPT-2 target-first token density modeling and/or multi-participant OOF banks. Under strict academic ML-only rules (zero LMs/LLMs/external OOFs), our `0.951267` Confirm10 stack reflects pure non-LM supervised modeling from scratch.
2. **Tabular Neural Redundancy on Strong GLM Baselines:** When compact 16-bin PLE ResNet (`06a_NN_Res`) and `TabM` (`06b_TabM`) were initialized with zero residual heads on Nested `Blend50`, their OOF predictions stayed nearly collinear with `Blend50` (`Spearman = 0.999988` and `0.999995`, `Pearson = 0.999990`), while `07_DCNv2` had `Spearman = 0.997919` (`Pearson = 0.999024`). Pruning those three offset neural nets alongside duplicate standalone trees and `05c` yielded a `+0.000029` improvement over the full 14-model stack (`exp08_stack_summary.json`: `delta_vs_all14 = +2.905e-05`, `p = 0.2033`), leaving `04_EBM` and `03d_StXGB` as cleaner, faster orthogonal anchors.
3. **Colab Execution Needed for Part A Deliverables:** Coursework notebooks `E8` and `E9` remain strictly unexecuted per instructions and await execution inside Google Colab.
