# S6E11 Submitted Versions Comparison (`V1` -> `V2` -> `Grandmaster`)

> **Important Validation Note:** Public Leaderboard (`LB`) scores are evaluated by Kaggle on the public test split, whereas local Cross-Validation (`OOF`) scores come from stratified splits of `data/train.csv` (`699,960` rows, `1.6700%` fraud) and use different fold protocols (`5-fold` in `V1`/`V2`, `10-fold` in `Grandmaster`). Private LB scores are unavailable and omitted intentionally.

## 1. Summary Metrics Table

| Archive Folder | Repo Origin | Public LB ROC AUC | Local CV Protocol | Local OOF ROC AUC | OOF LogLoss | Prediction Scale / Mean |
|---|---|---:|---|---:|---:|---|
| **`V1/`** | `versions/v1_frozen/` | **`0.95069`** | 5-Fold (`seed=42`) | `0.9508989` | — *(rank blend)* | Rank percentile (`mean = 0.500002`) |
| **`V2/`** | `research_v3/` / `V3` (`submission_v3.csv`) | **`0.95092`** | Nested 5-Fold (`seed=42`) | `0.9510771` | `0.0440757` | Calibrated prob (`mean = 0.016599`) |
| **`Grandmaster/`** | `independent_kaggle_research/` | **`0.95154`** | Protected 10-Fold (`seed=2026`) | **`0.951267`** (`CI [0.949405, 0.953129]`) | **`0.043960`** | Calibrated prob (`mean = 0.016607`) |

---

## 2. Architectural Evolution & Research Improvements

1. **`V1` -> `V2` (`0.95069` -> `0.95092` Public LB | `0.9508989` -> `0.9510771` 5-Fold OOF):**
   - **Resolved Non-Monotonic Risk Tails & Spline Knot Blindness:** Paired 10-knot Uniform cubic splines on `[merchant_risk_score, velocity_score]` with 7-knot Quantile splines and fixed `float32` decimal string precision tails via `np.round(..., 2)`.
   - **Independent Inner-TE Permutations:** Replaced shared inner folds with per-key independent permutations + 5-seed inner-TE bagging (`A3`, `A4`), removing collinear TE noise correlation.
   - **Strictly-Nested GLM Margin Residual Boosting (`eta_off=1.0`):** Added ultra-shallow (`3–4` leaves) residual LightGBM (`B5/B6`), CUDA XGBoost (`C2`), and GPU Symmetric CatBoost (`D2`) trained on leak-free nested GLM offsets alongside standalone `CatBoost Sym` (`D1`).
   - **Calibrated Convex Simplex + Platt Scaling:** Replaced ad-hoc `45/35/20` rank averaging with fold-standardized logit simplex weights (`10` models, `simplex_stdlogit`), restoring true calibrated fraud probabilities (`0.016599` mean) and improving `V1` on all `5/5` folds (`p = 0.0080`).

2. **`V2` -> `Grandmaster` (`0.95092` -> `0.95154` Public LB | `0.951267` Protected 10-Fold OOF):**
   - **Clean First-Principles Restart & Statistical Discovery:** Quantified sharp discrete spikes beyond smooth splines (`customer_age` `+0.001294`, `merchant_risk_score` `+0.000554`, `velocity_score` `+0.000529`) and `amount_usd` cent artifacts ($\chi^2 = 370.68, p = 4.99 \times 10^{-33}$) while flagging smooth negative controls (`time_of_day_hour`, `distance_from_home_km`) so exact keys aren't blindly applied everywhere.
   - **Vectorized Digit-Trie Sparse GLM (`02e_SpGLM`) + Multi-Scale Dense TE GLM (`02f_DeGLM`):** Built a joint `73,277`-column block-scaled Sparse Logistic Regression over multi-resolution digit prefixes/suffixes/cents + 10-knot splines, complemented by a 3-seed Dense TE (`m=[10, 50]`) GLM.
   - **Parsimonious 4-Family Orthogonal Lock:** Paired the two GLMs with Nested-OOF `Blend50` residual boosters (`03a_OffLGB`, `03b_OffXGB`, `03c_OffCAT`), high-bin (`max_bin=1024`) standalone Depth-2 XGBoost (`03d_StXGB`), and an orthogonal zero-TE `1,024`-bin Cyclic Explainable Boosting Machine (`04_EBM` GA2M).
   - **Residual-Decomposed Non-Negative Meta-Stacking (`ResDecomp_NonNeg`, `logit_eps = 1e-10`):** Eliminated extreme negative logit clipping ties (`1e-7` -> `1e-10`) and decomposed residual GBDTs as $\Delta_m = z_m - z_{\text{b50}}$ inside outer-fold-blind 10-fold CV, advancing OOF ROC AUC to **`0.951267`** (`+0.000451` over best single base model, $p = 1.21 \times 10^{-8}$) and reaching **`0.95154`** Public LB strictly without LLMs or competitor OOF blends.
