# S6E11 Fraud Detection — Three Submitted Solutions Release Archive

Clean, verified Google Drive release package preserving all three Kaggle Playground Series S6E11 (`playground-series-s6e11`) submitted ML solutions in unmodified archival form:

```text
final_release/
  README.md                  # Package overview, quick-start & data download guide
  VERSION_COMPARISON.md      # Verified LB + OOF comparison & architectural progression
  SHA256SUMS.txt             # Cryptographic SHA-256 manifest of all packaged files
  V1/                        # Public LB 0.95069 (Frozen V1 Tri-Model Rank Blend)
  V2/                        # Public LB 0.95092 (originally V3: 10-Model Nested Simplex)
  Grandmaster/               # Public LB 0.95154 (Independent 4-Family ResDecomp_NonNeg)
```

## Competition Data Setup (Excluded to Keep Drive Archive Compact)
Original Kaggle CSV files (`train.csv`, `test.csv`, `sample_submission.csv`, ~148 MB) and large intermediate `.npy`/`.npz` caches are excluded from this package and remain preserved in the local workspace. To reproduce any notebook from scratch, download the raw CSVs via Kaggle CLI:

```bash
kaggle competitions download -c playground-series-s6e11
mkdir -p data && unzip playground-series-s6e11.zip -d data/
```

## Integrity Verification
Verify byte-exact integrity of all packaged notebooks, submissions, and documents before running or uploading:

- **Linux / macOS / Git Bash:**
  ```bash
  cd final_release && sha256sum -c SHA256SUMS.txt
  ```
- **Windows PowerShell:**
  ```powershell
  Get-Content final_release\SHA256SUMS.txt | ForEach-Object {
      $expected, $rel = $_ -split '  ', 2
      $actual = (Get-FileHash (Join-Path final_release $rel) -Algorithm SHA256).Hash.ToLower()
      if ($actual -ne $expected) { throw "Mismatch: $rel" }
  }; "ALL SHA-256 CHECKS PASSED"
  ```

## Governance Compliance
All three solutions rely strictly on conventional tabular Machine Learning (`LogisticRegression` + splines/encodings, `LightGBM`, `XGBoost`, `CatBoost`, `ExplainableBoostingClassifier`) trained from scratch on official competition data — zero LLMs, zero pretrained language embeddings, zero external competitor OOF/submission blends, and zero Kaggle API credentials packaged inside this archive.
