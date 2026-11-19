$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..')).TrimEnd('\')
$rootPrefix = $taskRoot + '\'
$journalPath = Join-Path $PSScriptRoot 'changes.json'
$records = [Collections.Generic.List[object]]::new()
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'summary.json')) {
    throw 'This migration has already completed; this script is retained only as an audit record.'
}

function Safe-Path([string]$path) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $taskRoot $path))
    if (-not $absolute.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target is outside the workspace: $absolute"
    }
    $relative = $absolute.Substring($rootPrefix.Length)
    if ($relative.Split('\')[0] -in @('.git', '.agents', '.codex', '.aws')) {
        throw "Protected workspace directory: $absolute"
    }
    if (Test-Path -LiteralPath $absolute) {
        if ((Get-Item -Force -LiteralPath $absolute).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing a reparse point: $absolute"
        }
    }
    return $absolute
}

function Record-Change($record) {
    $records.Add($record)
    ConvertTo-Json -InputObject @($records.ToArray()) -Depth 8 |
        Set-Content -Encoding UTF8 -LiteralPath $journalPath
}

function Move-Logged([string]$from, [string]$to) {
    $source = Safe-Path $from
    $destination = Safe-Path $to
    if (-not (Test-Path -LiteralPath $source) -and (Test-Path -LiteralPath $destination) -and
        @($records | Where-Object { $_.action -eq 'move' -and $_.source -eq $from -and $_.destination -eq $to }).Count) { return }
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    $item = Get-Item -Force -LiteralPath $source
    $files = @(if ($item.PSIsContainer) { Get-ChildItem -LiteralPath $source -Recurse -File -Force } else { $item })
    $bytes = 0L
    foreach ($file in $files) { $bytes += $file.Length }
    New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
    Record-Change ([ordered]@{ action = 'move'; source = $from; destination = $to; files = $files.Count; bytes = $bytes })
    Move-Item -LiteralPath $source -Destination $destination
}

function Remove-Logged([string]$path, [string]$reason, [string]$duplicateOf = '') {
    $target = Safe-Path $path
    if (-not (Test-Path -LiteralPath $target) -and
        @($records | Where-Object { $_.action -eq 'delete' -and $_.source -eq $path }).Count) { return }
    $item = Get-Item -Force -LiteralPath $target
    $files = @(if ($item.PSIsContainer) { Get-ChildItem -LiteralPath $target -Recurse -File -Force } else { $item })
    if ($item.PSIsContainer) {
        $links = @(Get-ChildItem -LiteralPath $target -Recurse -Force |
            Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
        if ($links.Count) { throw "Refusing recursive deletion containing reparse points: $target" }
    }
    $inventory = @($files | ForEach-Object {
        [ordered]@{ path = $_.FullName.Substring($rootPrefix.Length); bytes = $_.Length }
    })
    $bytes = 0L
    foreach ($file in $files) { $bytes += $file.Length }
    Record-Change ([ordered]@{
        action = 'delete'; source = $path; reason = $reason; duplicate_of = $duplicateOf
        files = $files.Count; bytes = $bytes; removed_files = $inventory
    })
    Remove-Item -LiteralPath $target -Recurse -Force
}

function Remove-Duplicate([string]$path, [string]$canonical) {
    $left = Safe-Path $path
    $right = Safe-Path $canonical
    if (-not (Test-Path -LiteralPath $left) -and
        @($records | Where-Object { $_.action -eq 'delete' -and $_.source -eq $path -and $_.duplicate_of -eq $canonical }).Count) { return }
    if ((Get-FileHash -LiteralPath $left -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $right -Algorithm SHA256).Hash) {
        throw "Duplicate verification failed: $path versus $canonical"
    }
    Remove-Logged $path 'SHA256 verified duplicate' $canonical
}

if (Test-Path -LiteralPath $journalPath) {
    $previousRecords = Get-Content -Raw -Encoding UTF8 -LiteralPath $journalPath | ConvertFrom-Json
    foreach ($record in $previousRecords) { $records.Add($record) }
}
$run = 'INT2403E\s6e11_optuna\20261113T183409930114Z'
$baseline = 'INT2403E\artifacts'
$originalConfig = Get-Content -Raw -Encoding UTF8 -LiteralPath (Safe-Path "$run\run_config.json") | ConvertFrom-Json
$summary = [Collections.Generic.List[object]]::new()
foreach ($configFile in @(Get-ChildItem -LiteralPath (Safe-Path $baseline) -Recurse -Filter 'run_config.json' -File |
        Where-Object { $_.FullName -notmatch 'optuna_deps' })) {
    $summary.Add([ordered]@{
        original_path = $configFile.FullName.Substring($rootPrefix.Length)
        config = (Get-Content -Raw -Encoding UTF8 -LiteralPath $configFile.FullName | ConvertFrom-Json)
    })
}
New-Item -ItemType Directory -Path (Safe-Path 'legacy\experiment_summaries') -Force | Out-Null
ConvertTo-Json -InputObject @($summary.ToArray()) -Depth 12 |
    Set-Content -Encoding UTF8 -LiteralPath (Safe-Path 'legacy\experiment_summaries\smoke_runs.json')

# Preserve older unique files from the RAR before removing the redundant package.
$rarAudit = Get-Content -Raw -Encoding UTF8 -LiteralPath (Safe-Path 'reports\reorganization\rar_contents.json') | ConvertFrom-Json
if (-not $rarAudit.all_entries_verified) { throw 'RAR contents have not been verified.' }
if (Test-Path -LiteralPath (Safe-Path 's6e11_kaggle_ready.rar')) {
    foreach ($entry in $rarAudit.entries) {
        $checkPath = if ($entry.PSObject.Properties.Name -contains 'preserved_at') { $entry.preserved_at } else { $entry.duplicate_of }
        if ((Get-FileHash -LiteralPath (Safe-Path $checkPath) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw "RAR preservation check failed: $checkPath"
        }
    }
}
Remove-Logged 's6e11_kaggle_ready.rar' 'All archive members verified; unique older code/docs preserved under legacy/package_snapshots'

Move-Logged 's6e11_kaggle_ready\README_KAGGLE.md' 'legacy\docs\README_KAGGLE_old.md'
foreach ($file in @(Get-ChildItem -LiteralPath (Safe-Path 's6e11_kaggle_ready') -File)) {
    $canonical = $file.Name
    if ($file.Name -in @('run_config.json', 'fold_assignments.csv')) { $canonical = "$run\$($file.Name)" }
    elseif ($file.Name -in @('baseline_summary.csv', 'oof_logistic.npy', 'test_logistic.npy', 'oof_random_forest.npy', 'test_random_forest.npy')) {
        $canonical = "$baseline\$($file.Name)"
    }
    elseif ($file.Name -eq 's6e11_blend_run.py') { $canonical = 's6e11_blend.py' }
    elseif ($file.BaseName -match '^(.*)_(lightgbm|xgboost|catboost)$') {
        $canonical = "$run\$($Matches[2])\$($Matches[1])$($file.Extension)"
    }
    Remove-Duplicate "s6e11_kaggle_ready\$($file.Name)" $canonical
}
Remove-Logged 's6e11_kaggle_ready' 'Empty duplicate packaging directory'

$duplicateBaseline = Safe-Path 'INT2403E\artifacts(2)'
foreach ($file in @(Get-ChildItem -LiteralPath $duplicateBaseline -Recurse -File)) {
    $relative = $file.FullName.Substring($duplicateBaseline.Length + 1)
    $canonical = Safe-Path "$baseline\$relative"
    if ((Get-FileHash -LiteralPath $file.FullName).Hash -ne (Get-FileHash -LiteralPath $canonical).Hash) {
        throw "Baseline duplicate mismatch: $relative"
    }
}
Remove-Logged 'INT2403E\artifacts(2)' 'All files verified by SHA256 against baseline artifacts' $baseline

foreach ($directory in @('mini_input', 'mini_runs', 'optuna_test_runs', 's6e11_tuning', 'optuna_deps')) {
    Remove-Logged "$baseline\$directory" 'Regenerable smoke data/runs or vendored dependencies; run configurations summarized in legacy/experiment_summaries'
}
if (Test-Path -LiteralPath (Safe-Path '__pycache__')) { Remove-Logged '__pycache__' 'Regenerable Python bytecode' }

foreach ($file in @(Get-ChildItem -LiteralPath (Safe-Path 'INT2403E\__results___files') -File)) {
    Remove-Duplicate "INT2403E\__results___files\$($file.Name)" "__results___files\$($file.Name)"
}
Remove-Logged 'INT2403E\__results___files' 'Empty duplicate EDA images directory'
Remove-Duplicate 'submission.csv' "$run\submission.csv"
Remove-Duplicate 'INT2403E\submission.csv' "$run\submission.csv"

foreach ($name in @('train.csv', 'test.csv', 'sample_submission.csv')) { Move-Logged $name "data\$name" }
foreach ($name in @('credit-card-fraud-s6e11-guide.md', 's6e11_pure_ml_lessons.md')) {
    Move-Logged $name "docs\references\$name"
}
foreach ($name in @('s6e11_baselines_findings.md', 's6e11_eda_findings.md', 's6e11_ml_course_plan.md',
        's6e11_next_steps_plan.md', 's6e11_blend_README.md', 's6e11_tune_trees_README.md')) {
    Move-Logged $name "legacy\docs\$name"
}
foreach ($name in @('s6e11_baselines_kaggle.py', 's6e11_eda_kaggle.py', 's6e11_optuna_kaggle.py',
        's6e11_tune_trees_kaggle.py', 's6e11_blend.py')) {
    Move-Logged $name "legacy\scripts\$name"
}
Move-Logged 'eda.ipynb' 'legacy\notebooks\eda.ipynb'
Move-Logged '__results___files' 'legacy\notebooks\__results___files'
Move-Logged 'INT2403E\artifacts' 'legacy\runs\baselines'
Move-Logged 'INT2403E\s6e11_optuna' 'legacy\runs\optuna'
Move-Logged 'INT2403E\blend' 'legacy\runs\blend'
Remove-Logged 'INT2403E' 'Empty redundant nested workspace directory'

$dataManifest = [ordered]@{
    source = 'S6E11 competition CSV files verified against the original full Optuna run'
    train_rows = $originalConfig.train_rows; test_rows = $originalConfig.test_rows
    input_files = $originalConfig.input_files
}
$dataManifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath (Safe-Path 'data\manifest.json')
$removedFiles = 0L
$removedBytes = 0L
foreach ($record in $records.ToArray()) {
    if ($record.action -eq 'delete') { $removedFiles += $record.files; $removedBytes += $record.bytes }
}
[pscustomobject]@{
    Moves = @($records | Where-Object { $_.action -eq 'move' }).Count
    RemovedFiles = $removedFiles
    RemovedMB = [math]::Round(($removedBytes / 1MB), 1)
    Journal = $journalPath
}
