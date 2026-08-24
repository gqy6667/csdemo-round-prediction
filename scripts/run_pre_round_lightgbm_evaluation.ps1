[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ModelDir = "models\esta_full_m24",
    [string]$ReportDir = "reports\esta_full_m24",
    [int]$BootstrapSamples = 2000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m24_pre_round_lightgbm_evaluation `
    --project-root $ProjectRoot `
    --data data\processed\esta_full\pre_round.parquet `
    --kills data\interim\esta_full\kills.parquet `
    --model models\esta_full_m23\pre_round_lightgbm_tuned.joblib `
    --m23-summary reports\esta_full_m23\m23_summary.json `
    --m23-predictions reports\esta_full_m23\test_predictions.csv `
    --benchmarks benchmarks\external_round_model_metrics.csv `
    --model-dir $ModelDir `
    --report-dir $ReportDir `
    --bootstrap-samples $BootstrapSamples `
    --seed 42 `
    --folds 5 `
    --review-cases 30

if ($LASTEXITCODE -ne 0) {
    throw "M24 pre-round LightGBM evaluation failed with exit code $LASTEXITCODE"
}
