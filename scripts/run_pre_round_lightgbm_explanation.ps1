[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ReportDir = "reports\esta_full_m25",
    [int]$PermutationRepeats = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m25_pre_round_lightgbm_explanation `
    --project-root $ProjectRoot `
    --data data\processed\esta_full\pre_round.parquet `
    --model models\esta_full_m23\pre_round_lightgbm_tuned.joblib `
    --m24-summary reports\esta_full_m24\m24_summary.json `
    --m24-predictions reports\esta_full_m24\test_predictions_enriched.csv `
    --m12-report-dir reports\esta_full_m12 `
    --m24-external reports\esta_full_m24\external_benchmark_comparison.csv `
    --report-dir $ReportDir `
    --permutation-repeats $PermutationRepeats `
    --seed 42 `
    --case-features 10 `
    --shap-plot-rows 1500

if ($LASTEXITCODE -ne 0) {
    throw "M25 pre-round LightGBM explanation failed with exit code $LASTEXITCODE"
}
