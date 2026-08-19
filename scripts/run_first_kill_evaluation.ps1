[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ModelDir = "models\esta_full_m18",
    [string]$ReportDir = "reports\esta_full_m18",
    [int]$BootstrapSamples = 2000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m18_first_kill_evaluation `
    --project-root $ProjectRoot `
    --data data\processed\esta_full\first_kill.parquet `
    --model models\esta_full_m17\first_kill_xgboost_tuned.joblib `
    --m17-summary reports\esta_full_m17\m17_summary.json `
    --m17-predictions reports\esta_full_m17\test_predictions.csv `
    --m17-comparison reports\esta_full_m17\model_comparison.csv `
    --benchmarks benchmarks\external_first_kill_tuned_metrics.csv `
    --model-dir $ModelDir `
    --report-dir $ReportDir `
    --bootstrap-samples $BootstrapSamples `
    --seed 42 `
    --folds 5 `
    --review-cases 30

if ($LASTEXITCODE -ne 0) {
    throw "M18 first-kill evaluation stage failed with exit code $LASTEXITCODE"
}
