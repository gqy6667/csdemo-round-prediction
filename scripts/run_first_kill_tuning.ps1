[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ModelDir = "models\esta_full_m17",
    [string]$ReportDir = "reports\esta_full_m17"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m17_first_kill_tuning `
    --project-root $ProjectRoot `
    --data data\processed\esta_full\first_kill.parquet `
    --m16-summary reports\esta_full_m16\m16_summary.json `
    --m16-comparison reports\esta_full_m16\m16_model_comparison.csv `
    --m16-predictions reports\esta_full_m16\test_predictions.csv `
    --m16-encoded-columns reports\esta_full_m16\encoded_feature_columns.csv `
    --benchmarks benchmarks\external_first_kill_tuned_metrics.csv `
    --model-dir $ModelDir `
    --report-dir $ReportDir

if ($LASTEXITCODE -ne 0) {
    throw "M17 first-kill tuning stage failed with exit code $LASTEXITCODE"
}
