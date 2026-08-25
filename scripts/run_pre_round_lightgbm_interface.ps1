[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ReportDir = "reports\esta_full_m26"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m26_pre_round_lightgbm_interface `
    --project-root $ProjectRoot `
    --model models\esta_full_m23\pre_round_lightgbm_tuned.joblib `
    --calibrator models\esta_full_m24\pre_round_lightgbm_calibrator.joblib `
    --json-example examples\pre_round_snapshot.json `
    --csv-example examples\pre_round_snapshot.csv `
    --example-output examples\pre_round_lightgbm_prediction_output.json `
    --m24-summary reports\esta_full_m24\m24_summary.json `
    --m25-summary reports\esta_full_m25\m25_summary.json `
    --m25-external reports\esta_full_m25\external_benchmark_comparison.csv `
    --report-dir $ReportDir

if ($LASTEXITCODE -ne 0) {
    throw "M26 pre-round LightGBM interface failed with exit code $LASTEXITCODE"
}
