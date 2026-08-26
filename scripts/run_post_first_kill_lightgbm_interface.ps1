[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ReportDir = "reports\esta_full_m32"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m32_post_first_kill_lightgbm_interface `
    --project-root $ProjectRoot `
    --model models\esta_full_m29\post_first_kill_lightgbm_tuned.joblib `
    --calibrator models\esta_full_m30\post_first_kill_lightgbm_calibrator.joblib `
    --json-example examples\first_kill_snapshot.json `
    --csv-example examples\first_kill_snapshot.csv `
    --example-output examples\first_kill_lightgbm_prediction_output.json `
    --m30-summary reports\esta_full_m30\m30_summary.json `
    --m31-summary reports\esta_full_m31\m31_summary.json `
    --m31-external reports\esta_full_m31\external_benchmark_comparison.csv `
    --m31-external-markdown reports\esta_full_m31\external_benchmark_comparison.md `
    --report-dir $ReportDir

if ($LASTEXITCODE -ne 0) {
    throw "M32 post-first-kill LightGBM interface failed with exit code $LASTEXITCODE"
}
