[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ReportDir = "reports\esta_full_m20"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m20_first_kill_interface `
    --project-root $ProjectRoot `
    --model models\esta_full_m17\first_kill_xgboost_tuned.joblib `
    --calibrator models\esta_full_m18\first_kill_calibrator.joblib `
    --json-example examples\first_kill_snapshot.json `
    --csv-example examples\first_kill_snapshot.csv `
    --m18-summary reports\esta_full_m18\m18_summary.json `
    --m19-summary reports\esta_full_m19\m19_summary.json `
    --m17-comparison reports\esta_full_m17\model_comparison.csv `
    --benchmarks benchmarks\external_first_kill_tuned_metrics.csv `
    --report-dir $ReportDir

if ($LASTEXITCODE -ne 0) {
    throw "M20 first-kill prediction interface failed with exit code $LASTEXITCODE"
}
