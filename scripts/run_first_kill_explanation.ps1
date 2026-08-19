[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ReportDir = "reports\esta_full_m19",
    [int]$PermutationRepeats = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m19_first_kill_explanation `
    --project-root $ProjectRoot `
    --data data\processed\esta_full\first_kill.parquet `
    --model models\esta_full_m17\first_kill_xgboost_tuned.joblib `
    --m18-summary reports\esta_full_m18\m18_summary.json `
    --m17-comparison reports\esta_full_m17\model_comparison.csv `
    --benchmarks benchmarks\external_first_kill_tuned_metrics.csv `
    --report-dir $ReportDir `
    --permutation-repeats $PermutationRepeats `
    --seed 42 `
    --case-features 10 `
    --shap-plot-rows 1500

if ($LASTEXITCODE -ne 0) {
    throw "M19 first-kill explanation stage failed with exit code $LASTEXITCODE"
}
