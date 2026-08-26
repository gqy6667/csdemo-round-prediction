[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ReportDir = "reports\esta_full_m31",
    [int]$PermutationRepeats = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m31_post_first_kill_lightgbm_explanation `
    --project-root $ProjectRoot `
    --data data\processed\esta_full\first_kill.parquet `
    --model models\esta_full_m29\post_first_kill_lightgbm_tuned.joblib `
    --m30-summary reports\esta_full_m30\m30_summary.json `
    --m30-predictions reports\esta_full_m30\test_predictions_enriched.csv `
    --m19-report-dir reports\esta_full_m19 `
    --m30-external reports\esta_full_m30\external_benchmark_comparison.csv `
    --m30-external-markdown reports\esta_full_m30\external_benchmark_comparison.md `
    --report-dir $ReportDir `
    --permutation-repeats $PermutationRepeats `
    --seed 42 `
    --case-features 10

if ($LASTEXITCODE -ne 0) {
    throw "M31 post-first-kill LightGBM explanation failed with exit code $LASTEXITCODE"
}
