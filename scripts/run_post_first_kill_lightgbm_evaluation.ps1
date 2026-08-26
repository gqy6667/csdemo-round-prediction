param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [int]$BootstrapSamples = 2000,
    [int]$CalibrationFolds = 5,
    [int]$ReviewCases = 30,
    [switch]$SkipTests,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @(
    "-m", "src.csdemo.m30_post_first_kill_lightgbm_evaluation",
    "--project-root", $ProjectRoot,
    "--data", "data\processed\esta_full\first_kill.parquet",
    "--model", "models\esta_full_m29\post_first_kill_lightgbm_tuned.joblib",
    "--m29-summary", "reports\esta_full_m29\m29_summary.json",
    "--m29-predictions", "reports\esta_full_m29\test_predictions.csv",
    "--benchmarks", "benchmarks\external_first_kill_tuned_metrics.csv",
    "--model-dir", "models\esta_full_m30",
    "--report-dir", "reports\esta_full_m30",
    "--bootstrap-samples", $BootstrapSamples,
    "--calibration-folds", $CalibrationFolds,
    "--review-cases", $ReviewCases
)

if ($SkipTests) {
    $Arguments += "--skip-tests"
}
if ($SkipCompile) {
    $Arguments += "--skip-compile"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "M30 post-first-kill LightGBM evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
