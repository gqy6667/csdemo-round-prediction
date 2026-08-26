param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [switch]$SkipTests,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @(
    "-m", "src.csdemo.m28_post_first_kill_lightgbm_baseline",
    "--project-root", $ProjectRoot,
    "--data", "data\processed\esta_full\first_kill.parquet",
    "--model-dir", "models\esta_full_m28",
    "--report-dir", "reports\esta_full_m28",
    "--n-bootstrap", "2000"
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
        throw "M28 post-first-kill LightGBM baseline failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
