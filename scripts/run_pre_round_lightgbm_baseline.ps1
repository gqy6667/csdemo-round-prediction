param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @(
    "-m", "src.csdemo.m22_pre_round_lightgbm_baseline",
    "--project-root", $ProjectRoot,
    "--data", "data\processed\esta_full\pre_round.parquet",
    "--model-dir", "models\esta_full_m22",
    "--report-dir", "reports\esta_full_m22"
)

if ($SkipTests) {
    $Arguments += "--skip-tests"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "M22 LightGBM baseline failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
