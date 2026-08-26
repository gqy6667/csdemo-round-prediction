param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [switch]$SkipTests,
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @(
    "-m", "src.csdemo.m29_post_first_kill_lightgbm_tuning",
    "--project-root", $ProjectRoot,
    "--data", "data\processed\esta_full\first_kill.parquet",
    "--model-dir", "models\esta_full_m29",
    "--report-dir", "reports\esta_full_m29"
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
        throw "M29 post-first-kill LightGBM tuning failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
