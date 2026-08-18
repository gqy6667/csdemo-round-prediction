[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ModelDir = "models\esta_full_m16",
    [string]$ReportDir = "reports\esta_full_m16"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m16_first_kill_baselines `
    --project-root $ProjectRoot `
    --data data\processed\esta_full\first_kill.parquet `
    --m15-summary reports\esta_full_m15\m15_summary.json `
    --benchmarks benchmarks\external_first_kill_metrics.csv `
    --model-dir $ModelDir `
    --report-dir $ReportDir

if ($LASTEXITCODE -ne 0) {
    throw "M16 first-kill baseline stage failed with exit code $LASTEXITCODE"
}
