[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$ReportDir = "reports\esta_full_m15"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m src.csdemo.m15_first_kill_data `
    --project-root $ProjectRoot `
    --rounds data\interim\esta_full\rounds.parquet `
    --kills data\interim\esta_full\kills.parquet `
    --output data\processed\esta_full\first_kill.parquet `
    --split-manifest reports\esta_full_m14\split_assignments.csv `
    --report-dir $ReportDir `
    --m14-comparison reports\esta_full_m14\external_benchmark_comparison.csv

if ($LASTEXITCODE -ne 0) {
    throw "M15 first-kill data stage failed with exit code $LASTEXITCODE"
}
