[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$EstaRoot = "C:\project1\data\esta",
    [string]$ReportDir = "reports\esta_full_m21",
    [string]$ProgressReport = "reports\m6_to_m21_progress_report.md",
    [switch]$RebuildFirstKill,
    [switch]$FullRebuild,
    [int]$BootstrapSamples = 2000,
    [int]$PermutationRepeats = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

function Invoke-ProjectPython {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PythonArgs
    )

    & $Python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $PythonArgs"
    }
}

function Invoke-StageScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [hashtable]$Arguments = @{}
    )

    $ScriptPath = Join-Path $PSScriptRoot $Name
    & $ScriptPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage script failed with exit code ${LASTEXITCODE}: $Name"
    }
}

if ($FullRebuild) {
    Invoke-StageScript -Name "run_pre_round_pipeline.ps1" -Arguments @{
        Python = $Python
        EstaRoot = $EstaRoot
        FullRebuild = $true
        BootstrapSamples = $BootstrapSamples
    }
}

if ($FullRebuild -or $RebuildFirstKill) {
    Invoke-StageScript -Name "run_first_kill_data_stage.ps1" -Arguments @{
        Python = $Python
    }
    Invoke-StageScript -Name "run_first_kill_baselines.ps1" -Arguments @{
        Python = $Python
    }
    Invoke-StageScript -Name "run_first_kill_tuning.ps1" -Arguments @{
        Python = $Python
    }
    Invoke-StageScript -Name "run_first_kill_evaluation.ps1" -Arguments @{
        Python = $Python
        BootstrapSamples = $BootstrapSamples
    }
    Invoke-StageScript -Name "run_first_kill_explanation.ps1" -Arguments @{
        Python = $Python
        PermutationRepeats = $PermutationRepeats
    }
    Invoke-StageScript -Name "run_first_kill_interface.ps1" -Arguments @{
        Python = $Python
    }
}

Invoke-ProjectPython -m src.csdemo.m21_first_kill_acceptance `
    --project-root $ProjectRoot `
    --esta-root $EstaRoot `
    --report-dir $ReportDir `
    --progress-report $ProgressReport
