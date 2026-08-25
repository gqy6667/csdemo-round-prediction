[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$EstaRoot = "C:\project1\data\esta",
    [string]$ReportDir = "reports\esta_full_m27",
    [switch]$RebuildLightGBM,
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

if ($FullRebuild -or $RebuildLightGBM) {
    Invoke-StageScript -Name "run_pre_round_lightgbm_baseline.ps1" -Arguments @{
        Python = $Python
    }
    Invoke-StageScript -Name "run_pre_round_lightgbm_tuning.ps1" -Arguments @{
        Python = $Python
    }
    Invoke-StageScript -Name "run_pre_round_lightgbm_evaluation.ps1" -Arguments @{
        Python = $Python
        BootstrapSamples = $BootstrapSamples
    }
    Invoke-StageScript -Name "run_pre_round_lightgbm_explanation.ps1" -Arguments @{
        Python = $Python
        PermutationRepeats = $PermutationRepeats
    }
    Invoke-StageScript -Name "run_pre_round_lightgbm_interface.ps1" -Arguments @{
        Python = $Python
    }
}

& $Python -m src.csdemo.m27_pre_round_lightgbm_acceptance `
    --project-root $ProjectRoot `
    --report-dir $ReportDir

if ($LASTEXITCODE -ne 0) {
    throw "M27 pre-round LightGBM acceptance failed with exit code $LASTEXITCODE"
}
