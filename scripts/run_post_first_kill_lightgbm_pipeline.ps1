[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$EstaRoot = "C:\project1\data\esta",
    [string]$ReportDir = "reports\esta_full_m33",
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
    Invoke-StageScript -Name "run_first_kill_pipeline.ps1" -Arguments @{
        Python = $Python
        EstaRoot = $EstaRoot
        FullRebuild = $true
        BootstrapSamples = $BootstrapSamples
        PermutationRepeats = $PermutationRepeats
    }
}

if ($FullRebuild -or $RebuildLightGBM) {
    Invoke-StageScript -Name "run_post_first_kill_lightgbm_baseline.ps1" -Arguments @{
        Python = $Python
    }
    Invoke-StageScript -Name "run_post_first_kill_lightgbm_tuning.ps1" -Arguments @{
        Python = $Python
    }
    Invoke-StageScript -Name "run_post_first_kill_lightgbm_evaluation.ps1" -Arguments @{
        Python = $Python
        BootstrapSamples = $BootstrapSamples
    }
    Invoke-StageScript -Name "run_post_first_kill_lightgbm_explanation.ps1" -Arguments @{
        Python = $Python
        PermutationRepeats = $PermutationRepeats
    }
    Invoke-StageScript -Name "run_post_first_kill_lightgbm_interface.ps1" -Arguments @{
        Python = $Python
    }
}

& $Python -m src.csdemo.m33_post_first_kill_lightgbm_acceptance `
    --project-root $ProjectRoot `
    --report-dir $ReportDir

if ($LASTEXITCODE -ne 0) {
    throw "M33 post-first-kill LightGBM acceptance failed with exit code $LASTEXITCODE"
}
