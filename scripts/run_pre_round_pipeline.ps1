[CmdletBinding()]
param(
    [string]$Python = "C:\Users\admin\11\envs\game\python.exe",
    [string]$EstaRoot = "C:\project1\data\esta",
    [string]$ReportDir = "reports\esta_full_m14",
    [switch]$FullRebuild,
    [int]$BootstrapSamples = 2000
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

if ($FullRebuild) {
    if (-not (Test-Path -LiteralPath $EstaRoot -PathType Container)) {
        throw "ESTA dataset root not found: $EstaRoot"
    }

    Invoke-ProjectPython -m src.csdemo.esta_to_tables `
        --input $EstaRoot `
        --output data\interim\esta_full `
        --subsets lan online `
        --format parquet

    Invoke-ProjectPython -m src.csdemo.check_quality `
        --input data\interim\esta_full `
        --report-dir reports\data_quality\esta_full

    Invoke-ProjectPython -m src.csdemo.make_dataset `
        --input data\interim\esta_full `
        --output data\processed\esta_full `
        --format parquet `
        --quality-report-dir reports\data_quality\esta_full

    Invoke-ProjectPython -m src.csdemo.m6_analysis `
        --data data\processed\esta_full\pre_round.parquet `
        --report-dir reports\esta_full_m6

    Invoke-ProjectPython -m src.csdemo.m7_baselines `
        --data data\processed\esta_full\pre_round.parquet `
        --model-dir models\esta_full_m7 `
        --report-dir reports\esta_full_m7

    Invoke-ProjectPython -m src.csdemo.train_xgb `
        --task pre_round `
        --data data\processed\esta_full\pre_round.parquet `
        --model-dir models\esta_full_m8_tuned `
        --report-dir reports\esta_full_m8_tuned

    Invoke-ProjectPython -m src.csdemo.m9_evaluation `
        --data data\processed\esta_full\pre_round.parquet `
        --model models\esta_full_m8_tuned\pre_round_xgb.joblib `
        --report-dir reports\esta_full_m9 `
        --bootstrap-samples $BootstrapSamples `
        --seed 42

    Invoke-ProjectPython -m src.csdemo.m10_calibration `
        --data data\processed\esta_full\pre_round.parquet `
        --base-model models\esta_full_m8_tuned\pre_round_xgb.joblib `
        --model-dir models\esta_full_m10 `
        --report-dir reports\esta_full_m10 `
        --folds 5

    Invoke-ProjectPython -m src.csdemo.m11_robustness `
        --predictions reports\esta_full_m9\test_predictions.csv `
        --data data\processed\esta_full\pre_round.parquet `
        --kills data\interim\esta_full\kills.parquet `
        --report-dir reports\esta_full_m11 `
        --bootstrap-samples $BootstrapSamples `
        --seed 42 `
        --review-cases 30

    Invoke-ProjectPython -m src.csdemo.m12_explanation `
        --data data\processed\esta_full\pre_round.parquet `
        --model models\esta_full_m8_tuned\pre_round_xgb.joblib `
        --report-dir reports\esta_full_m12 `
        --permutation-repeats 20 `
        --seed 42 `
        --case-features 10 `
        --shap-plot-rows 1500

    Invoke-ProjectPython -m src.csdemo.m13_interface `
        --model models\esta_full_m8_tuned\pre_round_xgb.joblib `
        --calibrator models\esta_full_m10\pre_round_calibrator.joblib `
        --json-example examples\pre_round_snapshot.json `
        --csv-example examples\pre_round_snapshot.csv `
        --metrics reports\esta_full_m9\m9_summary.json `
        --benchmarks benchmarks\external_round_model_metrics.csv `
        --report-dir reports\esta_full_m13
}

Invoke-ProjectPython -m src.csdemo.m14_acceptance `
    --project-root $ProjectRoot `
    --esta-root $EstaRoot `
    --report-dir $ReportDir
