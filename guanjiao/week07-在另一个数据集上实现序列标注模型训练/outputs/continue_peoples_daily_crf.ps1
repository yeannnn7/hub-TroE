$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\SueGuan\Tools\Anaconda\envs\py312_learn\python.exe"
$BertPath = Join-Path $ProjectRoot "..\..\pretrain models\bert-base-chinese"
$LogDir = Join-Path $ProjectRoot "outputs\logs"
$CkptPath = Join-Path $ProjectRoot "outputs\checkpoints\best_peoples_daily_crf.pt"
$StatusLog = Join-Path $LogDir "peoples_daily_crf_background_status.log"
$EvalValidationLog = Join-Path $LogDir "peoples_daily_crf_eval_validation_stdout.txt"
$EvalTestLog = Join-Path $LogDir "peoples_daily_crf_eval_test_stdout.txt"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Status($Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Add-Content -Path $StatusLog -Value $line -Encoding UTF8
}

Write-Status "Monitor started for existing training PID 48420."

while (Get-Process -Id 48420 -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}

Write-Status "Training PID 48420 finished or is no longer visible."

if (Test-Path $CkptPath) {
    Write-Status "Checkpoint found: $CkptPath"
    & $Python (Join-Path $ProjectRoot "src\evaluate_peoples_daily.py") --use_crf --bert_path $BertPath --split validation 2>&1 |
        Tee-Object -FilePath $EvalValidationLog
    & $Python (Join-Path $ProjectRoot "src\evaluate_peoples_daily.py") --use_crf --bert_path $BertPath --split test 2>&1 |
        Tee-Object -FilePath $EvalTestLog
    Write-Status "Evaluation finished. Logs saved under outputs\logs."
} else {
    Write-Status "No checkpoint found after training process ended: $CkptPath"
}
