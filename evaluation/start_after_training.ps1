param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][int]$TrainingProcessId,
    [Parameter(Mandatory=$true)][string]$RunDir,
    [Parameter(Mandatory=$true)][string]$Spec,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Output "Waiting for training process $TrainingProcessId to exit cleanly."
while (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
    # Read-only polling works across the process permission boundary where
    # Wait-Process cannot open the existing training synchronization handle.
    Start-Sleep -Seconds 10
}
# The stop watcher may write its receipt just after the process exits.
for ($attempt=0; $attempt -lt 30; $attempt++) {
    if (Test-Path -LiteralPath (Join-Path $RunDir 'stop_at_2000_receipt.json')) { break }
    Start-Sleep -Seconds 2
}
Write-Output 'Checking saved iteration, then GPU preflight and league.'
& $Python evaluation/after_training.py --run-dir $RunDir --spec $Spec --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Evaluation stopped on verification/runtime failure. See launch_status.json; training was not resumed.' }
