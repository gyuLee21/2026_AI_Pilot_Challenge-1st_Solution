param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][int]$PreviousProcessId,
    [Parameter(Mandatory=$true)][string]$PreviousOutput,
    [Parameter(Mandatory=$true)][string]$Spec,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][string]$BenchmarkOutput
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
# Wait only for the preceding job; its report, not process exit, proves success.
if (Get-Process -Id $PreviousProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $PreviousProcessId
}
$progress = Get-Content -LiteralPath (Join-Path $PreviousOutput 'progress.json') -Raw | ConvertFrom-Json
$report = Get-Content -LiteralPath (Join-Path $PreviousOutput 'report.json') -Raw | ConvertFrom-Json
if (-not $progress.complete -or -not $report.complete) {
    throw 'Preceding league did not finish; head-on was not started.'
}
# Benchmark without the preceding GPU workload before choosing concurrency.
& $Python evaluation/parallel_tournament.py --spec $Spec --output $BenchmarkOutput --benchmark-only
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed; head-on was not started.' }
& $Python evaluation/parallel_tournament.py --spec $Spec --output $Output --benchmark-result (Join-Path $BenchmarkOutput 'benchmark.json')
if ($LASTEXITCODE -ne 0) { throw 'Head-on evaluation failed; completed pairs remain resumable.' }
