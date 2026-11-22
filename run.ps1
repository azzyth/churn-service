<#
.SYNOPSIS
    Task runner untuk churn-service (pengganti Makefile, karena `make` tidak tersedia).

.EXAMPLE
    .\run.ps1 check
    .\run.ps1 data
    .\run.ps1 train
    .\run.ps1 evaluate
    .\run.ps1 drift
    .\run.ps1 test
    .\run.ps1 serve
    .\run.ps1 all
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'check', 'download', 'data', 'train', 'evaluate', 'drift', 'test', 'serve', 'docker', 'all')]
    [string]$Task = 'check'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Pakai venv di parent folder kalau ada, kalau tidak pakai python di PATH.
$py = Join-Path (Split-Path -Parent $root) '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

function Invoke-Step([string]$Label, [scriptblock]$Block) {
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GAGAL: $Label (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Task) {
    'setup'    {
        Invoke-Step 'Install dependencies' { & $py -m pip install -r requirements.txt }
        Invoke-Step 'Install dev dependencies (kaggle CLI untuk unduh dataset)' {
            & $py -m pip install -r requirements-dev.txt
        }
    }
    'check'    { Invoke-Step 'Cek modul + config' { & $py -m src.check } }
    'download' { Invoke-Step 'Unduh dataset' { & $py -m src.data.download } }
    'data'     { Invoke-Step 'Bangun dataset + split + EDA' { & $py -m src.data.make_dataset } }
    'train'    { Invoke-Step 'Latih & bandingkan model' { & $py -m src.train } }
    'evaluate' { Invoke-Step 'Evaluasi + pilih threshold' { & $py -m src.evaluate } }
    'drift'    { Invoke-Step 'Drift report' { & $py -m src.monitor.drift } }
    'test'     { Invoke-Step 'pytest' { & $py -m pytest tests -q } }
    'serve'    { & $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 }
    'docker'   { Invoke-Step 'docker build' { & docker build -t churn-service:0.1.0 . } }
    'all' {
        Invoke-Step 'Cek modul + config' { & $py -m src.check }
        Invoke-Step 'Bangun dataset' { & $py -m src.data.make_dataset }
        Invoke-Step 'Latih model' { & $py -m src.train }
        Invoke-Step 'Evaluasi' { & $py -m src.evaluate }
        Invoke-Step 'Drift report' { & $py -m src.monitor.drift }
        Invoke-Step 'pytest' { & $py -m pytest tests -q }
    }
}
