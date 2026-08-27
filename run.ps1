# Windows equivalent of run.sh: creates the .venv, installs the project when it changed, starts the app.
#
#   .\run.ps1            (or double-click run.cmd)
#
# If PowerShell refuses to run scripts:  Set-ExecutionPolicy -Scope Process Bypass

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Stamp = Join-Path $Venv ".deps-installed"

if (-not (Test-Path $Python)) {
    Write-Host "Criando ambiente virtual..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv $Venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $Venv
    } else {
        Write-Error "Python 3 nao encontrado no PATH. Instale com: winget install --id Python.Python.3.12 -e"
        exit 1
    }
}

& $Python -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Inicializando pip no ambiente virtual..."
    & $Python -m ensurepip --upgrade
}

$PyProject = Join-Path $Root "pyproject.toml"
$NeedsInstall = -not (Test-Path $Stamp)
if (-not $NeedsInstall) {
    $NeedsInstall = (Get-Item $PyProject).LastWriteTime -gt (Get-Item $Stamp).LastWriteTime
}
if ($NeedsInstall) {
    Write-Host "Instalando dependencias do projeto..."
    & $Python -m pip install --upgrade pip
    & $Python -m pip install .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    New-Item -ItemType File -Path $Stamp -Force | Out-Null
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg nao encontrado no PATH. A interface abre, mas a exportacao nao vai funcionar. Instale com: winget install --id Gyan.FFmpeg -e"
}

& $Python -m space_video_enhancer @args
exit $LASTEXITCODE
