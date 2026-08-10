<#
.SYNOPSIS
    One-shot environment setup for youtube_dualsub.

.DESCRIPTION
    Installs uv, ffmpeg and the Python dependencies, then verifies the three
    things that actually break on Windows:
      1. ffmpeg on PATH
      2. the cuDNN 9 / cuBLAS DLLs CTranslate2 needs
      3. an Ollama model to translate with

    Safe to re-run.

.PARAMETER WithVocals
    Also install Demucs + CUDA torch (~3 GB download) for vocal isolation.

.EXAMPLE
    .\setup.ps1 -WithVocals
#>
[CmdletBinding()]
param(
    [switch]$WithVocals,
    [string]$OllamaModel = "gemma4:12b"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$problems = New-Object System.Collections.Generic.List[string]

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK   $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    WARN $msg" -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "    FAIL $msg" -ForegroundColor Red }

function Test-Command($name) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    return ($null -ne $c)
}

function Update-PathFromRegistry {
    # winget edits the persisted PATH, which this already-running shell never
    # sees. Re-reading it from the registry is what makes a freshly installed
    # tool usable without telling the user to open a new terminal.
    $parts = @(
        [Environment]::GetEnvironmentVariable("Path", "Machine")
        [Environment]::GetEnvironmentVariable("Path", "User")
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
    ) | Where-Object { $_ }
    $env:PATH = ($parts -join ";")
}

# ---------------------------------------------------------------- GPU ------
Write-Step "Checking GPU"
if (Test-Command "nvidia-smi") {
    $gpu = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | Select-Object -First 1
    Write-Ok $gpu
} else {
    Write-Bad "nvidia-smi not found. An NVIDIA GPU with a current driver is required."
    $problems.Add("No NVIDIA driver detected.")
}

# ---------------------------------------------------------------- uv -------
Write-Step "Checking uv"
if (Test-Command "uv") {
    Write-Ok (uv --version)
} else {
    Write-Warn2 "uv not found, installing via winget..."
    winget install --id astral-sh.uv --accept-source-agreements --accept-package-agreements
    Update-PathFromRegistry
    if (Test-Command "uv") {
        Write-Ok (uv --version)
    } else {
        Write-Bad "uv still not on PATH. Open a new terminal and re-run this script."
        exit 1
    }
}

# ---------------------------------------------------------------- ffmpeg ---
Write-Step "Checking ffmpeg"
if (Test-Command "ffmpeg") {
    Write-Ok ((ffmpeg -version) | Select-Object -First 1)
} else {
    Write-Warn2 "ffmpeg not found, installing via winget..."
    winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    Update-PathFromRegistry
    if (Test-Command "ffmpeg") {
        Write-Ok "ffmpeg installed."
    } else {
        Write-Warn2 "ffmpeg installed but not yet on PATH - open a NEW terminal before running the pipeline."
        $problems.Add("ffmpeg needs a new terminal session to appear on PATH.")
    }
}

# ---------------------------------------------------------------- deps -----
Write-Step "Installing Python dependencies (uv sync)"
if ($WithVocals) {
    Write-Host "    including the 'vocals' extra (Demucs + CUDA torch, large download)"
    uv sync --extra vocals
} else {
    uv sync
}
if (-not $?) {
    Write-Bad "uv sync failed."
    exit 1
}
Write-Ok "Dependencies installed."

# ---------------------------------------------------------------- cuDNN ----
Write-Step "Checking CUDA runtime DLLs for CTranslate2"
$cudnn = uv run python -c "from youtube_dualsub._cuda import diagnose; ok, msgs = diagnose(); print('\n'.join(msgs)); raise SystemExit(0 if ok else 1)"
Write-Host ($cudnn -join [Environment]::NewLine)
if ($LASTEXITCODE -eq 0) {
    Write-Ok "cuDNN 9 and cuBLAS are available."
} else {
    Write-Bad "Missing CUDA runtime DLLs - see the lines above for the exact fix."
    $problems.Add("CTranslate2 will fail until the missing nvidia-* wheels are installed.")
}

# ---------------------------------------------------------------- ollama ---
Write-Step "Checking Ollama"
if (Test-Command "ollama") {
    $models = (ollama list) -join "`n"
    $shortName = $OllamaModel.Split(":")[0]
    if ($models -match [regex]::Escape($OllamaModel)) {
        Write-Ok "Model '$OllamaModel' is present."
    } else {
        Write-Warn2 "Model '$OllamaModel' not found, pulling..."
        ollama pull $OllamaModel
        if ($?) { Write-Ok "Pulled $OllamaModel." } else { $problems.Add("Could not pull $OllamaModel.") }
    }
} else {
    Write-Bad "ollama not found. Install from https://ollama.com/download"
    $problems.Add("Ollama is required for translation.")
}

# ---------------------------------------------------------------- summary --
Write-Host ""
if ($problems.Count -eq 0) {
    Write-Host "Setup complete." -ForegroundColor Green
    Write-Host ""
    Write-Host "Try it:" -ForegroundColor White
    Write-Host "  uv run dualsub IqcS1d3eXYc --export srt"
    Write-Host "  uv run uvicorn youtube_dualsub.main:app --port 8756"
} else {
    Write-Host "Setup finished with $($problems.Count) problem(s):" -ForegroundColor Yellow
    foreach ($p in $problems) { Write-Host "  - $p" -ForegroundColor Yellow }
    exit 1
}
