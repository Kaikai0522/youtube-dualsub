<#
.SYNOPSIS
    Double-click launcher for youtube_dualsub.

.DESCRIPTION
    Brings up everything the Chrome extension needs, in the order that actually
    matters on this machine, and leaves the backend running in this window:

      1. uv on PATH            (setup.ps1 installs it; this only checks)
      2. no stale backend      (a running server holds the old code in memory)
      3. .venv matches uv.lock (uv sync)
      4. yt-dlp is not stale   (see below)
      5. ollama serve is up    (started hidden if it is not)
      6. the translate model is pulled
      7. the backend answers /api/health

    Closing this window stops the backend. That is the intended way to quit.

.NOTES
    Why step 4 exists: yt-dlp's version *is* a date, and YouTube breaks it every
    few weeks. When it goes stale the symptom is not an obvious error but
    "HTTP Error 403: Forbidden" on every video, hours after the last working
    run. Reading that date costs nothing and needs no network, so the launcher
    upgrades anything older than -YtDlpMaxAgeDays before the backend starts.

    The upgrade happens here and nowhere else. A running Python process holds
    the old yt-dlp module in memory, so upgrading mid-job would rewrite uv.lock
    and fix nothing until the next restart.

.PARAMETER UpdateYtDlp
    Upgrade yt-dlp regardless of its age.

.PARAMETER SkipYtDlpCheck
    Never touch yt-dlp. Use when offline.

.PARAMETER YtDlpMaxAgeDays
    Age at which yt-dlp is upgraded automatically. Default 14.
#>
[CmdletBinding()]
param(
    [switch]$UpdateYtDlp,
    [switch]$SkipYtDlpCheck,
    [int]$YtDlpMaxAgeDays = 14
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK   $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    WARN $msg" -ForegroundColor Yellow }
function Write-Bad($msg)  { Write-Host "    FAIL $msg" -ForegroundColor Red }

function Update-PathFromRegistry {
    # An installer edits the persisted PATH, which this already-running shell
    # never sees. Re-reading it is what makes a freshly installed tool usable
    # without telling the user to open a new terminal.
    $parts = @(
        [Environment]::GetEnvironmentVariable("Path", "Machine")
        [Environment]::GetEnvironmentVariable("Path", "User")
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
        "$env:LOCALAPPDATA\Programs\Ollama"
    ) | Where-Object { $_ }
    $env:PATH = ($parts -join ";")
}

function Test-Command($name) {
    return ($null -ne (Get-Command $name -ErrorAction SilentlyContinue))
}

function Test-Url($url) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Get-YtDlpVersion {
    try {
        $v = & uv run --no-sync python -c "import yt_dlp;print(yt_dlp.version.__version__)"
        if ($LASTEXITCODE -ne 0) { return $null }
        return ("$v").Trim()
    } catch {
        return $null
    }
}

function Get-YtDlpAgeDays($version) {
    # yt-dlp versions are dates: 2026.08.19, or 2026.8.20.234504 for a dev build.
    if (-not $version) { return $null }
    $p = ("$version").Split(".")
    if ($p.Count -lt 3) { return $null }
    try {
        $released = Get-Date -Year ([int]$p[0]) -Month ([int]$p[1]) -Day ([int]$p[2]) -Hour 0 -Minute 0 -Second 0
    } catch {
        return $null
    }
    return [int]((Get-Date) - $released).TotalDays
}

function Update-YtDlp {
    # Only yt-dlp moves. A bare `uv lock --upgrade` would re-resolve torch,
    # faster-whisper and 84 other packages, which is a new set of variables
    # rather than a fix.
    $before = Get-YtDlpVersion
    uv lock --upgrade-package yt-dlp
    if (-not $?) {
        Write-Warn2 "Could not reach PyPI. Continuing with yt-dlp $before."
        return
    }
    uv sync --frozen
    if (-not $?) {
        Write-Bad "uv sync failed after the yt-dlp upgrade."
        return
    }
    $after = Get-YtDlpVersion
    if ($after -ne $before) {
        Write-Ok "yt-dlp $before -> $after"
    } else {
        Write-Ok "yt-dlp $after is already the newest release."
    }
}

Write-Host "youtube_dualsub" -ForegroundColor White
Write-Host $PSScriptRoot -ForegroundColor DarkGray

# ---------------------------------------------------------------- uv -------
Write-Step "Checking uv"
Update-PathFromRegistry
if (-not (Test-Command "uv")) {
    Write-Bad "uv is not installed."
    Write-Host ""
    Write-Host "Run setup.ps1 once first (right-click -> Run with PowerShell)." -ForegroundColor Yellow
    exit 1
}
Write-Ok (uv --version)

# ------------------------------------------------------------ stale server -
Write-Step "Clearing any previous backend"
$stale = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like "*youtube_dualsub.main*" })
if ($stale.Count -gt 0) {
    foreach ($p in $stale) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Ok "Stopped $($stale.Count) old backend process(es)."
} else {
    Write-Ok "Nothing was running."
}

# ---------------------------------------------------------------- deps -----
Write-Step "Syncing dependencies"
uv sync
if (-not $?) {
    Write-Bad "uv sync failed. The messages above say why."
    exit 1
}
Write-Ok "Environment matches uv.lock."

# ---------------------------------------------------------------- yt-dlp ---
if ($SkipYtDlpCheck) {
    Write-Step "Skipping the yt-dlp check (-SkipYtDlpCheck)"
} else {
    Write-Step "Checking yt-dlp"
    $ver = Get-YtDlpVersion
    $age = Get-YtDlpAgeDays $ver
    if (-not $ver) {
        Write-Warn2 "Could not read the installed yt-dlp version; upgrading to be safe."
        Update-YtDlp
    } elseif ($UpdateYtDlp) {
        Write-Host "    yt-dlp $ver, upgrading because -UpdateYtDlp was given"
        Update-YtDlp
    } elseif ($null -ne $age -and $age -gt $YtDlpMaxAgeDays) {
        Write-Warn2 "yt-dlp $ver is $age days old - YouTube usually breaks it by now. Upgrading."
        Update-YtDlp
    } else {
        Write-Ok "yt-dlp $ver ($age days old)."
    }
}

# ---------------------------------------------------------------- config ---
Write-Step "Reading settings"
$cfg = @(& uv run --no-sync python -c "from youtube_dualsub.config import load_settings, LOCAL_CONFIG; s = load_settings(config_path=LOCAL_CONFIG); print(s.translate.model); print(s.server.host); print(s.server.port)")
if ($LASTEXITCODE -ne 0 -or $cfg.Count -lt 3) {
    Write-Bad "Could not read config.local.json / the config defaults."
    exit 1
}
$model = ("$($cfg[0])").Trim()
$httpHost = ("$($cfg[1])").Trim()
$port = ("$($cfg[2])").Trim()
$healthUrl = "http://${httpHost}:${port}/api/health"
Write-Ok "model=$model  server=${httpHost}:${port}"

# ---------------------------------------------------------------- ollama ---
Write-Step "Checking Ollama"
if (Test-Url "http://127.0.0.1:11434/api/version") {
    Write-Ok "Already serving."
} else {
    $exe = $null
    if (Test-Command "ollama") {
        $exe = (Get-Command ollama).Source
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") {
        $exe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    }
    if (-not $exe) {
        Write-Bad "Ollama is not installed. Get it from https://ollama.com/download"
        exit 1
    }
    Write-Host "    not running, starting 'ollama serve' in the background..."
    Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        if (Test-Url "http://127.0.0.1:11434/api/version") { break }
        Start-Sleep -Milliseconds 500
    }
    if (Test-Url "http://127.0.0.1:11434/api/version") {
        Write-Ok "Ollama is up."
    } else {
        Write-Bad "Ollama did not come up within 30s."
        exit 1
    }
}

# ---------------------------------------------------------------- model ----
Write-Step "Checking the translation model"
$installed = (& ollama list) -join "`n"
if ($installed -match [regex]::Escape($model)) {
    Write-Ok "$model is installed."
} else {
    Write-Warn2 "$model is missing. Pulling it now - this is a multi-GB download."
    ollama pull $model
    if (-not $?) {
        Write-Bad "Could not pull $model."
        exit 1
    }
    Write-Ok "Pulled $model."
}

# ---------------------------------------------------------------- backend --
function Start-Backend {
    # --no-sync: uv sync already ran above, and this keeps the backend from
    # silently re-resolving anything on the way up.
    return Start-Process -FilePath "uv" -ArgumentList @("run", "--no-sync", "python", "-m", "youtube_dualsub.main") -NoNewWindow -PassThru
}

function Wait-Healthy($proc, $url, $timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { return $false }
        if (Test-Url $url) { return $true }
        Start-Sleep -Milliseconds 700
    }
    return $false
}

Write-Step "Starting the backend"
$backend = Start-Backend
$healthy = Wait-Healthy $backend $healthUrl 60

if (-not $healthy) {
    # If it will not come up, a stale yt-dlp is the one cause this script can
    # fix by itself. Try it once, then fail honestly rather than looping.
    Write-Bad "The backend did not answer $healthUrl within 60s."
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
    if ($SkipYtDlpCheck) {
        Write-Warn2 "Not retrying: -SkipYtDlpCheck was given."
    } else {
        Write-Step "Retrying once after upgrading yt-dlp"
        Update-YtDlp
        $backend = Start-Backend
        $healthy = Wait-Healthy $backend $healthUrl 60
    }
}

if (-not $healthy) {
    Write-Bad "Still not healthy. The uvicorn output above is the real error."
    Write-Host ""
    Write-Host "Worth checking:" -ForegroundColor Yellow
    Write-Host "  - is something else already using port $port?" -ForegroundColor Yellow
    Write-Host "  - run setup.ps1 to re-verify ffmpeg and the CUDA DLLs" -ForegroundColor Yellow
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
    exit 1
}

Write-Host ""
Write-Host "  Ready - http://${httpHost}:${port}" -ForegroundColor Green
Write-Host "  Open a YouTube video; the extension talks to this window." -ForegroundColor Green
Write-Host "  Closing this window stops the backend." -ForegroundColor DarkGray
Write-Host ""

Wait-Process -Id $backend.Id
