<#
.SYNOPSIS
  Starts the training site on a Windows PC (Docker Engine inside WSL) and shows the link to share.

.DESCRIPTION
  For running this folder's Docker stack on Windows instead of a VM. Run it after every restart,
  or double-click start-on-windows.cmd next to it. It:
    1. starts WSL in the background and keeps it running (WSL otherwise stops when idle,
       taking the containers with it),
    2. waits until https://localhost/train answers,
    3. starts the quick tunnel (unless -NoTunnel) and waits for its trycloudflare.com link,
    4. prints the link and the access code, and copies the link to the clipboard.

  Setup comes first: WSL with Ubuntu, Docker Engine inside it, .env filled in, and one
  "docker compose up -d --build" (see README.md).

.PARAMETER Distro
  The WSL distribution that has Docker. Default: Ubuntu-22.04.

.PARAMETER NoTunnel
  Do not start the quick tunnel; only this PC can reach the site.

.PARAMETER StopTunnel
  Take the public link offline and exit. The site keeps running on this PC.
#>
param(
    [string]$Distro = "Ubuntu-22.04",
    [switch]$NoTunnel,
    [switch]$StopTunnel
)

# Not "Stop": in Windows PowerShell 5.1, anything docker writes to stderr (such as compose's
# progress lines) would then abort the script. Failures are checked explicitly instead.
$ErrorActionPreference = "Continue"
$Here = $PSScriptRoot

function Compose([string[]]$Arguments) {
    # Runs "docker compose" in this folder inside WSL, as root so the docker group does not matter.
    & wsl.exe -d $Distro --cd $Here -u root -- docker compose @Arguments
}

if (-not (Test-Path (Join-Path $Here ".env"))) {
    throw "No .env in $Here. Copy .env.example to .env and fill it in first (see README.md)."
}

if ($StopTunnel) {
    Compose @("--profile", "quick-tunnel", "stop", "quick-tunnel") | Out-Null
    Write-Host "Public link is offline. The site still runs at https://localhost/train" -ForegroundColor Yellow
    exit 0
}

# 1. Keep WSL running. One hidden "sleep infinity" is enough; do not start a second.
$keeper = Get-CimInstance Win32_Process -Filter "name='wsl.exe'" |
    Where-Object { $_.CommandLine -match "sleep infinity" -and $_.CommandLine -match [regex]::Escape($Distro) }
if ($keeper) {
    Write-Host "WSL is already being kept running."
} else {
    Write-Host "Starting WSL ($Distro) in the background..."
    Start-Process wsl.exe -ArgumentList "-d", $Distro, "-u", "root", "--exec", "sleep", "infinity" -WindowStyle Hidden
}

# 2. Wait for the site. curl.exe ships with Windows; -k because Caddy's certificate is self-signed.
Write-Host "Waiting for https://localhost/train ..."
$up = $false
for ($i = 0; $i -lt 60; $i++) {
    $code = & curl.exe -sk -o NUL -w "%{http_code}" --max-time 5 https://localhost/train 2>$null
    if ($code -eq "200") { $up = $true; break }
    Start-Sleep -Seconds 3
}
if (-not $up) {
    Write-Host "The site did not come up. Containers:" -ForegroundColor Red
    Compose @("--profile", "quick-tunnel", "ps", "-a")
    throw "Check the containers above; 'docker compose logs server' inside WSL shows server errors."
}
Write-Host "Site is up: https://localhost/train" -ForegroundColor Green

$code = (Get-Content (Join-Path $Here ".env") | Where-Object { $_ -match "^TRAINING_ACCESS_CODE=" }) -replace "^TRAINING_ACCESS_CODE=", ""

if ($NoTunnel) {
    Write-Host "Access code: $code"
    exit 0
}

# 3. Start the quick tunnel (a no-op if it is already running) and wait for its link.
Write-Host "Starting the quick tunnel..."
Compose @("--profile", "quick-tunnel", "up", "-d", "quick-tunnel") 2>$null | Out-Null
# The container's log keeps links from earlier starts, so read only since its latest start.
$id = (Compose @("ps", "-q", "quick-tunnel") | Select-Object -First 1)
if (-not $id) { throw "The quick-tunnel container is not running. Inside WSL: docker compose --profile quick-tunnel up -d" }
$since = (& wsl.exe -d $Distro -u root -- docker inspect -f "{{.State.StartedAt}}" $id.Trim()) | Select-Object -First 1
$link = $null
for ($i = 0; $i -lt 45; $i++) {
    $log = (Compose @("logs", "--no-log-prefix", "--since", $since.Trim(), "quick-tunnel") 2>&1) -join "`n"
    $found = [regex]::Matches($log, "https://[a-z0-9-]+\.trycloudflare\.com")
    if ($found.Count -gt 0 -and $log -match "Registered tunnel connection") {
        $link = $found[$found.Count - 1].Value
        break
    }
    Start-Sleep -Seconds 2
}
if (-not $link) {
    throw "The quick tunnel did not report a link. Inside WSL: docker compose logs quick-tunnel"
}

# Cloudflare can take a few seconds before a new link answers.
for ($i = 0; $i -lt 15; $i++) {
    $status = & curl.exe -s -o NUL -w "%{http_code}" --max-time 8 "$link/train" 2>$null
    if ($status -eq "200") { break }
    Start-Sleep -Seconds 2
}

Set-Clipboard -Value "$link"
Write-Host ""
Write-Host "Share this with your friend:" -ForegroundColor Green
Write-Host "  Link:        $link   (copied to the clipboard)"
Write-Host "  Access code: $code"
Write-Host ""
Write-Host "The link changes whenever the tunnel restarts. To take it offline:"
Write-Host "  .\start-on-windows.ps1 -StopTunnel"
