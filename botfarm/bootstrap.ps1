<#
.SYNOPSIS
    First thing to run on a bare Windows PC: git, a GitHub SSH key, the engine
    and the farm's repo cloned, then the farm's setup (setup.ps1).

.DESCRIPTION
    Safe to re-run; each step checks first:
      1. git (winget)
      2. an SSH key that GitHub accepts: made if missing; if GitHub doesn't
         know it yet, the public key goes on the clipboard and GitHub's
         "add SSH key" page opens; it waits until GitHub accepts it
      3. the engine repo and the farm's repo cloned under -Root (or left as
         they are when already there)
      4. botfarm\setup.ps1 -Farm <farm folder>

    Straight from GitHub, in a PowerShell window:

      & ([scriptblock]::Create((irm https://raw.githubusercontent.com/edi-spaghetti/wizard-eyes/main/botfarm/bootstrap.ps1))) `
          -FarmRepo git@github.com:OWNER/REPO.git -FarmFolder botfarm/FARM

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File bootstrap.ps1 -FarmRepo git@github.com:me/my-bots.git -FarmFolder botfarm/myfarm
#>
[CmdletBinding()]
param(
    # the bots' repo that holds the farm (SSH url)
    [Parameter(Mandatory = $true)][string]$FarmRepo,
    # the farm's folder inside that repo (with farm.json)
    [Parameter(Mandatory = $true)][string]$FarmFolder,
    [string]$EngineRepo = 'git@github.com:edi-spaghetti/wizard-eyes.git',
    # where the repos go
    [string]$Root = (Join-Path $env:USERPROFILE 'wizard-eyes'),
    [string]$Key = (Join-Path $env:USERPROFILE '.ssh\id_ed25519'),
    # only check and report
    [switch]$Look,
    # stop after cloning
    [switch]$NoSetup
)

$ErrorActionPreference = 'Stop'
function Write-Step([string]$t) { Write-Host "`n== $t" -ForegroundColor Cyan }
function Write-Ok([string]$t) { Write-Host "   OK   $t" -ForegroundColor Green }
function Write-Todo([string]$t) { Write-Host "   TODO $t" -ForegroundColor Yellow }
function Write-Info([string]$t) { Write-Host "        $t" }

# ssh -T git@github.com: exit 1 with "successfully authenticated" when the key works
function Test-GitHub {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $out = cmd /c 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -T git@github.com 2>&1'
    $ErrorActionPreference = $old
    ($out -join ' ') -match 'successfully authenticated'
}

function Update-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

Write-Step '1. git'
if (Get-Command git -ErrorAction SilentlyContinue) { Write-Ok (git --version) }
elseif ($Look) { Write-Todo 'git is not installed' }
else {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'winget is missing: install "App Installer" from the Microsoft Store, or git from https://git-scm.com, then run this again'
    }
    winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements --silent
    Update-Path
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git still not found; open a new PowerShell window and run this again' }
    Write-Ok (git --version)
}

Write-Step '2. GitHub SSH key'
if (Test-GitHub) { Write-Ok 'GitHub accepts this PC''s SSH key' }
elseif ($Look) { Write-Todo 'GitHub does not accept an SSH key from this PC yet' }
else {
    if (-not (Test-Path $Key)) {
        New-Item -ItemType Directory -Force (Split-Path $Key) | Out-Null
        # (PowerShell 5.1 drops an empty '' argument: '""' passes an empty passphrase)
        ssh-keygen -t ed25519 -N '""' -C "$env:USERNAME@$env:COMPUTERNAME wizard-eyes" -f $Key -q
        if (-not (Test-Path "$Key.pub")) { throw 'ssh-keygen failed' }
        Write-Ok "new key $Key"
    }
    Get-Content "$Key.pub" -Raw | Set-Clipboard
    Start-Process 'https://github.com/settings/ssh/new'
    Write-Host '   GitHub''s "Add new SSH key" page is open and the public key is on the clipboard:' -ForegroundColor Yellow
    Write-Host "   give it a title (e.g. $env:COMPUTERNAME), paste it into Key, press Add SSH key." -ForegroundColor Yellow
    $end = (Get-Date).AddMinutes(15)
    while (-not (Test-GitHub)) {
        if ((Get-Date) -gt $end) { throw 'GitHub still does not accept the key; run this again once it is added' }
        Start-Sleep 10
    }
    Write-Ok 'GitHub accepts the key'
}

Write-Step '3. Repos'
$dirs = @{}
foreach ($url in $EngineRepo, $FarmRepo) {
    $name = [IO.Path]::GetFileNameWithoutExtension(($url -split '[/:]')[-1])
    $dir = Join-Path $Root $name
    $dirs[$url] = $dir
    if (Test-Path (Join-Path $dir '.git')) { Write-Ok "$name at $dir"; continue }
    if ($Look) { Write-Todo "$name would be cloned to $dir"; continue }
    New-Item -ItemType Directory -Force $Root | Out-Null
    git clone -q $url $dir
    if ($LASTEXITCODE -ne 0) { throw "git clone $url failed" }
    Write-Ok "$name cloned to $dir"
}
$engine = $dirs[$EngineRepo]
$farm = Join-Path $dirs[$FarmRepo] $FarmFolder
if (-not $Look -and -not (Test-Path (Join-Path $farm 'farm.json'))) { throw "No farm.json in $farm (check -FarmFolder)" }

Write-Step '4. Farm setup'
$setup = Join-Path $engine 'botfarm\setup.ps1'
if ($NoSetup) { Write-Info "skipped (-NoSetup); next: powershell -ExecutionPolicy Bypass -File `"$setup`" -Farm `"$farm`""; return }
$a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $setup, '-Farm', $farm)
if ($Look) { if (Test-Path $setup) { $a += '-Look' } else { Write-Todo 'setup.ps1 not there yet'; return } }
& powershell.exe @a
exit $LASTEXITCODE
