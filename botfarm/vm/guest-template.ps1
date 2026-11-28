<#
Runs ELEVATED in a template VM (new-bot-vm.ps1 -Template, stage 8T, with Invoke-GuestElevated): the
machine-wide software every bot VM needs, installed quietly. Nothing is signed in. Re-runnable.
  - Visual C++ runtime (tesserocr needs msvcp140.dll)
  - WireGuard (the VPN client for Proton VPN's WireGuard configs; a VM's config goes in per clone)
The Jagex Launcher has no quiet mode: the host drives its installer with key presses (lib
Install-GuestJagexLauncher). Per-user things (git, repos, Python, RuneLite) are not here: each clone
has its own user, and guest-provision.ps1 sets them up there.
Prints one line per item: 'OK <item>' or 'FAIL <item>: <why>'.
#>
param([Parameter(Mandatory = $true)][string]$Drop)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Test-Signed([string]$File, [string]$Org) {
    $sig = Get-AuthenticodeSignature $File
    $sig.Status -eq 'Valid' -and $sig.SignerCertificate.Subject -match "O=$Org"
}

# --- Visual C++ runtime ---
$dll = "$env:SystemRoot\System32\msvcp140.dll"
if (-not (Test-Path $dll)) {
    try {
        $vc = Join-Path $Drop 'vc_redist.x64.exe'
        if (-not (Test-Path $vc)) { Invoke-WebRequest 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $vc -UseBasicParsing }
        if (-not (Test-Signed $vc 'Microsoft Corporation')) { throw 'bad signature' }
        $p = Start-Process $vc -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
        Remove-Item $vc -Force -EA 0
        if (-not (Test-Path $dll)) { throw "installer exit $($p.ExitCode), no msvcp140.dll" }
    } catch { "FAIL vcredist: $($_.Exception.Message)" }
}
if (Test-Path $dll) { 'OK vcredist' }

# --- WireGuard (the VPN: Proton VPN's WireGuard configs, one per VM; none in the template) ---
$wg = 'C:\Program Files\WireGuard\wireguard.exe'
if (-not (Test-Path $wg)) {
    try {
        $list = (Invoke-WebRequest 'https://download.wireguard.com/windows-client/' -UseBasicParsing).Content
        $msiName = [regex]::Matches($list, 'wireguard-amd64-[\d.]+\.msi') | ForEach-Object Value |
            Sort-Object { [version]($_ -replace '^wireguard-amd64-|\.msi$') } | Select-Object -Last 1
        if (-not $msiName) { throw 'no MSI on download.wireguard.com' }
        $msi = Join-Path $Drop $msiName
        Invoke-WebRequest "https://download.wireguard.com/windows-client/$msiName" -OutFile $msi -UseBasicParsing
        if (-not (Test-Signed $msi 'WireGuard LLC')) { throw "${msiName}: bad signature" }
        $p = Start-Process msiexec.exe -ArgumentList '/i', "`"$msi`"", '/qn', '/norestart', 'DO_NOT_LAUNCH=1' -Wait -PassThru
        Remove-Item $msi -Force -EA 0
        if (-not (Test-Path $wg)) { throw "msiexec exit $($p.ExitCode), no wireguard.exe" }
    } catch { "FAIL wireguard: $($_.Exception.Message)" }
}
if (Test-Path $wg) { 'OK wireguard' }
