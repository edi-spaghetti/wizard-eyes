<#
Runs ELEVATED in a bot VM (lib Start-GuestVpn, provider "proton"): puts the VM's Proton VPN WireGuard
config in place as a tunnel service and locks the VM's network to it.
  - the config (copied in by the host as -ConfFile, deleted here) goes to C:\ProgramData\WizardEyes\vpn,
    readable by SYSTEM and administrators only. Its AllowedIPs become 0.0.0.0/1 + 128.0.0.0/1: all IPv4
    through the tunnel, but not "/0", so WireGuard's own kill switch (which would also cut the VM off
    from the host at 10.0.2.2) stays out of it
  - our kill switch instead, Windows Firewall (group "WizardEyes VPN lock"): outbound is blocked by
    default; allowed are traffic from the tunnel's own address, WireGuard to the server's endpoint, the
    host (10.0.2.2) and DHCP. DNS to the NAT's resolver is blocked, so names go through the tunnel. A
    tunnel that is down leaves the VM with no internet, never the real address
  - the tunnel service (WireGuardTunnel$wizardeyes, automatic start, reconnects by itself)
  - waits up to 40 s for a handshake
Prints 'WG-OK <endpoint>' or 'WG-FAILED: <why>'. Re-runnable (a new config replaces the old).
#>
param([Parameter(Mandatory = $true)][string]$ConfFile, [string]$Tunnel = 'wizardeyes', [string]$HostIp = '10.0.2.2')

$ErrorActionPreference = 'Stop'
$wgExe = 'C:\Program Files\WireGuard\wireguard.exe'
$wgCli = 'C:\Program Files\WireGuard\wg.exe'
$dir = 'C:\ProgramData\WizardEyes\vpn'
$group = 'WizardEyes VPN lock'
try {
    if (-not (Test-Path $wgExe)) { throw 'WireGuard is not installed' }
    $text = [IO.File]::ReadAllText($ConfFile)
    Remove-Item $ConfFile -Force
    $address = if ($text -match '(?m)^\s*Address\s*=\s*([\d.]+)') { $Matches[1] } else { throw 'no IPv4 Address in the config' }
    if ($text -notmatch '(?m)^\s*Endpoint\s*=\s*([\d.]+):(\d+)\s*$') { throw 'the config needs an Endpoint = <IPv4>:<port>' }
    $epIp = $Matches[1]; $epPort = $Matches[2]
    if ($text -notmatch '(?m)^\s*PrivateKey\s*=') { throw 'no PrivateKey in the config' }
    $text = [regex]::Replace($text, '(?m)^(\s*AllowedIPs\s*=)[^\r\n]*', '$1 0.0.0.0/1, 128.0.0.0/1')
    # (IPv6 is left out: the VM's NAT has none, and the lock blocks it)
    $text = [regex]::Replace($text, '(?m)^(\s*Address\s*=\s*)([^,\r\n]+)[^\r\n]*', '$1$2')

    # the config, for SYSTEM and administrators only
    New-Item -ItemType Directory -Force $dir | Out-Null
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($who in 'NT AUTHORITY\SYSTEM', 'BUILTIN\Administrators') {
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($who, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')))
    }
    Set-Acl $dir $acl
    $conf = Join-Path $dir "$Tunnel.conf"
    [IO.File]::WriteAllText($conf, $text, (New-Object Text.UTF8Encoding($false)))

    # the lock
    Remove-NetFirewallRule -Group $group -EA SilentlyContinue
    $r = @{ Group = $group; Direction = 'Outbound'; Action = 'Allow'; Profile = 'Any' }
    New-NetFirewallRule @r -DisplayName 'WizardEyes VPN: through the tunnel' -LocalAddress $address | Out-Null
    New-NetFirewallRule @r -DisplayName 'WizardEyes VPN: to the server' -RemoteAddress $epIp -Protocol UDP -RemotePort $epPort | Out-Null
    New-NetFirewallRule @r -DisplayName 'WizardEyes VPN: the host' -RemoteAddress $HostIp | Out-Null
    New-NetFirewallRule @r -DisplayName 'WizardEyes VPN: DHCP' -Protocol UDP -LocalPort 68 -RemotePort 67 | Out-Null
    Set-NetFirewallProfile -All -Enabled True -DefaultOutboundAction Block
    # Windows' own outbound allow rules (Core Networking DNS, store apps, ...) would get past the block:
    # with the tunnel down, names still resolved through the home router (vm5, 2026-11-28). Off, all of
    # them: only the rules above let anything out
    $own = @(Get-NetFirewallRule -Direction Outbound -Action Allow -Enabled True -EA SilentlyContinue | Where-Object { $_.Group -ne $group })
    $own | Disable-NetFirewallRule -EA SilentlyContinue
    "lock: $($own.Count) of Windows' outbound allow rules switched off"

    # the tunnel service
    if (Get-Service "WireGuardTunnel`$$Tunnel" -EA SilentlyContinue) {
        & $wgExe /uninstalltunnelservice $Tunnel
        for ($i = 0; $i -lt 20 -and (Get-Service "WireGuardTunnel`$$Tunnel" -EA SilentlyContinue); $i++) { Start-Sleep 1 }
    }
    & $wgExe /installtunnelservice $conf
    # (wg.exe says "no such interface" on stderr until the tunnel is up: under 'Stop', PowerShell 5.1
    # would make that a terminating error)
    $ErrorActionPreference = 'Continue'
    $hs = 0
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep 1
        $line = (& $wgCli show $Tunnel latest-handshakes 2>$null) -join ' '
        if ($line -match '\s(\d+)\s*$' -and [long]$Matches[1] -gt 0) { $hs = [long]$Matches[1]; break }
    }
    if (-not $hs) { throw "no handshake with $epIp`:$epPort in 40 s" }
    "WG-OK ${epIp}:$epPort"
} catch {
    "WG-FAILED: $($_.Exception.Message)"
}
