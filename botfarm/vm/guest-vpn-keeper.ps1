<#
Runs in a bot VM (as the bot user, from a scheduled task at logon) and keeps
the ExpressVPN connection up: if the VPN isn't connected for a while it
connects again, and after a few failures it tries the next region in the
list. Network Lock (the kill switch) stays on, so the game has no internet
while the VPN is down; this brings it back.

Writes C:\wizard-eyes-provision\vpn.json (state, region, exit IP, seconds
down, failures) every few seconds for the host to read (bot-control status).
#>
param([string]$Region = 'smart',
      [string]$Fallback = '',
      [string]$Status = 'C:\wizard-eyes-provision\vpn.json',
      [int]$DownSeconds = 45)
$e = 'C:\Program Files\ExpressVPN\expressvpnctl.exe'
function Ctl { (& $e @args 2>&1 | Out-String).Trim() }
$regions = @($Region) + @($Fallback -split ',' | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })
$idx = 0; $fails = 0; $downSince = $null; $exit = ''; $exitAt = [datetime]::MinValue; $everUp = $false; $tried = $false

while ($true) {
    $state = Ctl get connectionstate
    $now = Get-Date
    if ($state -eq 'Connected') {
        $downSince = $null; $fails = 0; $everUp = $true
        # the address the world sees (the CLI's pubip is the ISP's): check now and then
        if (-not $exit -or ($now - $exitAt).TotalMinutes -gt 10) {
            try { $exit = (Invoke-RestMethod https://api.ipify.org -TimeoutSec 8).ToString().Trim(); $exitAt = $now } catch { }
        }
    } else {
        $exit = ''
        if (-not $downSince) { $downSince = $now }
        # at start-up connect at once; later, give a dropped tunnel its own chance to come back first
        if ((-not $everUp -and -not $tried) -or ($now - $downSince).TotalSeconds -ge $DownSeconds) {
            $tried = $true
            $fails++
            if ($fails -gt 3 -and $regions.Count -gt 1) { $idx = ($idx + 1) % $regions.Count; $fails = 1 }
            [void](Ctl connect $regions[$idx])
            $downSince = Get-Date       # give it time before judging again
        }
    }
    $o = [ordered]@{
        state = $state; region = (Ctl get region); exit_ip = $exit
        down_seconds = $(if ($downSince) { [int]($now - $downSince).TotalSeconds } else { 0 })
        failures = $fails; wanted = $regions[$idx]; updated = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    }
    try { [IO.File]::WriteAllText($Status, ($o | ConvertTo-Json -Compress)) } catch { }
    Start-Sleep 10
}
