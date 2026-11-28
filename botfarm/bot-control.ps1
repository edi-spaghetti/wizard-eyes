<#
Start / restart / stop / look at one bot of a farm, from any state (VM off,
Windows not logged in yet, client closed). The farm's page runs it for its
buttons and its watchdog:

    bot-control.ps1 -Role cleaner -Action start [-Farm <folder>]

The bot itself opens the launcher, logs in and gets going. Progress goes to
-Log (a line per step; the last one starts with OK or FAILED). -Action status
changes nothing: one line of JSON (VM on, bot running, its log's age and last
lines; -Shot also saves a screenshot of the VM).
#>
param([Parameter(Mandatory)][string]$Role,
      [Parameter(Mandatory)][ValidateSet('start', 'restart', 'stop', 'status')][string]$Action,
      [string]$Log, [string]$Shot, [string]$Farm)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm
$Bot = $fm.Roles[$Role]
if (-not $Bot) { throw "No role '$Role' in $($fm.Name) (roles: $($fm.Roles.Keys -join ', '))" }
$n = $Bot.Vm
$Coordination = $fm.Def.coordination

function Say($text) {
    $line = "{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $text
    if ($Log) { Add-Content -Path $Log -Value $line -Encoding utf8 } else { $line }
}
function Reported { # seconds since the bot last reported, huge if never
    try {
        $r = Invoke-RestMethod "$Coordination/state" -TimeoutSec 4
        $b = $r.bots.$Role      # its own report or the engine's heartbeat
        $last = [Math]::Max([double]$(if ($b.updated) { $b.updated } else { 0 }), [double]$(if ($b.alive) { $b.alive } else { 0 }))
        [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $last
    } catch { 1e9 }
}
function Get-LogTail([int]$Lines = 1) {
    # the log's last lines, without the progress backspaces
    ((Invoke-Guest $n ("`$t = Get-Content '$GuestDrop\$($Bot.Log)' -Raw -EA 0; if (`$t) { (`$t -split '[\r\n\x08]+' | ? { `$_.Trim() -and `$_ -notmatch 'Downloaded' } | select -last $Lines) }") 60 -AllowFail) -join "`n").Trim()
}

try {
    if ($Log) { Set-Content -Path $Log -Value '' -Encoding utf8 }
    $running = Test-VmRunning $n

    if ($Action -eq 'status') {
        $look = @{ vm = [bool]$running; running = $false; log_age = 0; lines = @() }
        if ($running) {
            if ($Shot) { Invoke-VBox @('controlvm', $n, 'screenshotpng', $Shot) -AllowFail | Out-Null }
            Use-VmPassword {
                $out = Invoke-Guest $n ("`$o = @{ running = (Get-ScheduledTask '$($Bot.Task)' -EA 0).State -eq 'Running'; log_age = 0; lines = @() }; `$p = '$GuestDrop\$($Bot.Log)'; " +
                    'if (Test-Path $p) { $o.log_age = [int]((Get-Date) - (Get-Item $p).LastWriteTime).TotalSeconds; ' +
                    '$f = [IO.File]::Open($p, "Open", "Read", "ReadWrite"); [void]$f.Seek(-[Math]::Min(12000, $f.Length), "End"); $t = (New-Object IO.StreamReader($f)).ReadToEnd(); $f.Close(); ' +
                    '$o.lines = @($t -replace "\x08", "" -replace "Time (\d+h)", "`n`$1" -replace "Sleeping @", "`nSleeping @" -split "[\r\n]+" | ? { $_.Trim() } | select -Last 40 | % { ($_ -replace "[^\x20-\x7e]", "?").Trim() }) }; ' +
                    '"LOOK" + ($o | ConvertTo-Json -Compress)') 60 -AllowFail
                $line = @($out | Where-Object { $_ -like 'LOOK{*' })[0]
                if (-not $line) { throw "no answer from $n" }
                $g = $line.Substring(4) | ConvertFrom-Json
                $look.running = $g.running; $look.log_age = $g.log_age; $look.lines = @($g.lines)
                if ($Bot.Vpn) { $look.vpn = Get-GuestVpnStatus $n }
            }
        }
        $look | ConvertTo-Json -Compress
        exit
    }

    # what the farm's watchdog should keep it doing
    New-Item -ItemType Directory -Force $fm.StateDir | Out-Null
    Set-Content -Path (Join-Path $fm.StateDir "wanted-$Role.local") -Value $(if ($Action -eq 'stop') { 'stop' } else { 'run' }) -Encoding ascii

    if ($Action -eq 'stop' -and -not $running) { Say "OK $n is off"; exit }

    if ($Action -ne 'stop') {
        [void](Get-Account $fm $Role)       # (fails early when it isn't set)
        # the same account must not play in two places: not on this PC as well
        $leaf = [regex]::Escape((Split-Path $Bot.Script -Leaf))
        $here = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $leaf }
        if ($here) { throw "the $Role is running on this PC as well: stop it first" }
        if (-not $running) {
            Say "starting $n"
            Start-FarmVm $n         # pinned to the fast cores (only possible at launch)
            Start-Sleep 8
        }
        Say 'waiting for Windows to be logged in'
        Wait-VmLoggedIn $n 6
        if (-not $running) { Start-Sleep 25 }       # the desktop settles
        $open = Get-Process VirtualBoxVM -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "$n*" }
        if (-not $open) { Show-VmWindow $n; Start-Sleep 12 }
        # moving the mouse over the VM's window mustn't move the bot's (a click captures it,
        # Right Ctrl frees it); VirtualBox forgets this when the window closes
        try { Say (powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'vm\vm-mouse.ps1') -Name $n -State off 2>&1 | Select-Object -First 1) }
        catch { Say "mouse integration: $($_.Exception.Message)" }
    }

    Use-VmPassword {
        # until the VM really answers (the first tries after it starts don't)
        $end = (Get-Date).AddMinutes(4)
        while ($true) {
            $out = try { (Invoke-Guest $n "'state=' + (Get-ScheduledTask '$($Bot.Task)' -EA 0).State" 60 -AllowFail) -join ' ' } catch { '' }
            if ($out -match 'state=(\w*)') { $state = $Matches[1]; break }
            if ((Get-Date) -gt $end) { throw "no answer from $n" }
            Start-Sleep 6
        }
        if ($running -and $Action -eq 'start' -and $state -eq 'Running' -and (Reported) -lt 180) {
            Say 'OK already running'; return
        }
        if ($state -eq 'Running') {
            Say 'stopping the bot'
            $leaf = [regex]::Escape((Split-Path $Bot.Script -Leaf))
            Invoke-Guest $n "Stop-ScheduledTask '$($Bot.Task)'; Start-Sleep 1; Get-CimInstance Win32_Process -Filter `"Name='python.exe'`" | ? { `$_.CommandLine -match '$leaf' } | % { Stop-Process -Id `$_.ProcessId -Force -EA 0 }" 60 -AllowFail | Out-Null
        }
        if ($Action -eq 'stop') { Say 'OK stopped'; return }

        # the VPN first, if the role has one: a bot must never start (or stay) outside it
        if ($Bot.Vpn) {
            Say "connecting the VPN ($(if ($Bot.Vpn.region) { $Bot.Vpn.region } else { $Bot.Vpn.provider }))"
            $vpn = Start-GuestVpn $fm $Bot { param($t) Say $t }
            Say "OK VPN connected, exit address $($vpn.exit_ip) ($($vpn.region))"
        }

        # settings sent at every start; the task made if it's missing
        Set-GuestText $n "$GuestDrop\$Role.cmd" (Get-RoleCmd $fm $Bot)
        if (-not $state) { Say "creating the scheduled task $($Bot.Task)" }
        [void](Install-RoleTask $fm $Bot)
        $since = Get-Date
        Say 'starting the bot (launcher, login)'
        Invoke-Guest $n "Remove-Item '$GuestDrop\$($Bot.Log)' -EA 0; Start-ScheduledTask '$($Bot.Task)'" 60 | Out-Null
        $end = (Get-Date).AddMinutes(14)
        $seen = ''
        while ($true) {
            Start-Sleep 8
            if ((Reported) -lt ((Get-Date) - $since).TotalSeconds - 2) { Say 'OK working'; break }
            $t = Get-LogTail 1
            $t = ($t -replace '^Time (\S+ \S+ \S+) \S+ *-? *', '$1 ').Trim()
            if ($t -match '^exit ') {
                $why = (Get-LogTail 2) -split "`n" | Select-Object -First 1
                throw "the bot stopped: $($why.Trim())"
            }
            if ($t -and $t -ne $seen) { $seen = $t; Say "  $($t.Substring(0, [Math]::Min(120, $t.Length)))" }
            if ((Get-Date) -gt $end) { throw 'the bot has not reported' }
        }
    }
} catch {
    Say "FAILED $($_.Exception.Message)"
    exit 1
}
