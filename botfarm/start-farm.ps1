<#
A farm's desktop icon (install.ps1 makes it): gets everything running, from any state.
  1. the farm's services (farm.json "services", in order), if down
     (a small window shows this, then closes)
  2. opens the farm's page, where the bots' progress shows
  3. each bot that isn't working: started through the page (bot-control.ps1:
     starts its VM if it's off, restarts a bot that's stuck). One that's working,
     at a meeting, logging in or waiting for a game update is left alone.
Safe to run again at any time. The icon runs it hidden (start-farm.vbs); a
message box only for what the page can't show. Log: <engine>\logs\start-farm.log
   -NoPage   don't open the page, no message box
   -Look     only say what it would do to the bots
#>
param([switch]$NoPage, [switch]$Look, [string]$Farm)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm
$Control = Join-Path $PSScriptRoot 'bot-control.ps1'
$Page = $fm.Def.page
$Service = $fm.Def.coordination
$Fresh = 480        # seconds since a report: working (as the watchdog)
$Meeting = 2400
$Login = 1500
New-Item -ItemType Directory -Force $fm.LogDir | Out-Null
$LogFile = Join-Path $fm.LogDir 'start-farm.log'
$Title = $fm.Name
$problems = @()
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

function Say($text) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    if ($script:ui) { $script:ui.text = $text.Substring(0, 1).ToUpper() + $text.Substring(1) }
}
function Show-Progress {
    # in its own thread, so it keeps moving while we wait
    $script:ui = [hashtable]::Synchronized(@{ text = 'Starting'; close = $false; title = $Title })
    $script:space = [runspacefactory]::CreateRunspace()
    $space.ApartmentState = 'STA'; $space.Open()
    $space.SessionStateProxy.SetVariable('ui', $script:ui)
    $script:shell = [powershell]::Create(); $shell.Runspace = $space
    [void]$shell.AddScript({
        Add-Type -AssemblyName System.Windows.Forms, System.Drawing
        $f = New-Object Windows.Forms.Form -Property @{ Text = $ui.title; ClientSize = '320,84'; FormBorderStyle = 'FixedDialog'; StartPosition = 'CenterScreen'
            MaximizeBox = $false; MinimizeBox = $false; ControlBox = $false; TopMost = $true; BackColor = 'White' }
        $l = New-Object Windows.Forms.Label -Property @{ Location = '16,14'; Size = '288,22'; Font = (New-Object Drawing.Font 'Segoe UI', 10); Text = 'Starting' }
        $b = New-Object Windows.Forms.ProgressBar -Property @{ Location = '16,46'; Size = '288,16'; Style = 'Marquee'; MarqueeAnimationSpeed = 25 }
        $f.Controls.AddRange(@($l, $b))
        $t = New-Object Windows.Forms.Timer -Property @{ Interval = 100 }
        $t.Add_Tick({ $l.Text = $ui.text; if ($ui.close) { $t.Stop(); $f.Close() } })
        # the script is started hidden, which Windows applies to its first window
        Add-Type -Namespace BotFarmUi -Name Win -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);'
        $f.Add_Shown({ [void][BotFarmUi.Win]::ShowWindow($f.Handle, 5); $f.Activate() })
        $t.Start()
        [void]$f.ShowDialog()
    })
    $script:wait = $shell.BeginInvoke()
}
function Hide-Progress {
    if (-not $script:ui) { return }
    $script:ui.close = $true
    [void]$script:wait.AsyncWaitHandle.WaitOne(3000)
    $script:ui = $null
}
function Answers($url) { try { [void](Invoke-RestMethod $url -TimeoutSec 5); $true } catch { $false } }
function Start-Service_($name, $task, $url, $match) {
    if (Answers $url) { Say "$name is running"; return }
    Say "starting $name"
    # hung, or a leftover: the port must be free
    Stop-ScheduledTask $task -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $match } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep 1
    Start-ScheduledTask $task
    $end = (Get-Date).AddSeconds(60)
    while (-not (Answers $url)) {
        if ((Get-Date) -gt $end) { throw "$name did not start (see wizard-eyes\logs)" }
        Start-Sleep 2
    }
    Say "$name is running"
}
function Get-Need($role) {
    # @(needs starting?, why)
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $c = (Invoke-RestMethod "$Page/control" -TimeoutSec 15).bots.$role
    if ($c.busy) { return @($false, 'already being started') }
    $bot = (Invoke-RestMethod "$Service/bots" -TimeoutSec 10).$role
    $age = if ($bot) { $now - $bot.updated } else { 1e9 }
    $out = powershell -NoProfile -ExecutionPolicy Bypass -File $Control -Role $role -Action status -Farm $fm.Dir
    $line = @($out | Where-Object { $_ -like '{*' })[0]
    if (-not $line) { return @($true, 'could not be looked at') }
    $look = $line | ConvertFrom-Json
    if (-not $look.vm) { return @($true, 'its VM is off') }
    if (-not $look.running) { return @($true, 'it is not running') }
    $lines = @($look.lines)
    $tail = ($lines | Select-Object -Last 6) -join "`n"
    if ($lines.Count -and $lines[-1] -match '\bSleeping @') { return @($false, 'paused (ctrl+p or Num Lock in its VM): left alone') }
    if ($age -lt $Fresh) { return @($false, 'working') }
    $meetings = @(Invoke-RestMethod "$Service/meetings" -TimeoutSec 10 | ForEach-Object { $_ })
    foreach ($m in $meetings) {
        if ($m.state -in 'accepted', 'met' -and $role -in $m.by, $m.to -and $now - $m.updated -lt $Meeting) { return @($false, 'at a meeting') }
    }
    if ($look.log_age -lt 600) {
        if ($tail -match 'game status page: .*trying again in') { return @($false, 'waiting for a game update') }
        if ($age -lt $Login -and $lines.Count -and $lines[-1] -match 'Client state:|Clicking (Play|Ok|Click)|Launcher|starting the client again|logging back in|Login screen') { return @($false, 'logging in') }
    }
    return @($true, "no report for $([int]($age / 60)) min")
}

# a second double-click while it's busy does nothing
$mutex = New-Object Threading.Mutex $false, "Local\BotFarmStart-$($fm.Name)"
try { $mine = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $mine = $true }
if (-not $mine) { exit }
try {
    Show-Progress
    foreach ($s in $fm.Def.services) {
        Start-Service_ $s.name $s.task $s.check ([regex]::Escape($s.module))
    }
    Hide-Progress
    if (-not $NoPage) { Start-Process $Page }

    $started = 0
    foreach ($role in $fm.Roles.Keys) {
        try {
            $need, $why = Get-Need $role
            if (-not $need) { Say "${role}: $why"; continue }
            if ($Look) { Say "${role}: would be started ($why)"; continue }
            if ($started) { Start-Sleep 20 }        # two VMs booting at once is slow
            Say "${role}: starting ($why)"
            [void](Invoke-RestMethod "$Page/control" -Method Post -ContentType 'application/json' -TimeoutSec 15 -Body (@{ role = $role; action = 'start' } | ConvertTo-Json))
            $started++
        } catch {
            # (the watchdog may have got there first: that's fine)
            $m = "$($_.ErrorDetails.Message) $($_.Exception.Message)"
            Say "${role}: not started: $m"
            if ($m -notmatch 'busy with') { $problems += "The $role was not started: $m" }
        }
    }
} catch {
    Say "FAILED $($_.Exception.Message)"
    $problems += $_.Exception.Message
} finally {
    Hide-Progress
    $mutex.ReleaseMutex()
}
if ($problems) {
    if (-not $NoPage) { [void][Windows.Forms.MessageBox]::Show(($problems -join "`n`n"), $Title, 'OK', 'Warning') }
    exit 1
}
