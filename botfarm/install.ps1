<#
Hooks a farm up on this PC (setup.ps1 runs it too). Safe to re-run:
  - the farm becomes this PC's default farm (scripts without -Farm use it)
  - its services (farm.json "services") as scheduled tasks for this user:
    run-service.ps1 in a hidden window, on battery too, no time limit, no
    trigger (started by the desktop icon, never by a reboot on its own)
  - its desktop icon: start-farm.ps1 through start-farm.vbs, with the farm's icon
Tasks whose action changes are re-registered; a running one keeps running
the old action until it's restarted (-Restart does that).

    install.ps1 [-Farm <folder>] [-Restart] [-Look]
#>
param([string]$Farm, [switch]$Restart, [switch]$Look)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm

function Say($t) { Write-Host "   $t" }

if (-not $Look) { Set-DefaultFarm $fm.Dir }
Say "default farm: $($fm.Dir)"

$runner = Join-Path $PSScriptRoot 'run-service.ps1'
foreach ($s in $fm.Def.services) {
    $arg = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`" -Service $($s.name) -Farm `"$($fm.Dir)`""
    $old = Get-ScheduledTask $s.task -ErrorAction SilentlyContinue
    $same = $old -and $old.Actions[0].Execute -eq 'powershell.exe' -and $old.Actions[0].Arguments -eq $arg
    if ($same) { Say "service $($s.name): task $($s.task) is up to date" }
    elseif ($Look) { Say "service $($s.name): task $($s.task) would be $(if ($old) { 'changed' } else { 'created' })" }
    else {
        $a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg
        $p = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
        $set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
        Register-ScheduledTask $s.task -Action $a -Principal $p -Settings $set -Force | Out-Null
        Say "service $($s.name): task $($s.task) $(if ($old) { 'changed' } else { 'created' })"
    }
    if ($Restart -and -not $Look) {
        Stop-ScheduledTask $s.task -ErrorAction SilentlyContinue
        $pat = [regex]::Escape($s.module)
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pat } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep 1
        Start-ScheduledTask $s.task
        $end = (Get-Date).AddSeconds(60)
        while (-not (Test-Url $s.check)) {
            if ((Get-Date) -gt $end) { throw "service $($s.name) did not start (see $($fm.LogDir)\$($s.name).log)" }
            Start-Sleep 2
        }
        Say "service $($s.name): restarted, answering"
    }
}

$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop "$($fm.Name).lnk"
$vbs = Join-Path $PSScriptRoot 'start-farm.vbs'
$icon = if ($fm.Def.icon) { Join-Path $fm.Dir $fm.Def.icon } else { '' }
$sh = New-Object -ComObject WScript.Shell
$want = @{ Target = "$env:SystemRoot\System32\wscript.exe"; Args = "`"$vbs`" `"$($fm.Dir)`""; Icon = if ($icon) { "$icon,0" } else { '' } }
$cur = if (Test-Path $lnk) { $sh.CreateShortcut($lnk) } else { $null }
if ($cur -and $cur.TargetPath -eq $want.Target -and $cur.Arguments -eq $want.Args -and $cur.IconLocation -eq $want.Icon) {
    Say "desktop icon $($fm.Name) is up to date"
} elseif ($Look) { Say "desktop icon $($fm.Name) would be $(if ($cur) { 'changed' } else { 'created' })" }
else {
    $s = $sh.CreateShortcut($lnk)
    $s.TargetPath = $want.Target
    $s.Arguments = $want.Args
    $s.WorkingDirectory = $PSScriptRoot
    if ($want.Icon) { $s.IconLocation = $want.Icon }
    $s.Description = "Start $($fm.Name)"
    $s.Save()
    Say "desktop icon $($fm.Name) $(if ($cur) { 'changed' } else { 'created' })"
}
