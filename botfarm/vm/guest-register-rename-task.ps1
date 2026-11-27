<#
Runs ELEVATED in a bot VM (the one admin click, see Get-RenameTaskCommand in
lib\BotFarm.psm1): registers the WizardEyesRename task (at startup, as SYSTEM)
and starts it. The task renames the computer from the first line of
C:\wizard-eyes-provision\name.txt, if that is a plain name (letters and digits,
3-15), deletes the file and restarts. The task's code is inside the task, not
in a folder the bot user can write; the name file is only data.

After this, renaming a VM (and every VM cloned from it) needs no admin prompt:
write name.txt, restart the VM.
#>
$log = 'C:\wizard-eyes-provision\register-rename-task.log'
try {
    $code = '$f = ''C:\wizard-eyes-provision\name.txt''; $n = (Get-Content $f -EA 0 | Select-Object -First 1); if ($n) { $n = $n.Trim() }; ' +
        'if ($n -match ''^[a-z0-9]{3,15}$'' -and $n -ne $env:COMPUTERNAME.ToLower()) { Rename-Computer -NewName $n -Force; Remove-Item $f -Force; Restart-Computer -Force }'
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($code))
    $a = New-ScheduledTaskAction -Execute powershell.exe -Argument "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $enc"
    $t = New-ScheduledTaskTrigger -AtStartup
    Register-ScheduledTask WizardEyesRename -Action $a -Trigger $t -User SYSTEM -RunLevel Highest -Force | Out-Null
    Set-Content $log 'registered' -Encoding ascii
    Start-ScheduledTask WizardEyesRename
    Add-Content $log 'started' -Encoding ascii
} catch {
    Set-Content $log "FAILED: $($_.Exception.Message)" -Encoding ascii
    throw
}
