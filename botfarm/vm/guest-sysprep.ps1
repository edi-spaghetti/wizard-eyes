<#
Runs ELEVATED in a template VM (new-bot-vm.ps1 -Template, stage 9T): generalizes Windows with sysprep, so
every clone makes its own machine SID, computer name and user at its first start (from the answer file
the host puts on a CD for that clone: lib New-CloneAnswerIso). Then the VM shuts down.
  - BitLocker off: Windows 11 encrypts the disk by itself (device encryption) and sysprep refuses to
    generalize an encrypted disk (0x80310039, tpl1 2026-11-27). Decrypted (waits for it) and kept off
    for good (PreventDeviceEncryption), also in the clones
  - the install's cached answer files go (they hold the password, and Windows would read them before
    the clone's)
  - sysprep /generalize /oobe /shutdown /mode:vm, started and left to it: success = the VM powers off
    (the host waits). Sysprep's exit code is 0 even when it fails, and GeneralizationState is 7 on a
    fresh install too: neither says anything. The VM must never be restarted before it is off: it
    would run Windows setup in the template
Prints 'SYSPREP-STARTED', or 'SYSPREP-FAILED: <why>' (BitLocker).
#>
$ErrorActionPreference = 'Continue'
$sp = "$env:SystemRoot\System32\Sysprep"

# --- BitLocker off, and kept off ---
$bl = 'HKLM:\SYSTEM\CurrentControlSet\Control\BitLocker'
if (-not (Test-Path $bl)) { New-Item $bl -Force | Out-Null }
New-ItemProperty $bl PreventDeviceEncryption -Value 1 -PropertyType DWord -Force | Out-Null
$v = Get-BitLockerVolume -MountPoint C: -EA SilentlyContinue
if ($v -and $v.VolumeStatus -ne 'FullyDecrypted') {
    "BitLocker on C: ($($v.VolumeStatus), $($v.EncryptionPercentage)%): decrypting"
    Disable-BitLocker -MountPoint C: | Out-Null
    $end = (Get-Date).AddMinutes(40)
    do {
        Start-Sleep 15
        $v = Get-BitLockerVolume -MountPoint C:
    } while ($v.VolumeStatus -ne 'FullyDecrypted' -and (Get-Date) -lt $end)
    if ($v.VolumeStatus -ne 'FullyDecrypted') { "SYSPREP-FAILED: C: still $($v.VolumeStatus) ($($v.EncryptionPercentage)%) after 40 min"; return }
    'BitLocker off (C: fully decrypted)'
} else { 'BitLocker: off' }

foreach ($f in "$env:SystemRoot\Panther\unattend.xml", "$env:SystemRoot\Panther\Unattend\unattend.xml",
               "$env:SystemRoot\Panther\autounattend.xml", "$env:SystemRoot\System32\Sysprep\unattend.xml") {
    if (Test-Path $f) { Remove-Item $f -Force; "removed $f" }
}

# --- one sysprep run, with its own /shutdown ---
# (no /quit trial run first: after a generalize with /quit, Windows hung on shutdown for an hour, twice,
# tpl1 2026-11-27/28; the host can't see into the VM after a generalize either, its Guest Additions
# link is gone). Started, not waited for: the host waits for the VM to power off, and if it doesn't,
# guest-sysprep-report.ps1 reads sysprep's logs (a refused generalize leaves the VM working as before).
Remove-Item "$sp\Panther\setupact.log", "$sp\Panther\setuperr.log" -Force -EA 0
Start-Process "$sp\sysprep.exe" -ArgumentList '/generalize', '/oobe', '/shutdown', '/mode:vm', '/quiet'
'SYSPREP-STARTED'
