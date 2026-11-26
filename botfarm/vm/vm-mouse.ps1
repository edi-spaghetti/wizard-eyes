# Mouse integration of a running VM's window: off = moving the mouse over the window
# doesn't reach the VM; click in it to use it, Right Ctrl to get the mouse back.
#   vm-mouse.ps1 -Name wizard-eyes-vm1 [-State off|on|show]
# Uses the window's own Input menu (VirtualBox has no command for it) and doesn't move
# the mouse. VirtualBox forgets it when the window closes: run again (bot-control does).
param([Parameter(Mandatory)][string]$Name, [ValidateSet('off', 'on', 'show')][string]$State = 'off')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [Windows.Automation.AutomationElement]; $T = [Windows.Automation.TreeScope]; $P = [Windows.Automation.PropertyCondition]
Add-Type -Namespace Vm -Name Win -MemberDefinition '[DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, int m, IntPtr w, IntPtr l);'
Add-Type -TypeDefinition 'using System; using System.Text; using System.Runtime.InteropServices; namespace Vm { public class Text { [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n); [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h); } }'
$w = $null
foreach ($id in @(Get-Process VirtualBoxVM -ErrorAction SilentlyContinue | ForEach-Object Id)) {
    # (an open menu can pass for the process's main window)
    foreach ($top in $A::RootElement.FindAll($T::Children, (New-Object $P($A::ProcessIdProperty, $id)))) {
        if ($top.Current.ClassName -like '*UIMachineWindow*' -or $top.Current.Name -match "^$([regex]::Escape($Name)) ") {
            $title = New-Object Text.StringBuilder 300
            [void][Vm.Text]::GetWindowText([IntPtr]$top.Current.NativeWindowHandle, $title, 300)
            if ("$title" -match "^$([regex]::Escape($Name))( |$)") { $w = $top; $proc = @{ Id = $id } }
        }
    }
}
if (-not $w) { "no window for $Name"; exit 2 }
function Close-Menus([IntPtr]$popup = [IntPtr]::Zero) {
    # Escape goes to the menu's own window alone, until it's gone
    for ($k = 0; $k -lt 12; $k++) {
        $h = $popup
        if ($h -eq [IntPtr]::Zero) {
            # (an open menu is the process's "main window")
            $h = (Get-Process -Id $proc.Id).MainWindowHandle
            if ($h -eq 0 -or $A::FromHandle($h).Current.ClassName -ne 'UIMenu') { return }
        }
        if (-not [Vm.Text]::IsWindowVisible($h)) { return }
        [void][Vm.Win]::PostMessage($h, 0x100, [IntPtr]0x1B, [IntPtr]0x10001)
        [void][Vm.Win]::PostMessage($h, 0x101, [IntPtr]0x1B, [IntPtr]0xC0010001)
        Start-Sleep -Milliseconds 250
    }
    "warning: $Name's menu is still open"
}
Close-Menus
Start-Sleep -Milliseconds 300
$menu = $w.FindFirst($T::Descendants, (New-Object $P($A::ClassNameProperty, 'UIActionMenuRuntimeInput')))
if (-not $menu) { throw 'no Input menu' }
$expand = $menu.GetCurrentPattern([Windows.Automation.ExpandCollapsePattern]::Pattern)
try { $expand.Expand() } catch { $menu.GetCurrentPattern([Windows.Automation.InvokePattern]::Pattern).Invoke() }
$item = $null
$popup = [IntPtr]::Zero
$end = (Get-Date).AddSeconds(4)
$byPid = New-Object $P($A::ProcessIdProperty, $proc.Id)
while (-not $item -and (Get-Date) -lt $end) {
    Start-Sleep -Milliseconds 200
    foreach ($top in $A::RootElement.FindAll($T::Children, $byPid)) {
        $item = $top.FindAll($T::Descendants, (New-Object $P($A::ControlTypeProperty, [Windows.Automation.ControlType]::MenuItem))) |
            Where-Object { $_.Current.Name -like '*Mouse*Integration*' } | Select-Object -First 1
        if ($item) { break }
    }
}
try {
    if (-not $item) { throw 'no Mouse Integration entry in the Input menu' }
    $popup = [IntPtr][Windows.Automation.TreeWalker]::RawViewWalker.GetParent($item).Current.NativeWindowHandle
    $names = @($item.GetSupportedPatterns() | ForEach-Object ProgrammaticName)
    $toggle = $null
    $on = $null
    if ($names -match 'TogglePattern') {
        $toggle = $item.GetCurrentPattern([Windows.Automation.TogglePattern]::Pattern)
        $on = $toggle.Current.ToggleState -eq 'On'
    }
    if ($null -eq $on) { throw "can't read its state ($($names -join ', '))" }
    if ($State -ne 'show' -and $on -ne ($State -eq 'on')) {
        if ($names -match 'InvokePattern') { $item.GetCurrentPattern([Windows.Automation.InvokePattern]::Pattern).Invoke() } else { $toggle.Toggle() }
        Start-Sleep -Milliseconds 400
        "$Name mouse integration: was $(if ($on) {'on'} else {'off'}), set $State"
        $item = 'done'
    } else { "$Name mouse integration: $(if ($on) {'on'} else {'off'})" }
} finally {
    Start-Sleep -Milliseconds 300
    Close-Menus $popup
    Close-Menus
}
