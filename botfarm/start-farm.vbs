' Runs start-farm.ps1 without a console window (a farm's desktop icon).
'   wscript.exe start-farm.vbs "<farm folder>"
Dim here, farm
here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
farm = ""
If WScript.Arguments.Count > 0 Then farm = " -Farm """ & WScript.Arguments(0) & """"
CreateObject("WScript.Shell").Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & here & "\start-farm.ps1""" & farm, 0, False
