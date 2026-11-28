Requirements
============

Host PC
-------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Needed
   * - Windows
     - Windows 11 (the scripts are Windows PowerShell 5.1). An account with admin rights.
   * - Memory
     - About 4–6 GB per running VM (``vm_defaults.memory_mb``) plus ~4 GB for the host.
       Tested: 32 GB with two farm VMs and one test VM. With four VMs up, the host ran out.
       Run at most one *test* VM next to a working farm.
   * - Disk
     - A fresh VM build checks for **30 GB free** on ``C:``. The template takes ~20 GB; each
       clone starts at ~2 GB and grows with use. The Windows ISO is ~8 GB.
   * - CPU
     - 4–6 cores per VM (``vm_defaults.cpus``). Bots are pinned to the fast cores.
   * - VirtualBox
     - **7.1 or newer** (7.0 can't boot current Windows 11 media under UEFI). Tested with 7.2.22.
       ``setup.ps1`` installs it with ``winget`` if missing.
   * - git, winget
     - git is installed by ``bootstrap.ps1`` / ``setup.ps1``; winget comes with Windows
       ("App Installer" in the Microsoft Store).
   * - Windows hypervisor
     - Must be **off while a VM installs Windows from the ISO** (Windows setup hangs under
       Hyper-V). Clones never need it off. ``vm\hypervisor.ps1 off`` / ``on`` (admin + reboot);
       ``setup.ps1`` offers it and turns it back on at the end.
   * - Background software
     - Watch for vendor tools eating memory. A Dell SupportAssist service once grew to 9 GB and
       starved a build; uninstalling it is recommended on a farm PC.

Downloads
---------

- **Windows 11 ISO**, English, x64, at
  ``%USERPROFILE%\VirtualBox VMs\iso\Win11_26H2_EnglishUK_x64.iso`` (or pass ``-Iso``).
  Edition index 6 (Pro: it honours the update policies). ``setup.ps1`` opens Microsoft's
  download page if it's missing. Windows in the VMs stays unactivated.
- Everything else is downloaded by the scripts: WireGuard (signature checked), the Visual C++
  runtime (signature checked), the Jagex Launcher (signature checked), RuneLite, Python 3.9,
  MinGit.

Accounts and subscriptions
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - What
     - Why
   * - GitHub
     - Cloning the repos (``bootstrap.ps1`` sets up an SSH key).
   * - Game accounts
     - One Jagex account per role. Members if the farm uses members worlds. Never the same
       account in two places at once.
   * - Proton VPN
     - A paid plan with WireGuard configs (one config = one device; each VM is one).
       Only if the farm's roles use ``"vpn": {"provider": "proton"}``.
   * - Telegram bot
     - Alerts to your phone (``farm.json`` ``"alerts"``). Made with @BotFather.
   * - Tailscale
     - The farm's page on your phone (``farm.json`` ``"phone"``). Optional.
   * - Google account with 2-Step Verification
     - Optional, planned: an app password will let the setup read Jagex's email codes. A normal
       Google password does not work for this.
