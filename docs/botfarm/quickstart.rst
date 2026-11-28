Step by step: a farm on a new PC
================================

This walks through a whole farm, from a bare Windows PC to bots working. Times are what
was measured on a 32 GB, 8-core PC (see :doc:`timings`).

.. contents:: Steps
   :local:
   :depth: 1

1. Before you start (10 minutes)
--------------------------------

Have these ready (see :doc:`requirements`):

- the bots' repo URL and the farm's folder in it (e.g. ``botfarm/herbitrage``)
- a GitHub login
- the game accounts' emails and passwords
- the Windows 11 ISO, or time to download it (~8 GB)
- for a VPN farm: a Proton VPN login
- for alerts: a Telegram account

Close anything memory-hungry. Expect one reboot if the Windows hypervisor is on.

2. Bootstrap: git, SSH key, repos (5–10 minutes)
------------------------------------------------

In a PowerShell window:

.. code-block:: powershell

   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/edi-spaghetti/wizard-eyes/main/botfarm/bootstrap.ps1))) `
       -FarmRepo git@github.com:OWNER/REPO.git -FarmFolder botfarm/FARM

It installs git, makes an SSH key (if GitHub doesn't know it, the key goes on your clipboard
and GitHub's "add SSH key" page opens; it waits until GitHub accepts it), clones both repos
under ``%USERPROFILE%\wizard-eyes`` and then runs ``setup.ps1``. You can stop it after the
clone with ``-NoSetup`` and run ``setup.ps1`` yourself.

3. Store the secrets (5 minutes, plus one Proton config per VM)
---------------------------------------------------------------

The setup asks for nothing secret through files or chats; you type secrets into Windows
Credential Manager yourself. Check what the farm needs:

.. code-block:: powershell

   cd $env:USERPROFILE\wizard-eyes\wizard-eyes
   powershell -ExecutionPolicy Bypass -File botfarm\secrets.ps1 -Farm ..\REPO\botfarm\FARM -Online

Every ``TODO`` line says how to store it. The usual ones:

.. code-block:: powershell

   # each role's game login (user = the account's email; it asks for the password)
   cmdkey /generic:wizard-eyes/jagex/ACCOUNT /user:you@example.com /pass

   # each VM's Proton WireGuard config (made on account.protonvpn.com, see the VPN page)
   powershell -ExecutionPolicy Bypass -File botfarm\vpn-config.ps1 -Vm wizard-eyes-vm1 -Path "$env:USERPROFILE\Downloads\vm1.conf"

Details: :doc:`secrets`, :doc:`vpn`.

4. Run the setup (about 70 minutes for the first VM, ~10 for each more)
-----------------------------------------------------------------------

.. code-block:: powershell

   cd $env:USERPROFILE\wizard-eyes\wizard-eyes
   powershell -ExecutionPolicy Bypass -File botfarm\setup.ps1 -Farm ..\REPO\botfarm\FARM

For VMs to be cloned from a template (much faster), the farm's ``farm.json`` needs
``"template": "wizard-eyes-tpl1"`` (any VM name; see :doc:`farm-json`). Without it each VM is
built from the ISO (~60 minutes each).

It goes through its steps in order and says ``OK`` or ``TODO`` for each (all steps:
:doc:`setup-steps`). The long parts:

- **template** (~60 minutes, only once per PC): a Windows VM installed from the ISO, locked
  down, with the machine-wide software, then generalized and powered off. If the hypervisor
  is on, it switches it off first and asks for a reboot; after the reboot the setup carries on
  by itself when you log in.
- **vm:NAME** (~9 minutes each): a clone of the template per role: its own computer name
  and Windows user, then git, the repos, Python, RuneLite and its profile.
- **vpn:ROLE** (~30 seconds each): the role's WireGuard tunnel and firewall lock, checked
  from inside the VM.

To leave it unattended, add ``-Yes``: it then answers "build now?" and "switch the
hypervisor off?" itself (it never ticks the by-hand checklist, see the next step).

To see what's done without changing anything: ``-Look``.

5. The by-hand checklist (5 minutes per VM)
-------------------------------------------

The setup stops at ``checklist:ROLE``: things only a person can do in each VM. Run
``setup.ps1`` again without ``-Yes``; it opens the VM's window and asks about each item:

- Jagex Launcher: log in with the role's account
- Jagex Launcher: RuneLite as the Old School client, play once, accept the game's terms
- whatever the farm's ``farm.json`` adds (bank PIN, friends list, ...)

Click into the VM window to use it; **Right Ctrl** gives the mouse back.

6. Services, alerts, phone, start (5–10 minutes)
------------------------------------------------

The remaining steps of the same run:

- **services**: makes the farm this PC's default, its services scheduled tasks, and a
  desktop icon
- **alerts**: Telegram bot token (from @BotFather) and pairing with your phone
- **phone**: Tailscale login and the page served to your phone
- **start**: services up and every bot started

Then the setup says ``Everything is set up.`` From now on the **desktop icon** gets the whole
farm going from any state (see :doc:`running`).

Adding a VM to an existing farm
-------------------------------

Add the role to ``farm.json`` (and its account to ``farm.local.json``), store its secrets,
then run ``setup.ps1`` again: done steps are skipped, the new VM is cloned from the template.
About 10 minutes plus its checklist.
