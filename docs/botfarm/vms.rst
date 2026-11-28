VMs: the template and its clones
================================

All VM work is done by ``botfarm\vm\new-bot-vm.ps1``. ``setup.ps1`` calls it; you can also
run it yourself.

.. code-block:: powershell

   # a template (once per PC)
   botfarm\vm\new-bot-vm.ps1 -Name wizard-eyes-tpl1 -Template -Farm <folder>

   # a bot VM cloned from it
   botfarm\vm\new-bot-vm.ps1 -Name wizard-eyes-vm5 -CloneFrom wizard-eyes-tpl1 -Farm <folder>

   # a bot VM straight from the ISO (no template)
   botfarm\vm\new-bot-vm.ps1 -Name wizard-eyes-vm5 -Farm <folder>

   # refresh an existing VM: repos, provisioning (it is running or gets started)
   botfarm\vm\new-bot-vm.ps1 -Name wizard-eyes-vm5 -Farm <folder>

It is safe to re-run: each stage checks whether it's done. Size and screen come from
``farm.json`` ``vm_defaults`` unless given (``-MemoryMB``, ``-Cpus``, ``-DiskGB``,
``-Width``, ``-Height``).

The template (about 60 minutes)
-------------------------------

A template is a Windows VM that every bot VM is cloned from. It holds what is the same for
all of them and nothing personal: no game login, no VPN config.

.. list-table::
   :header-rows: 1
   :widths: 8 62 30

   * - Stage
     - What happens
     - Time
   * - 1–4
     - Checks (VirtualBox 7.1+, the ISO, 30 GB free, hypervisor off), the VM (EFI, secure boot,
       TPM 2.0, network cable unplugged), an unattended Windows install with its own Windows
       user (a human name, e.g. ``gordon``)
     - 20–25 min
   * - 5–6
     - First login; the no-updates lockdown (Windows Update off for good, never sleep or lock,
       auto-logon, UAC consent prompt off), a reboot, checked
     - 5 min
   * - 7
     - Network on
     - seconds
   * - 8T
     - Machine-wide software, quietly: Visual C++ runtime, WireGuard; the Jagex Launcher (it
       has no quiet mode: the host presses its installer's keys through VirtualBox)
     - 5–8 min
   * - 9T
     - BitLocker off (Windows encrypts the disk by itself; sysprep refuses an encrypted disk),
       then **sysprep** ``/generalize /oobe /shutdown``: the VM powers off by itself and is
       marked as a template
     - 10–20 min

.. warning::

   **Never start a finished template.** Its next start runs Windows' first-run setup in the
   template itself, and it is spoiled (it has to be rebuilt). Clones are linked to it: never
   delete it while it has clones either.

The template is marked with the VirtualBox extradata ``wizard-eyes/template=sysprep``.
``farm.json`` ``"template": "wizard-eyes-tpl1"`` tells ``setup.ps1`` to use it.

A clone (about 9 minutes)
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 62 30

   * - Stage
     - What happens
     - Time
   * - 3
     - A linked clone (small: it keeps only its changes). An answer-file CD made for it holds
       its computer name (e.g. ``duchessmoth``) and its own Windows user (e.g. ``janet``).
       Network unplugged for the first start.
     - seconds
   * - 5
     - First start: Windows reads the CD, sets the name and makes the user, skips its first-run
       screens and logs in. The CD is ejected and deleted (it holds the password).
     - 2–4 min
   * - 5b
     - The template's user is removed; the lockdown runs again for the new user.
     - 1 min
   * - 6–7
     - Lockdown checked, admin without a prompt, machine name checked, network on.
     - 1 min
   * - 8
     - Provisioning in the VM: MinGit, the engine and the farm's repo (from this PC's
       committed ``main``), the read-only asset config, RuneLite, Python 3.9 + venv, OCR data,
       the reference RuneLite profile imported.
     - 4–6 min

Names
-----

- **Computer names** are random and never repeated (e.g. ``duchessmoth``), kept in
  ``%LOCALAPPDATA%\wizard-eyes\machine-names.json``.
- **Windows users** have a human name each (``janet``, ``gordon``, ...), kept in
  ``%LOCALAPPDATA%\wizard-eyes\vm-users.json``. VMs built before these existed keep the user
  ``bot``.
- All VMs share one Windows password, made once and kept in Credential Manager
  (``wizard-eyes/vm-user``). You never need to type it.

Getting code into running VMs
-----------------------------

``botfarm\vm\deploy-vm.ps1 [-Name <vm>]`` copies the **committed** code (HEAD) of the engine
and the farm's repo into the farm's running VMs. Bots keep their old code until restarted.

Looking at a VM
---------------

VMs run headless. To see one: VirtualBox Manager > the VM > **Show**, or let ``setup.ps1``'s
checklist step open it. ``bot-control.ps1 -Action status -Shot <file.png>`` saves a screenshot
without disturbing the bot.
