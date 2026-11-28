setup.ps1, step by step
=======================

.. code-block:: powershell

   powershell -ExecutionPolicy Bypass -File botfarm\setup.ps1 [-Farm <folder>] [-Look] [-Yes] [-Fresh] [-Skip <steps>] [-Iso <path>]

``-Farm``
   The farm's folder (with ``farm.json``). Default: ``$env:WIZARD_EYES_FARM``, then this
   PC's default farm.
``-Look``
   Only check and report; changes nothing. Exit code 1 if anything is left to do.
``-Yes``
   For a run with no one at it: answers "yes" to the questions that only ask permission
   (build a VM now, switch the hypervisor off, reboot). It never ticks the by-hand
   checklist; the run stops there.
``-Fresh``
   Build every VM from the ISO instead of cloning the farm's template.
``-Skip``
   Step names or patterns to leave out this time, e.g. ``-Skip hypervisor,vm:*``.

Each step first **checks**; if it isn't done it **acts**, then checks again. If it still
isn't done, the run stops there and says why: fix that and run it again. Done steps are
skipped, so running it again is always safe.

The steps
---------

.. list-table::
   :header-rows: 1
   :widths: 16 44 14 26

   * - Step
     - What it checks / does
     - Time
     - Needs a person?
   * - ``tools``
     - git, VirtualBox 7.1+, Tailscale (only for a phone page); installs missing ones with winget.
     - 2–5 min
     - admin prompts of the installers
   * - ``engine``
     - The engine's own ``setup.ps1``: Python 3.9, venv, packages, OCR data, the R2 asset
       config.
     - 5–15 min
     - no
   * - ``accounts``
     - ``farm.local.json`` has a game account per role and is git-ignored. Asks for missing
       account names.
     - 1 min
     - typing the names
   * - ``secrets``
     - Every secret the farm needs is in Credential Manager and looks right
       (``secrets.ps1``). Prints how to store what's missing.
     - seconds
     - typing them in (:doc:`secrets`)
   * - ``template``
     - Only if ``farm.json`` names a ``"template"``: the template VM exists, is generalized
       and powered off. Builds it if missing (ISO, hypervisor off, ``new-bot-vm.ps1 -Template``).
     - ~60 min, once
     - the ISO; a reboot if the hypervisor is on
   * - ``vm:NAME``
     - Each role's VM exists and, when running, has the engine venv and the farm's repo.
       Clones it from the template (or builds it from the ISO without one, or with ``-Fresh``).
     - ~9 min (clone), ~60 min (ISO)
     - no
   * - ``vpn:ROLE``
     - Roles with ``"vpn"``: the tunnel is connected, the firewall lock is on, the exit address
       isn't this PC's. Puts the tunnel in when it isn't.
     - ~30 s
     - the Proton config (stored before)
   * - ``checklist:ROLE``
     - The by-hand items for the role (Jagex login, game terms, the farm's own), ticked off one
       by one; also done once the role's bot is working.
     - ~5 min per VM
     - **yes**
   * - ``hypervisor``
     - The hypervisor and Memory Integrity back on, if the setup switched them off.
     - 1 min + reboot later
     - admin prompt
   * - ``services``
     - ``install.ps1``: default farm, services as scheduled tasks, desktop icon.
     - 1 min
     - no
   * - ``alerts``
     - Telegram: token stored, phone paired, a test message.
     - 3 min
     - making the bot, pairing
   * - ``phone``
     - Tailscale logged in, the page's host name and key, ``tailscale serve``.
     - 5 min
     - Tailscale login, saving the key
   * - ``start``
     - Every bot working (``start-farm.ps1``).
     - 3–5 min
     - no

What "done" means after a stop
------------------------------

The exit code is 1 whenever the run stopped before the end, including the expected stop at
``checklist:ROLE`` in a ``-Yes`` run. Read the last lines: ``Stopped at 'STEP'`` names the step.

Examples
--------

.. code-block:: powershell

   # what's left, no changes
   botfarm\setup.ps1 -Farm ..\REPO\botfarm\FARM -Look

   # unattended up to the by-hand items, leaving the host-wide steps alone
   botfarm\setup.ps1 -Farm ..\REPO\botfarm\FARM -Yes -Skip hypervisor,services,start

   # the by-hand items, interactively
   botfarm\setup.ps1 -Farm ..\REPO\botfarm\FARM
