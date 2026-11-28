farm.json reference
===================

A farm is a folder in the bots' repo:

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - File
     - What
     - In git
   * - ``farm.json``
     - roles (one bot each: VM, script, arguments), services, page, icon, checklists
     - yes
   * - ``farm.local.json``
     - this PC only: account names per role
     - no (``*.local.json`` ignored)
   * - an icon
     - for the desktop shortcut
     - yes

Example
-------

.. code-block:: json

   {
     "name": "Herbitrage",
     "state_dir": "data/planner",
     "page": "http://127.0.0.1:8766",
     "coordination": "http://127.0.0.1:8765",
     "icon": "herbitrage.ico",
     "deploy_paths": ["bots"],
     "template": "wizard-eyes-tpl1",
     "vm_defaults": {"memory_mb": 6144, "cpus": 6, "disk_gb": 64, "width": 1920, "height": 1200},
     "vpn": {"provider": "proton"},
     "roles": {
       "cleaner": {"vm": "wizard-eyes-vm1", "task": "WizardEyesHerbCleaner",
                   "script": "bots\\herblore\\herb-cleaner.py",
                   "args": ["--username", "{account:cleaner}", "--coordinator", "http://10.0.2.2:8765"]}
     },
     "services": [
       {"name": "coordination", "task": "WizardEyesCoordination", "module": "wizard_eyes.coordination",
        "args": ["serve", "--port", "8765"], "check": "http://127.0.0.1:8765/bots"}
     ],
     "checklist": {"all": ["..."], "cleaner": ["..."]}
   }

And ``farm.local.json``:

.. code-block:: json

   {"accounts": {"cleaner": "MyAccountName"}}

Keys
----

``name``, ``state_dir``
   The farm's name and its state folder (under the engine).
``page``, ``coordination``
   The farm's page and the coordination service the bots report to (on the host; VMs reach
   it at ``10.0.2.2``).
``template``
   The template VM to clone role VMs from (:doc:`vms`). Without it, VMs are built from the ISO.
``vm_defaults``
   Memory (MB), CPUs, disk (GB) and screen size of new VMs.
``vpn``
   ``{"provider": "proton"}`` for the whole farm, or on a role; ``"vpn": false`` on a role leaves
   it out (:doc:`vpn`).
``roles``
   One bot each: ``vm`` (its VirtualBox VM), ``task`` (its scheduled task in the VM),
   ``script`` (in the bots' repo) and ``args``.
``services``
   Host services, run as scheduled tasks by ``install.ps1``; ``check`` is a URL that answers
   when the service is up.
``alerts``, ``phone``
   Telegram alerts and the phone page (optional).
``deploy_paths``
   What ``deploy-vm.ps1`` copies of the bots' repo (default: all).
``checklist``
   Extra by-hand items: ``all`` for every role, or per role.

Placeholders in strings: ``{engine}``, ``{farm}``, ``{repo}``, ``{control}`` (the path of
``bot-control.ps1``) and ``{account:ROLE}`` (from ``farm.local.json``).

A role's bot runs in its VM as a scheduled task running ``C:\wizard-eyes-provision\ROLE.cmd``,
which ``bot-control.ps1`` writes from ``script`` and ``args`` at every start. Its log:
``C:\wizard-eyes-provision\ROLE.log``.
