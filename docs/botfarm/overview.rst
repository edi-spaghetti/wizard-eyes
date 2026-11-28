Overview
========

The engine runs *a* bot; the botfarm runs *bots*. A farm is a handful of bots, each
playing one game account in its own Windows VM, plus services on the host PC that watch
and control them.

What runs where
---------------

.. code-block:: text

   Host PC (Windows 11)
   ├── services (scheduled tasks): coordination, the farm's page, the watchdog
   ├── Telegram alerts, phone page through Tailscale (optional)
   ├── the template VM (powered off; clones are made from it)
   └── one VM per role
       ├── its own Windows user and computer name
       ├── WireGuard tunnel to Proton VPN + a firewall lock
       ├── Jagex Launcher, RuneLite, the engine (Python venv) and the farm's repo
       └── the role's bot, as a scheduled task on the VM's desktop

- A VM reaches the host's ``127.0.0.1`` at ``10.0.2.2`` (VirtualBox NAT), so bots report
  to the coordination service on the host.
- The host drives the VMs through VirtualBox (``VBoxManage guestcontrol``): no remote
  desktop, no SSH.

The two repos
-------------

- **The engine** (this repo, public): everything generic. The farm scripts are in
  ``botfarm/``, the Python side in ``python/wizard_eyes/botfarm/``.
- **The bots' repo** (yours, may be private): the bots and one folder per farm, e.g.
  ``botfarm/herbitrage``, holding ``farm.json`` (what to run) and ``farm.local.json`` (this
  PC only: account names; never in git). See :doc:`farm-json`.

How a farm gets built
---------------------

One script, ``botfarm/setup.ps1``, takes a PC from "repos cloned" to "bots working". It is a
list of steps; each one checks first and only acts when it isn't done, so it can be run
again at any time and carries on where it stopped. With ``-Yes`` it runs with no one at it
and stops only where a person is truly needed (see :doc:`setup-steps`).

The VMs come from a **template**: a Windows VM with everything machine-wide installed and
no logins, generalized with sysprep. Each role's VM is a linked clone of it that gives
itself its own name and user at its first start, then gets the per-user software. See
:doc:`vms`.

What it can't do for you
------------------------

By design, a few things always need a person:

- admin prompts on the host when tools are installed, and a reboot if the hypervisor must be switched off
- typing in passwords and keys (they never go into a chat or a file in git): see :doc:`secrets`
- making one Proton WireGuard config per VM on the Proton website: see :doc:`vpn`
- the first Jagex Launcher login of each account and the game's terms (being automated)
- CAPTCHAs and "new device" email codes, if they appear
- game-side preparation: membership, bank items, unlocks, friends lists
