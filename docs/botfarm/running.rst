Running the farm day to day
===========================

The desktop icon
----------------

``install.ps1`` (the ``services`` step) puts the farm's icon on the desktop. It runs
``start-farm.ps1``, which gets everything going from any state:

1. the farm's services (``farm.json`` ``"services"``), if they're down
2. opens the farm's page, where the bots' progress shows
3. starts each bot that isn't working (its VM too, if it's off); leaves alone bots that are
   working, at a meeting, logging in or waiting for a game update

Safe to click again at any time. Log: ``<engine>\logs\start-farm.log``.
``start-farm.ps1 -Look`` says what it would do.

One bot: bot-control.ps1
------------------------

.. code-block:: powershell

   botfarm\bot-control.ps1 -Role ROLE -Action start|restart|stop|status [-Farm <folder>] [-Log <file>] [-Shot <file.png>]

- ``start``: from any state: starts the VM if it's off, waits for Windows, connects the VPN
  (roles with ``"vpn"``), writes the role's command into the VM and starts its scheduled task.
  The bot opens the launcher, logs in and gets going.
- ``restart``: stops and starts the bot (e.g. after ``deploy-vm.ps1``).
- ``stop``: stops the bot; the watchdog then leaves it alone.
- ``status``: changes nothing; one line of JSON: VM on, bot running, its log's age and last
  lines, the VPN. ``-Shot`` also saves a screenshot.

The farm's page uses the same script for its buttons, and its **watchdog** for restarting
bots that are stuck.

Rests
-----

Bots take ``--sleep 23:00-07:00`` (every night, start and end moved by up to
``--rest-jitter`` minutes, default 45) and ``--breaks 50-130:3-15`` (a 3–15 minute break
every 50–130 minutes) in their ``farm.json`` ``args``. The heartbeat says when a bot rests, so
the watchdog leaves it alone.

New code
--------

Commit, then:

.. code-block:: powershell

   botfarm\vm\deploy-vm.ps1 [-Name <vm>]      # every running VM of the farm without -Name
   botfarm\bot-control.ps1 -Role ROLE -Action restart

Services after a change
-----------------------

``botfarm\install.ps1 -Restart`` re-registers changed service tasks and restarts them.
