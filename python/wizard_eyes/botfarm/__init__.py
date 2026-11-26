"""Python side of a bot farm (see the engine's ``botfarm/README.md``).

- ``farm``: reads a farm's ``farm.json`` (name, roles and their VMs, state folder)
- ``alerts``: Telegram alerts to the owner's phone (send only)
- ``remote``: the key and host name for a farm page reached from the phone
- ``live``: stills of the bots' VM screens
- ``control``: starts / stops a role's bot through ``botfarm/bot-control.ps1``
- ``watchdog``: restarts bots that stop getting on (farms add what progress
  is and their own stop reasons)
- ``page``: the farm's web page (bots, start / stop, live screens, watchdog);
  a farm's own page adds its routes, or runs this one as it is

Every bot a farm starts sends the engine's heartbeat (``wizard_eyes.heartbeat``),
so all of the above works for any script.

Farm-specific pages (e.g. a planner) use these; nothing here knows about any
one farm's bots.
"""
