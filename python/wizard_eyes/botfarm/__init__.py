"""Python side of a bot farm (see the engine's ``botfarm/README.md``).

- ``farm``: reads a farm's ``farm.json`` (name, roles and their VMs, state folder)
- ``alerts``: Telegram alerts to the owner's phone (send only)
- ``remote``: the key and host name for a farm page reached from the phone
- ``live``: stills of the bots' VM screens
- ``control``: starts / stops a role's bot through ``botfarm/bot-control.ps1``

Farm-specific pages (e.g. a planner) use these; nothing here knows about any
one farm's bots.
"""
