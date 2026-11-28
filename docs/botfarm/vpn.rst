VPN: Proton through WireGuard
=============================

A role with ``"vpn": {"provider": "proton"}`` in ``farm.json`` plays through Proton VPN. Each
VM has its own WireGuard config (its own key; one device on the Proton account), and a
firewall lock makes sure its real address never shows.

1. Make a config per VM (2 minutes each)
----------------------------------------

On `account.protonvpn.com <https://account.protonvpn.com>`_ > **Downloads** >
**WireGuard configuration**:

- platform **Windows**
- a name you'll recognise (the VM's name or its computer name)
- a server (e.g. one in your own country)
- **Create**, then download the ``.conf`` file

One config per VM, never the same config in two VMs. Proton plans have a device limit; each
connected VM counts.

2. Store it on the host (1 minute)
----------------------------------

In **your own** terminal (the file holds the VM's private key):

.. code-block:: powershell

   powershell -ExecutionPolicy Bypass -File botfarm\vpn-config.ps1 -Vm wizard-eyes-vm5 -Path "$env:USERPROFILE\Downloads\vm5.conf"

It checks the file (one ``[Interface]`` with a private key and an IPv4 address, one
``[Peer]`` with an ``Endpoint`` as IPv4:port), stores it in Credential Manager as
``wizard-eyes/vpn/<vm>``, deletes the file (``-Keep`` keeps it) and prints only the server
address.

3. It goes into the VM by itself (30 seconds)
---------------------------------------------

``setup.ps1`` (step ``vpn:ROLE``) and ``bot-control.ps1`` (before every bot start) put it in
when it isn't there or has changed (``botfarm\vm\guest-wireguard.ps1``, run elevated in
the VM):

- WireGuard (installed in the template) gets the config as a **tunnel service**: it starts
  with Windows and reconnects by itself. The config is stored in the VM for SYSTEM and
  administrators only.
- **The firewall lock**: outbound traffic is blocked by default; allowed are only traffic
  through the tunnel, WireGuard to the Proton server, the host (``10.0.2.2``) and DHCP.
  All of Windows' own outbound allow rules are switched off, so nothing (DNS included) goes
  around the tunnel.
- The exit address is checked: a bot never starts unless the tunnel is up, the lock is on and
  the address isn't the host's.

What was tested
---------------

- exit through Proton; the host's own address is different
- tunnel stopped: no internet and no name lookups at all, the host still reachable
- after a VM restart the tunnel is back by itself
- stopping and starting it again from the host

Stopping it on purpose
----------------------

Only through the farm's code (``Stop-GuestVpn`` in ``lib\BotFarm.psm1``): it closes the bot,
RuneLite and the Jagex Launcher first, then stops the tunnel. The lock stays on, so the VM has
no internet until the tunnel is back.

Status
------

``bot-control.ps1 -Role ROLE -Action status`` includes ``vpn``: ``state`` (Connected /
Connecting / Disconnected), ``exit_ip`` and ``lock``.
