Troubleshooting
===============

Problems seen while building the farm, and what the scripts now do about them. Most come
with a log: setup runs print to their console; VM builds print each stage.

Windows install hangs, or a VM never starts installing
------------------------------------------------------

The Windows hypervisor is on: Windows setup hangs under it. ``vm\hypervisor.ps1 status``;
``vm\hypervisor.ps1 off`` and reboot (``setup.ps1`` offers this). Clones don't need it off.

"Only N GB free on C:"
----------------------

A build from the ISO wants 30 GB free. Delete test VMs (VirtualBox Manager > Remove > Delete
all files). Remember a template can't be deleted while it has clones.

The template build: sysprep
---------------------------

- **Refused with 0x80310039**: the disk is BitLocker-encrypted (Windows 11 does it by
  itself). ``guest-sysprep.ps1`` decrypts first; this takes a few minutes.
- **The VM doesn't power off within 30 minutes**: the build stops and leaves the VM running.
  If the VM still answers, it prints sysprep's error log (an app installed for one user only
  is the usual cause; ``guest-sysprep-report.ps1 -Fix`` removes those). If it doesn't answer,
  sysprep has generalized and Windows hangs on the way down: power it off in VirtualBox,
  **don't start it**, and mark it (``Set-VmTemplate <vm>`` in ``lib\BotFarm.psm1``).
- After a generalize the VM's VirtualBox guest link stops working (``VERR_IO_NOT_READY``).
  That is expected; the host only waits for the power-off.

A template was started by mistake
---------------------------------

It shows "Getting ready" / "Just a moment…": it is running Windows setup in itself and is
spoiled. Delete it and its clones, and build it again (``setup.ps1`` does this when it's
missing).

A clone sits on "Just a moment…"
--------------------------------

Windows' first-run screens hung. Clones now start offline and skip those screens
(``SkipMachineOOBE`` / ``SkipUserOOBE`` in their answer file). If it happens anyway: delete
the clone and make it again; the template is not affected.

The VPN
-------

- **"no WireGuard config for <vm> in Credential Manager"**: make one on the Proton site and
  store it (:doc:`vpn`).
- **"did not connect within a minute"**: the Proton server may be down or the config revoked;
  make a new config for that VM, store it again, then ``bot-control.ps1 -Action start``.
- **The VM has no internet**: that is the lock doing its job while the tunnel is down.
  ``bot-control.ps1 -Action status`` shows ``vpn.state``.

Memory
------

If the host runs short, long background runs can be stopped. Run one test VM at most next
to a working farm, and check for vendor tools leaking memory (Task Manager > Details, sort by
memory).

Guest control errors
--------------------

``The specified user was not able to logon on guest``: the VM's Windows user isn't the one
the host expects (``%LOCALAPPDATA%\wizard-eyes\vm-users.json``), or Windows isn't logged in
yet. Wait for the login; check the file.
