Secrets
=======

Passwords, keys and tokens live in **Windows Credential Manager** on the host, typed in by
you in your own terminal. They are never written to the repos, never printed by the scripts,
and only go into a VM for as long as a step needs them (except where noted).

Check them
----------

.. code-block:: powershell

   powershell -ExecutionPolicy Bypass -File botfarm\secrets.ps1 -Farm <folder> [-Online]

It lists what the farm needs with ``OK``, ``TODO`` (missing or wrong, with how to store it)
or ``--`` (optional). ``-Online`` also asks Telegram whether the bot token works. Exit code 1
if anything needed is missing. ``setup.ps1`` runs the same check as its ``secrets`` step.

What there is
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Credential Manager target
     - What
     - How to store it
   * - ``wizard-eyes/vm-user``
     - The VMs' shared Windows password
     - made by the scripts at the first VM build
   * - ``wizard-eyes/jagex/<account>``
     - A game account's login (user = its email)
     - ``cmdkey /generic:wizard-eyes/jagex/<account> /user:<email> /pass``
   * - ``wizard-eyes/vpn/<vm>``
     - A VM's Proton WireGuard config
     - ``botfarm\vpn-config.ps1 -Vm <vm> -Path <file.conf>`` (:doc:`vpn`)
   * - ``wizard-eyes/telegram``
     - The Telegram bot token (``farm.json`` ``"alerts"``)
     - ``setup.ps1``'s ``alerts`` step, or ``cmdkey /generic:wizard-eyes/telegram /user:bot /pass``
   * - ``wizard-eyes/gmail``
     - Optional: a Google **app password** (16 letters) for Jagex's email codes
     - 2-Step Verification on, then myaccount.google.com/apppasswords;
       ``cmdkey /generic:wizard-eyes/gmail /user:<address> /pass``
   * - ``wizard-eyes/bank-pin/<account>``
     - A bank PIN, if the account has one. Kept **in the VM's** Credential Manager.
     - in the VM: ``cmdkey /generic:wizard-eyes/bank-pin/<account> /user:<account> /pass``

``cmdkey ... /pass`` with nothing after ``/pass`` asks for the value; nothing shows while you
type.

Files that hold secrets
-----------------------

- ``config\assets.local.json`` (engine, git-ignored): the asset storage keys. VMs get a
  read-only copy.
- ``farm.local.json`` (farm folder, git-ignored): account names only, no passwords.
