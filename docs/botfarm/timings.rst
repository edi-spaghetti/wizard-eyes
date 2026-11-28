How long things take
====================

Measured on the development PC (Windows 11, 32 GB, 8 cores, VirtualBox 7.2.22; VMs with
4 GB and 4 CPUs) in November 2026. Downloads depend on your connection.

.. list-table::
   :header-rows: 1
   :widths: 55 20 25

   * - What
     - Took
     - Notes
   * - Template from the ISO, start to powered off (``new-bot-vm -Template``)
     - ~60 min
     - once per PC
   * - … of which the unattended Windows install
     - 20–25 min
     -
   * - … the machine-wide software (VC++, WireGuard, Jagex Launcher)
     - 5–8 min
     -
   * - … BitLocker decryption + sysprep, until it powers off
     - 10–20 min
     -
   * - A clone of the template, provisioned (``new-bot-vm -CloneFrom``)
     - **9 min**
     - per VM
   * - A VM from the ISO without a template
     - ~60 min
     - per VM; needs the hypervisor off
   * - The VPN tunnel and lock into a VM
     - ~30 s
     - per VM, the first time
   * - One admin step in a VM (UAC elevation from the host)
     - ~20 s
     -
   * - ``setup.ps1 -Yes`` from nothing to the by-hand checklist, one role
     - **70.6 min**
     - template + clone + VPN
   * - ``setup.ps1 -Yes`` with the template already built, one role
     - **9.0 min**
     - clone + VPN
   * - The by-hand checklist (launcher login, terms)
     - ~5 min
     - per VM
   * - A bot's cold start (VM off → in game)
     - ~2.5 min
     - ``bot-control -Action start``

So a farm of N roles takes roughly **70 + 10 × (N − 1) minutes** unattended, plus about
5 minutes per VM at the end for the logins, plus the host steps (tools, engine, services,
alerts, phone: 15–30 minutes on a bare PC).
