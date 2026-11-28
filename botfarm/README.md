# Botfarm

Running several bots together on one PC, each in its own VirtualBox VM, with
services on the host (a noticeboard the bots report to, a page to watch and
control them), phone alerts and phone access. One level above a single bot
script: the engine runs a bot, the botfarm runs bots.

Everything here is generic. A **farm** is a folder in the bots' own repo
(e.g. `wizard-eyes-experiments/botfarm/herbitrage`) that says what to run:

| file | what | in git |
| --- | --- | --- |
| `farm.json` | roles (one bot each: VM, script, arguments), services, page, icon, checklists | yes |
| `farm.local.json` | this PC only: account names per role | no (`*.local.json` ignored) |
| an icon | for the desktop shortcut | yes |

**Step-by-step guide** (requirements, how long each part takes, troubleshooting): the Sphinx docs in
`docs/botfarm/`. Build them with `python -m pip install -r docs/requirements.txt` then
`python -m sphinx -b html docs docs/_build/html`.

## A new PC

In PowerShell (it asks for what only you can do: the GitHub key, accounts,
the Windows ISO, reboots, logins in the VMs, the Telegram token, Tailscale):

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/edi-spaghetti/wizard-eyes/main/botfarm/bootstrap.ps1))) `
    -FarmRepo git@github.com:OWNER/REPO.git -FarmFolder botfarm/FARM
```

Run `botfarm\setup.ps1` again at any time: done steps are skipped, `-Look`
says what's left.

## Scripts

| script | does |
| --- | --- |
| `bootstrap.ps1` | bare Windows PC: git, a GitHub SSH key, the engine + the farm's repo cloned, then `setup.ps1` (one-liner below) |
| `setup.ps1` | everything else, resumable, each step checks first: tools, engine setup, accounts, VMs built, by-hand checklists, hypervisor back on, services, Telegram, Tailscale phone page, bots started (`-Look` only reports, `-Skip 'vm:*'` leaves steps out) |
| `install.ps1` | default farm for this PC, services as scheduled tasks, desktop icon (`-Restart` restarts the services, `-Look` changes nothing) |
| `start-farm.ps1` | the desktop icon: services up, page open, bots that aren't working started |
| `bot-control.ps1` | start / restart / stop / status of one role, from any state (VM off, not logged in, stuck); the page's buttons and watchdog use it |
| `run-service.ps1` | a service's scheduled-task action |
| `vm/new-bot-vm.ps1` | build, clone (`-CloneFrom`) or refresh a bot VM: unattended Windows, no-updates lockdown, consent prompt off, unique machine name, a human user name of its own (new builds), repos, engine setup (`-Fixup` also does the prompt + name on a VM built before) |
| `vm/new-bot-vm.ps1 -Template` | a template to clone from: Windows, lockdown, VC++, WireGuard, Jagex Launcher (installer keys pressed from the host), no logins, then sysprep (`vm/guest-template.ps1`, `vm/guest-sysprep.ps1`); a clone of it gets its own computer name, Windows user and machine SID at first start from an answer-file CD |
| `secrets.ps1` | which secrets the farm needs in Credential Manager, whether each is there (never printed) |
| `vpn-config.ps1` | a VM's Proton VPN WireGuard config into Credential Manager (run it yourself) |
| `vm/deploy-vm.ps1` | committed code (HEAD) into the farm's running VMs |
| `vm/hypervisor.ps1` | Windows hypervisor off while VMs install, back on after (admin + reboot) |
| `vm/vm-mouse.ps1` | a VM window's mouse integration (off: the mouse over the window can't move the bot's) |
| `lib/BotFarm.psm1` | shared: farm settings, VirtualBox, guest control, CPU pinning |

Scripts take `-Farm <folder>`; without it they use `$env:WIZARD_EYES_FARM`
(services get it) and then this PC's default farm
(`%LOCALAPPDATA%\wizard-eyes\botfarm.json`, set by `install.ps1`).

## farm.json

```json
{
  "name": "Herbitrage",
  "state_dir": "data/planner",
  "page": "http://127.0.0.1:8766",
  "coordination": "http://127.0.0.1:8765",
  "icon": "herbitrage.ico",
  "deploy_paths": ["bots"],
  "vm_defaults": {"memory_mb": 6144, "cpus": 6, "disk_gb": 64, "width": 1920, "height": 1200},
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
```

- Strings may hold `{engine}`, `{farm}`, `{repo}`, `{control}` (path of
  `bot-control.ps1`) and `{account:ROLE}` (from `farm.local.json`).
- A role's bot runs in its VM as a scheduled task (`task`, made when missing)
  running `C:\wizard-eyes-provision\ROLE.cmd`, which `bot-control.ps1` writes
  from `script` + `args` at every start. Its log: `C:\wizard-eyes-provision\ROLE.log`.
- Rests: any bot takes `--sleep 23:00-07:00` (every night, start / end moved by up to
  `--rest-jitter` minutes, default 45) and `--breaks 50-130:3-15` (a 3-15 min break every
  50-130 min) in its `args` (see `wizard_eyes/rest.py`). The heartbeat says when it rests, so
  the watchdog leaves it alone. A login screen saying the account is banned stops the bot and
  the watchdog alerts (Telegram) instead of restarting it.
- VMs reach the host's 127.0.0.1 at `10.0.2.2`.
- Each new VM's Windows user has a human name of its own (`janet`, `gordon`, ...: made once per VM, kept in
  `%LOCALAPPDATA%\wizard-eyes\vm-users.json`, never repeated); a clone of a template gets its own at its
  first start (answer-file CD). The password is shared (Credential Manager `wizard-eyes/vm-user`). VMs from
  before 2026-11-27 keep `bot`.
- Admin work in a VM needs no one at it: guest control isn't elevated, so `Invoke-GuestElevated`
  (lib) starts it with RunAs on the VM's desktop. The UAC consent prompt is off in bot VMs
  (`ConsentPromptBehaviorAdmin=0`: set by the install's post-install command, the lockdown, or
  `Enable-GuestSilentAdmin`), so that is silent. Where a prompt still shows (VMs built before), the host
  presses Alt+Y through VirtualBox's virtual keyboard, which reaches the secure desktop.
- `vpn` (on a role, or for the whole farm; `false` on a role leaves it out):
  - `{"provider": "proton"}` (the only VPN since 2026-11-28; ExpressVPN was dropped): Proton VPN through WireGuard. Each VM has
    a WireGuard config of its own, made on account.protonvpn.com (Downloads > WireGuard configuration) and
    stored on the host with `vpn-config.ps1 -Vm <vm> -Path <file.conf>` (Credential Manager
    `wizard-eyes/vpn/<vm>`; run it in your own terminal). Before the bot starts, `bot-control.ps1` puts it
    into the VM (`vm/guest-wireguard.ps1`, elevated): a WireGuard tunnel service that starts with Windows
    and reconnects by itself, and a firewall lock: outbound traffic is blocked except through the tunnel,
    to the VPN server, to the host (10.0.2.2) and DHCP, so with the tunnel down the VM has no internet,
    never its real address. The config is kept in the VM for SYSTEM and administrators only. A bot never
    starts unless the tunnel is up, the lock is on and the exit address isn't the host's. WireGuard is in
    the template (`vm/guest-template.ps1`).
  `-Action status` adds `vpn` (state, exit address, `lock`).
- The ROLE.cmd also sets `WIZARD_EYES_ROLE` and `WIZARD_EYES_COORDINATOR`
  (`coordination` as the VM sees it): every engine `Application` then sends a
  heartbeat (`wizard_eyes.heartbeat`), so any script counts as working on the
  page and is watched, with no code of its own.
- The page (`page`) must answer `GET /control` and take
  `POST /control {"role", "action"}`; the coordination service is the
  engine's `wizard_eyes.coordination`. A farm with no page of its own runs the
  engine's (services entry `{"name": "page", "module": "wizard_eyes.botfarm.page",
  "args": ["--service", "http://127.0.0.1:8765", "--control", "{control}", "--watchdog"],
  "check": "http://127.0.0.1:8766/control", "after": "coordination"}`).
- `alerts` (`{"telegram": "wizard_eyes.botfarm.alerts", "credential": "..."}`,
  credential optional, default `wizard-eyes/telegram`) and `phone`
  (`{"port": 8767, "module": "wizard_eyes.botfarm.remote"}`): modules `setup.ps1`
  runs (`status`, `pair`, `test` / `host`, `key`) for those steps.

## Python: `wizard_eyes.botfarm`

For a farm's own page / services (the farm's code uses them; nothing in them
knows one farm's bots):

| module | what |
| --- | --- |
| `farm` | `Farm.load([folder])`: name, state folder, role -> VM, as `Get-Farm` finds it |
| `alerts` | Telegram, send only: `Telegram.for_farm`, `Alerts` (own thread, no repeats within 30 min), `Watch` (watchdog states + failed meetings -> texts); command `pair` / `test` / `status` |
| `remote` | the page from the phone through `tailscale serve`: key (hash only) + host name, login page and cookie named after the farm; command `host` / `key` / `off` / `status` |
| `live` | `Live(folder, farm.vms).picture(role)`: a VM's screen as a small JPEG, without touching the VM |
| `control` | `Control(script, folder, farm.roles)`: runs `bot-control.ps1` for the page's buttons, one job per role, `look` = `-Action status` |
| `watchdog` | `Watchdog(control, folder, roles, bots, inspect, meetings=, progress=, no_cure=)`: once a minute per wanted bot: heard from (report or heartbeat) and progress changing? else look, then wait / alert / restart (3 an hour, 8 a day). Progress default: the heartbeat's `progress` (`Application.progress`) |
| `page` | `FarmPage(name, roles, service, control, watchdog, live, html=, get=, post=)`: the farm page (`/`, `/control` with heartbeats + watchdog, `/live/ROLE`) plus the farm's routes; `python -m wizard_eyes.botfarm.page` runs it as it is (`page.html`) |

Commands: `python -m wizard_eyes.botfarm.alerts status [--farm DIR]`; state
(chat, remote settings) goes in the farm's `state_dir`.

## Rules learnt the hard way

- VMs start pinned to the fast cores (`Start-FarmVm`): unpinned they were ~6x slower.
- GUI bots run from scheduled tasks on the VM's logged-in desktop, never over guest control directly.
- Moving the mouse into a screen corner stops a bot (pyautogui fail-safe); a resized VM window breaks its bot.
- The same game account must never play in two places.
