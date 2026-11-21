Backlog
=======

Support other game client layouts
---------------------------------

The engine currently requires **Resizable - Modern**. Game objects (tabs,
inventory, bank, chat, minimap, orbs...) are positioned from
`config/clients.json`, which describes the modern layout only.

Goal: also support **Resizable - Classic** and **Fixed - Classic**.

What already exists:
- `wizard_eyes.login.detect_layout()` tells the three layouts apart from a
  single screenshot, using only the positions of tabs relative to each other
  (and to the chat "All" button), so it doesn't depend on window size or
  where the UI is. Tested on real captures of all three
  (`tests/unit/test_layout.py` rebuilds them synthetically).
- Start-up warns when the layout isn't modern (`startup_checks`).

Likely work:
- Per-layout container definitions in `config/clients.json` (side panel as
  two tab rows with the panel between them; fixed is a rigid 765x503 frame),
  selected from the detected layout at start-up.
- Check which existing templates differ per layout (tab icons scored
  0.93-0.99 across layouts against modern-captured templates, slightly
  different backgrounds) and capture per-layout variants where needed.
- Minimap / orbs / GPS: positions and possibly scale differ in fixed mode.
- The logout button is the X by the minimap in modern, but a door tab in the
  lower tab row in classic layouts.

Re-login when logged out mid-run
--------------------------------

`--username` logs in at start-up only. A bot that's disconnected or hits the
6-hour logout won't log back in. Cheap approach: when existing per-frame
checks notice the side panel tabs / chat buttons have gone, run
`login.classify()` once and, if at the login screen or lobby, call
`login.log_in()`.

Dynamic sun RuneLite plugin (separate project)
----------------------------------------------

A RuneLite plugin for a dynamic sun / sky, to be developed as its own task.
Relevant to the engine because sky and lighting colours affect colour-based
detection: the player's current-tile highlight is cyan (#00FFFF), and a
near-cyan skybox (#00FFFA, as in the original profile) was picked up as the
tile when the camera was tilted low. Whatever the plugin does, the sky
should stay clear of the colours the engine detects (see
`config/runelite_required.json`, which now flags a cyan overworld skybox).

Handle the RuneLite sidebar while a bot runs
--------------------------------------------

Opening the sidebar narrows the game canvas: by 31px for the icon strip,
273px with a panel open (measured on a 2576px window). Right-anchored UI
(tabs, minimap, gauges) moves left, and the player's position (worked out
from the canvas centre in `Player.get_bbox`) shifts by half that.

What exists: title-bar arrow templates `runelite_ui/sidebar_closed.npy` /
`sidebar_open.npy` (published) tell the states apart (1.0 vs <=0.68); the
sidebar's width can be read from its flat grey (strip grey 30, panel grey 40).

Plan (agreed approach "(a)"):
- `Client.get_bbox()` returns the canvas (window minus sidebar width), so
  everything derived from it follows.
- `client.update()` checks the arrow each frame; on a change, clear the
  cached client rect and reset `located` on template-located objects (tabs,
  gauges, bank...) so they re-locate next frame.
- Positions set by hand with `set_aoi` are left alone; bots relying on
  fixed hand-set positions on the right of the screen pause while the
  sidebar is open (see script review below).

Update existing bot scripts for the engine changes
--------------------------------------------------

Go through each script in `wizard-eyes-experiments` one by one and fix it
as we see fit. Engine changes to account for: `--username` launch/login,
start-up camera fix (`--skip-camera` for interface-only bots), RuneLite
sidebar handling, `--window-index` now meaning "Nth client".

`set_aoi` usage to review (about 40 calls), in three groups:
1. Fixed positions picked with the mouse at start-up; break if UI moves:
   `bots/1-tick-prayer.py`, `cyborgs/1-tick-prayer.py`,
   `cyborgs/one-screen-multi-clicker.py`, `cyborgs/circuit-training.py`,
   `cyborgs/auto-cast.py`, `cyborgs/herb-cleaner-delux.py`,
   `cyborgs/chaos-altar.py`, `cyborgs/auto-hot-key.py`,
   `cyborgs/farmstruction/plant-all.py`, `experiments/modulo_time_pray_flick.py`.
2. Set from a template match in the script; fine if re-matched, stale if
   matched once: `bots/fishing/fisher.py`, `bots/construction/oak-larders.py`,
   `bots/runecrafting/true_blood.py`, `bots/combat/house_portal_menu.py`,
   `bots/combat/slayer_rewards_menu.py`, `bots/construction/bagged-plants.py`,
   `bots/farming/tithe_farm.py`, `bots/fletching/bolt-fletcher.py`,
   `bots/crafting/edgeville-bracelets.py`.
3. Recomputed every frame; follow the UI automatically: agility course
   loot, `bots/mining/veins.py`, `bots/combat/barrows.py`,
   `bots/combat/slayer_task.py` loot, `bots/fishing/karambwanji.py`.

Slayer script: set up ground markers dynamically
------------------------------------------------

`bots/combat/slayer_task.py` relies on the Ground Markers plugin, which was
always set up by hand. With the engine now able to drive the RuneLite
sidebar (open it, open Configuration, search a plugin, edit settings - done
for the Skybox colour), the slayer bot could set up its markers itself per
task location (e.g. via the plugin's import/export, `groundMarker.showImportExport`
is on). The sidebar handling above is a prerequisite.
