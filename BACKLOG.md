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
