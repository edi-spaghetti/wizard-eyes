Wizard Eyes
===========

A colour bot engine for Runescape. The general goal of this project was to write
a fully fledged scripting engine that works entirely on visual information taken
from an unadulterated client. Hopefully avoid bans because all they see is official client.
No under-the-hood java hi-jinx necessary. Just see colour; click colour.

Since Jagex announced RuneLite as an 'official' client, some of its features
have been leveraged to make certain things easier,
but as I've learned things making this project, the plugins are actually becoming
less and less necessary - I may even start scaling back their use to bring it back
in line with the original plan.

Features include;

 - Auto-scaling coordinates based on window size and client
 - Application templates for easy script setup.
 - Input utilities for mouse and keyboard
 - Coordinate and mapping system for world coordinates with map-making tools.
 - Dynamic projection matrix for 3D world coordinates to 2D screen space
 - Dynamically generated UIs based on templating.
 - Supports minimap, minimap orbs, xp, context-aware mouse clicks, game screen 
   entity bounding boxes, interface support, inventory & other tabs,
   world hopping and much more.

One final note - I will not be releasing any finished scripts beyond a few examples in the scripts folder.
They are there to give working (TODO: fix half of them... lol) examples of some of the features of the engine.
I have had more fun writing code to play Runescape than I ever had playing it normally.
I want you to do the same.

I have personally botted my account to max with this code. You can too.

Setup
=====

pip
---

Use Python 3.9 (the pinned tesserocr wheel is built for cp39 on Windows).
Create a virtual environment, install the pinned packages, then install the
engine itself in editable mode so your own scripts can `import wizard_eyes`
from anywhere:

```commandline
py -3.9 -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m pip install -e .
```

You will also need tesseract data for OCR. The engine looks for a `tessdata`
folder *next to* this repo (i.e. `<repo>/../tessdata`), and only needs the
English model:

```commandline
mkdir ..\tessdata
curl -L -o ..\tessdata\eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
```

If you're not using any OCR features you can skip this step.

Assets
------

Templates, masks and map metadata live in `data/`, which isn't in git.
They're stored remotely (Cloudflare R2) and downloaded on demand: whenever
the engine loads a file from `data/`, it checks it against a manifest of
content hashes and fetches it if it's missing or out of date. The manifest
is re-checked once per run with a conditional request, so if nothing has
changed nothing is downloaded. If remote assets aren't configured, or the
network is down, the engine just uses whatever is on disk.

To configure, copy `config/assets.example.json` to
`config/assets.local.json` (gitignored) and fill in your R2 account id,
bucket and API token keys. Environment variables override the file:
`WIZARD_ASSETS_BACKEND` (`r2`, `local` or empty to disable),
`WIZARD_ASSETS_R2_ACCOUNT_ID`, `WIZARD_ASSETS_R2_BUCKET`,
`WIZARD_ASSETS_R2_ACCESS_KEY_ID`, `WIZARD_ASSETS_R2_SECRET_ACCESS_KEY`
and `WIZARD_ASSETS_LOCAL_DIR` (a folder used as storage by the `local`
backend, handy for testing).

```commandline
python utility/assets.py status              # compare local files with the manifest
python utility/assets.py sync                # download everything (or --prefix tabs/)
python utility/assets.py publish --dry-run   # show what would be uploaded
python utility/assets.py publish             # upload new/changed assets + manifest
```

Which files get published is controlled by `config/assets_include.json`.
Files are stored by content hash and never deleted remotely, so publishing
only uploads what changed.

Image Scaling
-------------

Make sure your monitor is set to 100% image scaling.

RuneLite
--------

I have tried to keep the amount of Runelite configuration to a bare minimum,
as the point of this project was to write scripts that work on the 'vanilla' client.

You must use the modern layout for the config to pick up interface locations.

Runelite requires the camera be a completely top down view, facing north if you're going to try to interact with
game entities like monsters, objects etc. It's not required if you're just working
with interfaces (e.g. bank, inventory etc.). The camera must also be set to default zoom.

Sometimes you might need to NPC highlight and ground item highlight plugins
for some additional support with those features.

Compatibility
=============

Currently, windows only. Probably not going to port to Linux.
The main things that would need to be updated for linux support are screen capture
and mouse and keyboard, everything is modular, so in theory it wouldn't be too hard
to refactor, but the main issue is that bot detection on Linux is significantly
stricter than on Windows, so it's overall better to stay on Windows.
