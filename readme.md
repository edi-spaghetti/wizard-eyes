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

Install pip packages the standard way

```commandline
pip install -r requirements.txt
```

Note that the tesserocr link is specific to my build of windows and python.
You may need to swap this for a different release.

AutoHotKey
----------

You will also need to download AutoHotKey v1.1.
It's deprecated now, and at some point I will upgrade (or even better remove this dependency entirely)
but for now the project is set up to use that specific version.

Download it from here: https://www.autohotkey.com/

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

Currently windows only. Probably not going to port to Linux.
The main things that would need to be updated for linux support are screen capture
and mouse and keyboard, everything is modular, so in theory it wouldn't be too hard
to refactor.
