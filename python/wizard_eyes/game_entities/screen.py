from typing import List
from dataclasses import dataclass

from . import player
from . import trees
from . import entity
from . import npcs
from . import items
from . import tile
from ..constants import DEFAULT_ZOOM
from ..game_objects.game_objects import GameObject


import cv2


@dataclass
class TileColour:
    lower: tuple
    upper: tuple
    name: str


class RedClick(GameObject):
    """Object to represent the red click animation that plays after
    doing a click action."""

    PATH_TEMPLATE = '{root}/data/game_screen/{name}.npy'

    TEMPLATES = ['red_click_1', 'red_click_2', 'red_click_3']

    def __init__(self, client, *_, **__):
        """Simplified init because this game object only operates one way."""
        super().__init__(client, client, template_names=self.TEMPLATES)
        self.invert = True
        self.match_method = cv2.TM_SQDIFF_NORMED


class GameScreen(object):
    """Container class for anything displayed within the main game screen."""

    def __init__(self, client, zoom=DEFAULT_ZOOM):
        self.client = client
        self._player = None
        self.default_npc = npcs.NPC
        self.zoom = zoom
        self.tile_marker = tile.TileMarker(zoom, self.client, self)
        self.npc_buffer: List[npcs.NPC] = []

    @property
    def player(self):
        if self._player is None:
            names = [f'player_marker_{self.zoom}',
                     'player_blue_splat', 'player_red_splat']
            _player = player.Player(
                'player', (0, 0), self.client, self, template_names=names)
            _player.load_masks(names)
            self._player = _player

        return self._player

    @property
    def tile_size(self):
        # assumes 100% top down view at default zoom
        template = self._player.templates[f'player_marker_{self.zoom}']
        width, _, _ = template.shape
        return width

    def add_to_buffer(self, npc):
        self.npc_buffer.append(npc)

    def create_game_entity(self, type_, *args,
                           entity_templates=None, **kwargs):
        """Factory method to create entities from this module."""

        if type_ in {'npc', 'npc_tag'}:
            # old NPC objects have already been initialised
            # re-use it to save CPU time of creating new objects
            if self.npc_buffer:
                npc = self.npc_buffer.pop(-1)
                name, key, *_ = args
                npc.name = name
                npc.key = key
            # otherwise create a new one
            else:
                npc = self.default_npc(*args, **kwargs)

            # TODO: re-implement optional default templates for NPCs
            #       for now they take up too much CPU and aren't used
            # templates = ['player_blue_splat', 'player_red_splat']
            # npc.load_templates(templates)
            # npc.load_masks(templates)

            return npc
        # TODO: tree factory
        elif type_ == 'oak':
            tree = trees.Oak(*args, **kwargs)
            return tree
        elif type_ == 'willow':
            tree = trees.Willow(*args, **kwargs)
            return tree
        elif type_ == 'blisterwood':
            tree = trees.Blisterwood(*args, **kwargs)
            return tree
        elif type_ == 'magic':
            tree = trees.Magic(*args, **kwargs)
            return tree
        elif type_ == 'item':
            item = items.GroundItem(*args, **kwargs)
            if entity_templates:
                item.load_templates(entity_templates)
                item.load_masks(entity_templates)
            return item
        else:
            _entity = entity.GameEntity(*args, **kwargs)
            if entity_templates:
                _entity.load_templates(entity_templates)
                _entity.load_masks(entity_templates)
            return _entity

    def is_clickable(self, x1, y1, x2, y2, allow_partial=False):
        """Validate bounding box can be clicked without accidentally clicking
        UI elements"""

        result = True

        corners = ((x1, y1), (x2, y2), (x2, y1), (x1, y2))
        partials = list()
        for corner in corners:
            offset = (self.client.margin_left, self.client.margin_top,
                      -self.client.margin_right, -self.client.margin_bottom)
            if self.client.is_inside(*corner, offset=offset):
                if allow_partial:
                    partials.append(True)
                else:
                    partials.append(False)
            else:
                if allow_partial:
                    partials.append(False)
                else:
                    return False

        if allow_partial and not any(partials):
            return False

        partials = list()
        fixed_ui = (self.client.banner, self.client.minimap,
                    self.client.tabs, self.client.chat)
        for element in fixed_ui:
            for corner in corners:
                if element.is_inside(*corner):
                    if allow_partial:
                        partials.append(False)
                    else:
                        return False
                else:
                    partials.append(True)
                # TODO: random chance if close to edge

        if allow_partial and not any(partials):
            return False

        # TODO: bank
        partials = list()
        dynamic_ui = (self.client.tabs, self.client.chat)
        for element in dynamic_ui:
            # TODO: method on AbstractInterface to determine if open
            #       for now, assume they are open
            for corner in corners:
                if element.is_inside(*corner):
                    if allow_partial:
                        partials.append(False)
                    else:
                        return False
                else:
                    partials.append(True)
                # TODO: random chance if close to edge

        if allow_partial and not any(partials):
            return False

        return result

    def find_highlighted_tiles(self, colours: List[TileColour]):

        cx1, cy1, _, _ = self.client.get_bbox()
        img = self.client.hsv_img

        tiles = []
        for colour in colours:
            mask = cv2.inRange(img, colour.lower, colour.upper)
            contours = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = contours[0] if len(contours) == 2 else contours[1]

            # TODO: this should be relative to zoom level
            #       player marker_tile shape +- 10%
            min_area = 45 ** 2
            max_area = 55 ** 2
            for c in contours:
                area = cv2.contourArea(c)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(c)
                    x1, y1, x2, y2 = cx1 + x, cy1 + y, cx1 + x + w, cy1 + y + h

                    tiles.append((colour, (x1, y1, x2, y2)))

        return tiles
