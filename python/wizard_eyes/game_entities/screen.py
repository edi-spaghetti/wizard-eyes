from typing import List
from dataclasses import dataclass

from . import player
from . import trees
from . import entity
from . import npcs
from . import items
from . import tile
from .colour import ColourCorrector
from ..constants import DEFAULT_ZOOM, DEFAULT_BRIGHTNESS
from ..game_objects.game_objects import GameObject


import cv2
import numpy


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
        self.tile_marker = tile.TileMarker(zoom, self.client, self.client)
        self.cc = ColourCorrector(DEFAULT_BRIGHTNESS, self.client, self.client)
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
        # assumes margin0% top down view at default zoom
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

    def find_highlighted_tiles(
            self, colours: List[TileColour], moe=.4, include_failures=False):
        """Find tiles of a given colour within the game screen.

        :param List[TileColour] colours: Colours of tile to find, and the name
            we should associate with that colour.
        :param moe: margin of error, relative % of the player tile size
        :param include_failures: if True, add failed matches to results
            with a different name in the format 'failed-<reason>-<colour>'

        :returns: Results in the form (<tile colour>, <bbox>)
        :rtype: List[Tuple[TileColour, Tuple[int, int, int, int]]
        """

        p = self.client.game_screen.player
        px1, py1, px2, py2 = p.tile_bbox()
        ph = py2 - py1 + 1
        pw = px2 - px1 + 1

        cx1, cy1, _, _ = self.client.get_bbox()
        img = self.client.hsv_img

        tiles = []
        # TODO: mask merging; red needs two hsv values to capture the full
        #       range, so we could have multiple colours with the same
        #       TileColour.name and then merge them.
        for colour in colours:
            mask = cv2.inRange(img, colour.lower, colour.upper)
            black = numpy.zeros_like(mask)

            # opencv have a habit of changing the number of return values for
            # this function between versions. FYI if this breaks in the future.
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for i, contour in enumerate(contours):
                x1, y1, w, h = cv2.boundingRect(contour)
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                condition1 = (
                    pw * (1 - moe) < w < pw * (1 + moe) and
                    ph * (1 - moe) < h < ph * (1 + moe))
                condition2 = (
                    hierarchy[0][i][2] < 0  # no parent
                )
                if not condition1:
                    if include_failures:
                        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)
                        failed = TileColour(
                            name=f'failed-{w}x{h}-{colour.name}',
                            lower=colour.lower,
                            upper=colour.upper)
                        tiles.append((failed, (x1, y1, x2, y2)))
                    continue
                if not condition2:
                    if include_failures:
                        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)
                        failed = TileColour(
                            name=f'failed-parent-{colour.name}',
                            lower=colour.lower,
                            upper=colour.upper)
                        tiles.append((failed, (x1, y1, x2, y2)))
                    continue

                margin = 10
                black = cv2.drawContours(
                    black, contours, i, (255, 255, 255), cv2.FILLED)
                # use only a section to greatly improve performance
                section = black[y1-margin:y2+margin, x1-margin:x2+margin]
                dst = cv2.cornerHarris(section, 5, 3, 0.04)
                ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
                dst = numpy.uint8(dst)
                (
                    ret, labels,
                    stats, centroids
                ) = cv2.connectedComponentsWithStats(dst)
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    100, 0.001
                )
                corners = cv2.cornerSubPix(
                    section,
                    numpy.float32(centroids),
                    (5, 5),
                    (-1, -1),
                    criteria
                )
                if len(corners) != 5:
                    if include_failures:
                        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)
                        failed = TileColour(
                            name=f'failed-notsquare-{colour.name}',
                            lower=colour.lower,
                            upper=colour.upper)
                        tiles.append((failed, (x1, y1, x2, y2)))
                    continue

                x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)

                tiles.append((colour, (x1, y1, x2, y2)))

        return tiles
