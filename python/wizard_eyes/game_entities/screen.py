from os import makedirs, listdir
from os.path import join, dirname
from typing import List, Union, Tuple
from dataclasses import dataclass
from uuid import uuid4

from . import player
from . import trees
from . import entity
from . import npcs
from . import items
from . import tile
from .colour import ColourCorrector
from ..constants import DEFAULT_ZOOM, DEFAULT_BRIGHTNESS, WHITEA
from ..game_objects.game_objects import GameObject


import cv2
import numpy


@dataclass
class TileColour:
    lower: tuple
    upper: tuple
    name: str


class ClickChecker(GameObject):
    """Object to verify the red/yellow click animation that plays after
    doing a click action. For now assume only one click at a time."""

    PATH_TEMPLATE = '{root}/data/game_screen/clicks/{name}.npy'

    TEMPLATES = ['red_0', 'red_1', 'red_2', 'red_3',
                 'yellow_0', 'yellow_1', 'yellow_2', 'yellow_3']

    DEFAULT_COLOUR = WHITEA

    def __init__(self, client, *_, **__):
        """Simplified init because this game object only operates one way."""
        super().__init__(client, client, template_names=self.TEMPLATES)
        self.invert = True
        self.match_method = cv2.TM_SQDIFF_NORMED
        self.active = False
        self.save_images = False
        """Whether to save images for debugging."""
        self.images = []
        """List of images to save as samples for debugging."""
        self.session_id = uuid4().hex[:8]
        """Unique ID for this session, images will be saved into a folder
        with this name."""
        self.x = -1
        self.y = -1
        self.state = ''
        self.red = None
        """Whether to look for a red click or a yellow click, or both!"""
        self.on_success = None
        """Callback to run on successfully finding a click icon."""
        self.on_failure = None
        """Callback to run on failing to find a click icon."""
        self.started_at = -float('inf')
        """Time when the click check started."""
        self.timeout_at = float('inf')
        """Time when the click checker should force a reset."""

    def draw(self):
        super().draw()
        if not self.get_bbox():
            return

        states = {'*state', 'click-state'}
        if self.client.args.show.intersection(states):
            x, y, _, _ = self.client.localise(self.x, self.y, 0, 0)
            state = self.state or 'none'
            cv2.putText(
                self.client.original_img, state, (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colour, 1
            )

    def update(self):
        """Manage state of the click checker, run the relevant callbacks.
        After the timeout is passed, the click checker will reset itself."""
        super().update()
        if self.client.time > self.timeout_at:
            self.reset()
        if not self.active:
            return

        self.state = self.identify(1)

        if self.save_images:
            img = self.img.copy()
            self.images.append(img)

        fail_condition = (
            self.client.time > self.timeout_at or
            'red' in self.state and self.red is False or
            'yellow' in self.state and self.red is True
        )
        if fail_condition:
            self.logger.debug('click check failed')
            self.do_image_save()
            self.on_failure()
            return

        success = False
        if self.red is True and 'red' in self.state:
            self.logger.debug('red click detected')
            success = True
        elif self.red is False and 'yellow' in self.state:
            self.logger.debug('yellow click detected')
            success = True
        elif self.state:
            self.logger.debug('click detected')
            success = True

        if success:
            self.do_image_save()
            self.on_success()

    def reset(self):
        """Reset the click checker to its default state."""
        self.active = False
        self.red = None
        self.on_success = None
        self.on_failure = None
        self.state = ''
        self.x = -1
        self.y = -1
        self.images = []
        self.started_at = -float('inf')
        self.timeout_at = float('inf')

    def do_image_save(self):
        """Save the images to a unique location on disk for debugging."""
        if not self.save_images:
            return

        target_dir = join(
            dirname(self.resolve_path()), 'samples', self.session_id
        )
        makedirs(target_dir, exist_ok=True)
        idx = len(listdir(target_dir))
        target_dir = join(target_dir, str(idx))
        makedirs(target_dir, exist_ok=True)

        for i, img in enumerate(self.images):
            path = join(target_dir, f'{i}.png')
            cv2.imwrite(path, img)

        self.logger.debug(
            f'saved {len(self.images)} images to {target_dir}')

    def start(self, x: int, y: int, red: Union[bool, None] = None,
              on_failure: callable = None, on_success: callable = None):
        """Start the click checker in a separate thread. It will automatically
        update every frame until the click animation is detected, or until
        the timeout is reached.

        :param int x: x coordinate of the click
        :param int y: y coordinate of the click
        :param bool red: If true consider a red click detection as success,
            yellow on false, and either on None.
        :param function on_failure: function to call if the click animation
            is not detected in time.
        :param function on_success: function to call if the click animation
            is detected in time.

        """

        if self.active:
            self.logger.error('click checker already active')
            return False

        # sanitise inputs
        if not callable(on_failure):
            on_failure = lambda: None
        if not callable(on_success):
            on_success = lambda: None

        self.on_failure = on_failure
        self.on_success = on_success
        self.started_at = self.client.time
        self.timeout_at = self.started_at + self.client.TICK
        self.red = red

        # set up red click bounding box
        h, w, = self.templates['red_0'].shape[:2]  # assume all same size
        offset = int(h / 2) + 2
        y1 = y - offset
        y2 = y + offset - 1
        x1 = x - offset
        x2 = x + offset - 1
        self.set_aoi(x1, y1, x2, y2)
        self.x = x
        self.y = y
        self.active = True

        return True


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
        self.player = player.Player('player', (0, 0), self.client, self)

    @property
    def tile_size(self):
        # assumes margin0% top down view at default zoom
        x1, y1, x2, y2 = self.player.tile_bbox()
        return x2 - x1 + 1

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

    def target_contains_points(
            self, target: GameObject, corners: Tuple[Tuple[int, int], ...],
            allow_partial: bool
    ):
        partials = []

        for corner in corners:
            if target.is_inside(*corner):
                partials.append(True)
            else:
                partials.append(False)

        if allow_partial and any(partials):
            return True
        else:
            return all(partials)

    def is_clickable(self, x1, y1, x2, y2, allow_partial=False):
        """Validate bounding box can be clicked without accidentally clicking
        UI elements"""

        corners = ((x1, y1), (x2, y2), (x2, y1), (x1, y2))

        # all points must be inside the client to be clickable
        if not self.target_contains_points(
                self.client, corners, allow_partial):
            return False

        # get a list of all the game screen objects that could block clicks
        blocking_elements = [self.client.banner, self.client.minimap]
        dynamic_ui = (self.client.tabs, self.client.chat, self.client.bank)
        for container in dynamic_ui:
            for widget in container.widgets:
                if widget.located:
                    blocking_elements.append(widget)

                if widget.interface.located:
                    blocking_elements.append(widget.interface)

        # check if any of the corners are inside a blocking element
        inside = list()
        for corner in corners:
            inside_something = False
            for element in blocking_elements:
                if element.is_inside(*corner):
                    inside_something = True
                    break
            inside.append(inside_something)

        # TODO: calculate what percentage inside other things is acceptable

        if allow_partial:
            return False in inside
        else:
            return set(inside) == {False}

    def find_highlighted_tiles(
            self, colours: List[TileColour], moe=.1, include_failures=False):
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
                if not section.any():
                    if include_failures:
                        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)
                        failed = TileColour(
                            name=f'failed-???-{colour.name}',
                            lower=colour.lower,
                            upper=colour.upper)
                        tiles.append((failed, (x1, y1, x2, y2)))
                    continue

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
