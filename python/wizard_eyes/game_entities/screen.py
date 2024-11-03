from enum import Enum
from os import makedirs, listdir
from os.path import join, dirname
from typing import List, Union, Tuple, Dict, Type, Optional
from collections import defaultdict
from dataclasses import dataclass
from uuid import uuid4
import math

from . import player
from . import trees
from . import entity
from . import npcs
from . import items
from . import tile
from . import other_player
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
            self.logger.debug('click checker already active')
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


class Grid(GameObject):
    """"""

    GRID_COLOURS = [
        (52, 52, 52, 255),
    ]

    def __init__(self, *args, **kwargs):
        self.parent: GameScreen = args[1]
        super().__init__(*args, **kwargs)
        self.set_aoi(*self.client.get_bbox())
        self._tiles: Dict[Tuple[int, int], numpy.ndarray] = {}
        self._player_contour = None

    def update(self):
        super().update()

        img_colour = self.client.get_img_at(
            self.get_bbox(), mode=self.client.BGRA
        )

        grid_mask = None
        for colour in self.GRID_COLOURS:
            if grid_mask is None:
                grid_mask = cv2.inRange(img_colour, colour, colour)
            else:
                grid_mask = cv2.bitwise_or(
                    grid_mask, cv2.inRange(img_colour, colour, colour)
                )
        if grid_mask is None:
            self.client.logger.warning('No colours defined for grid.')
            return

        player_mask = cv2.inRange(
            img_colour,
            self.parent.player.TILE_COLOURA,
            self.parent.player.TILE_COLOURA,
        )
        player_mask = cv2.dilate(
            player_mask, numpy.ones((5, 5), numpy.uint8), iterations=1
        )

        # find center of player tile
        player_contours = cv2.findContours(
            player_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[0]
        if not player_contours:
            self.client.logger.debug('Cannot find player tile??')
            return
        self._player_contour = None
        for i, contour in enumerate(player_contours):
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            player_area = cv2.contourArea(contour)
            if not (len(approx) == 4 and 18500 > player_area > 280):
                continue

            self._player_contour = contour
            break

        if self._player_contour is None:
            self.client.logger.debug('Cannot find player tile??')
            return

        # TODO: factor out internal attribute access
        px, py, pw, ph = cv2.boundingRect(self._player_contour)
        px1, py1, px2, py2 = self.globalise(px, py, px + pw - 1, py + ph - 1)
        self.parent.player._tile_bbox = (px1, py1, px2, py2)

        p_area = cv2.contourArea(contour)

        mask = cv2.bitwise_or(grid_mask, player_mask)
        mask = cv2.dilate(mask, numpy.ones((5, 5), numpy.uint8), iterations=2)
        contours, hierarchy = cv2.findContours(
            255 - mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        self._tiles = tiles = {}

        # test = numpy.zeros_like(mask)
        # cv2.drawContours(
        #     test, contours, -1, (255, 255, 255), 1
        # )

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(contour)

            cx = int(x + w / 2)
            cy = int(y + h / 2)

            # rx = (x - px) // pw
            # ry = (y - py) // ph

            # if not(len(approx) == 4 and p_area * 2.5 > area > 100):
            if not(p_area * 2.5 > area > 100):
                continue

            tiles[(cx, cy)] = contour
            # TODO: develop this shuffling algorithm to actually work.
            # self.shuffle_into_mapping(rx, ry, contour, tiles, px, py, pw, ph)

        return

    def shuffle_into_mapping(self, x, y, contour, mapping, px, py, pw, ph):

        if (x, y) in mapping:
            # we must determine which contour is closer to the player
            # and shuffle the other one further away

            # first determine if contours are mostly different in x or y
            # this will determine which direction we shuffle the further tile
            x1, y1, w1, h1 = cv2.boundingRect(mapping[(x, y)])
            dist_x1 = abs(x1 - px)
            dist_x = abs(x - px)
            dist_y1 = abs(y1 - py)
            dist_y = abs(y - py)

            if max(dist_x1, dist_x) > max(dist_y1, dist_y):
                # x is the main difference
                if dist_x1 > dist_x:
                    # the existing tile is further in x, we need to move it
                    if x1 - px > 0:
                        # we need to move the existing tile to the right
                        rx = x + 1
                        ry = y
                    else:
                        # we need to move the existing tile to the left
                        rx = x - 1
                        ry = y
                    existing_contour = mapping[(x, y)]
                    mapping[(x, y)] = contour
                    self.shuffle_into_mapping(
                        rx, ry, existing_contour, mapping, px, py, pw, ph
                    )
                else:
                    # the new tile is further in x, shuffle it over
                    if x - px > 0:
                        rx = x + 1
                        ry = y
                    else:
                        rx = x - 1
                        ry = y
                    self.shuffle_into_mapping(
                        rx, ry, contour, mapping, px, py, pw, ph
                    )
            else:
                # y is the main difference
                if dist_y1 > dist_y:
                    # the existing tile is further in y, we need to move it
                    if y1 - py > 0:
                        # we need to move the existing tile down
                        rx = x
                        ry = y + 1
                    else:
                        # we need to move the existing tile up
                        rx = x
                        ry = y - 1
                    existing_contour = mapping[(x, y)]
                    mapping[(x, y)] = contour
                    self.shuffle_into_mapping(
                        rx, ry, existing_contour, mapping, px, py, pw, ph
                    )
                else:
                    # the new tile is further in y, shuffle it over
                    if y - py > 0:
                        ry = y + 1
                        rx = x
                    else:
                        ry = y - 1
                        rx = x
                    self.shuffle_into_mapping(
                        rx, ry, contour, mapping, px, py, pw, ph
                    )

        else:
            # the space is unoccupied, so we can just slot it in
            mapping[(x, y)] = contour

    def draw(self):
        super().draw()
        if self.client.args.show.intersection({'grid'}):
            self._draw_grid()

    def _draw_grid(self):

        if self._player_contour is None:
            return

        for (rx, ry), contour in enumerate(self._tiles.items()):
            x, y, w, h = cv2.boundingRect(contour)

            if (rx, ry) == (0, 0):
                colour = self.parent.player.colour
            else:
                colour = (0, 255, 0)

            cv2.rectangle(
                self.client.original_img, (x, y), (x+w, y+h), colour, 1
            )
            cv2.putText(
                self.client.original_img,
                str((rx, ry)),
                (int(x + w / 2),
                 int(y + h / 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (0, 255, 0),
                1
            )

    def get_tile_bbox(self, x, y, ox=0, oy=0):

        if self._player_contour is None:
            return None

        px, py, pw, ph = cv2.boundingRect(self._player_contour)
        px1 = int(px + pw / 2)
        py1 = int(py + ph / 2)

        x_candidates = []
        y_candidates = []
        for cx, cy in self._tiles.keys():
            if abs(cx - px1) < pw / 2:
                y_candidates.append((cx, cy))
            if abs(cy - py1) < ph / 2:
                x_candidates.append((cx, cy))

        x_candidates.sort(key=lambda c: c[0])
        y_candidates.sort(key=lambda c: c[1])

        player_x_index = None
        player_x_dist = float('inf')
        player_y_index = None
        player_y_dist = float('inf')
        for i, (cx, cy) in enumerate(x_candidates):
            dist = math.sqrt(abs(cx - px1) ** 2 + abs(cy - py1) ** 2)
            if dist < player_x_dist:
                player_x_dist = dist
                player_x_index = i
        for i, (cx, cy) in enumerate(y_candidates):
            dist = math.sqrt(abs(cx - px1) ** 2 + abs(cy - py1) ** 2)
            if dist < player_y_dist:
                player_y_dist = dist
                player_y_index = i

        if player_x_index + x < 0 or player_y_index + y < 0:
            return None
        if player_x_index + x >= len(x_candidates) or player_y_index + y >= len(y_candidates):
            return None

        try:
            x_candidate = x_candidates[player_x_index + x]
            y_candidate = y_candidates[player_y_index + y]
        except IndexError:
            return None

        # find the closest tile to the candidate
        cx, _ = x_candidate
        _, cy = y_candidate
        min_dist = float('inf')
        candidate = None
        for (tx, ty), contour in self._tiles.items():
            dist = math.sqrt(abs(tx - cx) ** 2 + abs(ty - cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                candidate = contour

        if candidate is None:
            return None

        x, y, w, h = cv2.boundingRect(candidate)

        x1 = x + ox * w
        y1 = y + oy * h
        x2 = x1 + w - 1
        y2 = y1 + h - 1

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        return self.globalise(x1, y1, x2, y2)


class GameScreen(GameObject):
    """Container class for anything displayed within the main game screen."""

    def __init__(self, client, zoom=DEFAULT_ZOOM):
        super().__init__(client, client)
        self.setup_safe_area()
        self.client = client
        self._player = None
        self.default_npc = npcs.NPC
        self.grid = Grid(self.client, self)
        self.zoom = zoom
        self.tile_marker = tile.TileMarker(zoom, self.client, self.client)
        self.cc = ColourCorrector(DEFAULT_BRIGHTNESS, self.client, self.client)
        self.buffer: Dict[str, List[Type[npcs.NPC]]] = defaultdict(list)
        self.player = player.Player('player', (0, 0), self.client, self)

        self._custom_type = None
        self._custom_class = None

        class DistFromPlayerEnum(Enum):
            """Enum to represent methods for calculating distance from player."""
            screen = self.player.base_width
            tile = 1
            minimap = self.client.gauges.minimap.tile_size

        self.dfp = DistFromPlayerEnum
        """Enum to represent methods for calculating distance from player."""

    def update(self):
        super().update()
        self.player.update()
        self.grid.update()

    @property
    def tile_size(self):
        # assumes margin0% top down view at default zoom
        x1, y1, x2, y2 = self.player.tile_bbox()
        return x2 - x1 + 1

    def setup_safe_area(self):
        """Set up bounding box for game screen as a 10% margin inside client
        window."""

        x1, y1, x2, y2 = self.client.get_bbox()
        w = x2 - x1
        h = y2 - y1

        x1 = int(x1 + w * 0.1 + self.client.banner.height)
        y1 = int(y1 + h * 0.1)
        x2 = int(x2 - w * 0.1)  # TODO: check for runelite plugins sidebar
        y2 = int(y2 - h * 0.1)
        self.set_aoi(x1, y1, x2, y2)

    def add_to_buffer(self, npc: Type[npcs.NPC]):
        if issubclass(npc.__class__, npcs.NPC):
            npc.reset()  # noqa
            self.buffer[npc.class_name()].append(npc)
            return
        del npc

    def set_custom_type(self, type_: str):
        self._custom_type = type_

    def clear_custom_type(self):
        self._custom_type = None

    def set_custom_class(self, klass: Type[entity.GameEntity]):
        self._custom_class = klass

    def clear_custom_class(self):
        self._custom_class = None

    def create_game_entity(
            self,
            type_,
            *args,
            entity_templates=None,
            **kwargs
    ) -> entity.GameEntity:
        """Factory method to create entities from this module."""

        class_name_mapping = {
            'npc': npcs.NPC,
            'player': other_player.OtherPlayer,
        }

        if type_ in {'npc', 'npc-tag', 'npc-slayer', 'player'}:
            # old NPC objects have already been initialised
            # re-use it to save CPU time of creating new objects

            klass = class_name_mapping.get(type_, self.default_npc)

            if self.buffer[klass.class_name()]:
                npc = self.buffer[klass.class_name()].pop(-1)
                name, key, *_ = args
                npc.name = name
                npc.key = key
                npc.re_init()
            # otherwise create a new one
            else:
                npc = klass(*args, **kwargs)

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
        elif type_ == self._custom_type:
            _entity = self._custom_class(*args, **kwargs)
            if entity_templates:
                _entity.load_templates(entity_templates)
                _entity.load_masks(entity_templates)
            return _entity
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
                self, corners, allow_partial):
            return False

        # get a list of all the game screen objects that could block clicks
        blocking_elements = [self.client.banner, self.client.gauges]
        dynamic_ui = (self.client.tabs, self.client.chat, self.client.bank)
        for container in dynamic_ui:
            for widget in container.widgets:
                if not widget.state:
                    continue
                if widget.located:
                    blocking_elements.append(widget)
                if widget.interface.located and widget.selected:
                    blocking_elements.append(widget.interface)

        # check if any of the corners are inside a blocking element
        inside = list()
        for corner in corners:
            inside_something = False
            for element in blocking_elements:
                if element.is_inside(*corner):
                    inside_something = True
                    self.client.logger.debug(
                        f'blocked by: {element} at {corner}'
                    )
                    break
            inside.append(inside_something)

        # TODO: calculate what percentage inside other things is acceptable

        if allow_partial:
            return not all(inside)
        else:
            return not any(inside)

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

                try:
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
                    assert len(corners) == 5
                except AssertionError:
                    if include_failures:
                        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)
                        failed = TileColour(
                            name=f'failed-notsquare-{colour.name}',
                            lower=colour.lower,
                            upper=colour.upper)
                        tiles.append((failed, (x1, y1, x2, y2)))
                    continue
                except cv2.error:
                    if include_failures:
                        x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)
                        failed = TileColour(
                            name=f'failed-cornerharris-{colour.name}',
                            lower=colour.lower,
                            upper=colour.upper)
                        tiles.append((failed, (x1, y1, x2, y2)))
                    continue

                x1, y1, x2, y2 = self.client.globalise(x1, y1, x2, y2)

                tiles.append((colour, (x1, y1, x2, y2)))

        return tiles

    def find_grid(self):

        img = self.client.original_img
        greya = numpy.array([52, 52, 52, 255], dtype=numpy.uint8)
        mask = cv2.inRange(img, greya, greya)
        dilated = cv2.dilate(mask, numpy.ones((5, 5), numpy.uint8), iterations=1)

        contours, hierarchy = cv2.findContours(255 - dilated, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        output = numpy.zeros_like(img)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(contour)
            if area > 4000 or area < 100:
                continue
            if len(approx) == 4 and 2000 > area > 500:
                # cv2.drawContours(output, [c], -1, (255, 0, 0, 255))
                colour = (255, 0, 0)
            else:
                colour = (0, 0, 255)
            # cv2.rectangle(output, (x, y), (x+w, y+h), colour, 1)
            cv2.drawContours(output, [contour], -1, colour)
            cv2.putText(output, f'{i}', (int(x + w / 2), int(y + h / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1)
            cv2.putText(output, f'{area}', (int(x + w / 2), int(y + h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1)
        self.client.screen.show_img(output)
