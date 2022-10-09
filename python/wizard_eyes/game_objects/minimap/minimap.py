import math

import cv2
import numpy

from .gps import GielenorPositioningSystem
from ..game_objects import GameObject
from ..personal_menu import LogoutButton
from ...constants import (
    FILL,
    WHITE,
    BLACK,
)


class MiniMap(GameObject):
    """
    Represent the minimap orb in the top right of the screen.
    It is responsible for identifying templates in the minimap image, as well
    as running the GPS system.
    """

    PATH_TEMPLATE = '{root}/data/minimap/{name}.npy'
    RUNESCAPE_SURFACE = 0
    TAVERLY_DUNGEON = 20

    DEFAULT_COLOUR = (0, 0, 255, 255)

    def __init__(self, client, parent, logging_level=None, **kwargs):
        self.logout_button = LogoutButton(client, parent)
        super(MiniMap, self).__init__(
            client, parent, config_path='minimap.minimap',
            logging_level=logging_level, **kwargs,
        )

        self.updated_at = None
        self._img_colour = None

        self.gps = GielenorPositioningSystem(self.client, self)

        self._mask = None
        self.create_mask()

        # container for identified items/npcs/symbols etc.
        self._icons = dict()
        self._histograms = dict()

    @property
    def img_colour(self):
        """
        Slice the current client colour image on current object's bbox.
        This should only be used for npc/item etc. detection in minimap orb.
        Because these objects are so small, and the colours often quite close,
        template matching totally fails for some things unelss in colour.
        """
        if self.updated_at is None or self.updated_at < self.client.time:

            # slice the client colour image
            cx1, cy1, cx2, cy2 = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            img = self.client.original_img
            i_img = img[y1 - cy1:y2 - cy1 + 1, x1 - cx1:x2 - cx1 + 1]

            # process a copy of it
            i_img = i_img.copy()
            i_img = cv2.cvtColor(i_img, cv2.COLOR_BGRA2BGR)

            # update caching variables
            self._img_colour = i_img
            self.updated_at = self.client.time

        return self._img_colour

    def gen_histogram(self, img, mask):
        hist = cv2.calcHist(
            [img], [0, 1, 2], mask,
            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def load_histograms(self, config: dict):
        """
        Load histograms for cross-referencing template matching.
        Assumes templates and masks are already loaded.
        """

        histograms = dict()
        for name, data in config.items():
            template = self.templates.get(name)
            mask = self.masks.get(name)
            hist = self.gen_histogram(template, mask)
            histograms[name] = {'hist':  hist}
            histograms[name]['func'] = config.get(name, {}).get('func')
            histograms[name]['value'] = config.get(name, {}).get('value')

        self._histograms = histograms
        return histograms

    def update(self, auto_gps=True, threshold=0.99):
        """
        Basic update method for minimap. Should be run once per frame.
        Returns data from it's internal methods, which are run_gps and
        identify.

        :param auto_gps: If true, the coordinates will automatically be updated
            according to default parameters.
            If false, then it is is then up to the implementing application
            to do error filtering on these results.
        :param threshold: Value for template matching.
        """

        x, y = self.gps.update(auto=auto_gps)

        # TODO: auto entity generation
        icons = self.identify(threshold=threshold)

        return (x, y), icons

    # minimap icon detection methods

    def identify(self, threshold=1, method=cv2.TM_CCORR_NORMED):
        """
        Identify items/npcs/icons etc. on the minimap
        :param threshold:
        :param method: template matching method to use
        :return: A list of matches items of the format (item name, x, y)
            where x and y are tile coordinates relative to the player position
        """

        marked = set()
        results = set()
        assert method in (cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED)

        # reset mark on all icons, so know which ones we've checked
        for i in self._icons.values():
            i.refresh()

        for name, template in self.templates.items():

            try:
                ty, tx, tz = template.shape
                img = self.img_colour
            except ValueError:
                ty, tx = template.shape
                img = self.img

            func = self._histograms.get(name, {}).get('func')
            value = self._histograms.get(name, {}).get('value')

            matches = cv2.matchTemplate(
                img, template, method,
                mask=self.masks.get(name)
            )

            if method == cv2.TM_CCORR_NORMED:
                (my, mx) = numpy.where(matches >= threshold)
            else:  # TM_SQDIFF_NORMED
                (my, mx) = numpy.where(matches <= threshold)

            for y, x in zip(my, mx):

                cx = x + tx
                cy = y + ty

                if self._histograms and self._histograms.get(name):
                    candidate_img = self.img_colour[y:cy, x:cx]
                    candidate_hist = self.gen_histogram(
                        candidate_img, self.masks.get(name))
                    diff = cv2.compareHist(
                        self._histograms[name]['hist'], candidate_hist,
                        cv2.HISTCMP_CHISQR)
                    if not func(diff, value):
                        continue

                px, py, _, _ = self.client.game_screen.player.mm_bbox()
                mm_x, mm_y, _, _ = self.get_bbox()
                px = px - mm_x + 1
                py = py - mm_y + 1

                # calculate item relative pixel coordinate to player
                rx = x - px
                ry = y - py

                # guard statement prevents two templates matching the same
                # icon, which would cause duplicates
                if (rx, ry) in marked:
                    continue
                marked.add((rx, ry))
                results.add((name, (rx, ry)))

        return results

    def generate_entities(self, positions, entity_templates=None, 
                          offsets=None):
        """Generate game entities from results of :meth:`MiniMap.identify`"""

        checked = set()

        for name, (x, y) in positions:

            # rx = int((x - self.config['width'] / 2) * self.scale)
            # ry = int((y - self.config['height'] / 2) * self.scale)

            # convert pixel coordinate into tile coordinate
            tx = x // self.tile_size
            ty = y // self.tile_size

            # calculate icon's global map coordinate
            px, py = self.client.minimap.minimap.gps.get_coordinates()
            if isinstance(px, int) and isinstance(py, int):
                gx, gy = px + tx, py + ty
            else:
                gx = gy = None

            # key by pixel
            key = x, y

            added_on_adjacent = False
            try:
                icon = self._icons[key]

                # This usually happens when a tagged npc dies and is
                # untagged, so the coordinates match, but it should be a
                # different entity
                if icon.name != name:
                    continue

                icon.update()
                checked.add(key)
                continue
            except KeyError:

                # FIXME: calculate pixel position on map and use that to
                #        determine nearest candidate
                icon_copy = [i.key for i in self._icons.values()]
                max_dist = 1
                for icon_key in icon_copy:
                    # TODO: method to calc distance between coords
                    if (abs(tx - icon_key[0]) <= max_dist and
                            abs(ty - icon_key[1]) <= max_dist):
                        # move npc to updated key
                        icon = self._icons.pop(icon_key)
                        self._icons[key] = icon
                        icon.update(key=key)
                        added_on_adjacent = True
                        continue

            # finally if we still can't find it, we must have a new one
            if key not in checked and not added_on_adjacent:

                icon = self.client.game_screen.create_game_entity(
                    name, name, key, self.client, self.client,
                    entity_templates=entity_templates,
                )

                # add offsets and change default bounding box to click box
                # (which is where the offsets will be used)
                if offsets:
                    x1, y1, x2, y2 = offsets
                    if x1:
                        icon.x1_offset = x1
                    if y1:
                        icon.y1_offset = y1
                    if x2:
                        icon.x2_offset = x2
                    if y2:
                        icon.y2_offset = y2

                    icon.default_bbox = icon.click_box

                # set global coordinates on init
                icon.set_global_coordinates(gx, gy)

                icon.update(key)
                self._icons[key] = icon

        # do one final check to remove any that are no longer on screen
        keys = list(self._icons.keys())
        for k in keys:
            icon = self._icons[k]
            if not icon.checked:
                self._icons.pop(k)

        return self._icons.values()

    # GPS map matching methods

    @property
    def mask(self):
        """
        Mask to apply to minimap image to exclude the rim etc.
        Note to be confused with :attr:`MiniMap.masks` which refers to the
        template masks used for minimap icon identification.
        """
        if self._mask is None:
            self.create_mask()

        return self._mask

    @property
    def orb_xy(self):
        """The centre point of the minimap orb relative to minimap img."""
        y, x = self.config['height'] + 1, self.config['width'] + 1
        # half it to get centre point
        x //= 2
        y //= 2

        return x, y

    @property
    def orb_radius(self):
        """Pixel distance from orb centre to rim."""
        return self.config['width'] // 2 - self.config['padding']

    def create_mask(self):
        """
        Create a circular mask to exclude e.g. the orb rim.
        When we run any image processing on the minimap image it should only
        be the moving subsection of the world map with NPCs, items, etc.
        """

        # set mask to None, so we can be sure we're creating a new one
        self._mask = None

        y, x = self.config['height'] + 1, self.config['width'] + 1
        mask = numpy.zeros(
            shape=(y, x), dtype=numpy.dtype('uint8'))

        mask = cv2.circle(mask, self.orb_xy, self.orb_radius, WHITE, FILL)

        for name, values in self.config['orbs'].items():
            mask = cv2.circle(
                mask, (values['x'], values['y']), values['r'],
                BLACK, FILL
            )

        # TODO: mask out identified objects like NPCs so they don't affect GPS

        # cache and return
        self._mask = mask
        return mask

    def coordinate_to_pixel(self, c):
        return int(c * self.tile_size)

    def pixel_to_coordinate(self, p):
        return p // self.tile_size

    def coordinates_to_pixel_bbox(self, x, y):
        """
        Convert a global coordinate set into a pixel bounding box, assuming
        the box size should be on tile.
        TODO: move to Map class, since this has nothing to do with the minimap
              specifically.
        """
        x1 = self.coordinate_to_pixel(x)
        y1 = self.coordinate_to_pixel(y)
        x2 = x1 + int(self.tile_size) - 1
        y2 = y1 + int(self.tile_size) - 1

        return x1, y1, x2, y2

    def distance_between(self, u1, u2, as_pixels=False):
        """Calculate distance between coordinates."""

        x1, y1 = u1
        x2, y2 = u2

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        if as_pixels:
            dx *= self.tile_size
            dy *= self.tile_size

        return math.sqrt(dx**2 + dy**2)

    @property
    def tile_size(self):
        return self.config['tile_size']

    @property
    def scale(self):
        return self.config['scale']
