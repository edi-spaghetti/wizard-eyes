import math
from unittest.mock import ANY

import cv2
import numpy

from .gps import GielenorPositioningSystem
from ..game_objects import GameObject
from ...game_entities.entity import GameEntity
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
    DEFAULT_EXPECTED_AREA = ANY
    """When tracking minimap dots, determine what the expected area for each
    found contour should be."""

    def __init__(self, client, parent, logging_level=None, **kwargs):
        super(MiniMap, self).__init__(
            client, parent, config_path='minimap.minimap',
            logging_level=logging_level, **kwargs,
        )

        self.updated_at = None
        self._img_colour = None

        self.gps: GielenorPositioningSystem = GielenorPositioningSystem(
            self.client, self
        )

        self._mask = None
        self.create_mask()

        # container for identified items/npcs/symbols etc.
        self._icons = dict()
        self._histograms = dict()
        self._centres = None
        self._contours = None
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3), (1, 1)
        )
        self._template_idx = []
        """Index mapping of template names"""
        self._template_ranges = []
        """Colour ranges for each template"""

    def setup_thresolds(self, *names):
        """Create colour range and index mappings for currently loaded
        templates. Pass in the template names for colour templates.
        They will be loaded, but not cached. Templates indexes in the same
        order as the named passed in."""

        templates = self.load_templates(names, cache=False)

        for i, name in enumerate(names):
            template = templates[name]
            colours = []
            unique = numpy.unique(
                template.reshape(-1, template.shape[2]), axis=0)
            for colour in unique:
                if not colour.any():
                    continue
                colours.append(colour)
            self._template_idx.append(name)
            self._template_ranges.append(colours)

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

    def update(self, auto_gps=True,
               method=GielenorPositioningSystem.DEFAULT_MATCH,
               threshold=0.99):
        """
        Basic update method for minimap. Should be run once per frame.
        Returns data from it's internal methods, which are run_gps and
        identify.

        :param auto_gps: If true, the coordinates will automatically be updated
            according to default parameters.
            If false, then it is is then up to the implementing application
            to do error filtering on these results.
        :param int method: Enum for the matching method to use on gps.
        :param threshold: Value for template matching.
        """

        x, y = self.gps.update(auto=auto_gps, method=method)

        # TODO: auto entity generation
        icons = self.identify(threshold=threshold)

        return (x, y), icons

    # minimap icon detection methods

    def quick_identify(self, name, colour):
        """Finds a particular colour in the minimap image

        This method is quicker than identify, but less precise.
        Entities produced by this method should use the TOP_LEFT_KEY for their
        key_type attribute.

        :param str name: Name of the entity to identify.
        :param int colour: Greyscale colour to identify.

        """
        img = cv2.bitwise_and(self.img, self.mask)
        my, mx = numpy.where(img == colour)

        candidates = set()
        for y, x in zip(my, mx):
            # TODO: check proximity dynamically
            duplicate = False
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if (x + dx, y + dy) in candidates:
                        duplicate = True

            if not duplicate:
                candidates.add((x, y))

        # TODO: ensure we only pick top left pixel
        candidates = map(lambda c: (c[0] - 2, c[1] - 1), candidates)
        return map(lambda c: (name, self._relative_to_player(*c)), candidates)

    def threshold(self, img, template_range):
        """Apply a culmulative mask for a range of colours to the image."""
        mask = numpy.zeros(img.shape[:2], dtype=numpy.uint8)
        for colour in template_range:
            mask = cv2.bitwise_or(mask, cv2.inRange(img, colour, colour))

        return mask

    def find_centres(self, expected_area=ANY):
        """Find the centres of all entities in the minimap."""

        img = cv2.bitwise_and(self.img_colour, self.img_colour, mask=self.mask)
        results = []
        for idx, template_ranges in enumerate(self._template_ranges):
            processed = self.threshold(img, template_ranges)
            contours, _ = cv2.findContours(
                processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            accepted = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area == expected_area:
                    accepted.append(contour)
                elif area > expected_area:
                    # TODO: map tiles are 4 pixels wide, but if two
                    #       entities are adjacent their contours merge.
                    #       Attempt to split them here.
                    x, y, w, h = cv2.boundingRect(contour)
                    for cx in range(x, x + w, self.tile_size):
                        for cy in range(y, y + h, self.tile_size):
                            centre = (
                                round(cx + self.tile_size / 2),
                                round(cy + self.tile_size / 2),
                                idx
                            )
                            results.append(centre)
                    continue

            moments = [cv2.moments(c) for c in accepted]
            for i, m in enumerate(moments):
                try:
                    centre = (
                        round(m['m10'] / m['m00'] - self.width / 2),
                        round(m['m01'] / m['m00'] - self.height / 2),
                        idx
                )
                    results.append(centre)
                    self._contours.append(accepted[i])
                except ZeroDivisionError:
                    continue

        # self.msg.append(','.join(str(cv2.contourArea(c)) for c in self.contours))
        results = numpy.array(results)
        return results

    def track(self, objects):
        """Track entities in the minimap."""

        # no previous centres, create all new entities
        if self._centres is None:
            self._contours = []
            self._centres = []
            centres = self.find_centres(
                expected_area=self.DEFAULT_EXPECTED_AREA
            )
            objects = []
            for x, y, idx in centres:
                name = self._template_idx[idx]
                entity = self.client.game_screen.create_game_entity(
                    name, name, (x, y), self.client, self.client
                )
                entity.key_type = entity.CENTRED_KEY
                objects.append(entity)

            for obj in objects:
                obj.state_changed_at = self.client.time
                obj.update()

            self._centres = centres
            return objects

        # otherwise we must match previous centres with current ones
        for obj in objects:
            obj.checked = False

        centres = self.find_centres(
            expected_area=self.DEFAULT_EXPECTED_AREA
        )
        if len(centres) == 0:
            self._centres = None
            return []

        centres_array = numpy.array(centres[:, :2])
        m = len(self._centres)  # mature
        n = len(centres)        # new

        if len(self._centres) > 0:
            old_centres_array = numpy.array(self._centres[:, :2])
            object_keys_array = numpy.array([o.key for o in objects])

            distances = numpy.zeros((n, m))
            for i in range(n):
                diffs = centres_array[i] - old_centres_array
                squared = diffs ** 2
                sums = numpy.sum(squared, axis=1)
                distances[i, :] = numpy.sqrt(sums)

            t = self.client.time - self.client.last_time
            tick_fraction = t / self.client.TICK
            # max distance is 2 players running in opposite directions
            # multiply 2 for some extra leeway
            max_dist = numpy.ceil(
                4 * self.client.minimap.minimap.tile_size
                * tick_fraction) * 2
            for i in range(n):
                dist = numpy.min(distances[i, :])
                j = numpy.argmin(distances[i, :])
                min_dist = numpy.min(distances[:, j])
                if min_dist == dist and dist <= max_dist:
                    # find the object with this key
                    match = numpy.isin(object_keys_array, old_centres_array[j])
                    if not match.any():
                        continue
                    match = numpy.all(match, axis=1)
                    if not match.any():
                        continue
                    idx = numpy.argwhere(match)[0][0]
                    # then update it
                    objects[idx].name = self._template_idx[centres[i, 2]]
                    objects[idx].key = tuple(int(x) for x in centres_array[i])
                    objects[idx].update()

        # determine if there are old objects to remove
        old_objects = {o.key for o in objects if not o.checked}
        ok_objects = {o.key for o in objects if o.checked}
        # if less new that mature, old ones die
        i = 0
        while i < len(objects):
            if not objects[i].checked:
                obj = objects.pop(i)
                self.client.game_screen.add_to_buffer(obj)
            else:
                i += 1

        # finally check if there are new objects to create
        try:
            centre_set = set((x, y) for (x, y, idx) in centres)
        except ValueError:
            centre_set = set()
        brand_new = centre_set.difference(ok_objects) - old_objects
        for xy in brand_new:
            idx = numpy.argwhere(numpy.all(
                numpy.isin(centres_array, xy), axis=1))[0][0]
            name = self._template_idx[centres[idx, 2]]
            key = tuple(int(x) for x in xy)
            entity = self.client.game_screen.create_game_entity(
                name, name, key, self.client, self.client)
            entity.key_type = entity.CENTRED_KEY
            entity.state_changed_at = self.client.time
            entity.update()
            objects.append(entity)

        # reset the old centres ready for next frame
        self._centres = centres
        return objects

    def _relative_to_player(self, x, y):
        """Convert pixels found in minimap relative to player."""
        px, py, _, _ = self.client.game_screen.player.mm_bbox()
        mm_x, mm_y, _, _ = self.get_bbox()
        px = px - mm_x + 1
        py = py - mm_y + 1

        # calculate item relative pixel coordinate to player
        rx = x - px
        ry = y - py

        return rx, ry

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

                rx, ry = self._relative_to_player(x, y)

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
            px, py = self.client.minimap.minimap.gps.get_coordinates(real=True)
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
                icon.key_type = icon.TOP_LEFT_KEY

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
                old_icon = self._icons.pop(k)
                self.client.game_screen.add_to_buffer(old_icon)

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
    def patched_mask(self):
        """Mask with all minimap "dots" patched out."""

        mask = self.mask.copy()
        for template_ranges in self._template_ranges:
            patch = self.threshold(self.img_colour, template_ranges)
            patch = cv2.dilate(patch, self._kernel)
            patch = cv2.bitwise_not(patch)
            mask = cv2.bitwise_and(mask, patch)

        return mask

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
