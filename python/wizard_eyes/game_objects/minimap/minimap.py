import math
from collections import defaultdict

import cv2
import numpy

from ..game_objects import GameObject
from ..personal_menu import LogoutButton
from ...file_path_utils import get_root
from ...constants import (
    FILL,
    WHITE,
)


class MiniMap(GameObject):

    MAP_PATH_TEMPLATE = '{root}/data/maps/{z}/{x}_{y}.png'
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

        # settings for processing the map images + minimap on run
        self._canny_lower = 60
        self._canny_upper = 130

        # concatenated map images are stored here
        self._map_cache = None
        self._map_cache_original = None
        # map chunk images are cached here
        self._chunks = dict()
        self._chunks_original = dict()

        self._local_zone_radius = 25
        self._coordinates = None
        self._chunk_coordinates = None

        # TODO: configurable feature matching methods
        self._detector = self._create_detector()
        self._matcher = self._create_matcher()
        self._mask = self._create_mask()

        # container for identified items/npcs/symbols etc.
        self._icons = dict()

        # image for display
        self.gps_img = None
        self._map_img = None

    def process_img(self, img, grey=True, canny=True):
        if grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if canny:
            img = cv2.Canny(img, self._canny_lower, self._canny_upper)
        return img

    @property
    def map_img(self):
        return self._map_img

    def copy_map_img(self):
        """
        Create a fresh copy of the original map. Should be run once per
        frame so we can draw on it if we're showing.
        """
        self._map_img = self._map_cache_original.copy()

    def update(self, auto_gps=True):
        """
        Basic update method for minimap. Should be run once per frame.
        Returns data from it's internal methods, which are run_gps and
        identify.

        :param auto_gps: If true, the coordinates will automatically be updated
            according to default parameters.
            If false, then it is is then up to the implementing application
            to do error filtering on these results.
        """

        if self.client.args.show_map:
            self.copy_map_img()
        x, y = self.run_gps()

        if auto_gps:
            cx, cy = self.get_coordinates()
            # TODO: distance calculation, logging the last n gps updates so we
            #       can approximate speed
            # TODO: support teleportation
            # these numbers are fairly heuristic, but seem to work
            if abs(cx - x) < 4 and abs(cy - y) < 4:
                self.set_coordinates(x, y)

        # TODO: auto entity generation
        icons = self.identify()

        return (x, y), icons

    # minimap icon detection methods

    def identify(self, threshold=0.99):
        """
        Identify items/npcs/icons etc. on the minimap
        :param threshold:
        :return: A list of matches items of the format (item name, x, y)
            where x and y are tile coordinates relative to the player position
        """

        marked = set()
        results = set()

        # reset mark on all icons, so know which ones we've checked
        for i in self._icons.values():
            i.refresh()

        for name, template in self.templates.items():

            # for some reason masks cause way too many false matches,
            # so don't use a mask.
            matches = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED)

            (my, mx) = numpy.where(matches >= threshold)
            for y, x in zip(my, mx):

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

    def generate_entities(self, positions):
        """Generate game entities from results of :meth:`MiniMap.identify`"""

        checked = set()

        for name, (x, y) in positions:

            # rx = int((x - self.config['width'] / 2) * self.scale)
            # ry = int((y - self.config['height'] / 2) * self.scale)

            # convert pixel coordinate into tile coordinate
            tx = x // self.tile_size
            ty = y // self.tile_size

            # TODO: method to add coordinates
            # calculate icon's global map coordinate
            # v += tx
            # w += ty

            # key by pixel
            key = tx, ty

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
                    name, name, key, self.client, self.client)

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

    def run_gps(self, train_chunk=None):

        img = self.img  # sliced client image is already greyscale
        query_img = self.process_img(img, grey=False)
        kp1, des1 = self._detector.detectAndCompute(query_img, self._mask)

        if train_chunk:
            train_img = self.get_chunk(*train_chunk)
            radius = (self.chunk_shape_x // 2) // self.tile_size
        else:
            radius = self._local_zone_radius
            train_img = self.get_local_zone()

        try:
            kp2, des2 = self._detector.detectAndCompute(train_img, None)
            matches = self._matcher.match(des1, des2)
        except cv2.error:
            return

        self.logger.debug(f'got {len(matches)} matches')

        filtered_matches = self._filter_matches_by_grouping(matches, kp1, kp2)
        self.logger.debug(f'filtered to {len(filtered_matches)} matches')

        # determine pixel coordinate relationship of minimap to map for each
        # of the filtered matches, and pick the modal tile coordinate
        mapped_coords = defaultdict(int)
        for match in filtered_matches:
            tx, ty = self._map_key_points(match, kp1, kp2)
            mapped_coords[(tx, ty)] += 1
        sorted_mapped_coords = sorted(
            mapped_coords.items(), key=lambda item: item[1])
        (tx, ty), freq = sorted_mapped_coords[-1]
        self.logger.debug(f'got tile coord {tx, ty} (frequency: {freq})')

        # determine relative coordinate change to create new coordinates
        # local zone is radius tiles left and right of current tile, so
        # subtracting the radius gets us the relative change
        x, y = self._coordinates
        rx = tx - radius
        ry = ty - radius
        self.logger.debug(f'relative change: {rx, ry}')

        # it is the responsibility of the script to determine if a proposed
        # coordinate change is possible since the last time the gps was pinged.
        # TODO: record each time gps is pinged and calculate potential
        #       destinations since last gps pinged
        if abs(rx) > 4 or abs(ry) > 4:
            self.logger.debug(f'excessive position change: {rx, ry}')

        x = int(x + rx)
        y = int(y + ry)

        # GPS needs to be shown in a separate windows because it isn't
        # strictly part of the client image.
        if self.client.args.show_gps:

            train_img_copy = train_img.copy()
            ptx0, pty0 = int(tx * self.tile_size), int(ty * self.tile_size)
            ptx1 = ptx0 + self.tile_size - 1
            pty1 = pty0 + self.tile_size - 1

            self.logger.debug(f'position: {ptx0}, {pty0}')

            train_img_copy = cv2.rectangle(
                train_img_copy, (ptx0, pty0), (ptx1, pty1), WHITE, FILL)

            show_img = cv2.drawMatches(
                query_img, kp1, train_img_copy, kp2, filtered_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # set display image, it will be shown later in the application
            # event cycle
            self.gps_img = show_img

        # self._coordinates = new_coordinates
        return x, y

    def _create_detector(self):
        return cv2.ORB_create()

    def _create_matcher(self):
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

    def _create_mask(self):

        y, x = self.config['height'] + 1, self.config['width'] + 1
        mask = numpy.zeros(
            shape=(y, x), dtype=numpy.dtype('uint8'))

        mask = cv2.circle(mask, self.orb_xy, self.orb_radius, WHITE, FILL)

        # TODO: create additional cutouts for orbs that slightly overlay the
        #       minimap. Not hugely important, but may interfere with feature
        #       matching.

        return mask

    def set_coordinates(self, x, y):
        """
        x and y tiles
        """
        self._coordinates = x, y

    def coordinate_to_pixel(self, c):
        return int(c * self.tile_size)

    def pixel_to_coordinate(self, p):
        return p // self.tile_size

    def coordinates_to_pixel_bbox(self, x, y):
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

    def get_coordinates(self, as_pixels=False):
        x, y = self._coordinates

        if as_pixels:
            x = int(x * self.tile_size)
            y = int(y * self.tile_size)

        return x, y

    def _map_key_points(self, match, kp1, kp2):
        # get pixel coordinates of feature within minimap image
        x1, y1 = kp1[match.queryIdx].pt
        # get pixel coords of feature in main map
        x2, y2 = kp2[match.trainIdx].pt

        # calculate player coordinate in main map
        # TODO: check this, I think it's slightly off
        px = int((self.config['width'] / 2 - x1) * self.scale + x2)
        py = int((self.config['height'] / 2 - y1) * self.scale + y2)

        # convert player pixel coordinate into tile coordinate
        px //= self.tile_size
        py //= self.tile_size

        return px, py

    def _filter_matches_by_grouping(self, matches, kp1, kp2):

        # pre-filter matches in case we get lots of poor matches
        filtered_matches = [m for m in matches if m.distance < 70]

        groups = defaultdict(list)
        for m in filtered_matches:
            tx, ty = self._map_key_points(m, kp1, kp2)
            groups[(tx, ty)].append(m)

        # normalise the number of matches per group
        max_num_matches = max([len(v) for k, v in groups.items()], default=0)
        normalised_average = dict()
        for (k, v) in groups.items():
            average_distance = sum([m_.distance for m_ in v]) / len(v)
            normalised_len = self.client.screen.normalise(
                len(v), stop=max_num_matches)
            normalised_average[k] = (
                average_distance / normalised_len
            )

        sorted_normalised_average = sorted(
            [(k, v) for k, v in normalised_average.items()],
            # sort by normalised value, lower means more matches and lower
            key=lambda item: item[1])
        self.logger.debug(
            f'top 5 normalised matches: {sorted_normalised_average[:5]}')

        key, score = sorted_normalised_average[0]
        filtered_matches = groups[key]

        return filtered_matches

    @property
    def chunk_shape_x(self):
        return self.config['chunk_shape'][1]

    @property
    def chunk_shape_y(self):
        return self.config['chunk_shape'][0]

    @property
    def tile_size(self):
        return self.config['tile_size']

    @property
    def min_tile(self):
        return 0

    @property
    def max_tile(self):
        # assumes chunks are square
        return int(self.chunk_shape_x / self.tile_size) - 1

    @property
    def match_tolerance(self):
        return self.tile_size // 2

    def update_coordinate(self, dx, dy, cache=True):

        x, y, = self.get_coordinates()
        x = x + dx
        y = y + dy

        if cache:
            self.set_coordinates(x, y)
        return x, y

    def load_chunks(self, *chunks, fill_missing=None):

        for (x, y, z) in chunks:

            # attempt to load the map chunk from disk
            chunk_path = self.MAP_PATH_TEMPLATE.format(
                root=get_root(),
                x=x, y=y, z=z,
            )
            chunk = cv2.imread(chunk_path)

            # resolve if disk file does not exist
            if chunk is None:
                if fill_missing is None:
                    shape = self.config.get('chunk_shape', (256, 256))
                    chunk_processed = numpy.zeros(
                        shape=shape, dtype=numpy.dtype('uint8'))
                    shape = self.config.get(
                        'original_chunk_shape', (256, 256, 3))
                    chunk = numpy.zeros(
                        shape=shape, dtype=numpy.dtype('uint8')
                    )
                # TODO: implement requests method
                else:
                    raise NotImplementedError
            else:
                chunk_processed = self.process_img(chunk)

            # add to internal cache
            self._chunks_original[(x, y, z)] = chunk
            self._chunks[(x, y, z)] = chunk_processed

    def get_chunk(self, x, y, z, original=False):

        cache = self._chunks
        if original:
            cache = self._chunks_original

        chunk = cache.get((x, y, z))
        if chunk is None:
            self.load_chunks((x, y, z))
            chunk = cache.get((x, y, z))

        return chunk

    def _get_chunk_set_boundary(self, chunk_set):

        min_x = min_y = float('inf')
        max_x = max_y = -float('inf')
        z = None

        for (x, y, _z) in chunk_set:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

            # z is assumed to be constant
            z = _z

        return min_x, min_y, max_x, max_y, z

    def _arrange_chunk_matrix(self, chunks):

        min_x, min_y, max_x, max_y, z = self._get_chunk_set_boundary(chunks)

        chunk_list = list()
        # NOTE: chunks are numbered from bottom left, so we must iterate in
        #       the opposite direction
        for y in range(max_y, min_y - 1, -1):
            chunk_row = list()
            for x in range(min_x, max_x + 1):
                chunk_row.append((x, y, z))
            chunk_list.append(chunk_row)

        return chunk_list

    def create_map(self, chunk_set):
        """
        Determine chunks required to concatenate map chunks into a single
        image.

        Chunk set can actually just be top left and bottom right, missing
        chunks will be calculated.
        """
        chunk_matrix = self._arrange_chunk_matrix(chunk_set)

        processed_map = self.concatenate_chunks(chunk_matrix)
        colour_map = self.concatenate_chunks(chunk_matrix, original=True)

        x1, y1, x2, y2, z = self._get_chunk_set_boundary(chunk_set)
        self._chunk_coordinates = x1, y1, x2, y2, z

        self._map_cache = processed_map
        self._map_cache_original = colour_map

    def concatenate_chunks(self, chunk_matrix, original=False):

        col_data = list()
        for row in chunk_matrix:
            row_data = list()
            for (x, y, z) in row:
                chunk = self.get_chunk(x, y, z, original=original)
                row_data.append(chunk)
            row_data = numpy.concatenate(row_data, axis=1)
            col_data.append(row_data)
        concatenated_chunks = numpy.concatenate(col_data, axis=0)

        return concatenated_chunks

    def show_map(self):
        if self.client.args.show_map:
            img = self.map_img

            x1, y1 = self.get_coordinates(as_pixels=True)
            x2, y2 = x1 + self.tile_size - 1, y1 + self.tile_size - 1

            cv2.rectangle(
                img, (x1, y1), (x2, y2),
                self.colour, thickness=1
            )

            x1, y1, x2, y2 = self.local_bbox()
            offset = int(self._local_zone_radius * self.tile_size)
            cv2.rectangle(
                img, (x1, y1), (x2, y2),
                self.colour, thickness=1
            )

            cv2.circle(
                img, (x1 + offset, y1 + offset),
                self.orb_radius, self.colour, thickness=1)

    def local_bbox(self):
        """
        Bounding box of local zone.
        Since this refers to the currently loaded map, this is relative to
        that image.
        """
        x, y = self._coordinates
        radius = self._local_zone_radius

        x1 = int(x * self.tile_size) - int(radius * self.tile_size)
        y1 = int(y * self.tile_size) - int(radius * self.tile_size)
        #                                 +1 for current tile
        x2 = int(x * self.tile_size) + int((radius + 1) * self.tile_size)
        y2 = int(y * self.tile_size) + int((radius + 1) * self.tile_size)

        return x1, y1, x2,y2

    def get_local_zone(self, original=False):
        """
        Sub-matrix of map image around the given location.
        """

        img = self._map_cache
        if original:
            img = self._map_cache_original

        x1, y1, x2, y2 = self.local_bbox()
        local_zone = img[y1:y2, x1:x2]

        self.show_map()

        return local_zone

    def load_map_sections(self, sections):
        return numpy.concatenate(
            [numpy.concatenate(
                [cv2.imread(self.MAP_PATH_TEMPLATE.format(
                    root=get_root(), name=name))
                 for name in row], axis=1)
                for row in sections], axis=0
        )

    @property
    def scale(self):
        return self.config['scale']
