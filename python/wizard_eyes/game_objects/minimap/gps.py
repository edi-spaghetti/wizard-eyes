from collections import defaultdict
from copy import deepcopy
from itertools import islice
from typing import Union, Tuple
from os.path import exists

import cv2
import numpy

from ..game_objects import GameObject
from ...constants import WHITE, BLACK, FILL
from ...file_path_utils import get_root, load_pickle


class GielenorPositioningSystem(GameObject):
    """
    Manages maps and orb feature matching on those maps to determine the
    player's current location.
    """

    DEFAULT_COLOUR = (0, 0, 255, 255)
    PATH_TEMPLATE = '{root}/data/maps/meta/{name}.pickle'

    DEFAULT_MATCH = 0
    FEATURE_MATCH = 1
    TEMPLATE_MATCH = 2
    DEFAULT_METHOD = FEATURE_MATCH
    # DEFAULT_METHOD = TEMPLATE_MATCH

    TEMPLATE_METHOD = cv2.TM_CCORR_NORMED
    TEMPLATE_THRESHOLD = 0.88

    MASK_PATCH_TEMPLATES = None

    MOVEMENT_THRESHOLD = 4
    """int: number of tiles gps position is allowed to move before being
    considered too far away (and therefore an invalid update)"""

    def __init__(self, *args, tile_size=4, scale=1, **kwargs):
        super().__init__(*args, **kwargs)

        self._local_zone_radius = 25
        self._tile_size = tile_size
        self._scale = scale

        self._show_img = None
        self._coordinates = None
        self._coordinate_history = list()
        self._coordinate_history_size = 100
        self._chunk_coordinates = None
        self.current_map: Union["Map", None] = None
        self.maps = dict()
        self.confidence: Union[float, None] = None

        # setup for matching
        self._detector = self._create_detector()
        self._matcher = self._create_matcher()
        self._key_points_minimap = None
        self._key_points_map = None
        self._filtered_matches = None
        self._sorted_mapped_coords = None

    @property
    def match_methods(self):
        return {
            self.DEFAULT_MATCH: (
                self._feature_match
                if self.DEFAULT_METHOD == self.FEATURE_MATCH
                else self._template_match),
            self.FEATURE_MATCH: self._feature_match,
            self.TEMPLATE_MATCH: self._template_match,
        }

    @property
    def tile_size(self):
        """Size of a single game tile in pixels on the map."""
        return int(self._tile_size)

    @property
    def scale(self):
        """Ratio of minimap tile size to loaded map tile size"""
        return self._scale

    @property
    def img(self):
        """
        The image used by gps for feature matching. Should be pulled from the
        minimap and processed according to the settings on the current map.
        """
        img = self.parent.img_colour

        # TODO: img cache invalidation so we don't have to regenerate every
        #       time this property is called.
        img = self.current_map.process_img(img)

        return img

    @property
    def show_img(self):
        return self._show_img

    def set_coordinates(self, x, y, add_history=True):
        """
        record x and y tiles and add to history
        """
        self._coordinates = x, y

        # if we're map swapping, or teleporting, we don't want to add the
        # coordinate to history because it gives incorrect results for speed.
        if not add_history:
            return

        # add to history
        self._coordinate_history.append((self.client.time, (x, y)))
        if len(self._coordinate_history) > self._coordinate_history_size:
            # remove the oldest
            self._coordinate_history = self._coordinate_history[1:]

    def get_coordinates(self, as_pixels=False, real=False):
        try:
            x, y = self._coordinates
        except TypeError:
            return None, None

        if not real:
            x = round(x)
            y = round(y)

        if as_pixels:
            x = round(x * self.tile_size)
            y = round(y * self.tile_size)

        return x, y

    def update_coordinate(self, dx, dy, cache=True):

        x, y, = self.get_coordinates()
        x = x + dx
        y = y + dy

        if cache:
            self.set_coordinates(x, y)
        return x, y

    def clear_coordinate_history(self):
        self._coordinate_history = list()

    def calculate_average_speed(self, period=1.8):
        """Calculate the average speed over the last time period."""

        # get distances from history and keep track of when they we're recorded
        distances = list()
        current_period = 0
        for i, (time, (x, y)) in enumerate(self._coordinate_history[::-1], 1):

            # try to get the previous time-location (if it exists)
            try:
                time2, (x2, y2) = self._coordinate_history[-i - 1]
            except IndexError:
                break

            current_period += (time - time2)
            distances.append(
                self.client.minimap.minimap.distance_between((x, y), (x2, y2)))

            if current_period > period:
                break

        # try to calculate the speed over the given time period
        try:
            average_speed = sum(distances) / current_period
        except ZeroDivisionError:
            if distances:
                # we've somehow travelled a distance in zero time (teleport?)
                average_speed = float('inf')
            else:
                # we've travelled no distance in no time, usually on first
                # iteration - assume we're not moving.
                average_speed = 0

        return average_speed

    def local_bbox(self, real=False):
        """
        Bounding box of local zone.
        Since this refers to the currently loaded map, this is relative to
        that image.
        """
        x, y = self._coordinates
        radius = self._local_zone_radius

        # floored
        # x1 = int(x * self.tile_size - radius * self.tile_size)
        # y1 = int(y * self.tile_size - radius * self.tile_size)
        # #                                 +1 for current tile
        # x2 = int(x * self.tile_size + (radius + 1) * self.tile_size)
        # y2 = int(y * self.tile_size + (radius + 1) * self.tile_size)

        # # rounded
        # x1 = round(x * self.tile_size - radius * self.tile_size)
        # y1 = round(y * self.tile_size - radius * self.tile_size)
        # x2 = round(x * self.tile_size + (radius + 1) * self.tile_size)
        # y2 = round(y * self.tile_size + (radius + 1) * self.tile_size)
        #
        # # floating point
        x1 = (x * self.tile_size) - (radius * self.tile_size)
        y1 = (y * self.tile_size) - (radius * self.tile_size)
        x2 = (x * self.tile_size) + ((radius + 1) * self.tile_size)
        y2 = (y * self.tile_size) + ((radius + 1) * self.tile_size)

        if real:
            return x1, y1, x2, y2
        else:
            return round(x1), round(y1), round(x2), round(y2)

    def load_map(self, name, set_current=True, force_rebuild=False):
        """Load a map from meta data and chunk images on disk."""

        # if we've already loaded it before, and we don't want to rebuild,
        # load it from cache
        if name in self.maps and not force_rebuild:
            if set_current:
                self.current_map = self.maps[name]
                self.current_map.copy_original()
            return self.maps[name]

        # otherwise attempt to load it from disk
        path = self.PATH_TEMPLATE.format(root=get_root(), name=name)
        data = load_pickle(path)

        chunks = data.get('chunks', {})
        graph = data.get('graph', {})
        labels = data.get('labels', {})
        offsets = data.get(
            'offsets', (Map.DEFAULT_OFFSET_X, Map.DEFAULT_OFFSET_Y))
        map_object = Map(self.client, chunks, name=name, graph=graph,
                         labels=labels, offsets=offsets)

        self.maps[name] = map_object
        if set_current:
            self.current_map = map_object
            # ensure the map object has a copy of the original image in case
            # we need to draw to it.
            map_object.copy_original()

        return map_object

    # GPS feature matching

    def _create_detector(self):
        return cv2.ORB_create()

    def _create_matcher(self):
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_local_zone(self, original=False):
        """
        Sub-matrix of map image around the given location.
        """

        img = self.current_map.img
        if original:
            img = self.current_map.img_colour

        x1, y1, x2, y2 = self.local_bbox()
        local_zone = img[y1:y2, x1:x2]

        return local_zone

    def _map_key_points(self, match, kp1, kp2):
        # get pixel coordinates of feature within minimap image
        x1, y1 = kp1[match.queryIdx].pt
        # get pixel coords of feature in main map
        x2, y2 = kp2[match.trainIdx].pt

        # calculate player coordinate in main map
        px = (self.parent.config['width'] / 2 - x1) * self.scale + x2
        py = (self.parent.config['height'] / 2 - y1) * self.scale + y2

        # convert player pixel coordinate into tile coordinate
        px /= self.tile_size
        py /= self.tile_size

        return px, py

    def _filter_matches_by_grouping(self, matches, kp1, kp2):

        # pre-filter matches in case we get lots of poor matches
        filtered_matches = [m for m in matches if m.distance < 70]

        groups = defaultdict(list)
        for m in filtered_matches:
            kpx, kpy = self._map_key_points(m, kp1, kp2)
            groups[(kpx, kpy)].append(m)

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

        if not sorted_normalised_average:
            return list()

        key, score = sorted_normalised_average[0]
        filtered_matches = groups[key]

        return filtered_matches

    def get_mask(self):

        # TODO: cache parent mask copy and use unless patching
        mask = self.parent.mask.copy()

        # mask out self
        h, w = mask.shape
        x1 = round(w / 2)
        x2 = round(w / 2) + self.tile_size
        y1 = round(h / 2)
        y2 = round(h / 2) + self.tile_size
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), BLACK, thickness=FILL)

        if self.MASK_PATCH_TEMPLATES:
            matches = self.parent.identify(
                threshold=0, method=cv2.TM_SQDIFF_NORMED
            )
            for name, (x, y) in matches:

                try:
                    h, w, _ = self.parent.templates[name].shape
                except ValueError:
                    h, w = self.parent.templates[name].shape

                x2 = x + w - 1
                y2 = y + h - 1
                if name in self.MASK_PATCH_TEMPLATES:
                    mask = cv2.rectangle(
                        mask, (x, y), (x2, y2), BLACK, thickness=FILL
                    )

        return mask

    def _feature_match(self):
        """Use cv2 feature matching to find position on map.

        :returns: Tuple of x and y.
        :rtype: tuple[float, float]
        """
        query_img = self.img
        kp1, des1 = self._detector.detectAndCompute(
            query_img, self.get_mask()
        )

        radius = self._local_zone_radius
        train_img = self.get_local_zone()

        try:
            kp2, des2 = self._detector.detectAndCompute(train_img, None)
            matches = self._matcher.match(des1, des2)
        except cv2.error:
            return None, None

        # cache key points so we can display them for gps image
        self._key_points_minimap = kp1
        self._key_points_map = kp2

        self.logger.debug(f'got {len(matches)} matches')

        filtered_matches = self._filter_matches_by_grouping(matches, kp1, kp2)
        self.logger.debug(f'filtered to {len(filtered_matches)} matches')

        # cache filtered matches so we can display for gps image
        self._filtered_matches = filtered_matches

        # determine pixel coordinate relationship of minimap to map for each
        # of the filtered matches, and pick the modal tile coordinate
        mapped_coords = defaultdict(int)
        for match in filtered_matches:
            kpx, kpy = self._map_key_points(match, kp1, kp2)
            mapped_coords[(kpx, kpy)] += 1
        sorted_mapped_coords = sorted(
            mapped_coords.items(), key=lambda item: item[1])

        # cache mapped coords for debugging and showing gps
        self._sorted_mapped_coords = sorted_mapped_coords

        if not sorted_mapped_coords:
            return None, None

        (tx, ty), freq = sorted_mapped_coords[-1]
        self.confidence = freq / len(sorted_mapped_coords)
        self.logger.debug(f'coordinate {tx, ty} '
                          f'(confidence: {self.confidence})')

        # determine relative coordinate change to create new coordinates
        # local zone is radius tiles left and right of current tile, so
        # subtracting the radius gets us the relative change
        x, y = self._coordinates
        rx = tx - radius
        ry = ty - radius
        self.logger.debug(f'relative change: {rx, ry}')

        x = x + rx
        y = y + ry

        # I have no fucking clue....
        # probably wont work at other zoom levels than default 512
        x -= .875
        y -= .625

        # I have even less of a clue how different maps are differently offset
        x += self.current_map.offset_x
        y += self.current_map.offset_y

        return x, y

    def _template_match(self):
        """Use cv2 template matching to find position on map.

        :returns: Tuple of x and y.
        :rtype: tuple[float, float]
        """
        img = self.get_local_zone()
        template = self.img
        mask = self.get_mask()

        try:
            matches = cv2.matchTemplate(
                img, template, self.TEMPLATE_METHOD,
                mask=mask
            )
        except cv2.error:
            return None, None

        max_x = numpy.argmax(matches, axis=1)[0]
        max_y = numpy.argmax(matches, axis=0)[0]

        self.confidence = matches[max_y][max_x]

        x1, y1, _, _ = self.local_bbox()
        x = (max_x + template.shape[1] / 2 + x1) / self.tile_size
        y = (max_y + template.shape[0] / 2 + y1) / self.tile_size

        # no idea why we have to subtract these here either...
        x -= 1
        y -= .75

        return x, y

    def update(self, method=0, auto=True, draw=True):
        """Run image matching method to determine player coordinates in map."""

        try:
            match_method = self.match_methods[method]
        except KeyError:
            raise NotImplementedError(f'Unsupported method: {method}')

        x, y = match_method()

        # defer update display images (if necessary)
        if self.current_map:
            self.current_map.copy_original()
        if draw:
            self.client.add_draw_call(self.show_gps)
            self.client.add_draw_call(self.show_map)

        if auto:
            cx, cy = self.get_coordinates()
            try:
                assert x
                assert y
                condition = (
                        abs(cx - x) < self.MOVEMENT_THRESHOLD
                        and abs(cy - y) < self.MOVEMENT_THRESHOLD
                )
                if condition:
                    self.set_coordinates(x, y)
            except AssertionError:
                return None, None

        return x, y

    # path finding

    def calculate_path(self, start, end):
        """
        Find the shortest path between two nodes (if it exists) with
        dijkstra's algorithm.
        """

        graph = self.current_map.graph
        if start not in graph or end not in graph:
            return []
        # edge case if we calculate path from current position, which
        # happens to be a node already
        if start == end:
            return [start]

        unvisited = {node: float('inf') for node, _ in graph.items()}
        unvisited[start] = 0
        visited = {}
        parents = {}

        # dijkstra's algorithm
        while unvisited:
            min_node = min(unvisited, key=unvisited.get)
            for neighbour, distance in graph[min_node].items():

                # if we've visited it already, skip it
                if neighbour in visited:
                    continue

                min_dist = unvisited[min_node]
                neighbour_dist = graph[min_node].get(
                    neighbour, float('inf'))
                new_distance = min_dist + neighbour_dist

                if unvisited[neighbour] > new_distance:
                    unvisited[neighbour] = new_distance
                    parents[neighbour] = min_node

            visited[min_node] = unvisited[min_node]
            del unvisited[min_node]

            # if we hit our target node, we done
            if min_node == end:
                break

        # generate path from results
        path = [end]
        while True:
            key = parents[path[0]]
            path.insert(0, key)
            if key == start:
                break

        return path

    def get_route(self, start, end, checkpoints=None):
        """
        Calculate a direct route from start to end.
        Nodes can be referred to by label or x, y coordinate.

        :param start: Where the route should start from. You can supply any
            coordinate, and if it is not in the map graph, it will resolve to
            the nearest node. This way you can pass in the current player
            coordinates without having to be standing on a node.
        :param end: Where you would like to end up.
        :param list checkpoints: Optionally provide a list of nodes to visit
            before reaching the final destination.

        :raises: ValueError if start or end are not in the map graph.

        :return: List of nodes to visit before reaching the end. If no path
            exists between the start and end, and empty list will be returned.
        """

        if start not in self.current_map.graph:
            start = self.current_map.find(nearest=start)
        if not start:
            raise ValueError(f'Could not find start node: {start}')

        route = [start]

        # create a copy of this list, or every time we run the method,
        # it will be update in place!
        checkpoints = deepcopy(checkpoints or list())

        checkpoints.insert(0, start)
        checkpoints.append(end)
        for i, checkpoint in enumerate(checkpoints[:-1]):
            target = checkpoints[i + 1]

            # try to resolve labels, otherwise assume it's a raw coordinate
            try:
                checkpoint = self.current_map.label_to_node(
                    checkpoint, limit=1).pop()
                target = self.current_map.label_to_node(
                    target, limit=1).pop()
            except KeyError:
                raise ValueError(f'Could not find node: {checkpoint}')

            path = self.calculate_path(checkpoint, target)
            # skip first because it's same as last of previous path
            route.extend(path[1:])

        return route

    def sum_route(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Get the total length of a route on the current map in tiles.

        :param start: Starting map coordinates.
        :param end: Ending map coordinates.

        :return: Total length of route in tiles.
        :rtype: float

        """

        route = self.get_route(start, end)
        total = 0
        for i, node in enumerate(route[:-1]):
            target = route[i + 1]
            total += self.parent.distance_between(node, target)  # noqa

        return total

    # display

    def show_gps(self):
        # GPS needs to be shown in a separate windows because it isn't
        # strictly part of the client image.
        if self.client.args.show_gps:

            if self.current_map is None:
                return

            (tx, ty), _ = self._sorted_mapped_coords[-1]
            train_img = self.get_local_zone(original=True)
            query_img = self.img

            train_img_copy = train_img.copy()
            ptx0, pty0 = round(tx * self.tile_size), round(ty * self.tile_size)
            ptx1 = ptx0 + self.tile_size - 1
            pty1 = pty0 + self.tile_size - 1

            self.logger.debug(f'position: {ptx0}, {pty0}')

            train_img_copy = cv2.rectangle(
                train_img_copy, (ptx0, pty0), (ptx1, pty1), WHITE, FILL)

            show_img = cv2.drawMatches(
                query_img, self._key_points_minimap,
                train_img_copy, self._key_points_map,
                self._filtered_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # set display image, it will be shown later in the application
            # event cycle
            self._show_img = show_img

    def show_map(self):
        if self.client.args.show_map:
            if self.current_map is None:
                return

            img = self.current_map.img_colour

            x1, y1 = self.get_coordinates(as_pixels=True, real=True)
            x2, y2 = x1 + self.tile_size - 1, y1 + self.tile_size - 1

            cv2.rectangle(
                img, (x1, y1), (x2, y2),
                self.colour, thickness=1
            )

            x1, y1, x2, y2 = self.local_bbox()
            offset = round(self._local_zone_radius * self.tile_size)
            cv2.rectangle(
                img, (x1, y1), (x2, y2),
                self.colour, thickness=1
            )

            cv2.circle(
                img, (x1 + offset, y1 + offset),
                self.parent.orb_radius, self.colour, thickness=1)


class Map(object):
    """Represents a collection of chunks from the Runescape map."""

    PATH_TEMPLATE = '{root}/data/maps/{z}/{x}_{y}.png'

    DEFAULT_OFFSET_X = 0
    DEFAULT_OFFSET_Y = 0

    def __init__(self, client, chunk_set, name=None, graph=None, labels=None,
                 chunk_shape=(256, 256, 3),
                 offsets=(DEFAULT_OFFSET_X, DEFAULT_OFFSET_Y)):
        """
        Determine chunks required to concatenate map chunks into a single
        image.

        Chunk set can actually just be top left and bottom right, missing
        chunks will be calculated.
        """

        # init params
        self.client = client
        self.name = name
        self._chunk_set = chunk_set
        self._chunk_shape = chunk_shape
        self._graph = self.generate_graph(graph)
        self._labels_meta = None
        self._labels = None
        self.generate_labels(labels)

        # settings for processing the map images + minimap on run
        self._canny_lower = 60
        self._canny_upper = 130

        # some maps are more or less offset than others
        self.offset_x, self.offset_y = offsets

        # individual map chunk images are cached here
        self._chunks = dict()
        self._chunks_original = dict()

        # determine what chunk images are needed and load them in
        self._chunk_matrix = self._arrange_chunk_matrix()
        self._chunk_index_boundary = self._get_chunk_set_boundary()
        self._img = self.concatenate_chunks()
        self._img_original = self.concatenate_chunks(original=True)
        self._img_colour = None

    @property
    def img(self):
        return self._img

    @property
    def img_colour(self):
        return self._img_colour

    @property
    def graph(self):
        return self._graph

    @property
    def labels(self):
        return self._labels

    def generate_labels(self, labels):
        """
        Labels come in the following form,
        {
            <label>: {
                colour: <bgra_colour tuple[int, int, int, int]>
                nodes: {
                    <node tuple[int, int]>: {
                        x_offset: <x int>,
                        y_offset: <y int>,
                        size: <size float>,
                        width: <tile width int>,
                        height: <tile height int>,
                    },
                    ...
                }
            },
            ...
        }

        TODO: convert dict to pydantic data class

        Generate a mapping of node: <labels> so we can find node labels easily.
        """

        # keep a copy of the original in case we need it later
        self._labels_meta = deepcopy(labels)

        new_labels = defaultdict(set)
        for label, data in labels.items():
            nodes = data.get('nodes', dict())
            for node, meta in nodes.items():
                new_labels[node].add(label)

        self._labels = new_labels

    def label_to_node(self, u, limit=None):
        """
        Convert a text label into set of (x, y) tuples (if it exists).
        :returns: A set of (x, y) tuples.
        """
        if isinstance(u, tuple):
            return {u}

        nodes = set()
        for node, labels in self.labels.items():
            if u in labels:
                nodes.add(node)
        if limit:
            return set(islice(nodes, limit))
        else:
            return nodes

    def node_to_label(self, u, limit=None):
        """
        Convert an (x, y) tuple into a set of string label (if it exists)
        :returns: a set of label strings
        """

        if isinstance(u, str):
            return {u}

        labels = set()
        for label, data in self._labels_meta.items():
            nodes = data.get('nodes', dict())
            if u in nodes:
                labels.add(label)

        if limit:
            return set(islice(labels, limit))
        else:
            return labels

    def label_colour(self, label):
        """Get the colour for specified label or return a default."""

        meta = self._labels_meta.get(label, {})
        colour = meta.get('colour', WHITE)

        return colour

    def generate_graph(self, graph):
        """Converts simple graph into weighted graph by distance."""

        weight_graph = defaultdict(dict)
        mm = self.client.minimap.minimap
        for node, neighbours in graph.items():
            for neighbour in neighbours:
                # calculate distance and add to both edges
                distance = mm.distance_between(node, neighbour)
                weight_graph[node][neighbour] = distance
                weight_graph[neighbour][node] = distance

        return weight_graph

    def find(self, nearest=None, label=None):
        """
        Find node(s) by supplied parameters.
        :param tuple nearest: Find the nearest node to the supplied node
        :param str label: Filter search results by a regex pattern match on
            the string against node labels.
        """

        mm = self.client.minimap.minimap

        if label and label in self._labels_meta:
            keys = self._labels_meta[label]['nodes'].keys()
        else:
            keys = self.graph.keys()

        if nearest:
            try:
                return min(keys, key=lambda u: mm.distance_between(u, nearest))
            except ValueError:
                return None
        else:
            return keys

    def get_meta(self, label):
        """Get label metadata. See
        :meth:`wizard_eyes.game_objects.minimap.gps.Map.generate_labels`
        for structure of data.
        """
        meta = self._labels_meta.get(label, {})
        return meta

    def copy_original(self):
        """Reset the original colour image with a new copy."""
        self._img_colour = self._img_original.copy()

    def process_img(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = cv2.Canny(img, self._canny_lower, self._canny_upper)
        return img

    def _get_chunk_set_boundary(self):
        """Determine min and max chunk indices from init chunk set."""

        min_x = min_y = float('inf')
        max_x = max_y = -float('inf')
        z = None

        for (x, y, _z) in self._chunk_set:
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

    def _arrange_chunk_matrix(self):
        """
        Create a 2D array of chunk indices from the initial chunk set.
        If chunk set was just the top left and bottom right indices, this
        method will fill in the gaps to provide the full set, in order.
        """

        min_x, min_y, max_x, max_y, z = self._get_chunk_set_boundary()

        chunk_list = list()
        # NOTE: chunks are numbered from bottom left, so we must iterate in
        #       the opposite direction
        for y in range(max_y, min_y - 1, -1):
            chunk_row = list()
            for x in range(min_x, max_x + 1):
                chunk_row.append((x, y, z))
            chunk_list.append(chunk_row)

        return chunk_list

    def load_chunks(self, *chunks, fill_missing=None):

        for (x, y, z) in chunks:

            # attempt to load the map chunk from disk
            chunk_path = self.PATH_TEMPLATE.format(
                root=get_root(),
                x=x, y=y, z=z,
            )
            if exists(chunk_path):
                chunk = cv2.imread(chunk_path)
            else:
                chunk = None

            # resolve if disk file does not exist
            if chunk is None:
                if fill_missing is None:
                    shape = tuple(self._chunk_shape[:2])
                    chunk_processed = numpy.zeros(
                        shape=shape, dtype=numpy.dtype('uint8'))
                    shape = self._chunk_shape
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

    def concatenate_chunks(self, original=False):

        col_data = list()
        for row in self._chunk_matrix:
            row_data = list()
            for (x, y, z) in row:
                chunk = self.get_chunk(x, y, z, original=original)
                row_data.append(chunk)
            row_data = numpy.concatenate(row_data, axis=1)
            col_data.append(row_data)
        concatenated_chunks = numpy.concatenate(col_data, axis=0)

        return concatenated_chunks
