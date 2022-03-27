import cv2
import keyboard
import argparse
import pickle
import threading
from os.path import realpath, basename, splitext
from uuid import uuid4
from collections import defaultdict

from wizard_eyes.application import Application
from wizard_eyes.game_objects.minimap.gps import Map
from wizard_eyes.file_path_utils import get_root


lock = threading.Lock()


class Graph(dict):

    def __repr__(self):
        return str(self.__dict__)

    def __setitem__(self, key, item):
        # first link item to key
        if key in self.__dict__:
            self.__dict__[key].add(item)
        else:
            self.__dict__[key] = {item}
        # then link item to key
        if item in self.__dict__:
            self.__dict__[item].add(key)
        else:
            self.__dict__[item] = {key}

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        # first remove the node
        nodes = self.__dict__.pop(key)
        # then remove all nodes it was connected to
        for node in nodes:
            s = self.__dict__[node]
            self.__dict__[node] = s.difference({key})

    def __contains__(self, key):
        return key in self.__dict__

    def items(self):
        return self.__dict__.items()


class GraphMaker(Application):
    """
    Tool for loading in a map and then marking out a node graph for map
    coordinates that are connected.
    """

    MAP_PATH = '{root}/data/maps/meta/{name}.pickle'

    # colours are BGRA
    RED = (92, 92, 205, 255)  # indianred
    BLUE = (235, 206, 135, 255)  # skyblue
    YELLOW = (0, 215, 255, 255)  # gold

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.args = None
        self.y_offset = None
        self.x_offset = None
        self.c_size = None
        self.hotkey_modes = None
        self.labels_mode = None
        self.node_history = None
        self.graph = None
        self.distances = None
        self.labels = None
        self.path = None
        self.target = None
        self.cursor = None

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        group = parser.add_mutually_exclusive_group()
        group.add_argument('--map-name', help='name for map when saved.')
        group.add_argument('--map-path', help='path to save map to.')

        parser.add_argument(
            '--load-map', action='store_true', default=False,
            help='optional load from an existing map')

        parser.add_argument(
            '--start-xy', nargs=2, type=int,
            default=(133, 86),  # by the willow trees
            help='Specify starting coordinates'
        )

        parser.add_argument(
            '--end-xy', nargs=2, type=int,
            default=(89, 113),  # hosidius bank
            help='Specify end coordinates'
        )

        parser.add_argument(
            '--chunks', nargs=6, type=int,
            default=[26, 57, 0, 28, 55, 0],
            help='Specify chunks to load. Must be of format '
                 'top left (x, y, z) and bottom right (x, y, z)'
        )

        parser.add_argument(
            '--checkpoints', nargs='+',
            help='Calculate the path with checkpoints.'
        )

        args, _ = parser.parse_known_args()
        return args

    def add_node(self, v2):
        lock.acquire()

        if self.hotkey_mode == 'cursor':
            v2 = self.cursor

        # join the last node to the incoming one
        v1 = self.node_history[-1]
        self.graph[v1] = v2
        # cache distances for later use
        d = self.client.minimap.minimap.distance_between(v1, v2)
        self.distances[v1][v2] = d
        self.distances[v2][v1] = d
        # make the incoming node the new last node
        self.node_history.append(v2)

        lock.release()

    def remove_node(self, v1):
        lock.acquire()
        nh = self.node_history

        # remove node from node history
        del nh[nh.index(v1)]
        # remove it from the graph (it handles edges internally)
        del self.graph[v1]
        # remove node from distances dict
        neighbours = self.distances.pop(v1)
        for neighbour in neighbours:
            self.distances[neighbour].pop(v1)

        lock.release()

    def label_node(self, v):
        """Add a human readable label to a node."""

        lock.acquire()

        if self.hotkey_mode == 'cursor':
            v = self.cursor

        if v not in self.graph:
            print(f'Not a node: {v}')
            lock.release()
            return

        self.labels[v] = input('Label: ')

        lock.release()

    def draw_graph(self):

        lock.acquire()

        mm = self.client.minimap.minimap
        img = mm.gps.current_map.img_colour

        if not self.client.args.show_map:
            lock.release()
            return

        labels_mode = self.labels_mode[0]
        drawn = set()
        for node, neighbours in self.graph.items():
            for neighbour in neighbours:
                x, y = node
                ax1, ay1, ax2, ay2 = mm.coordinates_to_pixel_bbox(x, y)

                x, y = neighbour
                bx1, by1, bx2, by2 = mm.coordinates_to_pixel_bbox(x, y)

                if (node, neighbour) not in drawn:

                    # draw a line between nodes
                    # +1 so we draw to middle of node bbox
                    cv2.line(
                        img, (ax1+1, ay1+1), (bx1+1, by1+1), self.RED)

                    # draw the nodes on top so the line join is hidden
                    cv2.rectangle(
                        img, (ax1, ay1), (ax2, ay2), self.BLUE, -1)
                    cv2.rectangle(
                        img, (bx1, by1), (bx2, by2), self.BLUE, -1)

                    if not labels_mode:
                        continue

                    label = self.labels.get(node)
                    if label is None and labels_mode != 'labels only':
                        label = str(node)

                    # now write coords next to nodes
                    if label:
                        node_xy = (ax1 - self.x_offset, ay1 - self.y_offset)
                        cv2.putText(
                            img, label, node_xy,
                            cv2.FONT_HERSHEY_SIMPLEX, self.c_size, self.BLUE,
                            thickness=1
                        )

                    label = self.labels.get(neighbour)
                    if label is None and labels_mode != 'labels only':
                        label = str(neighbour)

                    neighbour_xy = (bx1 - self.x_offset, by1 - self.y_offset)
                    cv2.putText(
                        img, label, neighbour_xy,
                        cv2.FONT_HERSHEY_SIMPLEX, self.c_size, self.BLUE,
                        thickness=1
                    )

                drawn.add((neighbour, node))

        if self.hotkey_mode == 'cursor':
            x1, y1, x2, y2 = mm.coordinates_to_pixel_bbox(*self.cursor)
            cv2.rectangle(
                img, (x1, y1), (x2, y2), self.BLUE, -1)

        lock.release()

    def calculate_route(self, start, end):

        lock.acquire()

        # TODO: remove checkpoints once we pass them
        path = self.client.minimap.minimap.gps.get_route(
            start, end, checkpoints=self.args.checkpoints)

        self.path = path

        lock.release()

    def draw_path(self):

        lock.acquire()

        if not self.client.args.show_map:
            lock.release()
            return

        # draw to map
        mm = self.client.minimap.minimap
        img = mm.gps.current_map.img_colour
        for node in self.path:
            x, y = node
            ax1, ay1, ax2, ay2 = mm.coordinates_to_pixel_bbox(x, y)

            # draw a slightly larger, golden box
            cv2.rectangle(
                img, (ax1-1, ay1-1), (ax2+1, ay2+1), self.YELLOW, -1)

        lock.release()

    def get_map_path(self):
        if self.args.map_name:
            path = self.MAP_PATH.format(
                root=get_root(), name=self.args.map_name)
        elif self.args.map_path:
            path = self.args.map_path
        else:
            path = self.MAP_PATH.format(
                root=get_root(), name=uuid4().hex)

        return realpath(path)

    def save_map(self):

        gps = self.client.minimap.minimap.gps

        graph = dict()
        for k, v in self.graph.items():
            graph[k] = v
        data = dict(chunks=gps.current_map._chunk_set, graph=graph,
                    labels=self.labels)

        path = self.get_map_path()

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        self.client.minimap.logger.info(f'Saved to: {path}')

    def load_graph(self):

        new_graph = Graph()
        mm = self.client.minimap.minimap

        if self.args.load_map:
            path = self.get_map_path()
            with open(path, 'rb') as f:
                data = pickle.load(f)

            graph = data.get('graph', {})
            for node, neighbours in graph.items():
                for neighbour in neighbours:
                    # add the node to graph (internally it will add the
                    # reverse edge)
                    new_graph[node] = neighbour
                    # calculate distance and add to both edges
                    distance = mm.distance_between(node, neighbour)
                    self.distances[node][neighbour] = distance
                    self.distances[neighbour][node] = distance

            labels = data.get('labels', {})
            self.labels = labels

        return new_graph

    @property
    def hotkey_mode(self):
        return self.hotkey_modes[0]

    def cycle_labels_mode(self):

        lock.acquire()

        mode = self.labels_mode.pop(0)
        self.labels_mode.append(mode)

        lock.release()

    def cycle_hotkey_mode(self):

        lock.acquire()

        mode = self.hotkey_modes.pop(0)
        self.hotkey_modes.append(mode)

        method = getattr(self, f'{self.hotkey_mode}_hotkeys')
        try:
            method()
        except TypeError:
            print(f'Invalid mode: {self.hotkey_mode}')

        print(f'Mode: {self.hotkey_mode}')

        lock.release()

    def basic_hotkeys(self):

        gps = self.client.minimap.minimap.gps

        keyboard.add_hotkey(
            'subtract', lambda: self.remove_node(self.node_history[-1]))
        keyboard.add_hotkey(
            'plus', lambda: self.add_node(gps.get_coordinates())
        )
        keyboard.add_hotkey('asterisk', self.cycle_hotkey_mode)

        keyboard.add_hotkey('0', self.save_map)

        keyboard.add_hotkey(
            '.', lambda: self.client.save_img(
                name=self.args.map_name, original=True))

        keyboard.add_hotkey(
            'slash', lambda: self.label_node(gps.get_coordinates()))

        keyboard.add_hotkey('shift', self.set_current_node)

    def free_roam_hotkeys(self):

        keyboard.remove_all_hotkeys()
        self.basic_hotkeys()

    def cursor_hotkeys(self):
        keyboard.remove_all_hotkeys()

        self.basic_hotkeys()
        gps = self.client.minimap.minimap.gps

        self.cursor = gps.get_coordinates()
        keyboard.add_hotkey('4', lambda: self.move_cursor(-1, 0))
        keyboard.add_hotkey('6', lambda: self.move_cursor(1, 0))
        keyboard.add_hotkey('8', lambda: self.move_cursor(0, -1))
        keyboard.add_hotkey('2', lambda: self.move_cursor(0, 1))

    def set_current_node(self):
        """
        Set the node history to the current node without creating an edge.
        """

        lock.acquire()

        node = self.client.minimap.minimap.gps.get_coordinates()
        if self.hotkey_mode == 'cursor':
            node = self.cursor

        if node not in self.graph:
            print(f'Not a node: {node}')
            lock.release()
            return

        self.node_history = [node]

        lock.release()

    def move_cursor(self, dx, dy):
        """Move cursor object independent of the player."""
        lock.acquire()

        x, y = self.cursor
        x += dx
        y += dy
        self.cursor = x, y

        lock.release()

    def config_hotkeys(self):

        keyboard.remove_all_hotkeys()
        self.basic_hotkeys()

        keyboard.add_hotkey('7', self.cycle_labels_mode)

        x = 'x_offset'
        y = 'y_offset'
        keyboard.add_hotkey(
            '6', lambda: setattr(self, x, getattr(self, x) - 4))
        keyboard.add_hotkey(
            '4', lambda: setattr(self, x, getattr(self, x) + 4))
        keyboard.add_hotkey(
            '2', lambda: setattr(self, y, getattr(self, y) - 4))
        keyboard.add_hotkey(
            '8', lambda: setattr(self, y, getattr(self, y) + 4))

        s = 'c_size'
        keyboard.add_hotkey(
            '5', lambda: setattr(self, s, getattr(self, s) + 0.1))
        keyboard.add_hotkey(
            '0', lambda: setattr(self, s, getattr(self, s) - 0.1))

        keyboard.add_hotkey('7', self.cycle_labels_mode)

    def setup(self):
        """"""

        self.args = self.parse_args()
        self.y_offset = 4
        self.x_offset = 0
        self.c_size = 0.2
        self.hotkey_modes = ['free_roam', 'cursor', 'config']
        self.labels_mode = [True, False, 'labels only']

        # TODO: configurable init position
        start = tuple(self.args.start_xy)
        end = tuple(self.args.end_xy)
        self.node_history = [start]
        self.distances = defaultdict(dict)
        self.graph = self.load_graph()

        mm = self.client.minimap.minimap
        gps = mm.gps

        top_left = tuple(self.args.chunks[:3])
        bottom_right = tuple(self.args.chunks[3:])
        if self.args.map_name:
            gps.load_map(self.args.map_name)
        elif self.args.map_path:
            name, _ = splitext(basename(self.args.map_path))
            new_map = Map(self.client, {top_left, bottom_right}, name=name)
            gps.maps[name] = new_map
            gps.load_map(name)
        else:
            # this is a bit of a hack... we shouldn't need to use this anyway
            new_map = Map(self.client, {top_left, bottom_right})
            gps.maps[None] = new_map
            gps.load_map(None)

        gps.set_coordinates(*start)
        self.calculate_route(start, end)

        self.basic_hotkeys()

    def update(self):
        """"""

        mm = self.client.minimap.minimap
        gps = mm.gps
        mm.update()

        # recalculate route based on current position
        end = tuple(self.args.end_xy)
        self.calculate_route(gps.get_coordinates(), end)

        self.draw_path()
        self.draw_graph()
        self.msg.append(
            f'current: {gps.get_coordinates()} '
            f'last node: {self.node_history[-1]}')

    def action(self):
        """"""

        self.msg.append(str(self.path))


def main():
    app = GraphMaker()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
