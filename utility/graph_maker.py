import cv2
import keyboard
import argparse
import pickle
import threading
from os.path import dirname, realpath
from uuid import uuid4

from client import Application


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

    def items(self):
        return self.__dict__.items()


class GraphMaker(Application):
    """
    Tool for loading in a map and then marking out a node graph for map
    coordinates that are connected.
    """

    MAP_PATH = '{root}/../data/maps/{name}.pickle'

    # colours are BGRA
    RED = (92, 92, 205, 255)  # indianred
    BLUE = (235, 206, 135, 255)  # skyblue

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.args = None
        self.y_offset = None
        self.x_offset = None
        self.c_size = None
        self.show_labels = None
        self.node_history = None
        self.graph = None

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
            required=True,
            default=(133, 86),  # by the willow trees
            help='Specify starting coordinates'
        )

        parser.add_argument(
            '--chunks', nargs=6, type=int,
            required=True,
            default=[26, 57, 0, 28, 55, 0],
            help='Specify chunks to load. Must be of format '
                 'top left (x, y, z) and bottom right (x, y, z)'
        )

        args, _ = parser.parse_known_args()
        return args

    def add_node(self, v2):
        lock.acquire()
        v1 = self.node_history[-1]
        self.graph[v1] = v2
        self.node_history.append(v2)
        lock.release()

    def remove_node(self, v1):
        lock.acquire()
        nh = self.node_history
        del nh[nh.index(v1)]
        del self.graph[v1]
        lock.release()

    def draw_graph(self):

        mm = self.client.minimap.minimap

        lock.acquire()
        drawn = set()
        for node, neighbours in self.graph.items():
            for neighbour in neighbours:
                x, y = node
                ax1, ay1, ax2, ay2 = mm.coordinates_to_pixel_bbox(x, y)

                x, y = neighbour
                bx1, by1, bx2, by2 = mm.coordinates_to_pixel_bbox(x, y)

                if (node, neighbour) not in drawn:
                    # draw the nodes first
                    cv2.rectangle(
                        mm.map_img, (ax1, ay1), (ax2, ay2), self.BLUE, -1)
                    cv2.rectangle(
                        mm.map_img, (bx1, by1), (bx2, by2), self.BLUE, -1)
                    # then draw a line between them
                    # +1 so we draw to middle of node bbox
                    cv2.line(
                        mm.map_img, (ax1+1, ay1+1), (bx1+1, by1+1), self.RED)

                    if not self.show_labels:
                        continue

                    # now write coords next to nodes
                    node_xy = (ax1 - self.x_offset, ay1 - self.y_offset)
                    cv2.putText(
                        mm.map_img, str(node), node_xy,
                        cv2.FONT_HERSHEY_SIMPLEX, self.c_size, self.BLUE,
                        thickness=1
                    )
                    neighbour_xy = (bx1 - self.x_offset, by1 - self.y_offset)
                    cv2.putText(
                        mm.map_img, str(neighbour), neighbour_xy,
                        cv2.FONT_HERSHEY_SIMPLEX, self.c_size, self.BLUE,
                        thickness=1
                    )

                drawn.add((neighbour, node))
        lock.release()

    def get_map_path(self):
        if self.args.map_name:
            path = self.MAP_PATH.format(
                root=dirname(__file__), name=self.args.map_name)
        elif self.args.map_path:
            path = self.args.map_path
        else:
            path = self.MAP_PATH.format(
                root=dirname(__file__), name=uuid4().hex)

        return realpath(path)

    def save_map(self):
        graph = dict()
        for k, v in self.graph.items():
            graph[k] = v

        path = self.get_map_path()

        with open(path, 'wb') as f:
            pickle.dump(graph, f)

        self.client.minimap.logger.info(f'Saved to: {path}')

    def load_graph(self):

        new_graph = Graph()

        if self.args.load_map:
            path = self.get_map_path()
            with open(path, 'rb') as f:
                graph = pickle.load(f)

            for node, neighbours in graph.items():
                for neighbour in neighbours:
                    new_graph[node] = neighbour

        return new_graph

    def toggle_labels(self):
        self.show_labels = not self.show_labels

    def setup(self):
        """"""

        self.args = self.parse_args()
        self.y_offset = 4
        self.x_offset = 0
        self.c_size = 0.2
        self.show_labels = False

        # TODO: configurable init position
        start = tuple(self.args.start_xy)
        self.node_history = [start]
        self.graph = self.load_graph()

        mm = self.client.minimap.minimap
        nh = self.node_history

        top_left = tuple(self.args.chunks[:3])
        bottom_right = tuple(self.args.chunks[3:])
        mm.create_map({top_left, bottom_right})
        mm.set_coordinates(*start)

        keyboard.add_hotkey(
            'backspace', lambda: self.remove_node(nh[-1]))
        keyboard.add_hotkey(
            'plus', lambda: self.add_node(mm.get_coordinates())
        )
        keyboard.add_hotkey('end', self.save_map)

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

        keyboard.add_hotkey('7', self.toggle_labels)

    def update(self):
        """"""

        mm = self.client.minimap.minimap
        mm.update()
        self.draw_graph()
        self.msg.append(
            f'current: {mm.get_coordinates()} '
            f'last node: {self.node_history[-1]}')

    def action(self):
        """"""


def main():
    app = GraphMaker()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
