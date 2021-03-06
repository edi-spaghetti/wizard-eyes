import argparse
import pickle
import threading
import tkinter
from os.path import realpath, basename, splitext, isfile
from uuid import uuid4
from collections import defaultdict
from copy import deepcopy
from functools import wraps
from time import sleep

import cv2
import numpy
import keyboard

from wizard_eyes.application import Application
from wizard_eyes.game_objects.minimap.gps import Map
from wizard_eyes.file_path_utils import get_root


lock = threading.Lock()


def wait_lock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while lock.locked():
            sleep(0.001)
        lock.acquire()
        func(*args, **kwargs)
        lock.release()
    return wrapper


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

    def add_edgeless_node(self, key):
        self.__dict__[key] = set()


class MapMaker(Application):
    """
    Tool for loading in a map and then marking out a node graph for map
    coordinates that are connected.
    """

    MAP_PATH = '{root}/data/maps/meta/{name}.pickle'

    # colours are BGRA
    RED = (92, 92, 205, 255)  # indianred
    BLUE = (235, 206, 135, 255)  # skyblue
    YELLOW = (0, 215, 255, 255)  # gold
    GREEN = (0, 100, 0, 255)  # darkgreen

    DEFAULT_LABEL_COLOUR = (255, 255, 255, 255)  # white
    DEFAULT_LABEL_SETTINGS = dict(x_offset=0, y_offset=4, size=0.25)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.args = None
        self.labels_mode = None
        self.node_history = None
        self.graph = None
        self.distances = None
        self.labels = None
        self.label_settings = None
        self.label_backref = None
        self.path = None
        self.target = None
        self.cursor = None

        # label/node manager widgets
        self.map_manager = None
        self.gui_frames = None
        self.label_entries = None
        self.channels = ('blue', 'green', 'red')

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        group = parser.add_mutually_exclusive_group(required=True)
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

    @property
    def cursor_mode(self):
        """
        Attempt to get cursor mode from map manager widget.
        If the widget has not been initialised, assume cursor mode is off.
        """
        try:
            return self.label_entries.get('cursor_mode').get()
        except AttributeError:
            return False

    @property
    def current_node(self):
        if self.cursor_mode:
            return self.cursor
        else:
            return self.client.minimap.minimap.gps.get_coordinates()

    @property
    def current_label(self):
        try:
            return self.label_entries.get('label').get()
        except AttributeError:
            return

    # Async methods to edit map data

    def _add_node(self):
        """
        Add a node to the graph.
        Should be called from inside the :meth:`MapMaker.add_to_map` method,
        where the thread locks should already have been acquired.
        """

        if self.cursor_mode:
            v2 = self.cursor
        else:
            v2 = self.client.minimap.minimap.gps.get_coordinates()

        # join the last node to the incoming one
        v1 = self.node_history[-1]
        self.graph[v1] = v2
        # cache distances for later use
        d = self.client.minimap.minimap.distance_between(v1, v2)
        self.distances[v1][v2] = d
        self.distances[v2][v1] = d
        # make the incoming node the new last node
        self.node_history.append(v2)

    def _remove_node(self):
        """
        Remove the last node from node history.
        Should be called from inside the :meth:`MapMaker.remove_from_map`
        method, where the thread locks should already have been acquired.
        """

        if len(self.node_history) <= 1:
            print('Cannot remove last node. Use reset node history.')
            return

        # get the current node
        node = self.current_node
        if node not in self.graph:
            node = self.node_history[-1]

        # remove node from node history
        del self.node_history[-1]
        # remove it from the graph (it handles edges internally)
        del self.graph[node]
        # remove node from distances dict
        neighbours = self.distances.pop(node)
        for neighbour in neighbours:
            self.distances[neighbour].pop(node)

    @wait_lock
    def add_to_map(self):
        """
        Add a human readable label to a node. If the label is set to
        'graph' then the node will be added to the graph instead."""

        if not self.current_label:
            print('Must set a label.')
            return

        label = self.current_label
        print(f'Adding to {label}')
        if label == 'graph':
            self._add_node()
            return

        # determine what node we should be adding
        if self.cursor_mode:
            v = self.cursor
        else:
            v = self.client.minimap.minimap.gps.get_coordinates()

        # get the label settings
        try:
            colour = tuple([int(self.label_entries.get(c).get()) % 256
                            for c in self.channels])
        except (AttributeError, TypeError, ValueError):
            colour = self.DEFAULT_LABEL_COLOUR

        self.labels[label]['colour'] = colour
        self.labels[label]['nodes'][v] = deepcopy(self.DEFAULT_LABEL_SETTINGS)

        # add label to label settings, so the layer is displayed on draw
        if label not in self.label_settings:
            self._add_label_display(label)

        # set up backref for the new node
        # NOTE: nodes labeled twice will overwrite here
        self.label_backref[v] = label

    @wait_lock
    def remove_from_map(self):

        if not self.label_entries.get('label').get():
            print('Must set a label.')
            return

        label = self.label_entries.get('label').get()
        print(f'Removing from {label}')
        if label == 'graph':
            self._remove_node()
            return

        # determine what node we should be removing
        node = self.current_node
        if node not in self.labels[label]['nodes']:
            print(f'Not a labelled node: {node}')
            return

        # remove the node from labels
        del self.labels[label]['nodes'][node]
        if not self.labels[label]['nodes']:
            # if there's not more nodes left, remove whole label
            del self.labels[label]
            self._remove_label_display(label)

        # remove from backref
        # NOTE: nodes labeled twice will conflict here
        del self.label_backref[node]

    @wait_lock
    def reset_node_history(self):
        """
        Set the node history to the current node without creating an edge.
        """

        node = self.client.minimap.minimap.gps.get_coordinates()
        if self.cursor_mode:
            node = self.cursor

        if node not in self.graph:
            print(f'Not a node: {node}')
            return

        self.node_history = [node]

    @wait_lock
    def move_cursor(self, dx, dy):
        """Move cursor object independent of the player."""

        x, y = self.cursor
        x += dx
        y += dy
        self.cursor = x, y

    @wait_lock
    def cycle_labels_mode(self):
        mode = self.labels_mode.pop(0)
        self.labels_mode.append(mode)

    @wait_lock
    def reset_cursor(self):
        self.cursor = self.client.minimap.minimap.gps.get_coordinates()

    @wait_lock
    def toggle_label_display(self, name):
        self.label_settings[name] = not self.label_settings[name]

    def _add_label_display(self, name, enabled=True):
        self.label_settings[name] = enabled

        # TODO: figure out why checkboxes are not checked on creation
        var = tkinter.BooleanVar(value=enabled)
        checkbox = tkinter.Checkbutton(
            self.gui_frames['checkboxes'], text=name,
            variable=var, onvalue=True, offvalue=False,
            command=lambda label=name: self.toggle_label_display(label)
        )
        checkbox.pack()
        self.label_entries['label_config'][name] = checkbox

    def _remove_label_display(self, name):
        del self.label_settings[name]
        widget = self.label_entries['label_config'].pop(name)
        widget.destroy()

    def open_map_manager(self):

        self.gui_frames = dict()
        self.label_entries = dict()
        self.reset_cursor()

        root = tkinter.Tk()
        self.map_manager = root

        # colour setting for labels
        for colour in self.channels:
            int_var = tkinter.IntVar()
            colour_entry = tkinter.Entry(
                root, bg=colour, textvariable=int_var)
            colour_entry.pack(side=tkinter.LEFT)
            self.label_entries[colour] = int_var

        # cursor checkbox
        var = tkinter.BooleanVar()
        cursor_checkbox = tkinter.Checkbutton(
            root, text='cursor', variable=var, onvalue=True, offvalue=False,
            command=self.reset_cursor,
        )
        cursor_checkbox.pack()
        self.label_entries['cursor_mode'] = var

        # label entry box
        description = tkinter.Label(root, text='Label Name')
        description.config(font=('helvetica', 14))
        description.pack()
        entry = tkinter.Entry(root)
        entry.pack()
        self.label_entries['label'] = entry

        # add checkboxes for each label's display options
        self.label_entries['label_config'] = dict()
        self.gui_frames['checkboxes'] = tkinter.Frame(root)
        for label, enabled in self.label_settings.items():
            self._add_label_display(label, enabled=enabled)
        self.gui_frames['checkboxes'].pack()

        # add current node
        add_button = tkinter.Button(
            text='Add', command=self.add_to_map)
        add_button.pack()

        # remove current node
        remove_button = tkinter.Button(
            text='Remove', command=self.remove_from_map)
        remove_button.pack()

        # move the cursor around
        cursor = tkinter.Button(root, text='Cursor')
        cursor.bind('<Key-Left>', lambda e: self.move_cursor(-1, 0))
        cursor.bind('<Key-Right>', lambda e: self.move_cursor(1, 0))
        cursor.bind('<Key-Down>', lambda e: self.move_cursor(0, 1))
        cursor.bind('<Key-Up>', lambda e: self.move_cursor(0, -1))
        cursor.bind('<Button-1>', lambda e: cursor.focus_set())
        cursor.pack()

        # reset node history to current node
        reset_button = tkinter.Button(
            text='Reset Node History', command=self.reset_node_history)
        reset_button.pack()

        # save the map
        save_button = tkinter.Button(text='Save Map', command=self.save_map)
        save_button.pack()

        # the client image
        save_img_button = tkinter.Button(
            root,
            text='Save Client Image',
            command=lambda: self.client.save_img(
                name=self.args.map_name, original=True)
        )
        save_img_button.bind(
            '<Button-1>', lambda e: save_img_button.focus_set())
        save_img_button.bind('<Return>', lambda e: self.client.save_img(
                name=self.args.map_name, original=True))
        save_img_button.pack()

        # toggle labels mode
        labels_mode_button = tkinter.Button(
            text='Toggle Labels', command=self.cycle_labels_mode
        )
        labels_mode_button.pack()

        root.mainloop()

        # reset the widgets we keep track of once the manager is closed
        self.label_entries = None
        self.map_manager = None
        self.gui_frames = None

    def draw_label_legend(self):
        """
        Draw a table in top left of the map to display all the layers
        currently loaded, including tally of the number of nodes, and
        display whether or not the layer is selected.
        """

        img = self.client.minimap.minimap.gps.current_map.img_colour

        layers = sorted(self.label_settings.keys())
        text_settings = list()
        max_width = 0
        total_height = 0
        x_margin = y_margin = 5

        for i, layer in enumerate(layers):
            if layer == 'graph':
                size = len(self.graph.items())
                colour = self.BLUE
            else:
                size = len(self.labels[layer]['nodes'])
                colour = self.labels[layer]['colour']

            # calculate layer text settings
            status = '[x]'
            if not self.label_settings[layer]:
                status = '[ ]'
            text = f'{status} {layer} ({size})'
            thickness = 1 + (self.current_label == layer)
            size = 0.5
            (width, height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
            x = x_margin
            y = total_height + y_margin + height

            # add to list to be draw later
            text_settings.append((text, (x, y), colour, size, thickness))

            # update sum totals
            if width > max_width:
                max_width = width
            total_height += (y_margin + height)

        # broadcast a black background on to the image for the layers
        x = max_width + x_margin
        y = total_height + y_margin
        background = numpy.ones((y, x, 3), dtype=numpy.uint8)
        img[y_margin:y + y_margin, x_margin:x + x_margin] = background

        # draw the labels on top
        for text, (x, y), colour, size, thickness in text_settings:
            cv2.putText(
                img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, colour,
                thickness=thickness
            )

    def draw_labels(self):
        """
        Draw all node labels that are currently active.
        """

        labels_mode = self.labels_mode[0]
        if not labels_mode:
            return

        # draw label layers on side of screen
        self.draw_label_legend()

        if self.label_settings['graph']:
            for node, _ in self.graph.items():
                self.draw_label(node, self.BLUE)

        for label, data in self.labels.items():

            # if layer has been disabled, skip its draw method
            if not self.label_settings[label]:
                continue

            colour = data.get('colour', self.DEFAULT_LABEL_COLOUR)
            nodes = data.get('nodes', dict())
            for node, settings in nodes.items():
                self.draw_label(node, colour, **settings)

    def draw_label(self, node, colour, x_offset=0, y_offset=4, size=0.25):

        mm = self.client.minimap.minimap
        img = mm.gps.current_map.img_colour

        labels_mode = self.labels_mode[0]
        label = self.label_backref.get(node)
        if label is None and labels_mode != 'labels only':
            label = str(node)

        x1, y1, x2, y2 = mm.coordinates_to_pixel_bbox(*node)
        x = int((x1 + x2) / 2) + x_offset
        y = int((y1 + y2) / 2) + y_offset

        # first draw the node bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)

        # now write coordinates / label next to node if there is one
        if label and not labels_mode == 'nodes_only':
            cv2.putText(
                img, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, colour,
                thickness=1
            )

    def draw_edges(self):
        """
        Draw the graph nodes with edges between them.
        """

        mm = self.client.minimap.minimap
        img = mm.gps.current_map.img_colour

        drawn = set()
        for node, neighbours in self.graph.items():

            x, y = node
            ax1, ay1, ax2, ay2 = mm.coordinates_to_pixel_bbox(x, y)

            # # TODO: remove edgeless nodes
            # if not neighbours:
            #     # draw the node and label with no edges
            #     cv2.rectangle(
            #         img, (ax1, ay1), (ax2, ay2), self.BLUE, -1)

            for neighbour in neighbours:

                x, y = neighbour
                bx1, by1, bx2, by2 = mm.coordinates_to_pixel_bbox(x, y)

                if (node, neighbour) not in drawn:

                    # draw a line between nodes
                    # +1 so we draw to middle of node bbox
                    cv2.line(
                        img, (ax1+1, ay1+1), (bx1+1, by1+1), self.RED)

                    # # draw the nodes on top so the line join is hidden
                    # cv2.rectangle(
                    #     img, (ax1, ay1), (ax2, ay2), self.BLUE, -1)
                    # cv2.rectangle(
                    #     img, (bx1, by1), (bx2, by2), self.BLUE, -1)

                drawn.add((neighbour, node))

    def draw_nodes(self):
        """
        Draw all nodes recorded in current map.
        Special case is made for graph nodes so their edges can be drawn, as
        well as if a path exists on the current graph
        """

        mm = self.client.minimap.minimap
        img = mm.gps.current_map.img_colour

        if not self.client.args.show_map:
            return

        # draw the path first, so the graph nodes appear back-lit
        self.draw_path()
        self.draw_edges()

        # TODO: add cursor as a temporary label
        if self.cursor_mode:
            x1, y1, x2, y2 = mm.coordinates_to_pixel_bbox(*self.cursor)
            cv2.rectangle(
                img, (x1, y1), (x2, y2), self.BLUE, -1)

        # finally draw labels that are active
        self.draw_labels()

    @wait_lock
    def calculate_route(self, start, end):

        # TODO: remove checkpoints once we pass them
        path = self.client.minimap.minimap.gps.get_route(
            start, end, checkpoints=self.args.checkpoints)

        self.path = path

    def draw_path(self):
        """
        Draw the path (if there is one).
        """

        if not self.client.args.show_map:
            return

        if self.path is None:
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
        labels = dict()
        for k, v in self.labels.items():
            labels[k] = v
        data = dict(chunks=gps.current_map._chunk_set, graph=graph,
                    labels=labels)

        path = self.get_map_path()

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        self.client.minimap.logger.info(f'Saved to: {path}')

    def load_map(self):
        """
        Load map data from disk.
        It is expected to be a pickle dictionary of the following format,

        {
            graph: {
                <node tuple[int, int]>: <neighbours set[tuple[int, int], ...]>,
            },
            labels: {
                <label>: {
                    colour: <bgra_colour tuple[int, int, int, int]>
                    nodes: {
                        <node tuple[int, int]>: {
                            x_offset: <x int>,
                            y_offset: <y int>,
                            size: <size float>
                        },
                        ...
                    }
                },
                ...
            }
        }
        """

        gps = self.client.minimap.minimap.gps
        top_left = tuple(self.args.chunks[:3])
        bottom_right = tuple(self.args.chunks[3:])

        # first load the source data
        if self.args.load_map:
            path = self.get_map_path()
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {'chunks': {top_left, bottom_right},
                    'graph': {},
                    'labels': {}}
            path = self.get_map_path()
            if isfile(path):
                raise IOError(f'Cannot create new map - '
                              f'path already exists! {path}')
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        # then pop off the graph and turn it into our internal data structure
        try:
            graph_data = data.pop('graph')
        except KeyError:
            graph_data = {}
        self.graph = self.load_graph(graph_data)

        # what remains should be arbitrary layered label data
        try:
            label_data = data.pop('labels')
        except KeyError:
            label_data = {}
        # label settings for determining if layers should be displayed
        self.label_settings = dict(graph=True)
        # backref to easily find a node's label (if it has one).
        self.label_backref = dict()
        self.labels = self.load_labels(label_data)

        # now load this data into the gps system
        if self.args.map_name:
            gps.load_map(self.args.map_name)
        else:
            name, _ = splitext(basename(self.args.map_path))
            new_map = Map(self.client, {top_left, bottom_right}, name=name)
            gps.maps[name] = new_map
            gps.load_map(name)

    def load_labels(self, data):
        """
        Load labels data into a data structure more convenient for
        modification.

        :param dict data: information on where and how to draw labels.
            Should be a dict with the following format,
                {
                    <label>: {
                        colour: <bgra_colour tuple[int, int, int, int]>
                        nodes: {
                            <node tuple[int, int]>: {
                                x_offset: <x int>,
                                y_offset: <y int>,
                                size: <size float>
                            },
                            ...
                        }
                    },
                    ...
                }
        """

        labels = defaultdict(lambda: dict(colour=None, nodes=dict()))

        for label, label_data in data.items():
            colour = label_data.get('colour', self.DEFAULT_LABEL_COLOUR)
            nodes = label_data.get('nodes', dict())

            labels[label]['colour'] = colour
            labels[label]['nodes'] = nodes

            # add label to label settings, so the layer is displayed on draw
            self.label_settings[label] = True

            # set up backref for every node
            # NOTE: nodes labeled twice will overwrite here
            for node in nodes.keys():
                self.label_backref[node] = label

        return labels

    def load_graph(self, data):

        new_graph = Graph()
        mm = self.client.minimap.minimap

        for node, neighbours in data.items():

            # TODO: remove edgeless nodes, they are being handled by labels now
            if not neighbours:
                new_graph.add_edgeless_node(node)

            for neighbour in neighbours:
                # add the node to graph (internally it will add the
                # reverse edge)
                new_graph[node] = neighbour
                # TODO: pre-calculate weight graph
                # calculate distance and add to both edges
                distance = mm.distance_between(node, neighbour)
                self.distances[node][neighbour] = distance
                self.distances[neighbour][node] = distance

        return new_graph

    def basic_hotkeys(self):
        keyboard.add_hotkey('1', self.open_map_manager)

    def setup(self):
        """"""

        self.args = self.parse_args()
        self.labels_mode = [True, False, 'labels only', 'nodes_only']

        self.distances = defaultdict(dict)
        self.load_map()

        start = tuple(self.args.start_xy)
        end = tuple(self.args.end_xy)
        self.node_history = [start]

        self.client.minimap.minimap.gps.set_coordinates(*start)
        # we can only draw a route if we're loading an existing map
        # (where it is assumed at least some nodes on the graph have been
        # created)
        if self.args.load_map:
            self.calculate_route(start, end)

        self.basic_hotkeys()

    def update(self):
        """"""

        mm = self.client.minimap.minimap
        gps = mm.gps
        mm.update()

        # recalculate route based on current position
        end = tuple(self.args.end_xy)

        # we can only draw a route if we're loading an existing map
        # (where it is assumed at least some nodes on the graph have been
        # created)
        if self.args.load_map:
            self.calculate_route(gps.get_coordinates(), end)

        self.draw_nodes()

        self.msg.append(
            f'current: {gps.get_coordinates()} '
            f'last node: {self.node_history[-1]}')

    def action(self):
        """"""

        # self.msg.append(str(self.cursor_mode))
        self.msg.append(str(self.node_history))
        # self.msg.append(str(self.path))


def main():
    app = MapMaker()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
