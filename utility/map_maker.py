from argparse import ArgumentTypeError
import pickle
import threading
import tkinter
from os import listdir, makedirs
from os.path import realpath, basename, splitext, isfile, dirname
from uuid import uuid4
from collections import defaultdict
from copy import deepcopy
from functools import wraps, partial
from time import sleep
from typing import Union, List, Tuple, Literal

import cv2
import numpy
import keyboard

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtMultimedia import QVideoFrameFormat, QVideoFrame
from PySide6.QtCore import Signal
from PySide6.QtGui import QImage

from wizard_eyes.application import Application
from wizard_eyes.game_objects.gauges.gps import Map, MapData
from wizard_eyes.file_path_utils import get_root
from wizard_eyes.script_utils import int_or_str
from wizard_eyes.game_entities.entity import GameEntity


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


class BloodAltarMap(Map):
    """Custom Map class for Blood Altar. This is to circumvent some of the
    anti-botting techniques the blood altar has in place,
    such as there the water/blood, ground and bushes will alternate lighting
    styles independently. However we can just take the red channel and the
    outline of the island is already enough to locate outselves."""

    def process_img(self, img):
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(img, (175, 50, 20), (180, 255, 255))
        mask2 = cv2.inRange(img, (0, 50, 50), (10, 255, 255))

        img = cv2.bitwise_or(mask1, mask2)

        return img


class Visualiser(QWidget):
    """"""

    new_frame = Signal(numpy.ndarray)

    def __init__(self, map_maker, parent=None):
        super().__init__(parent)
        self.map_maker = map_maker

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.video = QVideoWidget()
        self.layout.addWidget(self.video)

        self.new_frame.connect(self.update_image)

    def update_image(self, img: numpy.ndarray):
        image = QImage(
            img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGBA8888
        )
        format_ = QVideoFrameFormat(
            image.size(),
            QVideoFrameFormat.pixelFormatFromImageFormat(image.format())
        )
        frame = QVideoFrame(format_)
        frame.map(QVideoFrame.ReadWrite)
        end = len(image.bits().tobytes())
        frame.bits(0)[:end] = image.bits()
        frame.unmap()
        self.video.videoSink().setVideoFrame(frame)


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

    HIGHLIGHT_ENTITY_COLOUR = RED

    DEFAULT_LABEL_COLOUR = (255, 255, 255, 255)  # white
    DEFAULT_LABEL_SETTINGS = dict(x_offset=0, y_offset=4, size=0.25,
                                  width=1, height=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.q_app = q_app
        # self.gui = Visualiser(self)
        # self._gui_thread = threading.Thread(target=self.open_gui)
        # self._gui_thread.start()

        self.args = None
        self.labels_mode = None
        self.node_history: List[Tuple[int, int,int]] = []
        self.graph = None
        self.distances = None
        self.labels = None
        self.label_settings = None
        self.label_backref = None
        self.path = None
        self.target = None
        self.cursor: Tuple[int, int, int] = 0, 0, 0
        self.update_minimap = True
        self.entities = None
        self.entity_mapping = None
        self.selected_entity: Union[GameEntity, None] = None
        self.offsets: List[int, float] = [
            Map.DEFAULT_OFFSET_X, Map.DEFAULT_OFFSET_Y]
        # self.grid_info_offset = (0, 0)

        # label/node manager widgets
        self.map_manager = None
        self.gui_frames = None
        self.label_entries = None
        self.channels = ('blue', 'green', 'red')

        # set up IO for minimap screenshots
        self.mm_img_dir = None
        self.mm_img_dir_idx = None
        self.mm_img_idx = None

    # def open_gui(self):
    #     self.gui.show()
    #     self.q_app.exec_()

    @property
    def mm_img_path(self):

        if self.mm_img_dir is None:
            # get base path
            path = self.client.gauges.minimap.resolve_path(
                root=get_root(),
                name=None
            )
            root = f'{dirname(path)}/recording'

            # determine dir idx
            if self.mm_img_dir_idx is None:
                idx = len(listdir(root))
                self.mm_img_dir_idx = idx
                self.mm_img_dir = f'{root}/{idx}'

        # ensure this dir exists (including if we toggled idx)
        makedirs(self.mm_img_dir, exist_ok=True)

        # determine image idx
        if self.mm_img_idx is None:
            self.mm_img_idx = len(listdir(self.mm_img_dir))

        return f'{self.mm_img_dir}/{self.mm_img_idx}.png'

    def add_default_start_xy_arg(self, parser):
        """Skip default start xy, we'll add it to mutually exclusive group
        later."""

    def add_default_map_name_arg(self, parser):
        """Skip map name arg, we'll add it later."""

    def region(self, string: str):
        """Parse a string into a region tuple.

        :param str string: string to parse
        :return: region tuple

        """
        try:
            x, y, z = map(int, string.split(','))
        except ValueError:
            raise ArgumentTypeError(
                f'Expected x,y,z but got {string}')
        return x, y, z

    def create_parser(self):
        gps = self.client.gauges.minimap.gps

        parser = super().create_parser()

        parser.add_argument(
            '--map-name',
            default='gielinor',
            help='name for map when saved.',
        )

        parser.add_argument(
            '--start-xy',
            required=gps.DEFAULT_METHOD != gps.GRID_INFO_MATCH,
            type=int_or_str,
            nargs=2,
            default=(3221, 3218),  # steps of lumby
            help='Specify starting coordinates by name or value',
        )

        parser.add_argument(
            '--end-xy',
            nargs=2,
            type=int,
            default=(3164, 3486),  # grand exchange
            help='Specify end coordinates'
        )

        parser.add_argument(
            '--checkpoints',
            nargs='+',
            help='Calculate the path with checkpoints.'
        )

        parser.add_argument(
            '--cycle-regions',
            type=self.region,
            nargs='+',
            default=[],
            help='When cycling through entities, only cycle entities that are'
                 'in these regions. This is useful when you are modifying '
                 'entity tile bases, but there are hundreds of unrelated '
                 'entities in other regions as part of the global map.'
        )

        return parser

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
            return self.client.gauges.minimap.gps.get_coordinates()

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
            v2 = self.client.gauges.minimap.gps.get_coordinates()

        # join the last node to the incoming one
        v1 = self.node_history[-1]
        self.graph[v1] = v2
        # cache distances for later use
        d = self.client.gauges.minimap.distance_between(v1, v2)
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
        try:
            del self.graph[node]
        except KeyError:
            pass
        # remove node from distances dict
        try:
            neighbours = self.distances.pop(node)
        except KeyError:
            neighbours = []

        for neighbour in neighbours:
            try:
                self.distances[neighbour].pop(node)
            except KeyError:
                pass

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
            v = self.client.gauges.minimap.gps.get_coordinates()

        # get the label settings
        try:
            colour = tuple(
                [
                    int(self.label_entries.get(c).get()) % 256
                    for c in self.channels
                ]
            )
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

        mm = self.client.gauges.minimap
        x, y, z = v
        gx, gy, gz = mm.gps.get_coordinates()
        key = (
            int((x - gx) * mm.tile_size),
            int((y - gy) * mm.tile_size),
        )

        entity = self.client.game_screen.create_game_entity(
            label, label, key, self.client, self.client
        )
        entity.set_global_coordinates(*v)

        print(entity, v)
        self.entities.append(entity)
        self.entity_mapping[label] = z

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

        node = self.client.gauges.minimap.gps.get_coordinates()
        if self.cursor_mode:
            node = self.cursor

        if node not in self.graph:
            print(f'Not a node: {node}')
            return

        self.node_history = [node]

    @wait_lock
    def set_to_cursor(self):
        """Set the current coordinates to wherever the cursor is"""

        if not self.cursor:
            print('No cursor')
        self.client.gauges.minimap.gps.set_coordinates(*self.cursor)
        print(f'Reset to {self.cursor}')
        self.node_history = [self.cursor]

    # @wait_lock
    # def set_grid_info_offset(self):
    #     """Set the grid info offset to the current coordinates."""
    #
    #     gx, gy, gz = self.client.gauges.grid_info.tile.coordinates()
    #     if gx == -1 or gy == -1:
    #         self.client.logger.warning('Grid info not found.')
    #         return
    #
    #     x, y = self.client.gauges.minimap.gps.get_coordinates()
    #
    #     dx = gx - x
    #     dy = gy - y
    #
    #     self.grid_info_offset = dx, dy

    @wait_lock
    def move_cursor(self, dx, dy):
        """Move cursor object independent of the player."""

        x, y, z = self.cursor
        x += dx
        y += dy
        self.cursor = x, y, z

    @wait_lock
    def update_map_offset(self, string_var: tkinter.StringVar, index: int):
        try:
            value = float(string_var.get())
        except ValueError:
            return True

        self.offsets[index] = value
        map_ = self.client.gauges.minimap.gps.current_map
        if index:
            self.client.logger.info(f'Updated y offset to: {value:.3f}')
            map_.offset_y = value
        else:
            self.client.logger.info(f'Updated x offset to: {value:.3f}')
            map_.offset_x = value

    @wait_lock
    def cycle_labels_mode(self):
        mode = self.labels_mode.pop(0)
        self.labels_mode.append(mode)

    @wait_lock
    def reset_cursor(self):
        self.cursor = self.client.gauges.minimap.gps.get_coordinates()

    @wait_lock
    def toggle_label_display(self, name):
        self.label_settings[name] = not self.label_settings[name]

    def _add_label_display(self, name, enabled=True):
        self.label_settings[name] = enabled

        # TODO: figure out why checkboxes are not checked on creation
        # var = tkinter.BooleanVar(value=enabled)
        # checkbox = tkinter.Checkbutton(
        #     self.gui_frames['checkboxes'], text=name,
        #     variable=var, onvalue=True, offvalue=False,
        #     command=lambda label=name: self.toggle_label_display(label)
        # )
        # checkbox.pack()
        # self.label_entries['label_config'][name] = checkbox

    def _remove_label_display(self, name):
        ...
        # del self.label_settings[name]
        # widget = self.label_entries['label_config'].pop(name)
        # widget.destroy()

    def toggle_update(self):
        """Turn minimap updating off, useful if you need to keep the map
        maker running, but you're temporarily going off map."""
        self.update_minimap = not self.update_minimap
        self.client.gauges.logger.info(
            f'Minimap update: {self.update_minimap}')

    def screenshot_minimap(self):
        """Captures the current image in minimap, crops out the circular border
        and saves to a unique location. Useful for re-mapping existing areas
        that are out-dated or created new maps for areas that don't exist
        on doogle maps."""

        # mask must be bgra
        mask = cv2.cvtColor(
            self.client.gauges.minimap.mask,
            cv2.COLOR_GRAY2BGRA,
        )

        lock.acquire()
        self.client.update()
        img = self.client.get_img_at(
            self.client.gauges.minimap.get_bbox(), mode=self.client.BGRA)
        lock.release()

        self.client.gauges.minimap.logger.debug(
            f'img: {img.shape}, mask: {mask.shape}')

        img = cv2.bitwise_and(img, mask)

        self.client.gauges.minimap.logger.info(f'Saving: {self.mm_img_path}')
        cv2.imwrite(self.mm_img_path, img)
        self.mm_img_idx += 1

    def open_map_manager(self):

        if self.map_manager:
            self.client.logger.warning('Map manager already open.')
            return

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

        # global offsets
        x_description = tkinter.Label(root, text='GlobalX')
        x_description.config(font=('helvetica', 14))
        x_description.pack()
        x_stringvar = tkinter.StringVar()
        x_entry = tkinter.Entry(root, textvariable=x_stringvar)
        x_entry.insert(tkinter.END, str(self.offsets[0]))
        x_stringvar.trace_add(
            'write', lambda *_: self.update_map_offset(x_stringvar, 0))
        x_entry.pack()
        y_description = tkinter.Label(root, text='GlobalY')
        y_description.config(font=('helvetica', 14))
        y_description.pack()
        y_stringvar = tkinter.StringVar()
        y_entry = tkinter.Entry(root, textvariable=y_stringvar)
        y_entry.insert(tkinter.END, str(self.offsets[1]))
        y_stringvar.trace_add(
            'write', lambda *_: self.update_map_offset(y_stringvar, 1))
        y_entry.pack()

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
        cursor.bind('<Key-Up>', lambda e: self.move_cursor(0, 1))
        cursor.bind('<Key-Down>', lambda e: self.move_cursor(0, -1))
        cursor.bind('<Button-1>', lambda e: cursor.focus_set())
        cursor.pack()

        # reset node history to current node
        reset_button = tkinter.Button(
            text='Reset Node History', command=self.reset_node_history)
        reset_button.pack()

        # rest current coordinates to wherever the curse is
        set_to_cursor_button = tkinter.Button(
            text='Set to Cursor', command=self.set_to_cursor)
        set_to_cursor_button.pack()

        # set_grid_info_offset_button = tkinter.Button(
        #     text='Set Grid Info Offset',
        #     command=self.set_grid_info_offset
        # )
        # set_grid_info_offset_button.pack()

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

        img = self.client.gauges.minimap.gps.current_map.img_copy

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

    def draw_label(self, node, colour, x_offset=0, y_offset=4, size=0.25, **_):

        gps = self.client.gauges.minimap.gps
        img = gps.current_map.img_copy

        labels_mode = self.labels_mode[0]
        label = self.label_backref.get(node)
        if label is None and labels_mode != 'labels only':
            label = str(node)

        x, y, z = node
        x1, y1 = gps.current_map.get_coordinate_pixel(*node)
        x2, y2 = gps.current_map.get_coordinate_pixel(x + 1, y - 1, z)

        # first draw the node bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)

        # now write coordinates / label next to node if there is one
        if label and not labels_mode == 'nodes_only':
            x = int((x1 + x2) / 2) + x_offset
            y = int((y1 + y2) / 2) + y_offset
            cv2.putText(
                img, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, colour,
                thickness=1
            )

    def draw_edges(self):
        """
        Draw the graph nodes with edges between them.
        """

        mm = self.client.gauges.minimap
        img = mm.gps.current_map.img_copy

        drawn = set()
        for node, neighbours in self.graph.items():

            x, y, z = node
            ax1, ay1 = mm.gps.current_map.get_coordinate_pixel(x, y, z)

            # # TODO: remove edgeless nodes
            # if not neighbours:
            #     # draw the node and label with no edges
            #     cv2.rectangle(
            #         img, (ax1, ay1), (ax2, ay2), self.BLUE, -1)

            for neighbour in neighbours:

                x, y, z = neighbour
                bx1, by1 = mm.gps.current_map.get_coordinate_pixel(x, y, z)

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

        mm = self.client.gauges.minimap
        img = mm.gps.current_map.img_copy

        if not self.client.args.show_map:
            return

        # draw the path first, so the graph nodes appear back-lit
        self.draw_path()
        self.draw_edges()

        # TODO: add cursor as a temporary label
        if self.cursor_mode:
            cx, cy, cz = self.cursor
            x1, y1 = mm.gps.current_map.get_coordinate_pixel(*self.cursor)
            x2, y2 = mm.gps.current_map.get_coordinate_pixel(cx + 1, cy - 1, cz)
            cv2.rectangle(
                img, (x1, y1), (x2, y2), self.BLUE, -1)

        # finally draw labels that are active
        self.draw_labels()

    @wait_lock
    def calculate_route(self, start, end):

        # TODO: remove checkpoints once we pass them
        path = self.client.gauges.minimap.gps.get_route(
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
        mm = self.client.gauges.minimap
        img = mm.gps.current_map.img_copy
        for node in self.path:
            x, y, z = node
            ax1, ay1, ax2, ay2 = mm.coordinates_to_pixel_bbox(x, y, z)

            # draw a slightly larger, golden box
            cv2.rectangle(
                img, (ax1-1, ay1-1), (ax2+1, ay2+1), self.YELLOW, -1)

    def get_map_path(self):
        path = self.MAP_PATH.format(
            root=get_root(),
            name=self.args.map_name
        )
        return realpath(path)

    def save_map(self):

        graph = dict()
        for k, v in self.graph.items():
            graph[k] = v
        labels = dict()
        for k, v in self.labels.items():
            labels[k] = v

        # todo: data model
        data = MapData(
            name=self.args.map_name,
            graph=graph,
            labels=labels,
        )

        path = self.get_map_path()

        with open(path, 'wb') as f:
            pickle.dump(data.dict(), f)

        self.client.gauges.logger.info(f'Saved to: {path}')

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

        gps = self.client.gauges.minimap.gps

        path = self.get_map_path()
        if isfile(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = MapData(name=self.args.map_name).dict()
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

        try:
            offsets = data.pop('offsets')
        except KeyError:
            offsets = [Map.DEFAULT_OFFSET_X, Map.DEFAULT_OFFSET_Y]
        self.offsets = offsets

        gps.load_map(self.args.map_name)

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
                                size: <size float>,
                                width: <tile width int>,
                                height: <tile height int>,
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
        mm = self.client.gauges.minimap

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

    @wait_lock
    def cycle_selected_entity(self, direction: Literal[-1, 1]):

        self.client.logger.info('cycling selected')

        if not self.selected_entity:
            return

        if not self.entities:
            return

        index = self.entities.index(self.selected_entity)

        total_index = 0
        while True:
            total_index += 1
            if total_index > len(self.entities):
                self.client.logger.info('No entities to cycle through.')
                return
            index += direction
            index %= len(self.entities)
            if self.args.cycle_regions:
                entity = self.entities[index]
                x, y, z = entity.get_global_coordinates()
                x //= self.client.gauges.minimap.gps.TILES_PER_REGION
                y //= self.client.gauges.minimap.gps.TILES_PER_REGION
                if (x, y, z) not in self.args.cycle_regions:
                    continue
            break

        self.selected_entity.colour = self.selected_entity.DEFAULT_COLOUR
        self.selected_entity = self.entities[index]
        self.selected_entity.colour = self.HIGHLIGHT_ENTITY_COLOUR
        self.client.logger.info(
            f'Selected: {self.selected_entity.name} ({index})'
        )

    @wait_lock
    def adjust_entity_tile_size(self, attribute, delta):
        current = getattr(self.selected_entity, attribute)
        setattr(self.selected_entity, attribute, current + delta)

        self.client.logger.info(f'Adjusted {attribute} by {delta}')

        for label, data in self.labels.items():
            if label != self.selected_entity.name:
                continue

            for node, node_data in data['nodes'].items():
                if node != self.selected_entity.get_global_coordinates():
                    continue

                node_data[attribute.replace('tile_', '')] = current + delta
                break

    @wait_lock
    def toggle_cam_drag(self):
        player = self.client.game_screen.player

        player.ADJUST_FOR_DRAG = not player.ADJUST_FOR_DRAG

    @wait_lock
    def toggle_gps_match(self):
        """Toggle the method used to do gps matching."""

        gps = self.client.gauges.minimap.gps
        if gps.DEFAULT_METHOD == gps.FEATURE_MATCH:
            gps.DEFAULT_METHOD = gps.TEMPLATE_MATCH
        else:
            gps.DEFAULT_METHOD = gps.FEATURE_MATCH

    def basic_hotkeys(self):
        keyboard.add_hotkey('0', self.toggle_update)
        keyboard.add_hotkey('1', self.open_map_manager)
        keyboard.add_hotkey('5', self.screenshot_minimap)

        keyboard.add_hotkey('tab', partial(self.cycle_selected_entity, 1))
        keyboard.add_hotkey('shift+tab', partial(self.cycle_selected_entity, -1))
        keyboard.add_hotkey(
            'a', lambda: self.adjust_entity_tile_size('tile_width', -1))
        keyboard.add_hotkey(
            'd', lambda: self.adjust_entity_tile_size('tile_width', 1))
        keyboard.add_hotkey(
            'w', lambda: self.adjust_entity_tile_size('tile_height', -1))
        keyboard.add_hotkey(
            's', lambda: self.adjust_entity_tile_size('tile_height', 1))

        keyboard.add_hotkey('*', self.toggle_cam_drag)

        keyboard.add_hotkey('/', self.toggle_gps_match)

    def vision_setup(self):
        """Set up the gps if we're using feature or template matching."""

        # mm = self.client.gauges.minimap
        # mm.setup_thresolds('npc', 'npc-slayer', 'npc-tag', 'player')
        self.load_map()

        start = tuple(self.args.start_xy)
        end = tuple(self.args.end_xy)
        self.node_history = [start]

        self.client.gauges.minimap.gps.set_coordinates(*start)
        # we can only draw a route if we're loading an existing map
        # (where it is assumed at least some nodes on the graph have been
        # created)
        if self.args.load_map:
            self.calculate_route(start, end)

        # self.basic_hotkeys()

        self.entities = list()
        if self.args.entities:
            for label, data in self.labels.items():
                entities = self._setup_game_entity(label, count=float('inf'))
                self.entities.extend(entities)
        self.entity_mapping = {
            e.id: e.get_global_coordinates()[2]
            for e in self.entities
        }
        if self.entities:
            self.selected_entity = self.entities[0]

    def ocr_setup(self):
        """"""

        # load map
        self.load_map()
        self.basic_hotkeys()

        self.entities = list()
        for label, data in self.labels.items():
            entities = self._setup_game_entity(label, count=float('inf'))
            self.entities.extend(entities)
        self.entity_mapping = {
            e.id: e.get_global_coordinates()[2]
            for e in self.entities
        }
        if self.entities:
            self.selected_entity = self.entities[0]

    def setup(self):
        """"""

        self.args = self.parse_args()
        self.labels_mode = [True, False, 'labels only', 'nodes_only']
        self.distances = defaultdict(dict)

        gps = self.client.gauges.minimap.gps
        if gps.DEFAULT_METHOD in {gps.FEATURE_MATCH, gps.TEMPLATE_MATCH}:
            self.vision_setup()
        elif gps.DEFAULT_METHOD == gps.GRID_INFO_MATCH:
            self.ocr_setup()

    def vision_update(self):
        """Update the application if we're using the vision system, i.e.
        feature or template matching, for locating the player on the map."""
        mm = self.client.gauges.minimap
        gps = mm.gps

        if self.update_minimap:
            self.client.gauges.update()

        # recalculate route based on current position
        end = tuple(self.args.end_xy)

        # we can only draw a route if we're loading an existing map
        # (where it is assumed at least some nodes on the graph have been
        # created)
        if self.args.load_map:
            x, y, z = gps.get_coordinates()
            self.calculate_route((x, y), end)

        try:
            self.draw_nodes()
        except RuntimeError:
            self.msg.append("Draw failure")

        if self.args.entities:
            self._update_game_entities(
                *self.entities,
                mapping=self.entity_mapping
            )
            self.client.game_screen.player.update()

        dx, dy = self.offsets
        # self.msg.append(f'global: {dx:.2f}, {dy:.2f}')
        self.msg.append(f'map: {mm.gps.current_map.offset_x:.2f},'
                        f' {mm.gps.current_map.offset_y:.2f}')

        x, y, z = gps.get_coordinates(real=True)
        confidence = gps.confidence or 0
        self.msg.append(
            f'method: {gps.DEFAULT_METHOD} '
            f'current: {x:.2f},{y:.2f} ({confidence:.2f})'
        )

    def ocr_update(self):
        """Update the application if we're using the OCR system for locating
        the player on the map."""

        gps = self.client.gauges.minimap.gps
        gi = self.client.gauges.grid_info
        g = self.client.gauges

        self.client.game_screen.update()
        g.update()
        x, y, z = gi.tile.coordinates()
        if (x, y, z) == (-1, -1, -1):
            self.msg.append('Grid info not found.')
            return

        if not self.node_history:
            self.node_history.append((x, y, z))

        self._update_game_entities(
            *self.entities,
            mapping=self.entity_mapping,
            map_by_z=True,
        )
        self.client.game_screen.player.update()

        self.draw_nodes()

        # self.gui.new_frame.emit(gps.current_map.img_copy)

    def update(self):
        """"""

        gps = self.client.gauges.minimap.gps
        if gps.DEFAULT_METHOD in {gps.FEATURE_MATCH, gps.TEMPLATE_MATCH}:
            self.vision_update()
        elif gps.DEFAULT_METHOD == gps.GRID_INFO_MATCH:
            self.ocr_update()

    def action(self):
        """"""

        gps = self.client.gauges.minimap.gps
        # self.msg.append(str(self.cursor_mode))
        # self.msg.append(str(self.path))
        self.msg.append(str(gps.get_coordinates()))
        self.msg.append(str(self.node_history))


def main():
    # q_app = QApplication([])
    app = MapMaker()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
