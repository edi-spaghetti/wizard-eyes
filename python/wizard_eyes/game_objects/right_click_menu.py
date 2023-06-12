from typing import List

import cv2
import numpy

from .game_objects import GameObject

from PIL import Image

class RightClickMenu(GameObject):

    ITEM_HEIGHT = 15
    ITEM_LR_MARGIN = 2
    OCR_READ_ITEMS = False

    PATH_TEMPLATE = '{root}/data/game_screen/menus/right_click/{name}.npy'
    MAX_DX = 300
    MAX_DY = 150

    BG_COLOUR = (71, 84, 93, 255)
    """Background colour of all menus. It doesn't seem to get
    cluster-fluttered, so it should be OK to keep as a constant."""

    DEFAULT_COLOUR = (255, 0, 255, 255)

    def __init__(self, client: 'Client', parent: GameObject, x, y,
                 *args, **kwargs):
        super().__init__(client, parent, *args, **kwargs)
        self.located = False
        self.x = x
        self.y = y
        self.items: List[MenuItem] = []
        self.load_templates(['top_left', 'bottom_right'])

        # invert the templates to get a better match
        self.templates['top_left'] = cv2.bitwise_not(
            self.templates['top_left'])
        self.templates['bottom_right'] = cv2.bitwise_not(
            self.templates['bottom_right'])

    @property
    def width(self):
        if self.located:
            x1, _, x2, _ = self.get_bbox()
            return x2 - x1 + 1
        else:
            return super().width

    @property
    def height(self):
        if self.located:
            _, y1, _, y2 = self.get_bbox()
            return y2 - y1 + 1
        else:
            return super().height

    def set_parent(self, new_parent: GameObject):
        self.parent = new_parent

    def locate(self):
        if self.x == -1 or self.y == -1:
            return False

        # find the top left, should be close to the x, y coords
        cx1, cy1, cx2, cy2 = self.client.get_bbox()
        x1 = max(self.x - self.MAX_DX, cx1)
        y1 = max(self.y - self.MAX_DY, cy1)
        x2 = min(self.x + self.MAX_DX, cx2)
        y2 = cy2  # could be a very long list to bottom of client

        img = self.client.get_img_at((x1, y1, x2, y2))
        img = cv2.bitwise_not(img)

        matches = cv2.matchTemplate(
            img, self.templates['top_left'], cv2.TM_SQDIFF_NORMED)
        my, mx = numpy.where(matches <= 0.01)
        if len(my) > 1:
            self.logger.warning('More than one TL bounding box detected')

        for x01, y01 in zip(mx, my):

            # find bottom right relative to top left
            a = x1 + x01
            b = y1 + y01
            img2 = self.client.get_img_at(
                (a, b, a + self.MAX_DX, self.MAX_DY), mode=self.client.BGRA)
            mask = cv2.inRange(img2, self.BG_COLOUR, self.BG_COLOUR)

            contours = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            rx1, ry1, rx2, ry2 = 0, 0, 0, 0
            # there should only be one contour detected, and it should be in
            # top left.
            if len(contours) > 1:
                self.logger.warning('More than one RC bounding box detected')

            for contour in contours:
                rx, ry, rxx, ryy = cv2.boundingRect(contour)
                if rx != 0 or ry != 0:
                    continue
                rx1 = rx
                ry1 = ry
                rx2 = rxx
                ry2 = ryy

            if rx1 == 0 and ry1 == 0 and rx2 == 0 and ry2 == 0:
                continue

            bbox = (x1 + x01, y1 + y01, x1 + x01 + rx2, y1 + y01 + ry2)
            self.set_aoi(*bbox)
            self.located = True

            self.create_items()

            return True

    def create_items(self):
        h, _ = self.templates['top_left'].shape
        items_height = self.height - h

        x1, y1, x2, y2 = self.get_bbox()

        num_items = items_height // self.ITEM_HEIGHT
        for i in range(num_items):
            ix1 = x1 + self.ITEM_LR_MARGIN
            iy1 = h + y1 + i * self.ITEM_HEIGHT
            ix2 = x2 - self.ITEM_LR_MARGIN
            iy2 = h + y1 + (i + 1) * self.ITEM_HEIGHT

            item = MenuItem(self.client, self, i)
            item.set_aoi(ix1, iy1, ix2, iy2)
            self.items.append(item)

    def reset(self):
        self.parent.context_menu = None
        self.parent = self.client
        self.x = -1
        self.y = -1
        self.clear_bbox()
        self.items = []
        self.located = False

    def update(self):

        if not self.located and self.x != -1 and self.y != -1:
            result = self.locate()
            if not result:
                return

        # moving the mouse outside context box destroys it
        if (not self.is_inside(*self.client.screen.mouse_xy) and
                self.client.screen.mouse_xy != (self.x, self.y)):
            self.reset()
            return

        super().update()

        for item in self.items:
            item.update()


class MenuItem(GameObject):

    DEFAULT_COLOUR = 210, 110, 180, 255

    def __init__(self, client: 'Client', parent: RightClickMenu, idx: int,
                 *args, **kwargs):
        super().__init__(client, client, *args, **kwargs)
        self.parent = parent
        self.idx = idx
        self.value = None
        self.value_changed_at = -float('inf')

    def click(self, *args, **kwargs):
        super().click(*args, **kwargs)
        # clicking a context menu item destroys the context menu
        self.parent.reset()

    def draw(self):
        bboxes = {'*bbox', 'rc-bbox', f'rc-{self.idx}-bbox'}
        if self.client.args.show.intersection(bboxes):
            self.draw_bbox()

        states = {'*state', 'rc-state', f'rc-{self.idx}-state'}
        if self.client.args.show.intersection(states):
            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            condition = (
                self.client.is_inside(x1, y1) and
                self.client.is_inside(x2, y2)
            )
            if condition:
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(
                    x1, y1, x2, y2, draw=True)

                # draw the state just under the bbox
                cv2.putText(
                    self.client.original_img, str(self.value),
                    (x1, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, self.colour,
                    thickness=1
                )

    def update(self):
        super().update()

        if self.parent.OCR_READ_ITEMS:
            img = Image.fromarray(self.img)
            self.client.ocr.SetImage(img)
            value = self.client.ocr.GetUTF8Text()
            value = value.strip().replace('\n', '').replace('\r', '').lower()

            if self.value != value:
                self.value_changed_at = self.client.time

            self.value = value
