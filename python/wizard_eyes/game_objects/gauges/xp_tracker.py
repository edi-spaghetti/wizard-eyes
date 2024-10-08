import cv2
import numpy

from ..game_objects import GameObject
from ...constants import REDA


class XPTracker(GameObject):

    PATH_TEMPLATE = '{root}/data/xp/{name}.npy'
    XP_DROP_SPEED = 60
    MATCH_THRESHOLD = 1
    USE_MASK = True
    INVERT = False
    MATCH_METHOD = cv2.TM_CCOEFF_NORMED
    DEFAULT_COLOUR = REDA

    def __init__(self, client, parent, *args, **kwargs):
        super().__init__(client, parent, *args, **kwargs)
        self._xp_drops = list()
        self._xp_drop_locations = list()
        self._img_colour = None
        self.updated_at = None
        self.located = False

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

    def find_xp_drops(self, *skills, tick=None, less_than=None):

        drops = filter(lambda nt: nt[0] in skills, self._xp_drops)
        if tick is not None:
            drops = filter(lambda nt: nt[1] == tick, drops)
        if less_than is not None:
            drops = filter(lambda nt: nt[1] < less_than, drops)

        drops = list(drops)
        return drops

    def show_xp(self):
        if 'xp' in self.client.args.show:

            px1, py1, _, py2 = self.get_bbox()
            for x, y, w, h in self._xp_drop_locations:

                x1 = x + px1
                y1 = y + py1
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(
                    x1, y1, x2, y2, draw=True)
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

    def update(self):
        if not self.located:
            return

        super().update()

        self._xp_drops = list()
        self._xp_drop_locations = list()
        px1, py1, _, py2 = self.get_bbox()
        for template_name in self.templates:
            template = self.templates.get(template_name)

            if self.INVERT:
                template = cv2.bitwise_not(template)

            if self.USE_MASK:
                mask = self.masks.get(template_name)
            else:
                mask = None

            # NOTE: we must use the colour image (?)
            img = self.img
            if self.INVERT:
                img = cv2.bitwise_not(img)

            matches = cv2.matchTemplate(
                img, template, self.MATCH_METHOD,
                mask=mask,
            )
            (my, mx) = numpy.where(matches >= self.MATCH_THRESHOLD)

            try:
                h, w, _ = template.shape
            except ValueError:
                h, w = template.shape

            for y, x in zip(my, mx):

                # add to records
                distance = (py2 - py1 + 1) - y  # from bottom of tracker
                estimated_ticks_ago = distance // self.XP_DROP_SPEED
                self._xp_drops.append((template_name, estimated_ticks_ago))
                self._xp_drop_locations.append((x, y, w, h))

        self.client.add_draw_call(self.show_xp)
