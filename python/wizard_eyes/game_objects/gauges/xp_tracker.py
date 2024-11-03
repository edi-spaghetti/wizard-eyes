import cv2
import numpy

from ...constants import REDA
from ...dynamic_menus.locatable import Locatable
from ..game_objects import GameObject


class XPTracker(Locatable):

    PATH_TEMPLATE = '{root}/data/gauges/xp/{name}.npy'

    ALPHA_MAPPING = {
        (0, 0, 255, 255): '_xp_area',
    }

    XP_DROP_SPEED = 60
    MATCH_THRESHOLD = 1
    USE_MASK = True
    INVERT = False
    MATCH_METHOD = cv2.TM_CCOEFF_NORMED
    DEFAULT_COLOUR = REDA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locatable_init()
        self.state = None

        self.elements = []
        self._xp_area = GameObject(self.client, self)
        self._xp_drops = list()
        self._xp_drop_locations = list()

    def track_skills(self, *skills):
        self.load_templates(skills)
        self.load_masks(skills)

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

            px1, py1, _, py2 = self._xp_area.get_bbox()
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
        prev_located = self.located
        result = super().update()
        self.state = result

        if not self.located:
            return

        # update subwidgets
        if not prev_located:
            for colour, bbox in self.iterate_alpha():
                element = getattr(self, self.ALPHA_MAPPING[colour])
                element.set_aoi(*bbox)
                element.colour = colour
                self.elements.append(element)
        for element in self.elements:
            element.update()

        self._xp_drops = list()
        self._xp_drop_locations = list()
        px1, py1, _, py2 = self._xp_area.get_bbox()
        for template_name in self.templates:
            template = self.templates.get(template_name)

            if self.INVERT:
                template = cv2.bitwise_not(template)

            if self.USE_MASK:
                mask = self.masks.get(template_name)
            else:
                mask = None

            img = self._xp_area.img
            if self.INVERT:
                img = cv2.bitwise_not(img)

            matches = cv2.matchTemplate(
                img, template, self.MATCH_METHOD,
                mask=mask,
            )

            if self.MATCH_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                (my, mx) = numpy.where(matches <= self.MATCH_THRESHOLD)
            elif self.MATCH_METHOD in [cv2.TM_CCORR, cv2.TM_CCORR_NORMED]:
                (my, mx) = numpy.where(matches >= self.MATCH_THRESHOLD)
            else:
                (my, mx) = numpy.where(matches == self.MATCH_THRESHOLD)

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
