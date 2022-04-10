
import cv2
import numpy

from ..game_objects import GameObject


class XPTracker(GameObject):

    PATH_TEMPLATE = '{root}/data/xp/{name}.npy'
    XP_DROP_SPEED = 60

    def __init__(self, client, parent, *args, **kwargs):
        super().__init__(client, parent, *args,
                         config_path='minimap.xp_tracker',
                         container_name='minimap', **kwargs)
        self._xp_drops = list()
        self._xp_drop_locations = list()

    def find_xp_drops(self, *skills, tick=None, less_than=None):

        drops = filter(lambda nt: nt[0] in skills, self._xp_drops)
        if tick is not None:
            drops = filter(lambda nt: nt[1] == tick, drops)
        if less_than is not None:
            drops = filter(lambda nt: nt[1] < less_than, drops)

        return list(drops)

    def show_xp(self):
        if 'xp' in self.client.args.show:

            px1, py1, _, py2 = self.get_bbox()
            for x, y, w, h in self._xp_drop_locations:

                x1 = x + px1
                y1 = y + py1
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)

    def update(self):
        super().update()

        self._xp_drops = list()
        self._xp_drop_locations = list()
        threshold = 0.99
        px1, py1, _, py2 = self.get_bbox()
        for template_name in self.templates:
            template = self.templates.get(template_name)
            mask = self.masks.get(template_name)

            matches = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED,
                mask=mask,
            )
            (my, mx) = numpy.where(matches >= threshold)

            h, w = template.shape
            for y, x in zip(my, mx):

                # add to records
                distance = (py2 - py1 + 1) - y  # from bottom of tracker
                estimated_ticks_ago = distance // self.XP_DROP_SPEED
                self._xp_drops.append((template_name, estimated_ticks_ago))
                self._xp_drop_locations.append((x, y, w, h))

        self.client.add_draw_call(self.show_xp)
