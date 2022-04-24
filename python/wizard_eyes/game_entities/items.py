import cv2
import numpy

from .entity import GameEntity


class GroundItem(GameEntity):

    DEFAULT_COLOUR = (0, 0, 255, 255)

    def __repr__(self):
        return f'GroundItem<{self.state} {self.key}>'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hitbox = None
        self._hitbox_info = None

    def _draw_hitbox(self):
        if f'ground_item_hitbox' in self.client.args.show:

            try:
                x1, y1, x2, y2 = self._hitbox
            except (ValueError, TypeError):
                # hitbox has not been set, nothing to draw
                return

            x1, y1, x2, y2 = self.localise(x1, y1, x2, y2)
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            cv2.rectangle(
                self.client.original_img,
                (x1, y1), (x2, y2),
                self.colour, 1)

            cv2.putText(
                self.client.original_img,
                str(self.state), (x1, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.25, self.colour
            )

    def update(self, key=None):
        super().update(key=key)

        self._hitbox_info = None

        x1, y1, x2, y2 = self.get_bbox()
        if (not self.client.is_inside(x1, y1)
                or not self.client.is_inside(x2, y2)):
            return

        for name, template in self.templates.items():
            mask = self.masks.get(name)

            try:
                matches = cv2.matchTemplate(
                    self.img, template,
                    cv2.TM_CCOEFF_NORMED, mask=mask)
            except cv2.error:
                return

            # TODO: configurable threshold for ground items
            (my, mx) = numpy.where(matches >= 0.6)
            for y, x in zip(my, mx):

                if name != self.state:
                    self.state = name
                    self.state_changed_at = self.client.time

                h, w = template.shape
                cx1, cy1, _, _ = self.client.get_bbox()
                x1, y1, _, _ = self.get_bbox()

                hx1, hy1, hx2, hy2 = (
                    x1 + x,
                    y1 + y - 1,
                    x1 + x + w,
                    y1 + y + h - 1
                )
                self._hitbox = hx1, hy1, hx2, hy2

                # cache draw call for later
                self.client.add_draw_call(self._draw_hitbox)

                return True

        return False
