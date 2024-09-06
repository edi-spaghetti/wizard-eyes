import cv2
import numpy

from .constants import NUMBERS
from ..game_objects import GameObject


class Orb(GameObject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value: int = 999
        self.located = False

    def get_red_threshold(self):
        hsv = self.client.get_img_at(self.get_bbox(), self.client.HSV)
        mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))

        return mask

    def update_value(self):
        """Checks for numerals in orb widget and sets a numerical value."""

        img = self.img
        _, img = cv2.threshold(img, 95, 255, cv2.THRESH_BINARY)

        # if image is black it's because very low numbers are out of threshold
        if not img.any():
            img = self.get_red_threshold()

        results = []
        for number, template in NUMBERS.items():
            matches = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

            _, mx = numpy.where(matches >= 0.99)
            for x in mx:
                results.append((number, x))

        results.sort(key=lambda n: n[1])
        stringified = map(lambda n: str(n[0]), results)
        number_str = ''.join(stringified)

        try:
            self.value = int(number_str)
        except ValueError:
            # if we still can't read any numbers just return a low default
            self.value = -1

    def update(self):
        super().update()
        self.update_value()

    def draw(self):
        states = {'*state', 'orb_state'}
        if self.client.args.show.intersection(states):
            x1, y1, x2, y2 = self.get_bbox()
            x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

            # TODO: manage this as configuration if we need to add more
            y_display_offset = 5

            cv2.putText(
                self.client.original_img, str(self.value),
                # convert relative to client image so we can draw
                (x1, y2 + y_display_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                self.colour, thickness=1
            )



class PrayerOrb(Orb):

    def __init__(self, client, parent, *args, **kwargs):
        super().__init__(client, parent, *args, **kwargs)


class HitPointsOrb(Orb):

    def __init__(self, client, parent, *args, **kwargs):
        super().__init__(client, parent, *args, **kwargs)


class RunEnergyOrb(Orb):

        def __init__(self, client, parent, *args, **kwargs):
            super().__init__(client, parent, *args, **kwargs)


class SpecialAttackOrb(Orb):

        def __init__(self, client, parent, *args, **kwargs):
            super().__init__(client, parent, *args, **kwargs)
