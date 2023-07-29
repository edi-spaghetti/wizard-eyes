import cv2
import numpy

from .constants import NUMBERS
from ..game_objects import GameObject


class Orb(GameObject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value: int = 999

    def get_red_threshold(self):

        x1, y1, x2, y2 = self.client.localise(*self.get_bbox())
        colour = self.client.original_img[y1+4:y2-4, x1+4:x1+25]
        hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))

        return mask

    def update_value(self):
        """Checks for numerals in orb widget and sets a numerical value."""

        img = self.img[5:-4, 4:25]
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


class PrayerOrb(Orb):

    def __init__(self, client, parent, *args, **kwargs):
        super().__init__(client, parent, *args,
                         config_path='minimap.prayer_orb',
                         container_name='minimap', **kwargs)


class HitPointsOrb(Orb):

    def __init__(self, client, parent, *args, **kwargs):
        super().__init__(client, parent, *args,
                         config_path='minimap.hitpoints_orb',
                         container_name='minimap', **kwargs)


class RunEnergyOrb(Orb):

        def __init__(self, client, parent, *args, **kwargs):
            super().__init__(client, parent, *args,
                            config_path='minimap.run_energy_orb',
                            container_name='minimap', **kwargs)


class SpecialAttackOrb(Orb):

        def __init__(self, client, parent, *args, **kwargs):
            super().__init__(client, parent, *args,
                            config_path='minimap.special_attack_orb',
                            container_name='minimap', **kwargs)
