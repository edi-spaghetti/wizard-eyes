from os.path import exists

import cv2
import numpy

from ...file_path_utils import get_root
from ..game_objects import GameObject
from .minimap import MiniMap
from .orbs import PrayerOrb, HitPointsOrb, RunEnergyOrb, SpecialAttackOrb
from .xp_tracker import XPTracker
from .grid_info import GridInfo


class OrbsContainer:

    def __init__(self, client, parent):
        self.prayer: PrayerOrb = PrayerOrb(client, parent)
        self.hitpoints: HitPointsOrb = HitPointsOrb(client, parent)
        self.run_energy: RunEnergyOrb = RunEnergyOrb(client, parent)
        self.special_attack: SpecialAttackOrb = SpecialAttackOrb(
            client, parent
        )

        self.children = [
            self.prayer, self.hitpoints, self.run_energy, self.special_attack
        ]

    @property
    def located(self):
        return any(o.located for o in self.children)

    def update(self):
        for orb in self.children:
            if orb.located:
                orb.update()


class GaugesWidget(GameObject):
    """Collection of various measuring devices, usually on top right.

    More commonly referred to by the dominant widget, the minimap.

    """

    PATH_TEMPLATE = '{root}/data/gauges/{name}.npy'

    ALPHA_MAPPING = {
        (0, 0, 255, 255): 'minimap',
        (0, 255, 0, 255): 'orb.hitpoints',
        (0, 254, 0, 255): 'orb.prayer',
        (0, 253, 0, 255): 'orb.run_energy',
        (0, 252, 0, 255): 'orb.special_attack',
    }

    def __init__(self, client):
        self.minimap = MiniMap(client, self)
        self.xp_tracker = XPTracker(client, self)
        self.orb: OrbsContainer = OrbsContainer(client, self)
        self.grid_info: GridInfo = GridInfo(client, self)
        super().__init__(client, client)

        self.alpha = self.get_alpha()
        self.frame = self.load_templates(['frame'], cache=False).get('frame')
        self.frame_mask = self.load_masks(['frame'], cache=False).get('frame')

        self.located = False
        self.children = [
            self.grid_info, self.minimap, self.xp_tracker, self.orb
        ]

    def update(self):
        super().update()
        if not self.located:
            self.located = self.locate()
        self.grid_info.update()
        self.xp_tracker.update()
        if not self.located:
            return
        self.minimap.update()
        self.orb.update()

    def get_alpha(self):
        path = self.resolve_path(
            name='alpha', root=get_root()).replace('.npy', '.png')
        if exists(path):
            return cv2.imread(path)
        return numpy.uint8([[[0, 0, 0]]])

    def locate_sub_widgets_by_alpha(self):
        if not self.alpha.any():
            return

        unique = numpy.unique(
            self.alpha.reshape(-1, self.alpha.shape[2]), axis=0)

        for bgr in unique:
            if not bgr.any():
                continue

            bgra = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2BGRA)
            colour = tuple(int(c) for c in bgra.reshape(4))

            sub_widget_path = self.ALPHA_MAPPING.get(colour)
            if not sub_widget_path:
                continue

            widget = self
            for token in sub_widget_path.split('.'):
                widget = getattr(widget, token)

            widget.DEFAULT_COLOUR = colour

            # set up bounding box
            ay, ax = numpy.where(
                numpy.all(
                    self.alpha == colour[:3], axis=-1
                )
            )
            x1, y1, x2, y2 = min(ax), min(ay), max(ax), max(ay)
            x1, y1, x2, y2 = self.globalise(x1, y1, x2, y2)
            widget.set_aoi(x1, y1, x2, y2)
            widget.located = True

    def locate(self):
        """Locate the interface itself. This method can also be used to check
        if the interface is currently open."""

        if self.frame is None:
            return False

        mask = self.frame_mask
        matches = cv2.matchTemplate(
            # must be client img, because we don't know where
            # the widget is yet
            self.client.img,
            self.frame,
            cv2.TM_CCOEFF_NORMED,
            mask=mask
        )

        (my, mx) = numpy.where(matches >= self.match_threshold)
        if len(mx) > 1:
            self.logger.warning(
                f'Found {len(mx)} matches for {self}, '
                'assuming the first one is correct.')

        # assume interfaces are unique and we only get one match
        for y, x in zip(my, mx):
            h, w = self.frame.shape
            x1, y1, x2, y2 = self.client.globalise(x, y, x + w - 1, y + h - 1)
            self.set_aoi(x1, y1, x2, y2)
            self.locate_sub_widgets_by_alpha()

            # some area of game screen are 'black' and cause false positives
            img = self.img.copy()
            img[img == 3] = 0
            if img.any():
                return True

        return False
