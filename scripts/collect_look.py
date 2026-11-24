"""Open the collection box through a Grand Exchange banker and log what it
shows (nothing is collected; the box is left open)::

    venv\\Scripts\\python scripts\\collect_look.py --username NAME --run-time 60
"""
import time

from wizard_eyes.application import Application
from wizard_eyes.interactions import UseOption
from wizard_eyes.navigation import map_labels
from wizard_eyes.navigation.still_npc import StillNPC

STILL_SECONDS = 1.3
OPEN_SECONDS = 8.0      # the banker may be a few steps away


class CollectLook(Application):

    def setup(self):
        self.args = getattr(self, 'args', None) or self.parse_args()
        self.client.mouse_options.add_colours('white', 'cyan', 'yellow')
        self.entity = self.use = None
        self.pos, self.still_since, self.used_at = None, 0.0, 0.0

    def update(self):
        self.client.gauges.update()
        self.client.game_screen.update()
        self.client.mouse_options.update()
        self.client.collection_box.update()
        if self.entity is not None:
            self._update_game_entities(self.entity)

    def finish(self, message):
        self.client.logger.info(f'COLLECT LOOK: {message}')
        self.stop()

    def action(self):
        cb = self.client.collection_box
        if cb.open:
            return self.finish([s for s in cb.slots
                                if s.status != 'empty'])
        if self.used_at:
            if time.time() - self.used_at > OPEN_SECONDS:
                self.finish('FAILED: the box did not open')
            return
        pos = tuple(int(i) for i in
                    self.client.gauges.minimap.gps.get_coordinates())
        if pos != self.pos:
            self.pos, self.still_since = pos, time.time()
        if pos[0] == -1 or time.time() - self.still_since < STILL_SECONDS:
            return
        if self.entity is None:
            found = map_labels.npc_map().find('ge_banker', pos, 10)
            if not found:
                return self.finish(f'FAILED: no banker near {pos}')
            self.entity = StillNPC(self, found[0])
            self.use = UseOption(self, self.entity, 'Collect', ['Banker'])
            return
        result = self.use.step()
        if result is False:
            self.finish('FAILED: no "Collect" on the banker')
        elif result:
            self.used_at = time.time()


def main():
    app = CollectLook()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
