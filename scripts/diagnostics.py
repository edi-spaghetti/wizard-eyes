"""Diagnostic bot: logs what the engine detects, never clicks.

Start-up (handled by Application): window found / launched via
``--username``, login, RuneLite profile + layout checks, camera fixed
(north, top-down, zoom). This script then logs, every ``--interval``
seconds and whenever something changes:

- login state (login screen / lobby / in game) and game layout
- camera state (north, top-down, zoom)
- side panel tabs: which are located and their state (e.g. selected)
- bank open/closed, mouse-over text, player tile, GPS coordinates
- timings for each update step

Run::

    venv\\Scripts\\python scripts\\diagnostics.py --username <name> --run-time 120
"""
import time
from typing import Callable, Dict

from wizard_eyes.application import Application
from wizard_eyes import camera, login


class Diagnostics(Application):

    def create_parser(self):
        parser = super().create_parser()
        parser.add_argument('--interval', type=float, default=5,
                            help='Seconds between full status reports.')
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_report = 0.0
        self.last_values: Dict[str, str] = {}
        self.timings: Dict[str, float] = {}

    def setup(self):
        log = self.client.logger
        log.info(f'Diagnostics: window "{self.client.title}", '
                 f'bbox {self.client.get_bbox()}')
        cam = getattr(self, 'camera_state', None)
        log.info(f'Diagnostics: camera after start-up checks: '
                 f'{cam.summary() if cam else "not checked"}')
        # make sure every tab widget tries to find itself
        for widget in self.client.tabs.widgets:
            widget.auto_locate = True

    def timed(self, name: str, func: Callable):
        t = time.perf_counter()
        try:
            func()
        except Exception as e:
            self.client.logger.warning(f'Diagnostics: {name} failed: {e!r}')
        self.timings[name] = (time.perf_counter() - t) * 1000

    def update(self):
        c = self.client
        self.timed('gauges', c.gauges.update)
        self.timed('game_screen', c.game_screen.update)
        self.timed('tabs', c.tabs.update)
        self.timed('bank', c.bank.update)
        self.timed('mouse_options', c.mouse_options.update)

    def safe(self, func: Callable, default='?'):
        try:
            return func()
        except Exception as e:
            return f'error: {e!r}'

    def collect(self) -> Dict[str, str]:
        c = self.client
        img = c.img
        gray = img if img.ndim == 2 else __import__('cv2').cvtColor(
            img, __import__('cv2').COLOR_BGRA2GRAY)
        values = {
            'login state': self.safe(lambda: login.classify(gray)[0]),
            'layout': self.safe(lambda: login.LAYOUT_NAMES[
                login.detect_layout(gray)]),
            'camera': self.safe(lambda: camera.camera_state(
                c._win_handle).summary()),
            'bank open': self.safe(lambda: c.bank.open),
            'mouse text': self.safe(lambda: c.mouse_options.state),
            'player tile': self.safe(lambda: c.game_screen.player.tile_bbox()),
            'gps': self.safe(lambda: c.gauges.minimap.gps.get_coordinates()),
        }
        for w in c.tabs.widgets:
            values[f'tab {w.name}'] = self.safe(
                lambda w=w: f'{w.state or "-"}' + ('' if w.located
                                                    else ' (not located)'))
        return values

    def action(self):
        """Report only; never clicks or presses anything."""
        now = time.time()
        values = self.collect()
        changed = {k: v for k, v in values.items()
                   if self.last_values.get(k) != str(v)}
        log = self.client.logger
        if now - self.last_report >= self.args.interval:
            self.last_report = now
            log.info('Diagnostics report:')
            for k, v in values.items():
                log.info(f'  {k:<22} {v}')
            log.info('  timings (ms)          ' + ', '.join(
                f'{k} {v:.0f}' for k, v in self.timings.items()))
        elif changed:
            for k, v in changed.items():
                log.info(f'Diagnostics: {k} -> {v}')
        self.last_values = {k: str(v) for k, v in values.items()}
        self.msg.append('diagnostics running')


def main():
    app = Diagnostics()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
