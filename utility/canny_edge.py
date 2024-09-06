import keyboard

from wizard_eyes.application import Application


class ExampleApp(Application):

    def setup(self):
        """"""

        mm = self.client.gauges.minimap
        self.client.gauges.minimap.create_map({(26, 57, 0), (28, 55, 0)})
        mm.set_coordinates(133, 86)

        lo = '_canny_lower'
        up = '_canny_upper'
        keyboard.add_hotkey('1', lambda: setattr(mm, lo, getattr(mm, lo) - 10))
        keyboard.add_hotkey('2', lambda: setattr(mm, lo, getattr(mm, lo) + 10))
        keyboard.add_hotkey('4', lambda: setattr(mm, up, getattr(mm, up) - 10))
        keyboard.add_hotkey('5', lambda: setattr(mm, up, getattr(mm, up) + 10))

        # keyboard.add_hotkey('4', lambda: mm.update_coordinate(-1, 0))
        # keyboard.add_hotkey('6', lambda: mm.update_coordinate(1, 0))
        # keyboard.add_hotkey('8', lambda: mm.update_coordinate(0, -1))
        # keyboard.add_hotkey('2', lambda: mm.update_coordinate(0, 1))

    def update(self):
        """"""

        mm = self.client.gauges.minimap
        old_xy = mm.get_coordinates()
        (x, y), _ = mm.update()

        self.msg.append(f'{old_xy} -> {x, y}')

        self.msg.append(f'{mm._canny_lower, mm._canny_upper}')

    def action(self):
        """"""


def main():
    app = ExampleApp()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
