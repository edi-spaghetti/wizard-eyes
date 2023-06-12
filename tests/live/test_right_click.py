from random import uniform

from wizard_eyes.application import Application
from wizard_eyes.game_objects.game_objects import GameObject


class TestRightClickApp(Application):
    """Demonstrates the right click menu with OCR."""

    def __init__(self, *args, **kwargs):
        """Set up some timers to click and click away, as well as a fake object
        we are going to click."""
        super().__init__(*args, **kwargs)
        self.click_timer = -float('inf')
        self.mouse_away_timer = -float('inf')
        self.object = GameObject(self.client, self.client)

    def setup(self):
        """Force args to show the bounding boxes and menu item states.
        We want OCR to be enabled for this right click menu, so turn it on.
        Set the fake object's bbox to be the center third of the screen.
        This is where we'll click to create right click menus."""

        self.client.args.show = {'mouse', '*bbox', '*state'}

        self.client.right_click_menu.OCR_READ_ITEMS = True

        x1, y1, x2, y2 = self.client.get_bbox()
        self.object.set_aoi(
            int(x1 + self.client.width / 3), int(y1 + self.client.height / 3),
            int(x2 - self.client.width / 3), int(y2 - self.client.height / 3))

    def update(self):
        """Just update the object, if a right click menu has been set up
        updates will automatically propagate down."""

        self.object.update()

    def action(self):
        """Click on the center third if it's time. Click away if it's time.
        This way we can see the right click menu in a variety of positions.
        The menu item text is logged to the console."""

        if self.client.time > self.click_timer:
            self.click_timer = self.client.time + uniform(3, 4)
            self.mouse_away_timer = self.client.time + uniform(2, 3)

            x, y = self.object.right_click(pause_before_click=True)
            self.object.set_context_menu(x, y)

        elif self.client.time > self.mouse_away_timer:
            self.mouse_away_timer = self.click_timer
            self.client.screen.mouse_away_object(self.client.right_click_menu)

        message = ', '.join([i.value for i in self.client.right_click_menu.items])
        self.msg.append(message)


def main():
    app = TestRightClickApp(msg_length=200)
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
