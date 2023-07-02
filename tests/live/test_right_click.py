import mouse

from wizard_eyes.application import Application


class TestRightClickApp(Application):
    """Demonstrates the right click menu with OCR."""

    def init_right_click_menu(self):
        """Whenever user right clicks inside the client try to find a context
        menu there."""
        x, y = self.client.screen.mouse_xy
        if not self.client.is_inside(x, y):
            return

        menu = self.client.set_context_menu(x, y)
        menu.OCR_READ_ITEMS = True

    def on_clicked_menu_item(self):
        """On left-clicking any of the menu items in a right click menu,
        the menu is destroyed. This is usually handled by
        :func:`wizard_eyes.game_objects.right_click_menu.MenuItem.click`, but
        we need to handle it here because the user is clicking manually."""

        rc = self.client.right_click_menu
        x, y = self.client.screen.mouse_xy
        if not rc.located:
            return

        for i, item in enumerate(rc.items):
            if item.is_inside(x, y):
                rc.logger.info(f'Clicked item {i}')
                rc.reset()
                return

    def setup(self):
        """Force args to show the bounding boxes and menu item states. Set up
        the event loop to check for right click actions."""

        self.client.args.show = {'mouse', '*bbox', '*state'}
        mouse.on_right_click(self.init_right_click_menu)
        mouse.on_click(self.on_clicked_menu_item)

    def update(self):
        """Just update the object, if a right click menu has been set up
        updates will automatically propagate down."""

        self.client.right_click_menu.update()

    def action(self):
        """Log menu item text to the console."""

        rc = self.client.right_click_menu

        if rc.located:
            message = ' | '.join([i.value for i in rc.items])
            self.msg.append(message)
        else:
            self.msg.append('Right click somewhere to locate a menu')


def main():
    app = TestRightClickApp(msg_length=200)
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
