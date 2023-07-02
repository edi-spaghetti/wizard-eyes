from wizard_eyes.application import Application


class TestChatOCR(Application):
    """Demonstrates the OCR feature on the chat menu to read text displayed
    in the chat dialogue box."""

    def setup(self):
        """Ensure we have flags to draw text, and set chat to auto locate."""

        self.client.args.show = {'*bbox', '*text', 'mouse'}

        chat = self.client.chat.all
        chat.auto_locate = True

    def update(self):
        """Update the text in the chat menu."""

        self.client.chat.update()
        if self.client.chat.all.located:
            self.client.chat.all.interface.read_text()

    def action(self):
        """Ensure chat tab is open, so we can read the text there. Log any
        text we find to the console."""

        chat = self.client.chat.all
        at = self.client.chat.active_tab

        if at != chat:
            self._click_tab(chat)
            self.msg.append('Switched to chat tab.')
        else:
            text = str(chat.interface.text)
            text = text.replace('\n', ' | ').replace('\r', '')
            self.msg.append(text)


def main():
    app = TestChatOCR(msg_length=200)
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
