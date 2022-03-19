"""
Completes a slayer task.
Assumes player is already geared up.
No banking.
Prayer flick enemy hits and piety for own attacks
"""

from client import Application


class Slayer(Application):

    def setup(self):
        """"""
        print('setting up')

    def update(self):
        """"""
        self.msg.append('update')

    def action(self):
        """"""
        self.msg.append('action')


def main():

    app = Slayer()
    app.run()


if __name__ == '__main__':
    main()
