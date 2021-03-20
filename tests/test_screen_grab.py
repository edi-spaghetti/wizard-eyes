from os.path import dirname, join

import client
from screen_tools import Screen


if __name__ == '__main__':
    c = client.Client('RuneLite')
    s = Screen()
    c.set_rect()
    c._client.activate()
    img = s.grab_screen(*c.get_bbox())
    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
