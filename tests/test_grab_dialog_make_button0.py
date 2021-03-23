import time
from os.path import dirname, join

import client
from screen_tools import Screen


if __name__ == '__main__':
    c = client.Client('RuneLite')
    c.dialog.add_make('lantadyme')
    s = Screen()
    c.set_rect()
    c._client.activate()

    t = time.time()
    img = s.grab_screen(*c.dialog.makes[0].get_bbox())
    path = join(dirname(__file__), '..', 'data', 'test.png')
    s.save_img(img, path)
    t = time.time() - t

    print(f'Got screen in {round(t, 2)} seconds')
