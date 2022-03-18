import time

from client import Client
c = Client('RuneLite')


if c.tabs.active_tab.name != 'prayer':
    c.tabs.prayer.click()
    time.sleep(1)
    c.update()

tmps = ['protect_melee', 'protect_melee_active']
c.tabs.prayer.interface.load_templates(tmps, cache=True)
c.tabs.prayer.interface.load_masks(tmps, cache=True)

tmps.reverse()
c.tabs.prayer.interface.locate_icons({'melee': {'templates': tmps}})

bbox = c.tabs.prayer.interface.icons['melee0'].get_bbox()
c.screen.show_img(c.screen.grab_screen(*bbox))
