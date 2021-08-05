import client
from game_objects import GameObject


def main():
    c = client.Client('RuneLite')


    bank_aoi = c.screen.gen_bbox()
    bank = GameObject(c, c)
    bank.set_aoi(*bank_aoi)

    range_aoi = c.screen.gen_bbox()
    range_ = GameObject


if __name__ == '__main__':
    main()