from wizard_eyes.application import Application


class Lumberjack(Application):

    HOSIDIUS_WILLOWS = {
        (26, 57, 0), (28, 55, 0)
    }

    def setup(self):
        """"""

        self.msg_length = 200
        mm = self.client.minimap.minimap
        mm.create_map(self.HOSIDIUS_WILLOWS)
        mm.set_coordinates(0, 0)
        mm.load_templates(['willow'])
        mm.load_masks(['willow'])

        self.willows = list()

        matches = mm.identify()
        for x, y in matches:

            key = x, y

            willow = self.client.game_screen.create_game_entity(
                'willow', 'willow', key, self.client, self.client
            )
            self.willows.append(willow)

    def update(self):
        """"""

        player = self.client.game_screen.player
        player.update()

        mm = self.client.minimap.minimap
        u, v = mm.run_gps()
        matches = mm.identify()

        cur_keys = {w.key for w in self.willows}

        cur_dx = 0
        cur_dy = 0
        if matches != cur_keys:
            margin = 10
            # TODO: use numpy
            for dy in (-margin, 0, margin):
                for dx in (-margin, 0, margin):
                    if dx == 0 and dy == 0:
                        continue
                    new_keys = {(x + dx, y + dy) for (x,y) in cur_keys}
                    if new_keys.issubset(matches):
                        cur_dx = dx
                        cur_dy = dy
            for willow in self.willows:
                x, y = willow.key
                willow.key = x + cur_dx, y + cur_dy

        for willow in self.willows:
            willow.update()

        # we need the negative of each delta because it's the world moving,
        # but we need to convert it to player movement
        dx, dy = - int(cur_dx // mm.tile_size), - int(cur_dy // mm.tile_size)
        self.msg.append(f'Delta: {cur_dx, cur_dy}')
        mm.update_coordinate(dx, dy)

        # if uv != u2:
        #     mm.colour = (255, 0, 0, 255)
        #     mm.get_local_zone(*uv, original=True)
        #     mm.colour = mm.DEFAULT_COLOUR

        # v1, w1, x1, y1, z1 = self.client.minimap.minimap.run_gps()
        # self.client.minimap.minimap.identify()
        # self.client.game_screen.player.update()
        #
        # v, w, x, y, z = self.client.minimap.minimap.get_coordinates()
        #
        # dv, dw = self.client.minimap.minimap.subtract_coordinates(
        #     (v1, w1, x1, y1, z1), (v, w, x, y, z)
        # )
        # self.msg.append(f'Delta: {dv,  dw}')
        # if abs(dv) <= 4 and abs(dw) <= 4:
        #     self.client.minimap.minimap.set_coordinates(v1, w1, x1, y1, z1)

    def action(self):
        """"""

        self.msg.append(
            f'Location: {self.client.minimap.minimap.get_coordinates()}')

        self.msg.append(f'{self.willows}')


def main():
    app = Lumberjack()
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
