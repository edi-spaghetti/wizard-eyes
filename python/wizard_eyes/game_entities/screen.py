from . import player
from . import trees
from . import entity
from . import npcs
from . import items
from . import tile
from ..constants import DEFAULT_ZOOM


class GameScreen(object):
    """Container class for anything displayed within the main game screen."""

    def __init__(self, client, zoom=DEFAULT_ZOOM):
        self.client = client
        self._player = None
        self.default_npc = npcs.NPC
        self.zoom = zoom
        self.tile_marker = tile.TileMarker(zoom, self.client, self)

    @property
    def player(self):
        if self._player is None:
            names = [f'player_marker_{self.zoom}',
                     'player_blue_splat', 'player_red_splat']
            _player = player.Player(
                'player', (0, 0), self.client, self, template_names=names)
            _player.load_masks(names)
            self._player = _player

        return self._player

    @property
    def tile_size(self):
        # assumes 100% top down view at default zoom
        template = self._player.templates[f'player_marker_{self.zoom}']
        width, _, _ = template.shape
        return width

    def create_game_entity(self, type_, *args,
                           entity_templates=None, **kwargs):
        """Factory method to create entities from this module."""

        if type_ in {'npc', 'npc_tag'}:
            npc = self.default_npc(*args, **kwargs)
            templates = ['player_blue_splat', 'player_red_splat']
            npc.load_templates(templates)
            npc.load_masks(templates)
            return npc
        # TODO: tree factory
        elif type_ == 'oak':
            tree = trees.Oak(*args, **kwargs)
            return tree
        elif type_ == 'willow':
            tree = trees.Willow(*args, **kwargs)
            return tree
        elif type_ == 'blisterwood':
            tree = trees.Blisterwood(*args, **kwargs)
            return tree
        elif type_ == 'magic':
            tree = trees.Magic(*args, **kwargs)
            return tree
        elif type_ == 'item':
            item = items.GroundItem(*args, **kwargs)
            if entity_templates:
                item.load_templates(entity_templates)
                item.load_masks(entity_templates)
            return item
        else:
            _entity = entity.GameEntity(*args, **kwargs)
            if entity_templates:
                _entity.load_templates(entity_templates)
                _entity.load_masks(entity_templates)
            return _entity

    def is_clickable(self, x1, y1, x2, y2):
        """Validate bounding box can be clicked without accidentally clicking
        UI elements"""

        result = True

        corners = ((x1, y1), (x2, y2), (x2, y1), (x1, y2))
        for corner in corners:
            offset = (self.client.margin_left, self.client.margin_top,
                      self.client.margin_right, self.client.margin_bottom)
            if not self.client.is_inside(*corner, offset=offset):
                return False

        fixed_ui = (self.client.banner, self.client.minimap,
                    self.client.tabs, self.client.chat)

        for element in fixed_ui:
            for corner in corners:
                if element.is_inside(*corner):
                    return False
                # TODO: random chance if close to edge

        # TODO: bank
        dynamic_ui = (self.client.tabs, self.client.chat)
        for element in dynamic_ui:
            # TODO: method on AbstractInterface to determine if open
            #       for now, assume they are open
            for corner in corners:
                if element.is_inside(*corner):
                    return False
                # TODO: random chance if close to edge

        return result
