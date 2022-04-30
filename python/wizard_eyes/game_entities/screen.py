from . import player
from . import trees
from . import entity
from . import npcs
from . import items


class GameScreen(object):
    """Container class for anything displayed within the main game screen."""

    def __init__(self, client):
        self.client = client
        self._player = None
        self.default_npc = npcs.NPC

    @property
    def player(self):
        if self._player is None:
            names = ['player_marker', 'player_blue_splat', 'player_red_splat']
            _player = player.Player(
                'player', (0, 0), self.client, self, template_names=names)
            _player.load_masks(names)
            self._player = _player

        return self._player

    @property
    def tile_size(self):
        # assumes 100% top down view at default zoom
        # TODO: set dynamically
        return 48

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
        elif type_ == 'willow':
            tree = trees.Willow(*args, **kwargs)
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
