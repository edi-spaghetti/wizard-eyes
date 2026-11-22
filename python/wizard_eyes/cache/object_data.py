"""Build the object index: every named or usable object on the map.

Each object placement in the map regions is kept if its definition, or any
of the definitions it can turn into (varbit / varp transforms, used by
doors and quest objects), has a name or a right-click option. Planes are
adjusted for bridges the same way as collision, so an object's plane is the
plane the player stands on next to it.

Output: flat arrays (one entry per placement) plus a dictionary of the
definitions they use, see ``wizard_eyes.navigation.object_map``.
"""

from typing import Dict, Iterable, Tuple

import numpy

from .maps import SETTING_BRIDGE, Region
from .objects import ObjectDefinition

ARRAY_TYPES = {
    'x': numpy.uint16,
    'y': numpy.uint16,
    'plane': numpy.uint8,
    'id': numpy.int32,
    'type': numpy.uint8,
    'orientation': numpy.uint8,
}


def is_useful(definition: ObjectDefinition) -> bool:
    return (definition.name not in ('null', '')
            or any(a for a in definition.actions))


def encode_definition(d: ObjectDefinition) -> list:
    """Compact JSON form: [name, actions, size_x, size_y, interact_type,
    access_block, transforms]."""
    return [d.name, d.actions, d.size_x, d.size_y, d.interact_type,
            d.access_block, d.transforms]


def build_object_data(regions: Iterable[Region],
                      objects: Dict[int, ObjectDefinition],
                      ) -> Tuple[Dict[str, numpy.ndarray], Dict[int, list]]:
    """(arrays, definitions) for every useful object placement."""
    useful: Dict[int, bool] = {}

    def keep(oid: int) -> bool:
        try:
            return useful[oid]
        except KeyError:
            d = objects.get(oid)
            ok = d is not None and (is_useful(d) or any(
                t in objects and is_useful(objects[t])
                for t in d.transforms))
            useful[oid] = ok
            return ok

    columns = {k: [] for k in ARRAY_TYPES}
    for region in regions:
        bridge = (region.settings[1] & SETTING_BRIDGE) != 0
        bx, by = region.base
        for loc in region.locations:
            if not keep(loc.id):
                continue
            plane = loc.plane - 1 if bridge[loc.x, loc.y] else loc.plane
            if plane < 0:
                continue
            columns['x'].append(bx + loc.x)
            columns['y'].append(by + loc.y)
            columns['plane'].append(plane)
            columns['id'].append(loc.id)
            columns['type'].append(loc.type)
            columns['orientation'].append(loc.orientation)

    arrays = {k: numpy.array(v, dtype=ARRAY_TYPES[k])
              for k, v in columns.items()}
    used = set(int(i) for i in numpy.unique(arrays['id']))
    for oid in list(used):
        used.update(t for t in objects[oid].transforms if t in objects)
    definitions = {oid: encode_definition(objects[oid])
                   for oid in sorted(used)}
    return arrays, definitions
