"""Object (scenery) definitions: index 2 (configs), group 6, one file each."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .store import Buffer, CacheStore

CONFIG_INDEX = 2
OBJECT_GROUP = 6


@dataclass
class ObjectDefinition:
    id: int
    name: str = 'null'
    size_x: int = 1
    size_y: int = 1
    interact_type: int = 2
    """0: doesn't block movement. Anything else blocks (walls block
    their edges, other objects the tiles they cover)."""
    blocks_projectile: bool = True
    wall_or_door: int = -1
    access_block: int = 0
    """Bit mask of the sides the object can be interacted from
    (1 north, 2 east, 4 south, 8 west; rotated with the object)."""
    actions: List[Optional[str]] = field(
        default_factory=lambda: [None] * 5)
    map_scene: int = -1
    map_area: int = -1
    category: int = -1
    varbit: int = -1
    varp: int = -1
    transforms: List[int] = field(default_factory=list)
    """Object ids this one turns into per varbit / varp value (-1 =
    nothing). Doors and quest objects use this."""
    params: Dict[int, object] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.interact_type != 0

    def size(self, orientation: int):
        """Tile size (x, y) after rotation."""
        if orientation in (1, 3):
            return self.size_y, self.size_x
        return self.size_x, self.size_y


class UnknownOpcode(ValueError):
    pass


def decode_object(oid: int, data: bytes) -> ObjectDefinition:
    d = ObjectDefinition(oid)
    buf = Buffer(data)
    while True:
        op = buf.u8()
        if op == 0:
            break
        elif op in (1, 6):
            # models with types; 6 uses 32-bit model ids
            n = buf.u8()
            for _ in range(n):
                buf.i32() if op == 6 else buf.u16()
                buf.u8()
        elif op in (5, 7):
            # models without types; 7 uses 32-bit model ids
            n = buf.u8()
            for _ in range(n):
                buf.i32() if op == 7 else buf.u16()
        elif op == 2:
            d.name = buf.string()
        elif op == 14:
            d.size_x = buf.u8()
        elif op == 15:
            d.size_y = buf.u8()
        elif op == 17:
            d.interact_type = 0
            d.blocks_projectile = False
        elif op == 18:
            d.blocks_projectile = False
        elif op == 19:
            d.wall_or_door = buf.u8()
        elif op in (21, 22, 23, 62, 64, 73, 74, 89, 90):
            # contoured ground, merge normals, occludes, rotated, no shadow,
            # obstructs ground, hollow, random anim start, defer anim
            pass
        elif op == 24:
            buf.u16()  # animation
        elif op == 27:
            d.interact_type = 1
        elif op in (28, 29, 39):
            buf.u8()  # decor displacement, ambient, contrast
        elif 30 <= op <= 34:
            action = buf.string()
            d.actions[op - 30] = None if action.lower() == 'hidden' \
                else action
        elif op in (40, 41):
            n = buf.u8()
            buf.pos += 4 * n  # recolour / retexture pairs
        elif op == 60:
            buf.u16()  # old map icon
        elif op == 61:
            d.category = buf.u16()
        elif op in (65, 66, 67):
            buf.u16()  # model scale
        elif op == 68:
            d.map_scene = buf.u16()
        elif op == 69:
            d.access_block = buf.u8()
        elif op in (70, 71, 72):
            buf.i16()  # model offset
        elif op == 75:
            buf.u8()  # supports items
        elif op in (77, 92, 97, 98):
            # 97 / 98 are 77 / 92 with 3-byte object ids
            read = buf.u16 if op in (77, 92) else buf.u24
            d.varbit = _none(buf.u16())
            d.varp = _none(buf.u16())
            default = _none(read()) if op in (92, 98) else -1
            n = buf.u8()
            d.transforms = [_none(read()) for _ in range(n + 1)]
            d.transforms.append(default)
        elif op == 78:
            buf.u16()  # ambient sound
            buf.u8()   # distance
            buf.u8()   # retain
        elif op == 79:
            buf.u16()
            buf.u16()
            buf.u8()   # distance
            buf.u8()   # retain
            n = buf.u8()
            buf.pos += 2 * n
        elif op in (95, 96):
            buf.u8()  # unknown, newer revisions
        elif op == 93:
            buf.pos += 6  # sound fade settings: (u8, u16) x 2
        elif op == 101:
            # action text that depends on a varbit / varp value range:
            # slot u8, varbit u16, varp u16, min i32, max i32, text
            buf.pos += 13
            buf.string()
        elif op == 100:
            # submenu entry: slot u8, index, text
            buf.u8()
            buf.smart_u16()
            buf.string()
        elif op == 102:
            # conditional submenu entry: slot u8, index u16, varbit u16,
            # varp u16, min i32, max i32, text
            buf.pos += 15
            buf.string()
        elif op == 81:
            buf.u8()  # contoured ground
        elif op == 82:
            d.map_area = buf.u16()
        elif op == 249:
            n = buf.u8()
            for _ in range(n):
                is_string = buf.u8()
                key = buf.u24()
                d.params[key] = buf.string() if is_string else buf.i32()
        else:
            raise UnknownOpcode(
                f'object {oid}: unknown opcode {op} at {buf.pos - 1}: '
                f'{data.hex()}')
    if buf.remaining():
        raise ValueError(f'object {oid}: {buf.remaining()} trailing bytes')
    return d


def _none(v: int) -> int:
    return -1 if v in (0xFFFF, 0xFFFFFF) else v


def load_objects(store: CacheStore) -> Dict[int, ObjectDefinition]:
    files = store.files(CONFIG_INDEX, OBJECT_GROUP)
    return {oid: decode_object(oid, data) for oid, data in files.items()}
