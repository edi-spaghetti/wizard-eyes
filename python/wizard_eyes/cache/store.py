"""Read-only access to the OSRS game cache (the ``main_file_cache`` files).

The client keeps a full copy of the cache on disk (RuneLite:
``~/.runelite/jagexcache/oldschool/LIVE``) and updates it whenever the game
updates, so reading it gives the current game data with no third party.

Layout:

- ``main_file_cache.idx<N>``: one 6-byte entry per group of index N
  (3-byte size, 3-byte first sector).
- ``main_file_cache.dat2``: 520-byte sectors holding the group data.
- ``idx255``: one reference table per index, listing its groups, the files
  in each group, revisions and (optionally) name hashes.

Each group is a container (compression byte, lengths, gzip/bzip2 body).
Groups holding several files are split by a chunk table at the end.
"""

import bz2
import gzip
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SECTOR_SIZE = 520
REFERENCE_INDEX = 255


def default_cache_dir() -> Path:
    """The cache RuneLite keeps up to date."""
    return Path(os.path.expanduser('~')) / '.runelite' / 'jagexcache' / \
        'oldschool' / 'LIVE'


def decompress(data: bytes) -> bytes:
    """Unpack a cache container. Raises ValueError if it's malformed."""
    compression = data[0]
    length = struct.unpack('>I', data[1:5])[0]
    if compression == 0:
        return data[5:5 + length]
    expected = struct.unpack('>I', data[5:9])[0]
    body = data[9:9 + length]
    if compression == 1:
        # cache bzip2 streams have their 'BZh1' header stripped
        out = bz2.decompress(b'BZh1' + body)
    elif compression == 2:
        out = gzip.decompress(body)
    else:
        raise ValueError(f'unknown compression {compression}')
    if len(out) != expected:
        raise ValueError(f'expected {expected} bytes, got {len(out)}')
    return out


def name_hash(name: str) -> int:
    """Java ``String.hashCode`` of the lower-cased name (signed 32 bit)."""
    h = 0
    for ch in name.lower():
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h & 0x80000000 else h


def split_group(data: bytes, file_count: int) -> List[bytes]:
    """Split a decompressed multi-file group into its files."""
    if file_count == 1:
        return [data]
    chunks = data[-1]
    pos = len(data) - 1 - chunks * file_count * 4
    parts: List[List[bytes]] = [[] for _ in range(file_count)]
    offset = 0
    for _ in range(chunks):
        size = 0
        for f in range(file_count):
            size += struct.unpack('>i', data[pos:pos + 4])[0]
            pos += 4
            parts[f].append(data[offset:offset + size])
            offset += size
    return [b''.join(p) for p in parts]


class Buffer:
    """Big-endian reader with the cache's variable-length integer types."""

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def u8(self) -> int:
        v = self.data[self.pos]
        self.pos += 1
        return v

    def i8(self) -> int:
        v = self.u8()
        return v - 256 if v > 127 else v

    def u16(self) -> int:
        v = struct.unpack_from('>H', self.data, self.pos)[0]
        self.pos += 2
        return v

    def i16(self) -> int:
        v = struct.unpack_from('>h', self.data, self.pos)[0]
        self.pos += 2
        return v

    def u24(self) -> int:
        v = int.from_bytes(self.data[self.pos:self.pos + 3], 'big')
        self.pos += 3
        return v

    def i32(self) -> int:
        v = struct.unpack_from('>i', self.data, self.pos)[0]
        self.pos += 4
        return v

    def f32(self) -> float:
        v = struct.unpack_from('>f', self.data, self.pos)[0]
        self.pos += 4
        return v

    def smart_u16(self) -> int:
        """1 byte if < 128, else 2 bytes minus 0x8000."""
        if self.data[self.pos] < 128:
            return self.u8()
        return self.u16() - 0x8000

    def smart_u32(self) -> int:
        """2 bytes if the top bit is clear, else 4 bytes with it masked."""
        if self.data[self.pos] & 0x80:
            return self.i32() & 0x7FFFFFFF
        return self.u16()

    def big_smart(self) -> int:
        """smart_u16 summed while it reads 32767 (location id deltas)."""
        total = 0
        value = self.smart_u16()
        while value == 32767:
            total += 32767
            value = self.smart_u16()
        return total + value

    def string(self) -> str:
        end = self.data.index(0, self.pos)
        s = self.data[self.pos:end].decode('cp1252')
        self.pos = end + 1
        return s


@dataclass
class ReferenceTable:
    """Groups and files of one index, from its idx255 entry."""

    protocol: int
    revision: int
    groups: List[int]
    files: Dict[int, List[int]] = field(default_factory=dict)
    names: Dict[int, int] = field(default_factory=dict)
    """name hash -> group id, only when the index stores names."""

    @classmethod
    def decode(cls, data: bytes) -> 'ReferenceTable':
        buf = Buffer(data)
        protocol = buf.u8()
        if protocol < 5 or protocol > 7:
            raise ValueError(f'unsupported reference table protocol '
                             f'{protocol}')
        revision = buf.i32() if protocol >= 6 else 0
        flags = buf.u8()
        read = buf.smart_u32 if protocol >= 7 else buf.u16

        count = read()
        groups, acc = [], 0
        for _ in range(count):
            acc += read()
            groups.append(acc)

        names = {}
        if flags & 0x1:
            for gid in groups:
                names[buf.i32()] = gid
        buf.pos += 4 * count            # crc
        if flags & 0x8:
            buf.pos += 4 * count        # hashes
        if flags & 0x2:
            buf.pos += 64 * count       # whirlpool digests
        if flags & 0x4:
            buf.pos += 8 * count        # compressed + uncompressed sizes
        buf.pos += 4 * count            # revisions

        counts = [read() for _ in range(count)]
        files = {}
        for gid, n in zip(groups, counts):
            ids, acc = [], 0
            for _ in range(n):
                acc += read()
                ids.append(acc)
            files[gid] = ids
        # file name hashes (flags & 1) follow; not needed

        return cls(protocol, revision, groups, files, names)


class CacheStore:
    """Read groups and files out of an on-disk cache."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_cache_dir()
        if not (self.path / 'main_file_cache.dat2').exists():
            raise FileNotFoundError(f'no cache at {self.path}')
        self._dat = open(self.path / 'main_file_cache.dat2', 'rb')
        self._idx: Dict[int, bytes] = {}
        self._tables: Dict[int, ReferenceTable] = {}

    def close(self):
        self._dat.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _index(self, index: int) -> bytes:
        if index not in self._idx:
            self._idx[index] = (
                self.path / f'main_file_cache.idx{index}').read_bytes()
        return self._idx[index]

    def raw(self, index: int, group: int) -> Optional[bytes]:
        """Raw container bytes of a group, or None if not downloaded."""
        idx = self._index(index)
        entry = idx[group * 6:group * 6 + 6]
        if len(entry) < 6:
            return None
        size = int.from_bytes(entry[:3], 'big')
        sector = int.from_bytes(entry[3:], 'big')
        if size == 0 or sector == 0:
            return None

        big = group > 0xFFFF
        header = 10 if big else 8
        out = bytearray()
        chunk = 0
        while len(out) < size:
            self._dat.seek(sector * SECTOR_SIZE)
            block = self._dat.read(SECTOR_SIZE)
            if big:
                g, c, nxt, ix = struct.unpack('>IH3sB', block[:10])
            else:
                g, c, nxt, ix = struct.unpack('>HH3sB', block[:8])
            if g != group or c != chunk or ix != index:
                raise ValueError(
                    f'corrupt sector for index {index} group {group}')
            take = min(SECTOR_SIZE - header, size - len(out))
            out += block[header:header + take]
            sector = int.from_bytes(nxt, 'big')
            chunk += 1
        return bytes(out)

    def table(self, index: int) -> ReferenceTable:
        if index not in self._tables:
            data = self.raw(REFERENCE_INDEX, index)
            if data is None:
                raise KeyError(f'no reference table for index {index}')
            self._tables[index] = ReferenceTable.decode(decompress(data))
        return self._tables[index]

    def group(self, index: int, group: int) -> Optional[bytes]:
        raw = self.raw(index, group)
        return None if raw is None else decompress(raw)

    def files(self, index: int, group: int) -> Optional[Dict[int, bytes]]:
        """All files of a group, keyed by file id."""
        data = self.group(index, group)
        if data is None:
            return None
        ids = self.table(index).files[group]
        return dict(zip(ids, split_group(data, len(ids))))
