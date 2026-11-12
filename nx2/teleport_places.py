"""The Teleport places' positions (gpu/vk/oot_tp_places.h): teleport_places.bin in the game folder, from the player's own dump.

    python -I nx2/teleport_places.py --romfs <romfs> --out <game folder>

Each place names its map and its arrival point (the start position its map change asks for). The point's position is read
from the map's actors in the player's RomFS (Banc/<map>/<map>_Static.bcett.byml.zs and _Dynamic, zstd with the game's
bcett.byml dictionary): the arrival point's HorsePos anchor where it links one (where the game stands Link on arrival), else
its BackPos or FieldDungeonEntranceGo anchor, else the point itself. Nothing made from the game is in the repository; the
file stays in the game folder, as the button prompts do. The menu uses the positions to know which place a screenshot is of.

Layout (little-endian): 'TPP1', u32 count, then count x {char key[48]; float x, y, z}. Needs zstandard (the builder's own).
"""
import argparse
import os
import re
import struct
import sys

import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
PLACES = os.path.join(HERE, "..", "gpu", "vk", "oot_tp_places.h")
LINKED = ("HorsePos", "BackPos", "FieldDungeonEntranceGo")   # (the anchors an arrival point links, nearest Link's spot first)


def zs_dictionary(romfs, name):
    """A dictionary out of Pack/ZsDic.pack.zs (a SARC, itself zstd without a dictionary), by its name"""
    with open(os.path.join(romfs, "Pack", "ZsDic.pack.zs"), "rb") as f:
        b = zstandard.ZstdDecompressor().decompressobj().decompress(f.read())
    data, = struct.unpack_from("<I", b, 12)
    hl, = struct.unpack_from("<H", b, 4)
    n, = struct.unpack_from("<H", b, hl + 6)
    names = hl + 12 + 16 * n + 8
    for i in range(n):
        attr, start, end = struct.unpack_from("<III", b, hl + 12 + 16 * i + 4)
        o = names + (attr & 0xFFFFFF) * 4
        if b[o:b.index(b"\0", o)].decode() == name:
            return b[data + start:data + end]
    raise SystemExit("teleport places: no %s in Pack/ZsDic.pack.zs" % name)


def byml(b):
    """A BYML (v7) document as Python values: what the map files need (dicts, arrays, strings, numbers)"""
    if b[:2] != b"YB":
        raise ValueError("not a little-endian BYML")
    keys_off, strs_off, root_off = struct.unpack_from("<III", b, 4)

    def table(off):
        if not off:
            return []
        n = struct.unpack_from("<I", b, off)[0] >> 8
        offs = struct.unpack_from("<%dI" % (n + 1), b, off + 4)
        return [b[off + offs[i]:b.index(b"\0", off + offs[i])].decode("utf-8", "replace") for i in range(n)]

    keys, strs = table(keys_off), table(strs_off)

    def value(t, v):
        if t == 0xA0:
            return strs[v]
        if t in (0xC0, 0xC1):
            return node(v)
        if t == 0xD0:
            return bool(v)
        if t == 0xD1:
            return struct.unpack("<i", struct.pack("<I", v))[0]
        if t == 0xD2:
            return struct.unpack("<f", struct.pack("<I", v))[0]
        if t == 0xD3:
            return v
        if t in (0xD4, 0xD5, 0xD6):
            return struct.unpack_from({0xD4: "<q", 0xD5: "<Q", 0xD6: "<d"}[t], b, v)[0]
        return None

    def node(off):
        kind, n = b[off], struct.unpack_from("<I", b, off)[0] >> 8
        if kind == 0xC0:
            vals = off + 4 + ((n + 3) & ~3)
            return [value(b[off + 4 + i], struct.unpack_from("<I", b, vals + 4 * i)[0]) for i in range(n)]
        if kind == 0xC1:
            out = {}
            for i in range(n):
                e = off + 4 + 8 * i
                out[keys[struct.unpack_from("<I", b, e)[0] & 0xFFFFFF]] = value(b[e + 3], struct.unpack_from("<I", b, e + 4)[0])
            return out
        return None

    return node(root_off)


def places():
    """[(map, arrival point, key)] from gpu/vk/oot_tp_places.h"""
    text = open(PLACES, encoding="utf-8").read()
    return re.findall(r'\{"(\w+)", "(\w+)", "(?:[^"\\]|\\.)*", "([^"]+)"\}', text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--romfs", required=True)
    ap.add_argument("--out", required=True, help="the game folder")
    a = ap.parse_args()
    decompress = zstandard.ZstdDecompressor(dict_data=zstandard.ZstdCompressionDict(zs_dictionary(a.romfs, "bcett.byml.zsdic")))
    maps, out, missing = {}, [], []
    for banc, spawn, key in places():
        if banc not in maps:
            base = re.sub(r"_(Child|Adult)$", "", banc)
            maps[banc] = []
            for part in ("Static", "Dynamic"):
                path = os.path.join(a.romfs, "Banc", base, "%s_%s.bcett.byml.zs" % (banc, part))
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        maps[banc] += (byml(decompress.decompressobj().decompress(f.read())) or {}).get("Actors", [])
        actors = maps[banc]
        point = next((t for t in actors if t.get("Name") == spawn), None)
        if not point:
            missing.append("%s at %s" % (banc, spawn))
            continue
        by_hash = {t.get("Hash"): t for t in actors}
        linked = {}
        for link in point.get("Links") or []:
            t = by_hash.get(link.get("Dst"))
            if t and t.get("Gyaml") == "DestinationAnchor":
                linked.setdefault(t.get("Name"), t)
        at = next((linked[n]["Translate"] for n in LINKED if n in linked), point.get("Translate"))
        out.append((key, at))
    dst = os.path.join(a.out, "teleport_places.bin")
    with open(dst + ".tmp", "wb") as f:
        f.write(b"TPP1" + struct.pack("<I", len(out)))
        for key, (x, y, z) in out:
            f.write(key.encode().ljust(48, b"\0")[:48] + struct.pack("<fff", x, y, z))
    os.replace(dst + ".tmp", dst)
    print("teleport places: %d positions -> %s%s" % (len(out), dst, "; not found: " + ", ".join(missing) if missing else ""),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
