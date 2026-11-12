"""The Teleport places (gpu/vk/oot_tp_places.h, nx2/teleport_places.py): the table holds names only, each area's places in
range and each key once; its old-Teleport arrivals (legacy) are host/hos_main.c's destinations; and the builder's BYML reader
reads what the game's map files are made of. No game: the map file here is made by the test.

python -I -B tests/test_teleport_places.py
"""
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))


def table():
    text = (ROOT / "gpu" / "vk" / "oot_tp_places.h").read_text(encoding="utf-8")
    regions = re.findall(r'"([^"]+)"', re.search(r"oot_tp_regions\[\] = \{(.*?)\};", text).group(1))
    areas = [(int(a), int(r), t, int(f), int(c)) for a, r, t, f, c in
             re.findall(r'^    \{(\d), (\d+), "((?:[^"\\]|\\.)*)", (\d+), (\d+)\},$', text, re.M)]
    places = re.findall(r'^    \{"(\w+)", "(\w+)", "((?:[^"\\]|\\.)*)", "([^"]+)", (\d)\},$', text, re.M)
    return text, regions, areas, places


def byml_doc():
    """A BYML v7 document: {"Actors": [{"Name": "Start", "Translate": [1.0, 2.0, 3.0]}]}"""
    keys = sorted(["Actors", "Name", "Translate"])
    strs = ["Start"]

    def table_node(items):
        head = struct.pack("<I", 0xC2 | len(items) << 8)
        offs, body, at = [], b"", 4 + 4 * (len(items) + 1)
        for s in items:
            offs.append(at + len(body))
            body += s.encode() + b"\0"
        offs.append(at + len(body))
        return head + struct.pack("<%dI" % len(offs), *offs) + body

    kt, st = table_node(keys), table_node(strs)
    while len(kt) % 4: kt += b"\0"
    while len(st) % 4: st += b"\0"
    base = 16 + len(kt) + len(st)
    # (the nodes, laid out after the tables: root dict, the Actors array, the actor dict, its Translate array)
    root_off = base
    arr_off = root_off + 4 + 8
    actor_off = arr_off + 4 + 4 + 4
    tr_off = actor_off + 4 + 8 * 2
    f = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
    root = struct.pack("<I", 0xC1 | 1 << 8) + struct.pack("<I", keys.index("Actors") | 0xC0 << 24) + struct.pack("<I", arr_off)
    arr = struct.pack("<I", 0xC0 | 1 << 8) + bytes([0xC1, 0, 0, 0]) + struct.pack("<I", actor_off)
    actor = (struct.pack("<I", 0xC1 | 2 << 8) + struct.pack("<II", keys.index("Name") | 0xA0 << 24, 0) +
             struct.pack("<II", keys.index("Translate") | 0xC0 << 24, tr_off))
    tr = struct.pack("<I", 0xC0 | 3 << 8) + bytes([0xD2] * 3 + [0]) + struct.pack("<III", f(1.0), f(2.0), f(3.0))
    return b"YB" + struct.pack("<H", 7) + struct.pack("<III", 16, 16 + len(kt), root_off) + kt + st + root + arr + actor + tr


def main():
    failed = []
    text, regions, areas, places = table()
    if re.search(r"\d+\.\d+f", text):
        failed.append("the table has coordinates (positions come from the player's dump)")
    if not places or not areas:
        failed.append("no places or areas read")
    seen = {}
    for banc, spawn, label, key, legacy in places:   # (a dungeon is in both ages' lists: the same place, one picture)
        if seen.setdefault(key, (banc, spawn)) != (banc, spawn):
            failed.append("%s names two places" % key)
    for adult, region, title, first, count in areas:
        if not 0 <= region < len(regions):
            failed.append("%s: no region %d" % (title, region))
        if count < 1 or first + count > len(places):
            failed.append("%s: its places out of the table" % title)
    hos = (ROOT / "host" / "hos_main.c").read_text(encoding="utf-8")
    dests = set(re.findall(r'"(\w+)", "(\w+)", "\w+"', hos[hos.index("oot_destinations[] ="):hos.index("};", hos.index("oot_destinations[] ="))]))
    for banc, spawn, label, key, legacy in places:
        if (legacy == "1") != ((banc, spawn) in dests):
            failed.append("%s: legacy %s, but %s at %s is %s destination" % (key, legacy, banc, spawn, "a" if (banc, spawn) in dests else "no"))
    import teleport_places
    doc = teleport_places.byml(byml_doc())
    actor = doc["Actors"][0]
    if actor.get("Name") != "Start" or [round(v, 3) for v in actor.get("Translate")] != [1.0, 2.0, 3.0]:
        failed.append("the BYML reader read %r" % doc)
    if [p[:3] for p in teleport_places.places()] != [(p[0], p[1], p[3]) for p in places]:
        failed.append("nx2/teleport_places.py reads the table differently")
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print("PASS: %d places in %d areas, names only; legacy places are the destinations; the BYML reader reads a map's actors"
          % (len(places), len(areas)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
