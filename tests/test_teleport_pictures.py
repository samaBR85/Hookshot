"""The Teleport page's pictures: every destination has its maps in gpu/vk/oot_teleport_pictures.h, in the order of
host/hos_main.c's oot_destinations, and every one of those maps can be taken as a screenshot (gpu/vk/oot_teleport_capture.h).

python -I -B tests/test_teleport_pictures.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def block(text, start):
    """The text between the { after start and its matching }"""
    at = text.index("{", text.index(start))
    depth = 0
    for i in range(at, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[at + 1:i]
    raise ValueError(start)


def main():
    main_c = (ROOT / "host" / "hos_main.c").read_text(encoding="utf-8")
    name = lambda m: "" if m in ("NULL", "nullptr") else m.strip('"')
    destinations = [(name(c), name(a)) for c, a in re.findall(
        r'\{(NULL|"\w+"), (?:NULL|"\w+"), (?:NULL|"\w+"),\s*(NULL|"\w+")', block(main_c, "oot_destinations[] ="))]
    pictures = (ROOT / "gpu" / "vk" / "oot_teleport_pictures.h").read_text(encoding="utf-8")
    maps = re.findall(r'\{\s*(nullptr|"\w+")\s*,\s*(nullptr|"\w+")\s*\}', block(pictures, "teleport_maps[][2] ="))
    maps = [(name(c), name(a)) for c, a in maps]
    capture = (ROOT / "gpu" / "vk" / "oot_teleport_capture.h").read_text(encoding="utf-8")
    capturable = re.findall(r'"(\w+)"', block(capture, "oot_capture_maps[]"))
    failed = []
    if len(maps) != len(destinations):
        failed.append("%d destinations, %d lines of pictures" % (len(destinations), len(maps)))
    for i, (d, m) in enumerate(zip(destinations, maps)):
        if d != m:
            failed.append("destination %d: hos_main.c has %s, the pictures %s" % (i + 1, d, m))
    for m in sorted({x for pair in maps for x in pair if x} - set(capturable)):
        failed.append("%s has no screenshot slot in oot_teleport_capture.h" % m)
    if len(set(capturable)) != len(capturable):
        failed.append("oot_teleport_capture.h lists a map twice")
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print("PASS: %d destinations, %d maps, every one can be taken" % (len(maps), len(capturable)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
