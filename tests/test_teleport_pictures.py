"""The Teleport page's pictures: every destination has its maps in gpu/vk/oot_teleport_pictures.h, in the order of
host/hos_main.c's oot_destinations, and every one of those maps can be taken as a screenshot (gpu/vk/oot_teleport_capture.h).
The screenshots' files are read and written by the menu's picture worker only, and taken only while the Debug tools are on.

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
    # The pictures' files are the picture worker's alone: the replay's thread (a present) hands the pixels over, the game's
    # thread hands the arrival over, and neither opens, reads or writes a file (a ~389 KB picture, at a present, was a hitch)
    menu = (ROOT / "gpu" / "vk" / "vk_menu.cpp").read_text(encoding="utf-8")
    file_work = re.compile(r"\b(fopen|fread|fwrite|read_picture_file|write_picture_file|picture_read_now|filesystem)\b")
    store = block(menu, "void menu_store_capture(int map")
    if file_work.search(store) or "picture_job(" not in store:
        failed.append("menu_store_capture (the replay's thread) must only hand the picture to picture_job")
    option_set = block(menu, "void menu_option_set(int id, int value)")
    arrived = option_set[option_set.index("OOT_TP_CAPTURE_ID"):]
    arrived = arrived[:arrived.index("return;\n    }") + 1]
    if file_work.search(arrived) or "picture_job(" not in arrived:
        failed.append("menu_option_set's OOT_TP_CAPTURE_ID (the game's thread) must only hand the arrival to picture_job")
    if "teleport_captures_on()" not in arrived:
        failed.append("menu_option_set's OOT_TP_CAPTURE_ID must take screenshots only while teleport_captures_on()")
    # And the game side watches for arrivals only while the menu takes screenshots (the Debug tools on)
    tick = block(main_c, "static void oot_capture_tick(Ctx *c, int64_t now)")
    if "rx_option(HOS_OPT_TELEPORT_PICTURES, 0)" not in tick.split("oot_gamedata_string")[0]:
        failed.append("oot_capture_tick must ask rx_option(HOS_OPT_TELEPORT_PICTURES) before anything else")
    replay = (ROOT / "gpu" / "vk" / "vk_replay.cpp").read_text(encoding="utf-8")
    if "if (id == HOS_OPT_TELEPORT_PICTURES) return menu_teleport_captures_on();" not in replay:
        failed.append("the renderer's option() must answer HOS_OPT_TELEPORT_PICTURES with menu_teleport_captures_on()")
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print("PASS: %d destinations, %d maps, every one can be taken" % (len(maps), len(capturable)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
