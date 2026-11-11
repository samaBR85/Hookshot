"""Game-free: the item wheel's files, made at the end of every build (nx2/make_game.py runs nx2/wheel_icons.py and
nx2/wheel_sounds.py), are what the game reads: wheel/icons.bin for gpu/vk/vk_menu.cpp and wheel/sounds.bin for
host/nn_audio.c, with the same magic, version and record layout; every sound the host plays is one the build writes; and
wheel_sounds.py finds the names and sounds of a BARS archive.

python -I tests/test_wheel_assets.py
"""
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import wheel_sounds  # noqa: E402


def read(*p):
    return (ROOT.joinpath(*p)).read_text(encoding="utf-8")


def fail(msg):
    raise SystemExit("FAIL: " + msg)


def build_runs_them():
    make = read("nx2", "make_game.py")
    for script, out in (("wheel_icons.py", "icons.bin"), ("wheel_sounds.py", "sounds.bin")):
        if not re.search(r'os\.path\.join\(HERE, "%s"\), "--romfs", romfs, "--out", os\.path\.join\(wheel, "%s"\)'
                         % (re.escape(script), re.escape(out)), make):
            fail("nx2/make_game.py does not run nx2/%s into wheel/%s" % (script, out))
    if 'wheel = os.path.join(out, "wheel")' not in make:
        fail("nx2/make_game.py's wheel folder is not <game folder>/wheel")


def icons_match():
    writer, reader = read("nx2", "wheel_icons.py"), read("gpu", "vk", "vk_menu.cpp")
    if 'f.write(b"HWIC" + struct.pack("<III", 1, len(icons), a.size))' not in writer:
        fail("nx2/wheel_icons.py's header is not HWIC, version 1, count, size")
    if 'name.encode()[:47].ljust(48, b"\\0") + px' not in writer:
        fail("nx2/wheel_icons.py's record is not name[48] then the pixels")
    if '"wheel/icons.bin"' not in reader or '!std::memcmp(magic, "HWIC", 4)' not in reader or "hd[0] == 1" not in reader:
        fail("gpu/vk/vk_menu.cpp does not read wheel/icons.bin as HWIC version 1")
    if not re.search(r"struct WheelIcon \{\s*char name\[48\];", reader):
        fail("gpu/vk/vk_menu.cpp's WheelIcon name is not 48 bytes")


def sounds_match():
    writer, reader = read("nx2", "wheel_sounds.py"), read("host", "nn_audio.c")
    head = 48 + struct.calcsize("<II16h")
    if 'f.write(b"HWSN" + struct.pack("<II", 1, len(blobs))' not in writer:
        fail("nx2/wheel_sounds.py's header is not HWSN, version 1, count")
    if 'name.encode().ljust(48, b"\\0")[:48] + struct.pack("<II16h", rate, len(left), *coefs) + pcm' not in writer:
        fail("nx2/wheel_sounds.py's record is not name[48], rate, frames, coefs[16], pcm")
    if '"wheel/sounds.bin"' not in reader or '!memcmp(b, "HWSN", 4)' not in reader:
        fail("host/nn_audio.c does not read wheel/sounds.bin as HWSN")
    if "b + at + %d" % head not in reader or "at += %d + (long)frames * 4" % head not in reader:
        fail("host/nn_audio.c's sound record is not %d bytes before its samples" % head)
    played = set(re.findall(r'"(SE_Sys_[A-Za-z0-9_]+)"', read("host", "nn_audio.c") + read("host", "oot_wheel.c")))
    missing = played - set(wheel_sounds.WANTED)
    if not played or missing:
        fail("the host plays sounds the build does not write: %s" % sorted(missing))


def bars_entries():
    names = ["SE_Sys_ShortCut_Open00", "SE_Sys_ShortCut_Close00"]
    amtas, bwavs = [], []
    for n in names:
        body = b"AMTA" + b"\0" * 4 + b"\0" * 8 + b"\0" * 16 + n.encode() + b"\0"
        body = body[:8] + struct.pack("<I", len(body)) + body[12:]
        amtas.append(body)
        bwavs.append(b"BWAV" + b"\0" * 12)
    at = 0x10 + 4 * len(names) + 8 * len(names)
    table, blob = b"", b""
    for a, w in zip(amtas, bwavs):
        table += struct.pack("<II", at + len(blob), at + len(blob) + len(a))
        blob += a + w
    bars = b"BARS" + b"\0" * 8 + struct.pack("<I", len(names)) + b"\0" * (4 * len(names)) + table + blob
    found = wheel_sounds.entries(bars)
    if sorted(found) != sorted(names):
        fail("wheel_sounds.entries found %s, not %s" % (sorted(found), sorted(names)))
    for n in names:
        if bars[found[n]:found[n] + 4] != b"BWAV":
            fail("wheel_sounds.entries' offset for %s is not its BWAV" % n)


def main():
    build_runs_them()
    icons_match()
    sounds_match()
    bars_entries()
    print("PASS: the build makes the item wheel's icons and sounds as the game reads them")


if __name__ == "__main__":
    sys.exit(main())
