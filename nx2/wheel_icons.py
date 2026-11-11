"""(wheel mod) The item wheel's icons: the game's own item icons (UI/Tex/Icon/<item name>.bntx.zs), decoded with the decoders
nx2/prompt_assets.py uses for the button prompts, scaled down and written to one file the renderer reads (gpu/vk/vk_menu.cpp).
make_game.py runs it at the end of every build. It stays on this PC, made from the user's RomFS like the prompts.

    python -I nx2/wheel_icons.py --romfs <dir> --out <game folder>/wheel/icons.bin [--size 128] [--dump <dir>]

icons.bin: "HWIC", u32 version 1, u32 count, u32 size; then for each icon: char name[48] (NUL-padded game name), size*size RGBA
pixels, straight alpha, rows top down.
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_assets as pa  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--romfs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--dump", help="also write each icon as PNG here (to look at)")
    p.add_argument("--only", nargs="*", help="these names only")
    a = p.parse_args()
    from PIL import Image
    dictionary = pa.zs_dictionary(a.romfs)
    folder = os.path.join(a.romfs, "UI", "Tex", "Icon")
    icons = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".bntx.zs"):
            continue
        name = f[:-len(".bntx.zs")]
        if a.only and name not in a.only:
            continue
        b = pa.read_zs(a.romfs, os.path.join("UI", "Tex", "Icon", f), dictionary)
        tex = pa.bntx_textures(b)
        if not tex:
            print("wheel icons: %s has no texture" % name)
            continue
        tname, t = next(iter(tex.items()))
        _, fmt, w, h, _, _ = t
        try:
            im = pa.decode(b, t).convert("RGBA")
        except Exception as e:   # (a format the prompts' decoder does not know)
            print("wheel icons: %s: format %#x %dx%d not decoded (%s)" % (name, fmt, w, h, e))
            continue
        im = im.resize((a.size, a.size), Image.LANCZOS)
        if a.dump:
            os.makedirs(a.dump, exist_ok=True)
            im.save(os.path.join(a.dump, name + ".png"))
        icons.append((name, im.tobytes()))
        if a.only:
            print("wheel icons: %s: %s format %#x %dx%d, %d textures" % (name, tname, fmt, w, h, len(tex)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"HWIC" + struct.pack("<III", 1, len(icons), a.size))
        for name, px in icons:
            f.write(name.encode()[:47].ljust(48, b"\0") + px)
    print("wheel icons: %d icons at %d px in %s" % (len(icons), a.size, a.out))


if __name__ == "__main__":
    main()
