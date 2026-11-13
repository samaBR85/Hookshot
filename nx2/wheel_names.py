"""(hookshot) The item wheel's names (gpu/vk/vk_menu.cpp): every item's name as the game writes it, in each of its
languages, from the game's own messages (Mals/<region><lang>.Product.110.sarc.zs, ActorMsg/PouchContent.msbt: the
<item>_Name entries), so the wheel names an item as the game does in the language it runs in.

out: "HWNM", u32 version 1, u32 languages; then for each language: char code[8] (language.txt's: "pt-BR", NUL-padded),
u32 count, and count records of char name[48] (the game's item name, "Item_Arrow_Ice") and char text[64] (UTF-8,
NUL-padded, cut at a character's start).

python -I nx2/wheel_names.py --romfs <romfs> --out <game folder>/wheel/names.bin
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_assets as pa  # noqa: E402 (zs_dictionary, read_zs)

# the message archives by language.txt's codes (gpu/vk/vk_menu_i18n.cpp kGameLangs)
LANGS = {"en-US": "USen", "en-GB": "EUen", "fr": "EUfr", "fr-CA": "USfr", "de": "EUde", "it": "EUit", "es": "EUes",
         "es-419": "USes", "pt-BR": "USpt", "nl": "EUnl", "pl": "EUpl", "ru": "EUru", "ja": "JPja", "ko": "KRko",
         "zh-Hans": "CNzh", "zh-Hant": "TWzh"}
KINDS = ("Item_", "Weapon_", "Armor_", "Pouch_")   # (the pouch's contents the item row can hold: Pouch_ the bow with an arrow)


def sarc(b):
    """the files of a SARC by name"""
    if b[:4] != b"SARC":
        raise ValueError("not a SARC")
    hl, = struct.unpack_from("<H", b, 4)
    data, = struct.unpack_from("<I", b, 12)
    n, = struct.unpack_from("<H", b, hl + 6)
    names = hl + 12 + 16 * n + 8
    out = {}
    for i in range(n):
        attr, start, end = struct.unpack_from("<III", b, hl + 12 + 16 * i + 4)
        o = names + (attr & 0xFFFFFF) * 4
        out[b[o:b.index(b"\0", o)].decode()] = b[data + start:data + end]
    return out


def text_of(raw, utf8, e):
    """a message's text without its control tags (0x0E: group, type, a length and that many bytes; 0x0F: a closing tag)"""
    if utf8:
        o, k = bytearray(), 0
        while k < len(raw):
            c = raw[k]
            if c == 0x0E:
                ln, = struct.unpack_from(e + "H", raw, k + 5)
                k += 7 + ln
            elif c == 0x0F:
                k += 5
            elif c == 0:
                break
            else:
                o.append(c)
                k += 1
        return o.decode("utf-8", "replace")
    s, k = [], 0
    while k + 1 < len(raw):
        c, = struct.unpack_from(e + "H", raw, k)
        if c == 0x0E:
            ln, = struct.unpack_from(e + "H", raw, k + 6)
            k += 8 + ln
        elif c == 0x0F:
            k += 6
        elif c == 0:
            break
        else:
            s.append(chr(c))
            k += 2
    return "".join(s)


def msbt(b):
    """an MSBT's texts by label"""
    if b[:8] != b"MsgStdBn":
        raise ValueError("not an MSBT")
    e = "<" if b[8:10] == b"\xff\xfe" else ">"
    utf8 = b[12] == 0
    sections, = struct.unpack_from(e + "H", b, 14)
    p, labels, texts = 0x20, {}, []
    for _ in range(sections):
        magic, size = b[p:p + 4], struct.unpack_from(e + "I", b, p + 4)[0]
        body = p + 16
        if magic == b"LBL1":
            groups, = struct.unpack_from(e + "I", b, body)
            for g in range(groups):
                count, off = struct.unpack_from(e + "II", b, body + 4 + 8 * g)
                q = body + off
                for _ in range(count):
                    ln = b[q]
                    labels[b[q + 1:q + 1 + ln].decode()] = struct.unpack_from(e + "I", b, q + 1 + ln)[0]
                    q += 1 + ln + 4
        elif magic == b"TXT2":
            n, = struct.unpack_from(e + "I", b, body)
            offs = [struct.unpack_from(e + "I", b, body + 4 + 4 * i)[0] for i in range(n)] + [size]
            texts = [text_of(b[body + offs[i]:body + offs[i + 1]], utf8, e) for i in range(n)]
        p = (body + size + 15) & ~15
    return {label: texts[i] for label, i in labels.items() if i < len(texts)}


def cut(text, n):
    """text as UTF-8 in at most n bytes, cut at a character's start"""
    b = text.encode()
    if len(b) <= n:
        return b
    b = b[:n]
    while b and (b[-1] & 0xC0) == 0x80:
        b = b[:-1]
    return b[:-1] if b and b[-1] >= 0xC0 else b


def names(romfs, dictionary, region):
    b = pa.read_zs(romfs, os.path.join("Mals", f"{region}.Product.110.sarc.zs"), dictionary)
    m = msbt(sarc(b)["ActorMsg/PouchContent.msbt"])
    return {k[:-len("_Name")]: v for k, v in m.items() if k.endswith("_Name") and k.startswith(KINDS) and v.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--romfs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dictionary = pa.zs_dictionary(a.romfs)
    langs = []
    for code, region in LANGS.items():
        try:
            langs.append((code, names(a.romfs, dictionary, region)))
        except Exception as e:   # (a language this game's files do not have)
            print(f"wheel names: {code} ({region}) not read: {e}", file=sys.stderr)
    if not langs:
        raise SystemExit("wheel names: no language read")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"HWNM" + struct.pack("<II", 1, len(langs)))
        for code, m in langs:
            f.write(code.encode().ljust(8, b"\0") + struct.pack("<I", len(m)))
            for name in sorted(m):
                f.write(name.encode()[:47].ljust(48, b"\0") + cut(m[name], 63).ljust(64, b"\0"))
    print(f"wheel names: {len(langs)} languages, {len(langs[0][1])} names each, in {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
