"""(hookshot) The item wheel's sounds: the item row's own UI sounds (Sound/Resource/UI_Static.Product.111.bars.zs, the
"ShortCut" entries), decoded from DSP-ADPCM to 16-bit PCM and written to one file the host reads (host/oot_wheel.c, mixed
in by host/nn_audio.c). make_game.py runs it at the end of every build. It stays on this PC, made from the user's RomFS like the icons.

    python -I nx2/wheel_sounds.py --romfs <dir> --out <game folder>/wheel/sounds.bin [--wav <dir>] [--list]

sounds.bin: "HWSN", u32 version 1, u32 count; then for each sound: char name[48] (NUL-padded game name), u32 rate,
u32 frames, int16 coefficients[16] (its first channel's DSP-ADPCM ones: the host knows the game's own voice of the sound
by them), then frames x 2 int16 (stereo, interleaved; a mono sound is copied to both sides).
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_assets as pa  # noqa: E402

ARCHIVE = os.path.join("Sound", "Resource", "UI_Static.Product.111.bars.zs")
WANTED = ("SE_Sys_ShortCut_Open00", "SE_Sys_ShortCut_Close00", "SE_Sys_ShortCut_CursolMove_00", "SE_Sys_ShortCut_ItemSelect00")


def entries(bars):
    """{name: BWAV bytes} of a BARS archive (v1.2: header 0x10, u32 name hashes x count, then (AMTA, BWAV) offsets)"""
    if bars[:4] != b"BARS":
        raise ValueError("not a BARS archive")
    count, = struct.unpack_from("<I", bars, 0xc)
    table = 0x10 + 4 * count
    out = {}
    for i in range(count):
        amta, bwav = struct.unpack_from("<II", bars, table + 8 * i)
        if bars[amta:amta + 4] != b"AMTA" or bars[bwav:bwav + 4] != b"BWAV":
            continue
        size, = struct.unpack_from("<I", bars, amta + 8)
        name = bars[amta:amta + size].rstrip(b"\0").rsplit(b"\0", 1)[-1].decode("ascii", "replace")
        out[name] = bwav
    return out


def decode_bwav(b, at=0):
    """(rate, [channel samples], first channel's 16 coefficients) of a BWAV at b[at:] (u16 at 0xc: 1 = a prefetch, only its first frames): u16 codec (1 = DSP-ADPCM, 0 = PCM16) per channel info of 0x4c bytes
    from 0x10 (codec, pan, rate, frames, frames, 16 coefficients, data offset from the BWAV, ...)"""
    if b[at:at + 4] != b"BWAV":
        raise ValueError("not a BWAV")
    prefetch, channels = struct.unpack_from("<HH", b, at + 0xc)
    rate, chans, first = 0, [], None
    for ch in range(channels):
        info = at + 0x10 + 0x4c * ch
        codec, _pan, rate, frames, held = struct.unpack_from("<HHIII", b, info)
        if prefetch:   # (only the start is here, the rest is in Sound/Resource/Stream/<name>.bwav)
            frames = held
        coefs = struct.unpack_from("<16h", b, info + 0x10)
        first = first or coefs
        data, = struct.unpack_from("<I", b, info + 0x30)
        data += at
        if codec == 0:
            chans.append(list(struct.unpack_from("<%dh" % frames, b, data)))
            continue
        if codec != 1:
            raise ValueError("codec %d" % codec)
        out, h1, h2, p = [], 0, 0, data
        while len(out) < frames:
            ps = b[p]
            scale, pred = 1 << (ps & 0xf), (ps >> 4) & 7
            c1, c2 = coefs[2 * pred], coefs[2 * pred + 1]
            for k in range(14):
                if len(out) >= frames:
                    break
                byte = b[p + 1 + k // 2]
                nib = (byte >> 4) if k % 2 == 0 else (byte & 0xf)
                if nib >= 8:
                    nib -= 16
                s = ((nib * scale) << 11) + 1024 + c1 * h1 + c2 * h2
                s >>= 11
                s = -32768 if s < -32768 else 32767 if s > 32767 else s
                out.append(s)
                h2, h1 = h1, s
            p += 8
        chans.append(out)
    return rate, chans, first or (0,) * 16


def write_wav(path, rate, left, right):
    n = len(left)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + 4 * n) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 2, rate, rate * 4, 4, 16))
        f.write(b"data" + struct.pack("<I", 4 * n))
        f.write(b"".join(struct.pack("<hh", l, r) for l, r in zip(left, right)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--romfs", required=True)
    p.add_argument("--out")
    p.add_argument("--wav", help="also write each sound as a .wav here (to listen to)")
    p.add_argument("--list", action="store_true", help="list the archive's names")
    a = p.parse_args()
    bars = pa.read_zs(a.romfs, ARCHIVE, pa.zs_dictionary(a.romfs))
    found = entries(bars)
    if a.list:
        for n in sorted(found):
            print(n)
        return
    blobs = []
    for name in WANTED:
        if name not in found:
            print("wheel sounds: %s is not in %s" % (name, ARCHIVE))
            continue
        stream = os.path.join(a.romfs, "Sound", "Resource", "Stream", name + ".bwav")
        if struct.unpack_from("<H", bars, found[name] + 0xc)[0] and os.path.exists(stream):
            with open(stream, "rb") as f:
                rate, chans, coefs = decode_bwav(f.read())
        else:
            rate, chans, coefs = decode_bwav(bars, found[name])
        left, right = chans[0], chans[1] if len(chans) > 1 else chans[0]
        peak = max(max(abs(x) for x in left), 1)
        print("wheel sounds: %s, %d Hz, %d ch, %.3f s, peak %d" % (name, rate, len(chans), len(left) / rate, peak))
        if a.wav:
            os.makedirs(a.wav, exist_ok=True)
            write_wav(os.path.join(a.wav, name + ".wav"), rate, left, right)
        pcm = struct.pack("<%dh" % (2 * len(left)), *[s for pair in zip(left, right) for s in pair])
        blobs.append(name.encode().ljust(48, b"\0")[:48] + struct.pack("<II16h", rate, len(left), *coefs) + pcm)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "wb") as f:
            f.write(b"HWSN" + struct.pack("<II", 1, len(blobs)) + b"".join(blobs))
        print("wheel sounds: %d sounds -> %s" % (len(blobs), a.out))


if __name__ == "__main__":
    main()
