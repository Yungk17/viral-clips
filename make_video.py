#!/usr/bin/env python3
"""Beat-synced 9:16 cinematic music-video generator.

Usage: python make_video.py --audio assets/song.wav --photo assets/photo.jpg --out out/video.mp4

Only uniform crop/scale is applied to the source (no warping), so the face is never distorted.
"""
import argparse, subprocess, math, os
import numpy as np
import librosa
from PIL import Image

W, H, FPS = 1080, 1920, 30


def ease(x):  # smoothstep
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def analyze(audio, start_hint, length):
    y, sr = librosa.load(audio, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats)
    # snap start to a beat; downbeat grid = every 4th beat from that start
    i0 = int(np.argmin(np.abs(beats - start_hint)))
    start = float(beats[i0])
    n = int(length / (60 / tempo))
    n -= n % 4
    sel = beats[i0:i0 + n + 1]
    return tempo, start, sel - start  # beat times relative to start


# (cx, cy, zoom) in source-fraction coords; zoom=1 is full frame. y=0 top.
SHOTS = {
    'wide':    (0.50, 0.50, 1.00),
    'face':    (0.24, 0.13, 2.30),
    'chain':   (0.24, 0.27, 1.90),
    'hands':   (0.20, 0.42, 1.80),
    'flowers': (0.80, 0.37, 1.65),
    'stand':   (0.76, 0.72, 1.60),
    'shoes':   (0.30, 0.90, 1.90),
    'torso':   (0.28, 0.33, 1.45),
    'shadow':  (0.55, 0.22, 1.50),
}
# (shot, beats, push-in direction +1 in / -1 out, extra drift x)
PLAN = [
    ('wide', 2, +1), ('face', 2, +1), ('chain', 2, +1), ('hands', 2, +1),
    ('flowers', 2, -1), ('torso', 2, +1),                      # bars 1-3 (build)
    ('face', 1, +1), ('chain', 1, +1), ('hands', 1, +1), ('face', 1, -1),  # bar 4 fast
    ('wide', 4, +1),                                           # drop hold
    ('shoes', 2, -1), ('stand', 2, +1), ('face', 2, +1), ('shadow', 2, -1),
    ('chain', 1, +1), ('face', 1, +1), ('hands', 2, -1),
    ('flowers', 2, +1), ('torso', 2, -1), ('face', 4, +1),
    ('wide', 2, -1), ('chain', 2, +1), ('face', 2, +1), ('wide', 4, +1),
]


def build_timeline(beats, tempo):
    spb = 60 / tempo
    shots, b = [], 0
    for name, nb, d in PLAN:
        if b >= len(beats) - 1:
            break
        nb = min(nb, len(beats) - 1 - b)
        shots.append(dict(name=name, t0=beats[b], t1=beats[b + nb], d=d, b0=b, nb=nb))
        b += nb
    # make last shot run to the end
    if b < len(beats) - 1:
        shots[-1]['t1'] = beats[-1]; shots[-1]['nb'] += len(beats) - 1 - b
    return shots, spb


def make_src(photo):
    im = Image.open(photo).convert('RGB')
    return im.resize((im.width * 2, im.height * 2), Image.LANCZOS)


def crop_frame(src, cx, cy, zoom, shift=(0, 0)):
    sw, sh = src.size
    cw, ch = sw / zoom, sh / zoom
    x = cx * sw - cw / 2 + shift[0] * sw
    y = cy * sh - ch / 2 + shift[1] * sh
    x = min(max(x, 0), sw - cw); y = min(max(y, 0), sh - ch)
    return src.resize((W, H), Image.LANCZOS, box=(x, y, x + cw, y + ch))


def grade(a, vig, rng, t):
    """a: float32 HxWx3 0..1. Cinematic look: contrast S-curve, warm highlights / cool shadows, vignette, grain."""
    lum = a.mean(axis=2, keepdims=True)
    lum = lum * lum * (3 - 2 * lum) * 0.55 + lum * 0.45          # gentle S-curve
    lum = np.clip((lum - 0.03) * 1.06, 0, 1)
    shadows = np.array([0.93, 1.00, 1.08], np.float32)             # cool
    highs = np.array([1.10, 1.02, 0.88], np.float32)               # warm gold
    k = lum
    tint = shadows * (1 - k) + highs * k
    out = lum * tint
    out = out * vig
    g = rng.standard_normal((H // 2, W // 2, 1)).astype(np.float32) * 0.022
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)
    out = out + g * (1 - np.abs(out - 0.5) * 1.2)
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', default='assets/song.wav')
    ap.add_argument('--photo', default='assets/photo.jpg')
    ap.add_argument('--out', default='out/promo.mp4')
    ap.add_argument('--start', type=float, default=22.6, help='approx start (s), snapped to a beat')
    ap.add_argument('--length', type=float, default=25.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

    tempo, start, beats = analyze(a.audio, a.start, a.length)
    shots, spb = build_timeline(beats, tempo)
    dur = beats[-1]
    nframes = int(dur * FPS)
    print(f'tempo {tempo:.1f}  start {start:.2f}s  dur {dur:.2f}s  shots {len(shots)}')

    src = make_src(a.photo)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = (1 - 0.55 * np.clip(d - 0.35, 0, 1) ** 1.6)[..., None].astype(np.float32)
    rng = np.random.default_rng(7)
    beat_arr = np.asarray(beats)

    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', a.audio,
        '-map', '0:v', '-map', '1:a',
        '-af', f'afade=t=in:d=0.15,afade=t=out:st={dur - 1.0:.3f}:d=1.0',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-maxrate', '14M', '-bufsize', '28M', '-pix_fmt', 'yuv420p',
        '-profile:v', 'high', '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '256k', '-shortest', a.out], stdin=subprocess.PIPE)

    def shot_frame(s, t, speed_ramp):
        cx, cy, z = SHOTS[s['name']]
        p = (t - s['t0']) / max(s['t1'] - s['t0'], 1e-6)
        p = ease(p) * 0.5 + p * 0.5                        # soft ease -> slow-motion feel at ends
        span = 0.10 + 0.04 * min(s['nb'], 4) / 4          # push amount
        zz = z * (1 + span * (p if s['d'] > 0 else 1 - p))
        drift = (0.012 * (p - 0.5)) * (1 if s['b0'] % 2 else -1)
        # beat punch: exponential decay after nearest previous beat
        pb = beat_arr[beat_arr <= t].max() if (beat_arr <= t).any() else 0
        punch = math.exp(-(t - pb) / (spb * 0.22)) * 0.035
        down = (int(np.searchsorted(beat_arr, pb)) % 4 == 0)
        zz *= 1 + punch * (1.7 if down else 1.0)
        return crop_frame(src, cx, cy, zz, (drift, 0))

    for f in range(nframes):
        t = f / FPS
        idx = next(i for i, s in enumerate(shots) if t < s['t1'] or i == len(shots) - 1)
        s = shots[idx]
        im = shot_frame(s, t, None)
        arr = np.asarray(im, np.float32) / 255
        # transition at start of shot: crossfade (on phrase starts) or whip-zoom blur
        tt = t - s['t0']
        if idx > 0 and tt < 0.20:
            prev = shot_frame(shots[idx - 1], min(t, shots[idx - 1]['t1']), None)
            pa = np.asarray(prev, np.float32) / 255
            if s['b0'] % 8 == 0:                          # smooth dissolve on phrase boundaries
                m = ease(tt / 0.20)
                arr = pa * (1 - m) + arr * m
            else:                                         # quick punch-through blend
                m = ease(tt / 0.08)
                arr = pa * (1 - m) + arr * m
        out = grade(arr, vig, rng, t)
        # flash on downbeat cuts
        if idx > 0 and s['b0'] % 4 == 0:
            fl = math.exp(-tt / 0.07) * (0.55 if s['b0'] % 16 == 0 else 0.28)
            out = out + (1 - out) * fl * np.array([1.0, 0.96, 0.88], np.float32)
        # fade in / out
        out *= min(1, t / 0.25) * min(1, (dur - t) / 0.6)
        ff.stdin.write((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    ff.stdin.close(); ff.wait()
    print('wrote', a.out)


if __name__ == '__main__':
    main()
