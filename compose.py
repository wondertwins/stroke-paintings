"""All five CEO paintings on one canvas: painted together, held, then a camera tour of each."""
import math, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

FPS = 30; OW, OH = 1920, 1200; K = 3                       # master canvas is K x the output
NF = 330                                                    # painting frames (11 s)
HOLD = 150; XF = 15; MOVE = 36; STAY = 66
BG = (20, 18, 18)                                           # BGR
N = {'podium': 5417, 'boardroom': 7767, 'walkout': 6411, 'mirror': 6741, 'stage': 6473}
h, wP, wL, g, mx, my = 582, 466, 932, 12, 16, 12
RECT = {'podium': (mx, my, wP, h), 'boardroom': (mx + wP + g, my, wL, h), 'mirror': (mx + wP + wL + 2 * g, my, wP, h),
        'walkout': (mx, my + h + g, wL, h), 'stage': (mx + wL + 2 * g, my + h + g, wL, h)}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def labels(img, counts, k):
    p = Image.fromarray(img[:, :, ::-1]); d = ImageDraw.Draw(p); f = ImageFont.truetype(FONT, 22 * k)
    for s, (x, y, w, hh) in RECT.items():
        d.text(((x + 12) * k, (y + hh - 38) * k), f'{counts[s]:,} strokes', font=f, fill=(255, 255, 255), stroke_width=2 * k, stroke_fill=(0, 0, 0))
    return np.ascontiguousarray(np.asarray(p)[:, :, ::-1])


def canvas(tiles, k):
    c = np.empty((OH * k, OW * k, 3), np.uint8); c[:] = BG
    for s, (x, y, w, hh) in RECT.items():
        c[y * k:(y + hh) * k, x * k:(x + w) * k] = cv2.resize(tiles[s], (w * k, hh * k), interpolation=cv2.INTER_AREA)
    return c


ease = lambda t: t * t * t * (t * (6 * t - 15) + 10)


def main(out):
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'slow', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    put = lambda fr: ff.stdin.write(fr.tobytes())
    last = None
    for i in range(NF + 1):                                 # 1. all five painted at once
        tiles = {s: cv2.imread(f'vid3/{s}/frames/f{i:04d}.png') for s in RECT}
        cnt = {s: int(round(N[s] * (i / NF) ** 1.25)) for s in RECT}
        last = labels(canvas(tiles, 1), cnt, 1); put(last)
    master = labels(canvas({s: cv2.imread(f'final/{s}/painting.png') for s in RECT}, K), N, K)
    half = cv2.resize(master, (OW * K // 2, OH * K // 2), interpolation=cv2.INTER_AREA)

    def view(cx, cy, cw):                                   # camera: centre and width in output units
        sc = OW / cw
        src, ks = (master, K) if sc * 2 > K else (half, K / 2)
        r = sc / ks
        M = np.float32([[r, 0, OW / 2 - cx * ks * r], [0, r, OH / 2 - cy * ks * r]])
        return cv2.warpAffine(src, M, (OW, OH), flags=cv2.INTER_AREA if r < 0.999 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
    full = (OW / 2, OH / 2, OW)
    still = view(*full)
    for i in range(HOLD):                                   # 2. hold on the finished canvas
        a = min(1.0, (i + 1) / XF)
        put(still if a >= 1 else cv2.addWeighted(last, 1 - a, still, a, 0))

    def shots(s):
        x, y, w, hh = RECT[s]
        if w > hh:
            return (x + w / 2, y + hh / 2, w * 1.05), (x + w / 2, y + hh / 2, w * 1.01)
        cw = w * 1.36; ch = cw * OH / OW; lo = y + ch / 2 - 6; hi = y + hh - ch / 2 + 6
        return (x + w / 2, lo + 0.10 * (hi - lo), cw), (x + w / 2, lo + 0.95 * (hi - lo), cw)
    cam = full
    for s in ('podium', 'boardroom', 'mirror', 'stage', 'walkout'):   # 3. visit each one
        a, b = shots(s)
        for i in range(MOVE):
            t = ease((i + 1) / MOVE)
            put(view(cam[0] + (a[0] - cam[0]) * t, cam[1] + (a[1] - cam[1]) * t, math.exp(math.log(cam[2]) + (math.log(a[2]) - math.log(cam[2])) * t)))
        for i in range(STAY):
            t = ease((i + 1) / STAY) if a[1] != b[1] else (i + 1) / STAY
            put(view(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))
        cam = b
    for i in range(MOVE + 9):                               # 4. back out to all five
        t = ease((i + 1) / (MOVE + 9))
        put(view(cam[0] + (full[0] - cam[0]) * t, cam[1] + (full[1] - cam[1]) * t, math.exp(math.log(cam[2]) + (math.log(full[2]) - math.log(cam[2])) * t)))
    for i in range(60):
        put(still)
    ff.stdin.close(); ff.wait()


if __name__ == '__main__':
    main(sys.argv[1])
