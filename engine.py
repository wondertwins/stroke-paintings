"""Stroke engine shared by the original (no reference image) paintings.

Strokes are polylines with a width, colour, opacity, dryness and taper; they are rasterised one by one
with bristle texture onto a blank canvas, in the order they were added.
"""
import argparse, math, os, random
import numpy as np, cv2
from numba import njit
from PIL import Image, ImageDraw, ImageFont

W, H = 3000, 1875
S = float(H)            # scene units: y in [0,1], x in [0,AR]
AR = W / H
MAXP = 160
DENS = 1.0
R = random.Random(20261112)
g = R.gauss; U_ = R.uniform


def setup(w, h, seed=20261112, dens=1.0):
    global W, H, S, AR, DENS
    W, H = w, h; S = float(h); AR = w / h; DENS = dens
    R.seed(seed); _hal[0] = 0
    del strokes[:]; del layer_marks[:]

_lat = np.random.RandomState(5).rand(256, 256)


@njit(cache=True)
def _vn(x, y, lat):
    xi = math.floor(x); yi = math.floor(y)
    fx = x - xi; fy = y - yi
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    i0 = int(xi) & 255; i1 = (i0 + 1) & 255; j0 = int(yi) & 255; j1 = (j0 + 1) & 255
    a = lat[j0, i0] * (1 - fx) + lat[j0, i1] * fx
    b = lat[j1, i0] * (1 - fx) + lat[j1, i1] * fx
    return a * (1 - fy) + b * fy


@njit(cache=True)
def _fbm(x, y, lat, octv):
    s = 0.0; amp = 0.5; tot = 0.0; f = 1.0
    for _ in range(octv):
        s += amp * _vn(x * f + 17.3 * f, y * f + 5.1 * f, lat); tot += amp
        amp *= 0.5; f *= 2.0
    return s / tot


def N2(x, y, o=4):
    return _fbm(x, y, _lat, o)


@njit(cache=True, fastmath=True)
def render(can, hgt, A, U, T, P, NP, WD, COL, OP, DRY, T0, T1, CM, i0, i1):
    Hh, Ww, _ = can.shape
    cum = np.empty(MAXP)
    bris = np.empty(512)
    for i in range(i0, i1):
        M = NP[i]; r = 0.5 * WD[i]
        px = P[i, :, 0]; py = P[i, :, 1]
        cum[0] = 0.0
        for k in range(1, M):
            cum[k] = cum[k - 1] + math.sqrt((px[k] - px[k - 1]) ** 2 + (py[k] - py[k - 1]) ** 2)
        Ltot = cum[M - 1]
        if Ltot < 1e-3:
            continue
        t0 = T0[i]; t1 = T1[i]
        gx0 = Ww; gy0 = Hh; gx1 = -1; gy1 = -1
        for k in range(M - 1):
            ax = px[k]; ay = py[k]; vx = px[k + 1] - ax; vy = py[k + 1] - ay
            l2 = vx * vx + vy * vy
            if l2 < 1e-9:
                continue
            sl = math.sqrt(l2)
            bx0 = int(math.floor(min(ax, ax + vx) - r - 1.5)); bx1 = int(math.ceil(max(ax, ax + vx) + r + 1.5))
            by0 = int(math.floor(min(ay, ay + vy) - r - 1.5)); by1 = int(math.ceil(max(ay, ay + vy) + r + 1.5))
            if bx0 < 0: bx0 = 0
            if by0 < 0: by0 = 0
            if bx1 > Ww - 1: bx1 = Ww - 1
            if by1 > Hh - 1: by1 = Hh - 1
            if bx1 < bx0 or by1 < by0:
                continue
            if bx0 < gx0: gx0 = bx0
            if by0 < gy0: gy0 = by0
            if bx1 > gx1: gx1 = bx1
            if by1 > gy1: gy1 = by1
            for y in range(by0, by1 + 1):
                fyy = y + 0.5 - ay
                for x in range(bx0, bx1 + 1):
                    fxx = x + 0.5 - ax
                    t = (fxx * vx + fyy * vy) / l2
                    if t < 0.0: t = 0.0
                    elif t > 1.0: t = 1.0
                    ddx = fxx - t * vx; ddy = fyy - t * vy
                    d = math.sqrt(ddx * ddx + ddy * ddy)
                    if d > r + 0.5:
                        continue
                    arc = (cum[k] + t * sl) / Ltot
                    s1 = arc / 0.12
                    if s1 > 1.0: s1 = 1.0
                    s1 = s1 * s1 * (3 - 2 * s1)
                    s2 = (1.0 - arc) / 0.30
                    if s2 > 1.0: s2 = 1.0
                    s2 = s2 * s2 * (3 - 2 * s2)
                    reff = r * (t0 + (1 - t0) * s1) * (t1 + (1 - t1) * s2)
                    a = reff + 0.5 - d
                    if a <= 0.0:
                        continue
                    if a > 1.0: a = 1.0
                    u = d / (reff + 1e-6)
                    if u > 1.0: u = 1.0
                    if vx * fyy - vy * fxx < 0.0: u = -u
                    cur = A[y, x]
                    if a > cur + 1e-6 or (a >= cur - 1e-6 and abs(u) < abs(U[y, x])):
                        A[y, x] = a; U[y, x] = u; T[y, x] = arc
        if gx1 < 0:
            continue
        w = WD[i]
        nbr = int(w * 0.55) + 4
        if nbr > 500: nbr = 500
        prev = np.random.rand()
        for k in range(nbr):
            prev = 0.45 * prev + 0.55 * np.random.rand()
            bris[k] = prev
        mn = 1.0; mx = 0.0; bm = 0.0
        for k in range(nbr):
            if bris[k] < mn: mn = bris[k]
            if bris[k] > mx: mx = bris[k]
        for k in range(nbr):
            bris[k] = (bris[k] - mn) / (mx - mn + 1e-6); bm += bris[k]
        bm /= nbr
        opac = OP[i]; dry = DRY[i]; cmod = CM[i]
        tk = math.sqrt(w / 10.0)
        if tk > 1.6: tk = 1.6
        if tk < 0.45: tk = 0.45
        for y in range(gy0, gy1 + 1):
            for x in range(gx0, gx1 + 1):
                a = A[y, x]
                if a <= 0.0:
                    continue
                fpos = (U[y, x] + 1.0) * 0.5 * (nbr - 1)
                j = int(fpos)
                if j > nbr - 2: j = nbr - 2
                fr = fpos - j
                bv = bris[j] * (1 - fr) + bris[j + 1] * fr
                thr = (T[y, x] - 0.45) * 1.5 * dry
                if thr > 0.0:
                    gg = (bv - thr) / 0.18 + 1.0
                    if gg < 0.0: gg = 0.0
                    if gg < 1.0: a *= gg
                a *= opac * (0.9 + 0.1 * bv)
                m = 1.0 + cmod * (bv - bm)
                for c in range(3):
                    nv = can[y, x, c] + a * (COL[i, c] * m - can[y, x, c])
                    if nv < 0.0: nv = 0.0
                    if nv > 1.0: nv = 1.0
                    can[y, x, c] = nv
                hgt[y, x] += a * (tk * (0.35 + 0.65 * bv) - hgt[y, x])
                A[y, x] = 0.0; U[y, x] = 0.0


# ----------------------------------------------------------------------------- helpers
strokes = []
layer_marks = []


def add(pts, w, col, opac=0.95, dry=0.3, t0=0.62, t1=0.42, cmod=0.07):
    if len(pts) < 2:
        return
    if len(pts) > MAXP:
        pts = pts[:MAXP]
    strokes.append((pts, float(w), (min(max(col[0], 0), 1), min(max(col[1], 0), 1), min(max(col[2], 0), 1)),
                    opac, dry, t0, t1, cmod))


def layer(name):
    layer_marks.append((name, len(strokes)))


def mix(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def jit(c, a=0.025, l=0.05):
    k = 1 + l * g(0, 1)
    return (c[0] * k + a * g(0, 1), c[1] * k + a * g(0, 1), c[2] * k + a * g(0, 1))


def sstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(stops, t):
    if t <= stops[0][0]: return stops[0][1]
    for k in range(1, len(stops)):
        if t <= stops[k][0]:
            a, b = stops[k - 1], stops[k]
            return mix(a[1], b[1], (t - a[0]) / (b[0] - a[0]))
    return stops[-1][1]


def pl(points, x):
    if x <= points[0][0]: return points[0][1]
    for k in range(1, len(points)):
        if x <= points[k][0]:
            a, b = points[k - 1], points[k]
            return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])
    return points[-1][1]


def halton(i, b):
    f = 1.0; r = 0.0
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r


_hal = [0]


def hpt(x0, x1, y0, y1):
    _hal[0] += 1
    i = _hal[0]
    return x0 + (x1 - x0) * halton(i, 2), y0 + (y1 - y0) * halton(i, 3)


def trace(x, y, angfn, length, step, inside=None, both=True, smooth=0.55):
    def go(sgn):
        out = []
        a = angfn(x, y)
        dx, dy = math.cos(a) * sgn, math.sin(a) * sgn
        qx, qy = x, y
        n = int((length / 2 if both else length) / step)
        for _ in range(n):
            a = angfn(qx, qy)
            ex, ey = math.cos(a), math.sin(a)
            if ex * dx + ey * dy < 0: ex, ey = -ex, -ey
            ex = smooth * ex + (1 - smooth) * dx; ey = smooth * ey + (1 - smooth) * dy
            nn = math.hypot(ex, ey) + 1e-9
            ex /= nn; ey /= nn
            nx, ny = qx + ex * step, qy + ey * step
            if inside is not None and not inside(nx, ny): break
            if not (-0.1 < nx < AR + 0.1 and -0.1 < ny < 1.1): break
            out.append((nx, ny)); qx, qy, dx, dy = nx, ny, ex, ey
        return out
    f = go(1.0)
    b = go(-1.0) if both else []
    pts = b[::-1] + [(x, y)] + f
    if R.random() < 0.5 and both:
        pts = pts[::-1]
    return pts


def seg(x0, y0, x1, y1, n=2):
    return [(x0 + (x1 - x0) * k / (n - 1), y0 + (y1 - y0) * k / (n - 1)) for k in range(n)]


def arc(cx, cy, r, a0, a1, n=None):
    n = n or max(4, int(abs(a1 - a0) * r * S / 10) + 2)
    n = min(n, MAXP)
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / (n - 1)), cy + r * math.sin(a0 + (a1 - a0) * k / (n - 1))) for k in range(n)]


def px(w):
    return w / S



def sd_ell(x, y, cx, cy, a, b):
    return (math.hypot((x - cx) / a, (y - cy) / b) - 1.0) * min(a, b)


def sd_cap(x, y, ax, ay, bx, by, r):
    vx = bx - ax; vy = by - ay; wx = x - ax; wy = y - ay
    t = min(max((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-12), 0.0), 1.0)
    return math.hypot(wx - t * vx, wy - t * vy) - r


def sd_box(x, y, cx, cy, hx, hy, r=0.0):
    dx = abs(x - cx) - hx + r; dy = abs(y - cy) - hy + r
    return math.hypot(max(dx, 0.0), max(dy, 0.0)) + min(max(dx, dy), 0.0) - r


def sd_rbox(x, y, cx, cy, hx, hy, rot, r=0.0):
    c = math.cos(rot); s = math.sin(rot)
    lx = (x - cx) * c + (y - cy) * s; ly = -(x - cx) * s + (y - cy) * c
    return sd_box(lx, ly, 0.0, 0.0, hx, hy, r)


def sd_poly(x, y, P):
    d = 1e9; s = 1.0; n = len(P); j = n - 1
    for i in range(n):
        ex = P[j][0] - P[i][0]; ey = P[j][1] - P[i][1]; wx = x - P[i][0]; wy = y - P[i][1]
        t = min(max((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0), 1.0)
        bx = wx - ex * t; by = wy - ey * t
        d = min(d, bx * bx + by * by)
        c1 = y >= P[i][1]; c2 = y < P[j][1]; c3 = ex * wy > ey * wx
        if (c1 and c2 and c3) or ((not c1) and (not c2) and (not c3)): s = -s
        j = i
    return s * math.sqrt(d)


def tex(c, x, y, a=0.22, f=30.0):
    k = 1 - a / 2 + a * N2(x * f, y * f, 3)
    return (c[0] * k, c[1] * k, c[2] * k)


def block(sd, bbox, colfn, w, a, margin=0.3, seglen=(3.0, 6.0), jc=(0.008, 0.03)):
    """Block a shape in solidly: rows of overlapping strokes, laid in random order."""
    x0, y0, x1, y1 = bbox
    cxb = (x0 + x1) / 2; cyb = (y0 + y1) / 2; Rb = 0.5 * math.hypot(x1 - x0, y1 - y0)
    wu = px(w); m = wu * margin; st = wu * 0.4
    dx, dy = math.cos(a), math.sin(a); nx, ny = -dy, dx
    out = []
    o = -Rb
    while o <= Rb:
        oo = o + wu * 0.08 * g(0, 1)
        run = []
        t = -Rb
        while t <= Rb + st:
            x = cxb + oo * nx + t * dx; y = cyb + oo * ny + t * dy
            ok = t <= Rb and (x0 - wu <= x <= x1 + wu) and (y0 - wu <= y <= y1 + wu) and sd(x, y) < -m
            if ok: run.append((x, y))
            if (not ok) and run:
                i = 0
                while i < len(run):
                    L = max(2, int(U_(seglen[0], seglen[1]) / 0.4))
                    ch = run[i:i + L + 1]
                    if len(ch) == 1:
                        ch = [(ch[0][0] - dx * st * 0.5, ch[0][1] - dy * st * 0.5), (ch[0][0] + dx * st * 0.5, ch[0][1] + dy * st * 0.5)]
                    out.append(ch); i += L
                run = []
            t += st
        o += wu * 0.55
    R.shuffle(out)
    for ch in out:
        if R.random() < 0.5: ch = ch[::-1]
        mid = ch[len(ch) // 2]
        add(ch[:MAXP], w * U_(0.92, 1.1), jit(colfn(mid[0], mid[1]), jc[0], jc[1]), 0.985, 0.12, 0.85, 0.8)


def fill(sd, bbox, colfn, n, w, ang=0.0, lmul=(3, 6), margin=0.35, opac=0.97, dry=0.25, t0=0.62, t1=0.42,
         jc=(0.012, 0.04), aj=0.12, flow=False, dens=True, base=None, bang=None, seglen=(3.0, 6.0)):
    x0, y0, x1, y1 = bbox
    if base is None: base = w[1]
    if base:
        block(sd, bbox, colfn, base, bang if bang is not None else (math.pi / 2 if callable(ang) else ang), min(margin, 0.3), seglen=seglen)
    n = int(n * DENS) if dens else n
    k = 0; tries = 0
    while k < n and tries < n * 80 + 300:
        tries += 1
        x, y = hpt(x0, x1, y0, y1)
        ww = U_(w[0], w[1]); m = px(ww) * margin
        if sd(x, y) > -m: continue
        st = max(px(ww) * 0.5, px(2.0))
        ins = lambda a, b: sd(a, b) < -m
        if flow:
            pts = trace(x, y, ang, px(ww) * U_(lmul[0], lmul[1]), st, inside=ins)
            a0 = ang(x, y)
        else:
            a0 = (ang(x, y) if callable(ang) else ang) + aj * g(0, 1)
            pts = trace(x, y, lambda a, b: a0, px(ww) * U_(lmul[0], lmul[1]), st, inside=ins)
        if len(pts) < 2:
            pts = [(x - 0.5 * st * math.cos(a0), y - 0.5 * st * math.sin(a0)), (x + 0.5 * st * math.cos(a0), y + 0.5 * st * math.sin(a0))]
        add(pts, ww, jit(colfn(x, y), jc[0], jc[1]), opac, dry, t0, t1)
        k += 1


def rim(sd, bbox, n, colfn, w=(4, 7), light=None, thresh=0.15, inset=0.004, ln=(0.02, 0.06), opac=0.95, band=0.03,
        ymax=None, ymin=None, tol=0.008):
    x0, y0, x1, y1 = bbox
    k = 0; tries = 0; e = 0.002

    def grad(a, b):
        gx = sd(a + e, b) - sd(a - e, b); gy = sd(a, b + e) - sd(a, b - e)
        gn = math.hypot(gx, gy) + 1e-12
        return gx / gn, gy / gn
    while k < n and tries < n * 300 + 300:
        tries += 1
        x = U_(x0, x1); y = U_(y0, y1)
        d = sd(x, y)
        if abs(d) > band: continue
        gx, gy = grad(x, y)
        x -= (d + inset) * gx; y -= (d + inset) * gy
        if ymax is not None and y > ymax: continue
        if ymin is not None and y < ymin: continue
        s = 1.0
        if light is not None:
            lx = light[0] - x; ly = light[1] - y; ll = math.hypot(lx, ly) + 1e-9
            dot = (gx * lx + gy * ly) / ll
            if dot < thresh: continue
            s = sstep((dot - thresh) / (1 - thresh))
            if R.random() > 0.2 + 0.8 * s: continue

        def ang(a, b):
            hx, hy = grad(a, b)
            return math.atan2(hx, -hy)
        pts = trace(x, y, ang, U_(ln[0], ln[1]), px(3), inside=lambda a, b: abs(sd(a, b) + inset) < tol)
        if len(pts) < 2: continue
        add(pts, U_(w[0], w[1]), jit(colfn(x, y, s), 0.01, 0.03), opac, 0.3, 0.3, 0.3)
        k += 1


def hband(y0, y1, x0, x1, colfn, n, w, ln=(0.06, 0.2), opac=0.97, dry=0.25, aj=0.01):
    for _ in range(int(n * DENS)):
        ww = U_(w[0], w[1]); r = px(ww) * 0.5
        y = U_(y0 + r * 0.7, y1 - r * 0.7) if (y1 - y0) > 1.4 * r else (y0 + y1) / 2
        x = U_(x0, x1); L = U_(ln[0], ln[1]); a = aj * g(0, 1)
        add(seg(x - L / 2, y - L / 2 * math.sin(a), x + L / 2, y + L / 2 * math.sin(a), max(3, int(L * S / 20))), ww,
            jit(colfn(x, y), 0.012, 0.04), opac, dry, 0.7, 0.6)


def line(pts, w, col, opac=1.0, dry=0.0, t0=0.9, t1=0.9):
    add(list(pts), w, col, opac, dry, t0, t1)


def dot(x, y, w, col, opac=1.0):
    add([(x - px(w) * 0.15, y), (x + px(w) * 0.15, y)], w, col, opac, 0.0, 0.95, 0.95)


def relief(can, hgt, strength=0.3):
    h = cv2.GaussianBlur(hgt, (0, 0), 0.8)
    hx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    hy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    sh = np.clip(-(hx * 0.7 + hy * 0.7) * strength, -0.12, 0.12)
    return np.clip(can * (1.0 + sh[..., None]) + np.maximum(sh, 0)[..., None] * 0.15, 0, 1)


def to_u8(a):
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def finish(out, frames=0, scale=1.0, ground=(0.93, 0.905, 0.85), relief_k=0.3, quiet=False, label=True):
    os.makedirs(out + '/frames', exist_ok=True)
    layer('end')
    N = len(strokes)
    if not quiet:
        for k in range(len(layer_marks) - 1):
            print(f'{layer_marks[k][0]:>34}: {layer_marks[k + 1][1] - layer_marks[k][1]}')
    print('TOTAL STROKES', N)
    w = int(round(W * scale)); h = int(round(H * scale)); sc = S * scale
    P = np.zeros((N, MAXP, 2), np.float32); NP = np.zeros(N, np.int64)
    WD = np.zeros(N); COL = np.zeros((N, 3)); OP = np.zeros(N); DRY = np.zeros(N); T0 = np.zeros(N); T1 = np.zeros(N); CM = np.zeros(N)
    for i, (pts, wd, c, op, dr, t0, t1, cm) in enumerate(strokes):
        NP[i] = len(pts); P[i, :len(pts)] = np.array(pts, np.float32) * sc
        WD[i] = wd * scale; COL[i] = c; OP[i] = op; DRY[i] = dr; T0[i] = t0; T1[i] = t1; CM[i] = cm
    can = np.empty((h, w, 3), np.float32); can[:] = np.array(ground, np.float32)
    hgt = np.zeros((h, w), np.float32)
    A = np.zeros((h, w), np.float32); Ub = np.zeros((h, w), np.float32); Tb = np.zeros((h, w), np.float32)
    marks = sorted(set([0] + [int(round(N * (k / frames) ** 1.25)) for k in range(1, frames + 1)])) if frames else [0, N]
    fw = 1440 if w >= h else 1080; fh = round(h * fw / w); fh += fh % 2
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)

    def frame(idx, n):
        small = cv2.resize(to_u8(can), (fw, fh), interpolation=cv2.INTER_AREA)
        p = Image.fromarray(small); d = ImageDraw.Draw(p)
        if label:
            d.text((30, fh - 64), f'{n:,} strokes', font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        p.save(f'{out}/frames/f{idx:04d}.png', compress_level=1)
    if frames: frame(0, 0)
    for k in range(1, len(marks)):
        render(can, hgt, A, Ub, Tb, P, NP, WD, COL, OP, DRY, T0, T1, CM, marks[k - 1], marks[k])
        if frames: frame(k, marks[k])
    fin = to_u8(relief(can, hgt, relief_k))
    Image.fromarray(fin).save(out + '/painting.png')
    Image.fromarray(fin).save(out + '/painting.jpg', quality=95, subsampling=0)
    sw = 1400 if w >= h else 1000
    Image.fromarray(fin).resize((sw, round(sw * h / w)), Image.LANCZOS).save(out + '/small.png')
    return N
