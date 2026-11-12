"""An original painting, no reference image: 'Lighthouse at dusk'.

Every brush stroke (path, width, colour) is decided directly from a composition written in code and
laid down back-to-front like a painter would: sky, clouds, stars, moon, hills, sea, cliff, lighthouse,
boat, foreground grass. There is no target picture and no correction step - the strokes are the painting.
"""
import argparse, math, os, random
import numpy as np, cv2
from numba import njit
from PIL import Image, ImageDraw, ImageFont

W, H = 3000, 2250
S = float(H)            # scene units: y in [0,1], x in [0,AR]
AR = W / H
MAXP = 160
R = random.Random(20261110)
g = R.gauss; U_ = R.uniform

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
            if not (-0.05 < nx < AR + 0.05 and -0.05 < ny < 1.05): break
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


# ----------------------------------------------------------------------------- the scene
HZ = 0.60
SUN = (0.40, HZ + 0.01)
MOON = (0.27, 0.175); MOONR = 0.046
LH = 1.035
SKY = [(0.0, (0.045, 0.06, 0.21)), (0.28, (0.10, 0.15, 0.40)), (0.52, (0.30, 0.25, 0.55)), (0.70, (0.70, 0.36, 0.50)),
       (0.85, (0.97, 0.55, 0.30)), (1.0, (1.0, 0.82, 0.46))]
VORT = [(0.72, 0.20, 0.12, 0.18), (1.24, 0.11, 0.075, 0.10), (MOON[0], MOON[1], 0.10, 0.14)]
CLIFF = [(0.56, 1.02), (0.667, 0.93), (0.75, 0.80), (0.80, 0.66), (0.853, 0.585), (0.93, 0.555), (1.07, 0.537), (1.2, 0.527), (1.34, 0.517)]
GRASS = [(0.76, 1.02), (0.90, 0.905), (1.05, 0.845), (1.2, 0.805), (1.34, 0.775)]


def sky_t(x, y):
    t = y / HZ
    d = math.hypot((x - SUN[0]) * 0.72, y - SUN[1])
    gl = math.exp(-(d / 0.33) ** 2)
    far = min(1.0, abs(x - SUN[0]) / 0.95)
    return t * (1 - 0.30 * far) + 0.24 * gl - 0.05 + 0.05 * (N2(x * 2.5, y * 2.5) - 0.5)


def sky(x, y):
    return ramp(SKY, sky_t(x, y))


def psi(x, y):
    v = y + 0.035 * math.sin(2.2 * x + 0.6) + 0.018 * math.sin(5.1 * x + 1.7 + 3 * y)
    fade = sstep((HZ - y) / 0.22)
    for (cx, cy, s, gm) in VORT:
        v += gm * fade * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (s * s))
    return v


def sky_ang(x, y):
    e = 0.004
    px_ = (psi(x + e, y) - psi(x - e, y)); py_ = (psi(x, y + e) - psi(x, y - e))
    return math.atan2(-px_, py_)


def band(x, y):
    return math.sin(2 * math.pi * psi(x, y) * 17 + 11 * N2(x * 2.2, y * 2.2)) * (0.45 + 1.1 * N2(x * 4 + 9, y * 4 + 2, 2))


def cliff_top(x):
    return pl(CLIFF, x) + 0.010 * (N2(x * 16, 3.3, 3) - 0.5) * 2


def grass_top(x):
    return pl(GRASS, x) + 0.012 * (N2(x * 9, 8.8, 3) - 0.5) * 2


def in_cliff(x, y): return y > cliff_top(x)
def in_grass(x, y): return y > grass_top(x)
def ridge_a(x): return HZ - 0.018 - 0.05 * N2(x * 2.3 + 7, 1.5, 3) * (0.5 + 0.5 * sstep((x - 0.15) / 0.5))
def ridge_b(x): return HZ - 0.105 * sstep(1 - x / 0.36) ** 1.3 - 0.012 * (N2(x * 14, 4.1, 3) - 0.3) * sstep(1 - x / 0.40)


def sea(x, y):
    d = (y - HZ) / (1 - HZ)
    ys = max(0.03, HZ - (y - HZ) * 1.05)
    sc = sky(x + 0.02 * math.sin(60 * y), ys)
    deep = (0.035, 0.12, 0.21)
    c = mix((sc[0] * 0.62, sc[1] * 0.62, sc[2] * 0.66), deep, 0.22 + 0.55 * d)
    n = N2(x * 5, y * 45)
    k = 0.72 + 0.56 * n
    return (c[0] * k, c[1] * k, c[2] * k)


def rock(x, y):
    dl = max(0.0, y - pl(CLIFF, x))
    sm = sstep((0.92 - x) / 0.14)
    lit = math.exp(-dl / (0.022 + 0.20 * sm))
    n = N2(x * 5 + y * 3, y * 20 - x * 7)
    f = N2(x * 21 + 3, y * 21 + 9, 3)
    Lg = min(1.0, max(0.0, lit * (0.25 + 1.5 * f) - 0.15 * (1 - sm)))
    sh = mix((0.11, 0.075, 0.16), (0.40, 0.25, 0.37), n ** 1.3)
    sh = mix(sh, (0.15, 0.20, 0.40), 0.32 * f)
    ledge = sstep((N2(x * 4 + y * 2 + 5, y * 30 - x * 9) - 0.60) * 7)
    sh = mix(sh, (0.74, 0.42, 0.40), 0.55 * ledge)
    k = 1 - 0.40 * sstep((y - 0.62) / 0.28)
    sh = (sh[0] * k, sh[1] * k, sh[2] * k)
    lt = mix((0.74, 0.36, 0.28), (1.0, 0.68, 0.42), f)
    return mix(sh, lt, Lg ** 1.15)


# ----------------------------------------------------------------------------- painting
def paint_sky():
    def sky_region(x, y):
        return y < HZ + 0.015 and not (y > pl(CLIFF, x) + 0.05)
    layer('sky block-in')
    n = 0
    while n < 170:
        x, y = hpt(-0.02, AR + 0.02, -0.02, HZ + 0.02)
        if not sky_region(x, y): continue
        w = U_(120, 175)
        add(trace(x, y, sky_ang, px(w) * U_(4, 7), px(w) * 0.5), w, jit(sky(x, y), 0.012, 0.03), opac=0.97, dry=0.25)
        n += 1
    layer('sky')
    n = 0
    while n < 950:
        x, y = hpt(-0.02, AR + 0.02, -0.02, HZ + 0.02)
        if not sky_region(x, y): continue
        w = U_(40, 78)
        b = band(x, y); c = sky(x, y); t = sky_t(x, y)
        c = mix(c, (0.40, 0.62, 0.90) if t < 0.5 else (1.0, 0.90, 0.66), 0.16 * max(b, 0))
        c = mix(c, (0.02, 0.03, 0.15) if t < 0.5 else (0.55, 0.20, 0.42), 0.16 * max(-b, 0))
        add(trace(x, y, sky_ang, px(w) * U_(5, 10), px(w) * 0.5), w, jit(c, 0.02, 0.04), opac=U_(0.8, 0.97), dry=U_(0.2, 0.7))
        n += 1
    layer('sky swirls')
    n = 0
    while n < 2300:
        x, y = hpt(-0.02, AR + 0.02, -0.02, HZ + 0.01)
        if not sky_region(x, y): continue
        b = band(x, y); t = sky_t(x, y); c = sky(x, y)
        if abs(b) < 0.25 and R.random() < 0.6: continue
        w = U_(15, 32)
        if b > 0:
            tint = (0.50, 0.78, 0.98) if t < 0.42 else ((0.86, 0.70, 0.92) if t < 0.62 else (1.0, 0.93, 0.70))
            c = mix(c, tint, 0.42 * b + 0.05)
        else:
            tint = (0.02, 0.02, 0.13) if t < 0.42 else ((0.22, 0.12, 0.42) if t < 0.62 else (0.80, 0.26, 0.30))
            c = mix(c, tint, 0.38 * (-b))
        add(trace(x, y, sky_ang, px(w) * U_(6, 13), px(w) * 0.5), w, jit(c, 0.025, 0.05), opac=U_(0.75, 0.95), dry=U_(0.3, 0.9))
        n += 1


def cloud_d(x, y):
    return N2(x * 1.7 + 3.1, y * 8.5 + 1.3) + 0.22 * math.exp(-((y - 0.455) / 0.075) ** 2) - 0.70


def paint_clouds():
    layer('clouds')
    n = 0
    while n < 620:
        x, y = hpt(-0.02, AR + 0.02, 0.28, HZ - 0.02)
        cd = cloud_d(x, y)
        if cd < 0: continue
        sp = math.exp(-(math.hypot(x - SUN[0], y - SUN[1]) / 0.55) ** 2)
        under = sstep((cloud_d(x, y - 0.014) - cloud_d(x, y + 0.014)) * 9 + 0.5)
        lit = min(1.0, 0.15 + 1.0 * sp * (0.35 + 0.65 * under) + 0.25 * under)
        c = mix((0.30, 0.22, 0.44), (1.0, 0.56, 0.36), lit)
        if lit > 0.75: c = mix(c, (1.0, 0.86, 0.55), (lit - 0.75) * 3)
        w = U_(16, 46)
        ang = lambda a, b_: 0.35 * sky_ang(a, b_) + 0.05 * g(0, 1)
        add(trace(x, y, ang, px(w) * U_(4, 9), px(w) * 0.5), w, jit(c, 0.025, 0.05), opac=U_(0.7, 0.95), dry=U_(0.3, 0.9))
        n += 1
    n = 0
    while n < 220:
        x, y = hpt(-0.02, AR + 0.02, 0.30, HZ - 0.02)
        if not (cloud_d(x, y) > 0 and cloud_d(x, y + 0.012) < 0.01): continue
        sp = math.exp(-(math.hypot(x - SUN[0], y - SUN[1]) / 0.6) ** 2)
        c = mix((0.85, 0.50, 0.62), (1.0, 0.88, 0.58), sp)
        w = U_(8, 16)
        add(trace(x, y, lambda a, b_: 0.03 * g(0, 1), px(w) * U_(5, 11), px(w) * 0.6), w, jit(c, 0.02, 0.04), opac=0.9, dry=0.8)
        n += 1


def paint_stars_moon():
    layer('stars')
    placed = []
    tries = 0
    while len(placed) < 32 and tries < 5000:
        tries += 1
        x, y = U_(0.02, AR - 0.02), U_(0.02, 0.44)
        if sky_t(x, y) > 0.47: continue
        if math.hypot(x - MOON[0], y - MOON[1]) < 0.20: continue
        if abs(x - LH) < 0.07 and y > 0.14: continue
        if any(math.hypot(x - a, y - b_) < 0.10 for a, b_ in placed): continue
        placed.append((x, y))
    for (x, y) in placed:
        big = U_(0.55, 1.5)
        base = sky(x, y)
        warm = R.random() < 0.7
        core = (1.0, 0.95, 0.70) if warm else (0.86, 0.94, 1.0)
        for ring in range(3):
            rr = (0.011 + 0.0085 * ring) * big
            nd = 4 + ring + (1 if big > 1 else 0)
            ph = U_(0, 6.28)
            tint = mix(core, (0.55, 0.72, 0.98), ring / 2.5)
            for k in range(nd):
                a0 = ph + 6.283 * k / nd + 0.25 * g(0, 1)
                sp = 6.283 / nd * U_(0.45, 0.8)
                add(arc(x, y, rr * U_(0.92, 1.08), a0, a0 + sp), U_(9, 14) * big ** 0.5, jit(mix(base, tint, 0.78 - 0.23 * ring), 0.02, 0.03),
                    opac=0.92, dry=0.5, t0=0.5, t1=0.5)
        add(arc(x, y, 0.0022 * big, 0, 6.3, 9), 13 * big, jit(core, 0.01, 0.02), opac=1.0, dry=0.0, t0=0.95, t1=0.95)
        add(seg(x - 0.001, y - 0.001, x + 0.001, y - 0.001, 3), 7 * big, (1.0, 1.0, 0.93), opac=1.0, dry=0.0, t0=0.95, t1=0.95)
    layer('moon')
    mx, my = MOON
    for k in range(150):
        f = (k / 150.0)
        rr = MOONR * (1.12 + 2.3 * f ** 1.15) + 0.004 * g(0, 1)
        a0 = U_(0, 6.28); sp = U_(0.35, 0.9) / (1 + 1.5 * f)
        px_, py_ = mx + rr * math.cos(a0), my + rr * math.sin(a0)
        tint = mix((1.0, 0.95, 0.72), (0.55, 0.72, 1.0), f * 1.2)
        add(arc(mx, my, rr, a0, a0 + sp), U_(14, 24), jit(mix(sky(px_, py_), tint, 0.70 * (1 - f) ** 1.3 + 0.08), 0.02, 0.03),
            opac=0.9, dry=0.5, t0=0.5, t1=0.5)
    for k in range(26):                                   # earthshine disc
        rr = MOONR * math.sqrt(R.random()) * 0.97
        a0 = U_(0, 6.28)
        add(arc(mx, my, max(rr, 0.004), a0, a0 + U_(1.0, 2.5)), 13, jit((0.33, 0.33, 0.56), 0.02, 0.04), opac=0.9)
    dx, dy = SUN[0] - mx, SUN[1] - my
    dn = math.hypot(dx, dy); dx /= dn; dy /= dn
    c2 = (mx - 0.42 * MOONR * dx, my - 0.42 * MOONR * dy)

    def in_cres(x, y):
        return math.hypot(x - mx, y - my) < MOONR and math.hypot(x - c2[0], y - c2[1]) > MOONR * 0.97
    n = 0
    while n < 70:
        rr = MOONR * U_(0.55, 0.985); a0 = math.atan2(dy, dx) + U_(-1.5, 1.5)
        x, y = mx + rr * math.cos(a0), my + rr * math.sin(a0)
        if not in_cres(x, y): continue
        pts = trace(x, y, lambda a, b_: math.atan2(b_ - my, a - mx) + math.pi / 2, 0.06, px(5), inside=in_cres)
        add(pts, U_(7, 11), jit((1.0, 0.96, 0.80), 0.012, 0.02), opac=1.0, dry=0.1, t0=0.5, t1=0.5)
        n += 1


def paint_beam():
    layer('lantern glow and beam')
    ox, oy = LH, 0.243
    for k in range(110):
        f = k / 110.0
        rr = 0.026 + 0.105 * f ** 1.2 + 0.003 * g(0, 1)
        a0 = U_(0, 6.28)
        px_, py_ = ox + rr * math.cos(a0), oy + rr * math.sin(a0)
        add(arc(ox, oy, rr, a0, a0 + U_(0.35, 0.9) / (1 + f)), U_(12, 20), jit(mix(sky(px_, py_), (1.0, 0.90, 0.58), 0.72 * (1 - f) ** 1.3 + 0.06), 0.02, 0.03),
            opac=0.9, dry=0.5, t0=0.5, t1=0.5)
    for (a_c, n, lmax) in [(math.pi - 0.035, 60, 0.62), (-0.035, 18, 0.35)]:
        for k in range(n):
            a = a_c + U_(-0.055, 0.055)
            r0 = U_(0.03, 0.10); r1 = U_(0.22, lmax)
            add(seg(ox + r0 * math.cos(a), oy + r0 * math.sin(a), ox + r1 * math.cos(a), oy + r1 * math.sin(a), 12),
                U_(14, 34), jit((1.0, 0.93, 0.66), 0.01, 0.02), opac=U_(0.10, 0.22), dry=0.9, t0=0.25, t1=0.9, cmod=0.2)


def paint_hills():
    layer('far hills')
    n = 0
    while n < 260:
        x, y = hpt(-0.02, AR + 0.02, HZ - 0.075, HZ + 0.004)
        if y < ridge_a(x): continue
        sp = math.exp(-(abs(x - SUN[0]) / 0.45) ** 2)
        c = mix(mix((0.40, 0.30, 0.52), (0.30, 0.24, 0.46), (y - ridge_a(x)) / 0.05), (0.98, 0.62, 0.42), 0.62 * sp)
        w = U_(12, 24)
        e = 0.01
        ang = lambda a, b_: math.atan2(ridge_a(a + e) - ridge_a(a - e), 2 * e) * 0.8 + 0.06 * g(0, 1)
        add(trace(x, y, ang, px(w) * U_(4, 8), px(w) * 0.5, inside=lambda a, b_: b_ > ridge_a(a) and b_ < HZ + 0.006), w, jit(c, 0.02, 0.04))
        n += 1
    layer('headland')
    n = 0
    while n < 230:
        x, y = hpt(-0.02, 0.42, HZ - 0.12, HZ + 0.006)
        if y < ridge_b(x): continue
        dtop = y - ridge_b(x)
        c = mix((0.36, 0.20, 0.34), (0.13, 0.10, 0.22), sstep(dtop / 0.03))
        c = mix(c, (0.10, 0.16, 0.20), 0.3 * N2(x * 30, y * 30))
        w = U_(11, 24)
        e = 0.01
        ang = lambda a, b_: math.atan2(ridge_b(a + e) - ridge_b(a - e), 2 * e) + 0.15 * g(0, 1)
        add(trace(x, y, ang, px(w) * U_(3, 7), px(w) * 0.5, inside=lambda a, b_: b_ > ridge_b(a) and b_ < HZ + 0.008), w, jit(c, 0.02, 0.05))
        n += 1
    for k in range(9):                                   # village lights
        x = U_(0.03, 0.25); y = HZ - U_(0.004, 0.02)
        if y < ridge_b(x) + 0.006: continue
        add(seg(x - 0.003, y, x + 0.003, y, 3), U_(6, 9), (1.0, 0.86, 0.45), opac=1.0, dry=0, t0=0.9, t1=0.9)


def paint_sea():
    def sea_region(x, y):
        return y > HZ and not (y > pl(CLIFF, x) + 0.06)
    hang = lambda a, b_: 0.035 * math.sin(a * 23 + b_ * 60) + 0.02 * g(0, 1)
    layer('sea block-in')
    n = 0
    while n < 150:
        x, y = hpt(-0.02, AR, HZ, 1.02)
        if not sea_region(x, y): continue
        d = (y - HZ) / (1 - HZ)
        w = 45 + 110 * d
        add(trace(x, y, hang, px(w) * U_(5, 9), px(w) * 0.5), w, jit(sea(x, y), 0.012, 0.03), opac=0.97, dry=0.2)
        n += 1
    layer('sea')
    n = 0
    while n < 1350:
        x, y = hpt(-0.02, AR, 0, 1)
        d = y ** 1.6; y = HZ + 0.002 + d * (1.02 - HZ)
        if not sea_region(x, y): continue
        w = (9 + 55 * d) * U_(0.7, 1.3)
        add(trace(x, y, hang, px(w) * U_(5, 11), px(w) * 0.5, inside=lambda a, b_: b_ > HZ + 0.001), w, jit(sea(x, y), 0.02, 0.06),
            opac=U_(0.8, 0.97), dry=U_(0.2, 0.8))
        n += 1
    layer('waves')
    n = 0
    while n < 950:
        x, y = hpt(-0.02, AR, 0, 1)
        d = y ** 1.5; y = HZ + 0.004 + d * (1.02 - HZ)
        if not sea_region(x, y): continue
        wv = N2(x * 5, y * 45)
        ys = max(0.03, HZ - (y - HZ) * 1.05)
        sc = sky(x, ys)
        if wv > 0.5:
            c = mix(sc, (0.80, 0.80, 0.92), 0.25); c = (c[0] * 0.95, c[1] * 0.95, c[2] * 1.0)
        else:
            c = mix((0.02, 0.07, 0.15), (0.05, 0.20, 0.27), R.random())
        w = (5 + 24 * d) * U_(0.7, 1.3)
        add(trace(x, y, hang, px(w) * U_(4, 10), px(w) * 0.5, inside=lambda a, b_: b_ > HZ + 0.002), w, jit(c, 0.02, 0.05),
            opac=U_(0.6, 0.9), dry=U_(0.5, 1.0), t0=0.3, t1=0.3)
        n += 1
    layer('sun glitter')
    n = 0
    while n < 760:
        d = R.random() ** 1.7
        y = HZ + 0.003 + d * (1.0 - HZ)
        hw = 0.022 + 0.17 * d
        dxn = g(0, 0.5)
        x = SUN[0] + hw * dxn + 0.01 * math.sin(y * 70)
        if not sea_region(x, y): continue
        core = math.exp(-dxn * dxn * 2.2)
        c = mix((0.98, 0.50, 0.22), (1.0, 0.90, 0.60), core * (1 - 0.35 * d))
        w = (5 + 19 * d) * U_(0.7, 1.3)
        Ln = px(w) * U_(2.5, 7)
        a = 0.03 * g(0, 1)
        add(seg(x - Ln / 2 * math.cos(a), y - Ln / 2 * math.sin(a), x + Ln / 2 * math.cos(a), y + Ln / 2 * math.sin(a), 4), w,
            jit(c, 0.02, 0.03), opac=U_(0.8, 1.0), dry=U_(0.2, 0.9), t0=0.35, t1=0.35)
        n += 1


def paint_boat():
    layer('sailboat')
    bx, by = 0.545, 0.672
    for k in range(16):                                  # reflection
        y = by + 0.006 + 0.055 * R.random() ** 1.3
        x = bx + 0.012 + U_(-0.03, 0.03) * (1 - (y - by) / 0.09)
        c = (0.96, 0.78, 0.60) if R.random() < 0.55 else (0.08, 0.07, 0.14)
        add(seg(x - U_(0.006, 0.016), y, x + U_(0.006, 0.016), y + 0.001 * g(0, 1), 3), U_(5, 9), jit(c, 0.02, 0.05), opac=0.7, dry=0.8, t0=0.3, t1=0.3)
    mast = bx + 0.012
    tri_m = [(mast + 0.003, by - 0.088), (mast + 0.003, by - 0.012), (mast + 0.050, by - 0.012)]
    tri_j = [(mast - 0.003, by - 0.078), (mast - 0.003, by - 0.012), (mast - 0.036, by - 0.012)]
    for tri, lit, shade, sgn in [(tri_m, (1.0, 0.84, 0.64), (0.82, 0.56, 0.52), 1), (tri_j, (0.98, 0.72, 0.52), (0.72, 0.46, 0.50), -1)]:
        (ax, ay), (bx_, by_), (cx, cy) = tri
        for k in range(15):
            f = (k + 0.5) / 15.0
            x = ax + (cx - ax) * f * 0.93 + 0.0015 * sgn
            ytop = ay + (cy - ay) * f + 0.004
            if cy - ytop < 0.006: continue
            add(seg(x + 0.003 * sgn * 0, ytop, x + 0.002 * sgn, cy, 5), U_(8, 11), jit(mix(lit, shade, f), 0.015, 0.03), opac=0.97, dry=0.2, t0=0.5, t1=0.8)
    add(seg(mast, by - 0.094, mast, by, 4), 5, (0.16, 0.10, 0.14), opac=1, dry=0, t0=0.8, t1=0.9)
    for k in range(5):
        yy = by - 0.004 + 0.003 * k
        sh = 0.004 * k
        add([(bx - 0.036 + sh, yy), (bx + 0.012, yy + 0.0015), (bx + 0.058 - sh * 0.6, yy - 0.001)], 9,
            jit((0.13, 0.08, 0.13) if k else (0.70, 0.22, 0.16), 0.01, 0.03), opac=1, dry=0.1, t0=0.5, t1=0.5)


def paint_cliff():
    layer('cliff block-in')
    def sang(x, y):
        e = 0.01
        sl = math.atan2(pl(CLIFF, x + e) - pl(CLIFF, x - e), 2 * e)
        dl = y - pl(CLIFF, x)
        k = math.exp(-dl / 0.05)
        return k * sl + (1 - k) * (-0.38) + 0.9 * (N2(x * 7, y * 7) - 0.5)
    n = 0
    while n < 420:
        x, y = hpt(0.54, AR + 0.02, 0.50, 1.02)
        if not in_cliff(x, y) or y > pl(GRASS, x) + 0.06: continue
        w = U_(38, 80)
        add(trace(x, y, sang, px(w) * U_(3, 5.5), px(w) * 0.5, inside=lambda a, b_: b_ > pl(CLIFF, a) + 0.004), w, jit(rock(x, y), 0.012, 0.04), opac=1.0, dry=0.0, t0=0.8, t1=0.7)
        n += 1
    layer('cliff')
    n = 0
    while n < 900:
        x, y = hpt(0.54, AR + 0.02, 0.50, 1.02)
        if not in_cliff(x, y) or y > pl(GRASS, x) + 0.04: continue
        w = U_(10, 26)
        add(trace(x, y, sang, px(w) * U_(2.5, 6), px(w) * 0.5, inside=in_cliff), w, jit(rock(x, y), 0.025, 0.08), opac=U_(0.85, 1), dry=U_(0.2, 0.8))
        n += 1
    layer('cliff detail')
    n = 0
    while n < 420:
        x, y = hpt(0.54, AR + 0.02, 0.50, 1.02)
        if not in_cliff(x, y) or y > pl(GRASS, x) + 0.03: continue
        c = rock(x, y)
        lum = (c[0] + c[1] + c[2]) / 3
        w = U_(5, 10)
        if lum > 0.30 or R.random() < 0.55:
            c2 = mix(c, (1.0, 0.82, 0.58), 0.50) if lum > 0.30 else mix(c, (0.62, 0.42, 0.56), 0.5)
            add(trace(x, y, sang, px(w) * U_(3, 8), px(w) * 0.6, inside=in_cliff), w, jit(c2, 0.02, 0.04), opac=0.95, dry=0.7, t0=0.3, t1=0.3)
        else:
            c2 = mix(c, (0.04, 0.03, 0.09), 0.6)
            a0 = sang(x, y) + (1.1 if R.random() < 0.4 else 0.0) + 0.35 * g(0, 1)
            add(trace(x, y, lambda a, b_: a0, px(w) * U_(4, 12), px(w) * 0.6, inside=in_cliff), w, jit(c2, 0.01, 0.04),
                opac=0.8, dry=0.8, t0=0.3, t1=0.2)
        n += 1


def paint_lighthouse():
    layer('lighthouse')
    yb, yt = 0.548, 0.276
    hb, ht = 0.031, 0.019

    def hw(y): return ht + (hb - ht) * (y - yt) / (yb - yt)
    def inside(x, y): return yt <= y <= yb and abs(x - LH) < hw(y)

    def col(x, y):
        s = (x - LH) / hw(y)
        f = (y - yt) / (yb - yt)
        red = (0.14 < f < 0.34) or (0.56 < f < 0.78)
        lit = sstep(0.55 - 0.85 * s + 0.25 * (N2(x * 60, y * 25) - 0.5))
        if red: return mix((0.40, 0.10, 0.20), (0.93, 0.27, 0.16), lit)
        return mix((0.50, 0.47, 0.70), (1.0, 0.90, 0.76), lit)

    def tang(x, y):
        s = (x - LH) / hw(y)
        return math.pi / 2 - s * math.atan2(hb - ht, yb - yt)
    for rep, (wlo, whi, cnt) in enumerate([(20, 26, 60), (9, 14, 300)]):
        n = 0
        while n < cnt:
            x, y = hpt(LH - hb, LH + hb, yt, yb)
            if not inside(x, y): continue
            w = U_(wlo, whi)
            add(trace(x, y, tang, U_(0.022, 0.05), px(w) * 0.5, inside=lambda a, b_: inside(a, b_) and
                      (((0.14 < (b_ - yt) / (yb - yt) < 0.34) or (0.56 < (b_ - yt) / (yb - yt) < 0.78)) ==
                       ((0.14 < (y - yt) / (yb - yt) < 0.34) or (0.56 < (y - yt) / (yb - yt) < 0.78)))),
                w, jit(col(x, y), 0.015, 0.04), opac=0.98, dry=0.15, t0=0.7, t1=0.7)
            n += 1
    for (xx, yy) in [(LH - 0.004, 0.36), (LH + 0.002, 0.455)]:        # windows
        add(seg(xx, yy, xx, yy + 0.014, 3), 9, (0.10, 0.08, 0.18), opac=1, dry=0, t0=0.9, t1=0.9)
    add(seg(LH - 0.003, yb - 0.026, LH - 0.003, yb - 0.004, 3), 16, (0.12, 0.08, 0.14), opac=1, dry=0, t0=0.8, t1=0.9)  # door
    # lantern room
    ly0, ly1, lw = 0.226, 0.266, 0.0155
    for k in range(16):
        x = LH + lw * U_(-0.85, 0.85)
        c = mix((1.0, 0.98, 0.82), (1.0, 0.80, 0.34), abs(x - LH) / lw)
        add(seg(x, ly0, x, ly1, 4), U_(9, 13), jit(c, 0.01, 0.02), opac=1, dry=0, t0=0.9, t1=0.9)
    for xx in (-0.0155, -0.005, 0.0055, 0.0155):
        add(seg(LH + xx, ly0, LH + xx, ly1, 3), 4.5, (0.14, 0.09, 0.14), opac=1, dry=0, t0=0.9, t1=0.9)
    # gallery
    for k in range(4):
        yy = 0.266 + 0.0035 * k
        add(seg(LH - 0.029, yy, LH + 0.029, yy, 5), 9, jit((0.13, 0.10, 0.18) if k != 0 else (0.52, 0.30, 0.34), 0.01, 0.03), opac=1, dry=0.1, t0=0.8, t1=0.8)
    for k in range(9):
        xx = LH - 0.028 + 0.007 * k
        add(seg(xx, 0.254, xx, 0.268, 3), 3.2, (0.12, 0.09, 0.15), opac=1, dry=0, t0=0.9, t1=0.9)
    add(seg(LH - 0.029, 0.254, LH + 0.029, 0.254, 5), 3.5, (0.14, 0.10, 0.16), opac=1, dry=0, t0=0.9, t1=0.9)
    # roof
    for k in range(9):
        f = (k + 0.5) / 9.0
        x0 = LH + 0.021 * (2 * f - 1)
        c = mix((0.80, 0.32, 0.20), (0.22, 0.07, 0.13), sstep(f * 1.3))
        add(seg(x0, 0.2265, LH + 0.002 * (2 * f - 1), 0.196, 4), 11, jit(c, 0.01, 0.03), opac=1, dry=0.1, t0=0.9, t1=0.35)
    add(seg(LH, 0.198, LH, 0.176, 3), 5, (0.16, 0.08, 0.12), opac=1, dry=0, t0=0.9, t1=0.5)
    add(seg(LH - 0.022, 0.227, LH + 0.022, 0.227, 4), 6, (0.16, 0.09, 0.14), opac=1, dry=0, t0=0.9, t1=0.9)

    layer('cottage')
    x0, x1, y0, y1 = 1.125, 1.225, 0.497, 0.540
    for k in range(22):                                   # wall
        x = x0 + (x1 - x0) * (k + 0.5) / 22
        lit = 1 - sstep((x - x0) / 0.03)
        add(seg(x, y0, x, y1, 3), 13, jit(mix((0.46, 0.41, 0.60), (1.0, 0.80, 0.62), lit), 0.015, 0.03), opac=1, dry=0.1, t0=0.9, t1=0.9)
    for k in range(8):                                    # roof
        f = k / 7.0
        yy = y0 + 0.002 - 0.030 * (1 - f)
        ins = 0.022 * (1 - f)
        add(seg(x0 - 0.006 + ins, yy, x1 + 0.006 - ins * 0.4, yy, 5), 10, jit(mix((0.72, 0.30, 0.22), (0.34, 0.13, 0.18), 1 - f), 0.012, 0.04), opac=1, dry=0.2, t0=0.7, t1=0.7)
    add(seg(1.197, 0.447, 1.197, 0.475, 3), 14, (0.36, 0.20, 0.24), opac=1, dry=0, t0=0.9, t1=0.9)   # chimney
    for xx in (1.155, 1.195):
        add(seg(xx, 0.509, xx, 0.524, 3), 13, (1.0, 0.84, 0.40), opac=1, dry=0, t0=0.9, t1=0.9)
        add(seg(xx, 0.511, xx, 0.522, 3), 5, (1.0, 0.97, 0.75), opac=1, dry=0, t0=0.9, t1=0.9)

    layer('clifftop grass')
    for k in range(330):
        x = U_(0.84, AR)
        y = cliff_top(x) + U_(-0.002, 0.016)
        lit = sstep(1.1 - (y - cliff_top(x)) / 0.012) * (0.4 + 0.6 * R.random())
        c = mix((0.10, 0.14, 0.10), (0.78, 0.55, 0.22), lit * 0.9)
        blade(x, y, U_(0.010, 0.026), U_(4.5, 8), c)


def blade(x, y, ln, w, c, lean=None):
    a = -math.pi / 2 + (lean if lean is not None else (0.25 + 0.45 * g(0, 1)))
    bend = U_(0.3, 1.3) * (1 if a > -math.pi / 2 else -1)
    pts = [(x, y)]
    n = 7
    for k in range(n):
        a2 = a + bend * (k / n) ** 1.5
        x += math.cos(a2) * ln / n; y += math.sin(a2) * ln / n
        pts.append((x, y))
    add(pts, w, jit(c, 0.02, 0.06), opac=0.97, dry=0.1, t0=1.0, t1=0.12)


def paint_foreground():
    layer('foreground block-in')
    def gang(x, y):
        e = 0.01
        return math.atan2(pl(GRASS, x + e) - pl(GRASS, x - e), 2 * e) + 0.5 * (N2(x * 6, y * 6) - 0.5)
    n = 0
    while n < 170:
        x, y = hpt(0.72, AR + 0.02, 0.76, 1.03)
        if not in_grass(x, y): continue
        w = U_(40, 75)
        c = mix((0.07, 0.10, 0.09), (0.17, 0.15, 0.10), N2(x * 8, y * 8))
        add(trace(x, y, gang, px(w) * U_(3, 6), px(w) * 0.5, inside=lambda a, b_: b_ > pl(GRASS, a) - 0.004), w, jit(c, 0.012, 0.06), opac=0.98)
        n += 1
    layer('grass')
    bl = []
    n = 0
    while n < 1500:
        x, y = hpt(0.72, AR + 0.02, 0.76, 1.06)
        if not y > grass_top(x) + 0.004: continue
        bl.append((y, x)); n += 1
    bl.sort()
    fl = []
    for (y, x) in bl:
        dt = y - grass_top(x)
        sc = 0.7 + 1.5 * (y - 0.77) / 0.25
        rim = math.exp(-dt / 0.035)
        r_ = R.random()
        if r_ < 0.20 + 0.55 * rim:
            c = mix((0.50, 0.42, 0.16), (0.96, 0.70, 0.28), R.random() * (0.4 + 0.6 * rim))
        elif r_ < 0.80:
            c = mix((0.07, 0.11, 0.07), (0.24, 0.31, 0.12), R.random())
        else:
            c = mix((0.10, 0.17, 0.22), (0.22, 0.30, 0.34), R.random())
        wind = 0.35 + 0.5 * (N2(x * 5, y * 5) - 0.5) + 0.3 * g(0, 1)
        blade(x, y, U_(0.035, 0.085) * sc, U_(5, 9) * sc ** 0.8, c, lean=wind)
        if R.random() < 0.085:
            fl.append((x + U_(-0.01, 0.01), y - U_(0.03, 0.07) * sc, sc))
        if len(fl) and R.random() < 0.5:
            fx, fy, fs = fl.pop(0)
            kind = R.random()
            if kind < 0.72:
                c1, c2 = (0.60, 0.08, 0.10), (0.98, 0.30, 0.13)
            elif kind < 0.9:
                c1, c2 = (0.75, 0.55, 0.20), (1.0, 0.90, 0.50)
            else:
                c1, c2 = (0.55, 0.50, 0.70), (0.96, 0.92, 0.95)
            rad = px(U_(8, 13) * fs)
            add(seg(fx, fy + rad * 0.5, fx + U_(-0.004, 0.004), fy + rad * 4.5, 3), 3.5 * fs ** 0.5, (0.16, 0.24, 0.10), opac=1, dry=0, t0=0.9, t1=0.9)
            ph = U_(0, 6.28)
            for k in range(5):
                a0 = ph + 6.283 * k / 5 + 0.3 * g(0, 1)
                up = 0.5 - 0.5 * math.sin(a0)
                add(seg(fx + 0.15 * rad * math.cos(a0), fy + 0.15 * rad * math.sin(a0), fx + rad * math.cos(a0), fy + rad * 0.8 * math.sin(a0), 3),
                    rad * S * 1.05, jit(mix(c1, c2, up * U_(0.6, 1.0)), 0.02, 0.06), opac=1, dry=0, t0=0.6, t1=0.9)
            add(seg(fx - rad * 0.1, fy, fx + rad * 0.1, fy, 3), rad * S * 0.5, (0.10, 0.05, 0.09) if kind < 0.72 else (0.85, 0.55, 0.12), opac=1, dry=0, t0=0.9, t1=0.9)


def paint_birds():
    layer('gulls')
    for (x, y, s) in [(0.66, 0.40, 1.0), (0.715, 0.435, 0.8), (0.60, 0.455, 0.7), (0.76, 0.385, 0.6), (0.69, 0.47, 0.5), (0.635, 0.375, 0.55)]:
        sp = 0.017 * s
        add([(x - sp, y - 0.003 * s), (x - sp * 0.5, y - 0.008 * s), (x, y)], 5 * s ** 0.5, (0.08, 0.06, 0.14), opac=1, dry=0, t0=0.3, t1=0.9)
        add([(x, y), (x + sp * 0.5, y - 0.009 * s), (x + sp, y - 0.005 * s)], 5 * s ** 0.5, (0.08, 0.06, 0.14), opac=1, dry=0, t0=0.9, t1=0.3)


def relief(can, hgt, strength=0.3):
    h = cv2.GaussianBlur(hgt, (0, 0), 0.8)
    hx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    hy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    sh = np.clip(-(hx * 0.7 + hy * 0.7) * strength, -0.12, 0.12)
    return np.clip(can * (1.0 + sh[..., None]) + np.maximum(sh, 0)[..., None] * 0.15, 0, 1)


def to_u8(a):
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='orig'); ap.add_argument('--frames', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out + '/frames', exist_ok=True)
    paint_sky(); paint_clouds(); paint_stars_moon(); paint_beam(); paint_hills(); paint_sea(); paint_boat()
    paint_cliff(); paint_lighthouse(); paint_birds(); paint_foreground()
    layer('end')
    N = len(strokes)
    for k in range(len(layer_marks) - 1):
        print(f'{layer_marks[k][0]:>24}: {layer_marks[k + 1][1] - layer_marks[k][1]}')
    print('TOTAL STROKES', N)
    P = np.zeros((N, MAXP, 2), np.float32); NP = np.zeros(N, np.int64)
    WD = np.zeros(N); COL = np.zeros((N, 3)); OP = np.zeros(N); DRY = np.zeros(N); T0 = np.zeros(N); T1 = np.zeros(N); CM = np.zeros(N)
    for i, (pts, w, c, op, dr, t0, t1, cm) in enumerate(strokes):
        NP[i] = len(pts); P[i, :len(pts)] = np.array(pts, np.float32) * S
        WD[i] = w; COL[i] = c; OP[i] = op; DRY[i] = dr; T0[i] = t0; T1[i] = t1; CM[i] = cm
    can = np.empty((H, W, 3), np.float32); can[:] = np.array([0.93, 0.905, 0.85], np.float32)
    hgt = np.zeros((H, W), np.float32)
    A = np.zeros((H, W), np.float32); Ub = np.zeros((H, W), np.float32); Tb = np.zeros((H, W), np.float32)
    if a.frames:
        marks = sorted(set([0] + [int(round(N * (k / a.frames) ** 1.25)) for k in range(1, a.frames + 1)]))
    else:
        marks = [0, N]
    fw = 1440; fh = round(H * fw / W); fh += fh % 2
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)

    def frame(idx, n):
        small = cv2.resize(to_u8(can), (fw, fh), interpolation=cv2.INTER_AREA)
        p = Image.fromarray(small); d = ImageDraw.Draw(p)
        d.text((30, fh - 64), f'{n:,} strokes', font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        p.save(f'{a.out}/frames/f{idx:04d}.png', compress_level=1)
    if a.frames: frame(0, 0)
    for k in range(1, len(marks)):
        render(can, hgt, A, Ub, Tb, P, NP, WD, COL, OP, DRY, T0, T1, CM, marks[k - 1], marks[k])
        if a.frames: frame(k, marks[k])
    fin = to_u8(relief(can, hgt))
    Image.fromarray(fin).save(a.out + '/painting.png')
    Image.fromarray(fin).save(a.out + '/painting.jpg', quality=95, subsampling=0)
    Image.fromarray(fin).resize((1400, 1050), Image.LANCZOS).save(a.out + '/small.png')


if __name__ == '__main__':
    main()
