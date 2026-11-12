"""An original painting, no reference image: an evening commuter train.

A row of passengers bowed over their phones, backlit by a low sun they are not looking at; a child is.
Every brush stroke (path, width, colour) is decided directly from a composition written in code and
laid down back-to-front like a painter would. There is no target picture and no correction step.
"""
import argparse, math, os, random
import numpy as np, cv2
from numba import njit
from PIL import Image, ImageDraw, ImageFont

W, H = 3000, 1875
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


# ----------------------------------------------------------------------------- shapes & fills
DENS = 1.3
SUN = (0.735, 0.30)          # the sun as seen through the middle window
G0 = (0.735, 0.43)           # point on the horizon under the sun: floor shadows fan out from here
WT, WB = 0.135, 0.52         # window top / sill
WINS = [(0.03, 0.47), (0.53, 0.97), (1.03, 1.47)]
SEAT_T, SEAT_F0, SEAT_F1, FLOOR0, FEET = 0.655, 0.673, 0.712, 0.77, 0.845
CYAN = (0.60, 0.88, 1.0)
SIL = []                     # silhouettes of the seated passengers


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


def sd_win(x, y):
    return min(sd_box(x, y, (a + b) / 2, (WT + WB) / 2, (b - a) / 2, (WB - WT) / 2, 0.03) for a, b in WINS)


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
         jc=(0.012, 0.04), aj=0.12, flow=False, dens=True, base=None, bang=None):
    x0, y0, x1, y1 = bbox
    if base is None: base = w[1]
    if base:
        block(sd, bbox, colfn, base, bang if bang is not None else (math.pi / 2 if callable(ang) else ang), min(margin, 0.3))
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


# ----------------------------------------------------------------------------- the world outside
SKY2 = [(0.0, (0.25, 0.24, 0.50)), (0.30, (0.50, 0.32, 0.56)), (0.55, (0.90, 0.46, 0.44)), (0.80, (1.0, 0.64, 0.36)), (1.0, (1.0, 0.80, 0.48))]


def sky2(x, y):
    t = (y - WT) / (WB - WT)
    c = ramp(SKY2, t + 0.06 * (N2(x * 3, y * 9) - 0.5))
    d = math.hypot(x - SUN[0], y - SUN[1])
    c = mix(c, (1.0, 0.90, 0.60), 0.88 * math.exp(-(d / 0.17) ** 2))
    cl = N2(x * 2.0 + 4, y * 24 + 2)
    if cl > 0.58:
        k = sstep((cl - 0.58) / 0.10)
        under = sstep((N2(x * 2.0 + 4, (y - 0.012) * 24 + 2) - cl) * 12 + 0.5)
        cc = mix((0.40, 0.22, 0.42), (1.0, 0.60, 0.40), under * (0.4 + 0.6 * math.exp(-(d / 0.5) ** 2)))
        c = mix(c, cc, 0.8 * k)
    return c


def make_buildings(seed, hmin, hmax, wmin, wmax, sunmax):
    rr = random.Random(seed); x = -0.03; out = []
    while x < AR + 0.03:
        w = rr.uniform(wmin, wmax); h = rr.uniform(hmin, hmax)
        if rr.random() < 0.18: h *= 1.45
        cxm = x + w / 2
        if abs(cxm - SUN[0]) < 0.12: h = min(h, sunmax)
        out.append((cxm, w / 2 * 0.94, 0.53 - h))
        x += w
    return out


FAR = make_buildings(3, 0.06, 0.15, 0.035, 0.075, 0.11)
NEAR = make_buildings(11, 0.025, 0.085, 0.05, 0.13, 0.06)


def sd_bl(B):
    return lambda x, y: min(sd_box(x, y, c, (t + 0.60) / 2, hw, (0.60 - t) / 2) for c, hw, t in B)


def paint_ground():
    layer('toned ground')
    for k in range(150):
        x, y = hpt(-0.05, AR + 0.05, -0.02, 1.02)
        c = mix((0.10, 0.10, 0.17), (0.16, 0.13, 0.20), N2(x * 2, y * 2))
        L = U_(0.35, 0.7); a = 0.12 * g(0, 1)
        add(seg(x - L / 2, y - L / 2 * math.sin(a), x + L / 2, y + L / 2 * math.sin(a), 12), U_(170, 250), jit(c, 0.01, 0.05), 0.97, 0.3)


def paint_view():
    layer('evening sky')
    bb = (0.0, WT, AR, WB)
    fill(sd_win, bb, sky2, 300, (26, 46), ang=0.0, lmul=(4, 9), aj=0.04, margin=0.2)
    fill(sd_win, bb, sky2, 330, (9, 17), ang=0.0, lmul=(5, 12), aj=0.04, margin=0.3, dry=0.6, base=0)
    layer('city passing by')
    sdf, sdn = sd_bl(FAR), sd_bl(NEAR)

    def glare(x, y): return math.exp(-(math.hypot(x - SUN[0], y - SUN[1]) / 0.20) ** 2)

    def farc(x, y):
        c = mix((0.40, 0.21, 0.38), (0.98, 0.62, 0.40), 0.75 * glare(x, y))
        return mix(c, (0.80, 0.42, 0.38), 0.25 * sstep((y - 0.42) / 0.1))

    def nearc(x, y):
        return mix((0.12, 0.065, 0.16), (0.55, 0.22, 0.20), 0.8 * glare(x, y))
    fill(lambda x, y: max(sdf(x, y), sd_win(x, y) + 0.004), (0, 0.30, AR, WB), farc, 230, (9, 17), ang=0.0, lmul=(2.5, 6), aj=0.02, margin=0.3)
    fill(lambda x, y: max(sdn(x, y), sd_win(x, y) + 0.004), (0, 0.38, AR, WB), nearc, 230, (9, 17), ang=0.0, lmul=(3, 8), aj=0.02, margin=0.3)
    n = 0
    while n < 150:                                           # lit windows of other people's evenings
        x, y = hpt(0, AR, 0.36, WB - 0.012)
        if sd_win(x, y) > -0.012: continue
        near = sdn(x, y) < -0.006
        if not near and not sdf(x, y) < -0.006: continue
        c = (1.0, 0.86, 0.50) if R.random() < 0.8 else (0.70, 0.90, 1.0)
        add(seg(x - 0.004, y, x + U_(0.004, 0.010), y, 3), U_(3.5, 5.5), jit(c, 0.02, 0.05), 0.95, 0.2, 0.6, 0.3)
        n += 1
    layer('the sun')
    for k in range(150):
        f = k / 150.0
        rr = 0.028 + 0.19 * f ** 1.2 + 0.004 * g(0, 1)
        a0 = U_(0, 6.283); sp = U_(0.4, 1.0) / (1 + 1.5 * f)
        qx, qy = SUN[0] + rr * math.cos(a0), SUN[1] + rr * math.sin(a0)
        if sd_win(qx, qy) > -0.01: continue
        add(arc(SUN[0], SUN[1], rr, a0, a0 + sp), U_(18, 34), jit((1.0, 0.92, 0.66), 0.01, 0.02),
            0.05 + 0.36 * (1 - f) ** 1.6, 0.2, 0.5, 0.5)
    for k in range(26):
        rr = 0.026 * math.sqrt(R.random()); a0 = U_(0, 6.283)
        add(arc(SUN[0], SUN[1], max(rr, 0.003), a0, a0 + U_(1.5, 3.5)), 13, jit((1.0, 0.985, 0.86), 0.008, 0.01), 1.0, 0.0, 0.8, 0.8)


# ----------------------------------------------------------------------------- the carriage
def wallc(x, y):
    c = mix((0.085, 0.10, 0.17), (0.16, 0.18, 0.27), N2(x * 6, y * 6))
    d = math.hypot(x - SUN[0], y - SUN[1])
    c = mix(c, (0.90, 0.44, 0.22), 0.34 * math.exp(-(d / 0.30) ** 2))
    return mix(c, (0.34, 0.50, 0.52), 0.5 * math.exp(-max(0.0, y - 0.03) / 0.06))


def paint_wall():
    layer('carriage wall')
    sdw = lambda x, y: max(-sd_win(x, y), 0.098 - y, y - 0.54)
    fill(sdw, (-0.02, 0.098, AR + 0.02, 0.54), wallc, 430, (14, 30),
         ang=lambda x, y: math.pi / 2 if WT + 0.02 < y < WB - 0.02 else 0.0, lmul=(3, 7), margin=0.3, aj=0.05, bang=math.pi / 2, base=26)
    layer('window frames')
    bb = (0.0, WT - 0.03, AR, WB + 0.03)
    rim(sd_win, bb, 170, lambda x, y, s: (0.03, 0.03, 0.065), w=(7, 10), inset=-0.005, ln=(0.06, 0.16))

    def bounce(x, y, s):
        d = math.hypot(x - SUN[0], y - SUN[1])
        return mix((0.26, 0.20, 0.30), (1.0, 0.66, 0.32), math.exp(-(d / 0.42) ** 2))
    rim(sd_win, bb, 150, bounce, w=(3, 5), inset=-0.0125, ln=(0.04, 0.12))


def glyph(pts, ox, oy, sx, sy, w, col):
    line([(ox + p[0] * sx, oy + p[1] * sy) for p in pts], w, col)


def paint_ceiling():
    layer('ceiling and strip light')
    hband(0.0, 0.046, -0.02, AR + 0.02, lambda x, y: mix((0.05, 0.06, 0.10), (0.20, 0.32, 0.33), math.exp(-abs(y - 0.02) / 0.02) * 0.8), 70, (22, 34), ln=(0.15, 0.4))
    for x0 in (0.04, 0.44, 0.84, 1.24):
        for k in range(5):
            add(seg(x0 - 0.01, 0.019 + 0.004 * g(0, 1), x0 + 0.33, 0.019 + 0.004 * g(0, 1), 14), U_(22, 34), (0.45, 0.78, 0.74), 0.22, 0.6, 0.5, 0.5)
        for k in range(3):
            add(seg(x0 + 0.004 * k, 0.019, x0 + 0.32 - 0.004 * k, 0.019, 14), 9 - 2 * k, (0.80 + 0.07 * k, 1.0, 0.95), 1.0, 0.0, 0.8, 0.8)
    layer('adverts')
    bgs = [(0.52, 0.11, 0.11), (0.07, 0.13, 0.30), (0.50, 0.18, 0.34), (0.55, 0.40, 0.10), (0.08, 0.30, 0.22), (0.26, 0.16, 0.40)]
    INK = (0.86, 0.90, 0.86)
    for i in range(6):
        pcx = 0.145 + 0.262 * i; hw = 0.118; y0, y1 = 0.050, 0.096; cy = (y0 + y1) / 2
        hband(y0, y1, pcx - hw + 0.02, pcx + hw - 0.02, lambda x, y, c=bgs[i]: tex(c, x, y, 0.25), 16, (13, 17), ln=(0.05, 0.11))
        ox = pcx - 0.085
        if i == 0:                       # % SALE
            add(arc(ox - 0.012, cy - 0.011, 0.006, 0, 6.3, 9), 4.5, (1.0, 0.86, 0.35), 1, 0, .9, .9)
            add(arc(ox + 0.012, cy + 0.011, 0.006, 0, 6.3, 9), 4.5, (1.0, 0.86, 0.35), 1, 0, .9, .9)
            line([(ox + 0.014, cy - 0.018), (ox - 0.014, cy + 0.018)], 5, (1.0, 0.86, 0.35))
        elif i == 1:                     # 24/7
            G = {'2': [(0, .2), (.3, 0), (.8, 0), (1, .25), (.8, .5), (0, 1), (1, 1)], '4': [(.75, 1), (.75, 0), (0, .7), (1, .7)],
                 '/': [(.9, 0), (.1, 1)], '7': [(0, 0), (1, 0), (.35, 1)]}
            for k, ch in enumerate('24/7'):
                glyph(G[ch], ox - 0.028 + 0.021 * k, cy - 0.016, 0.014, 0.032, 4.5, INK)
        elif i == 2:                     # like
            line([(ox - 0.016, cy - 0.006), (ox, cy + 0.016), (ox + 0.016, cy - 0.006)], 6, INK)
            dot(ox - 0.008, cy - 0.008, 12, INK); dot(ox + 0.008, cy - 0.008, 12, INK)
        elif i == 3:                     # coffee to go
            for k in range(4):
                line([(ox - 0.011 + 0.002 * k, cy - 0.008 + 0.006 * k), (ox + 0.011 - 0.002 * k, cy - 0.008 + 0.006 * k)], 7, (0.20, 0.10, 0.06))
            line([(ox - 0.014, cy - 0.012), (ox + 0.014, cy - 0.012)], 4, INK)
            line([(ox - 0.004, cy - 0.018), (ox - 0.001, cy - 0.024), (ox - 0.005, cy - 0.03)], 2.5, INK, 0.8)
        elif i == 4:                     # growth
            line([(ox - 0.022, cy + 0.014), (ox - 0.008, cy + 0.002), (ox + 0.002, cy + 0.009), (ox + 0.02, cy - 0.014)], 4.5, INK)
            line([(ox + 0.009, cy - 0.014), (ox + 0.02, cy - 0.014), (ox + 0.02, cy - 0.003)], 4.5, INK)
        else:                            # smile
            add(arc(ox, cy, 0.016, 0, 6.3, 16), 4.5, (1.0, 0.84, 0.30), 1, 0, .9, .9)
            dot(ox - 0.006, cy - 0.005, 5, (1.0, 0.84, 0.30)); dot(ox + 0.006, cy - 0.005, 5, (1.0, 0.84, 0.30))
            add(arc(ox, cy + 0.001, 0.009, 0.4, 2.74, 7), 3.5, (1.0, 0.84, 0.30), 1, 0, .9, .9)
        for k in range(3):               # the small print
            yy = cy - 0.012 + 0.012 * k
            line([(pcx - 0.04, yy), (pcx - 0.04 + U_(0.07, 0.135), yy)], 5 if k == 0 else 3.2, mix(bgs[i], INK, 0.75 if k == 0 else 0.45), 0.95)
    layer('grab rail and straps')
    line(seg(-0.02, 0.117, AR + 0.02, 0.117, 30), 7, (0.045, 0.05, 0.085))
    line(seg(-0.02, 0.1145, AR + 0.02, 0.1145, 30), 2.5, (0.50, 0.42, 0.42), 0.9)
    x = 0.07
    while x < 1.26:
        sw = 0.005 * g(0, 1)
        line([(x, 0.118), (x + sw, 0.160)], 6, (0.07, 0.07, 0.11))
        add(arc(x + sw, 0.176, 0.015, -1.57, 4.72, 14), 5, (0.06, 0.06, 0.10), 1, 0, .9, .9)
        add(arc(x + sw, 0.176, 0.017, -1.2, 0.3, 6), 2.5, (1.0, 0.66, 0.36), 0.9, 0, .5, .5)
        x += 0.118


OCC = [(0.15, 0.085), (0.38, 0.085), (0.84, 0.085), (0.957, 0.036), (1.06, 0.092)]


def floor_lit(x, y):
    f = (FEET - G0[1]) / (y - G0[1])
    xq = G0[0] + (x - G0[0]) * f
    pen = 0.006 + 0.035 * abs(y - FEET)
    m = 1.0
    for c, hw in OCC:
        m = min(m, sstep((abs(xq - c) - hw) / pen + 0.5))
    return m * sstep((y - 0.795) / 0.05)


def floorc(x, y):
    L = floor_lit(x, y)
    dark = mix((0.05, 0.055, 0.095), (0.11, 0.11, 0.17), N2(x * 9, y * 40))
    I = L * (0.62 + 0.38 * math.exp(-((x - SUN[0]) / 0.6) ** 2)) * (0.78 + 0.34 * N2(x * 40 + 3, y * 40 + 7, 2))
    I = min(1.0, I)
    return mix(dark, mix((0.80, 0.36, 0.13), (1.0, 0.74, 0.34), sstep(I * 1.5 - 0.45)), I)


def paint_bench_floor():
    layer('bench')

    def backc(x, y):
        c = mix((0.03, 0.11, 0.14), (0.07, 0.21, 0.24), N2(x * 7, y * 7))
        return mix(c, (0.55, 0.36, 0.24), 0.30 * math.exp(-(y - 0.54) / 0.02) * math.exp(-((x - SUN[0]) / 0.5) ** 2))
    fill(lambda x, y: sd_box(x, y, 0.8, 0.598, 0.86, 0.062), (-0.02, 0.536, AR + 0.02, 0.66), backc, 300, (16, 30), ang=math.pi / 2, lmul=(2, 4), margin=0.3)
    for xb in (0.265, 0.495, 0.725, 0.95, 1.175, 1.405):
        line([(xb, 0.542), (xb + 0.002 * g(0, 1), 0.655)], 5, (0.015, 0.05, 0.07), 0.9)
    for k in range(30):
        x = U_(0, AR)
        line([(x, 0.5385), (x + U_(0.03, 0.09), 0.5385)], 3.2, mix((0.20, 0.34, 0.36), (1.0, 0.70, 0.38), math.exp(-((x - SUN[0]) / 0.45) ** 2)), 0.9)
    hband(SEAT_T, SEAT_F0, -0.02, AR + 0.02, lambda x, y: mix((0.11, 0.28, 0.30), (0.86, 0.56, 0.30), 0.7 * math.exp(-((x - SUN[0]) / 0.33) ** 2)), 60, (11, 15))
    hband(SEAT_F0, SEAT_F1, -0.02, AR + 0.02, lambda x, y: tex((0.045, 0.15, 0.18), x, y, 0.4, 12), 90, (16, 24))
    for k in range(26):
        x = U_(0, AR)
        line([(x, SEAT_F0), (x + U_(0.04, 0.12), SEAT_F0)], 3, (0.26, 0.50, 0.50), 0.85)
    hband(SEAT_F1, FLOOR0 + 0.004, -0.02, AR + 0.02, lambda x, y: tex((0.022, 0.026, 0.05), x, y, 0.5, 9), 90, (24, 36))
    for xb in (0.03, 0.495, 0.95, 1.405):
        line([(xb, SEAT_F1), (xb, FLOOR0 + 0.004)], 9, (0.07, 0.08, 0.12))
    layer('floor and long shadows')
    rad = lambda x, y: math.atan2(y - G0[1], x - G0[0])
    sdfl = lambda x, y: sd_box(x, y, 0.8, 0.90, 0.9, 0.13)
    fill(sdfl, (-0.02, FLOOR0, AR + 0.02, 1.03), floorc, 620, (18, 36), ang=rad, lmul=(4, 8), margin=0.2, aj=0.05, base=30)
    fill(sdfl, (-0.02, FLOOR0, AR + 0.02, 1.03), floorc, 620, (7, 13), ang=rad, lmul=(5, 12), margin=0.2, aj=0.04, dry=0.7, base=0)
    n = 0
    while n < 150:                                           # glints where the low sun rakes the scuffed floor
        x, y = hpt(0, AR, 0.83, 1.0)
        if floor_lit(x, y) < 0.9: continue
        a = rad(x, y); L = U_(0.01, 0.035)
        add(seg(x, y, x + L * math.cos(a), y + L * math.sin(a), 3), U_(3, 5), jit((1.0, 0.90, 0.60), 0.01, 0.02), 0.8, 0.6, 0.3, 0.3)
        n += 1


# ----------------------------------------------------------------------------- passengers
def sunrim(x, y, s):
    return mix((0.92, 0.40, 0.16), (1.0, 0.84, 0.52), s)


def glow(c, x, y, src, amt=1.0, rad=0.075):
    d = math.hypot(x - src[0], y - src[1])
    return mix(c, CYAN, min(0.9, amt * math.exp(-(d / rad) ** 2)))


def legs_and_shoes(cx, pants, shoe, src):
    sds = []
    for s in (-1, 1):
        hip = (cx + s * 0.036, 0.664); knee = (cx + s * 0.052, 0.708); ank = (cx + s * 0.056, 0.828)
        shin = lambda x, y, k=knee, a=ank: sd_cap(x, y, k[0], k[1], a[0], a[1], 0.027)
        thigh = lambda x, y, h=hip, k=knee: sd_cap(x, y, h[0], h[1], k[0], k[1], 0.037)
        sh = lambda x, y, s=s: sd_ell(x, y, cx + s * 0.061, 0.852, 0.036, 0.020)
        fill(shin, (cx - 0.1, 0.68, cx + 0.1, 0.86), lambda x, y: tex(pants, x, y, 0.5), 42, (11, 17), ang=math.pi / 2, lmul=(3, 6), aj=0.08)
        fill(thigh, (cx - 0.1, 0.62, cx + 0.1, 0.75), lambda x, y: glow(tex(pants, x, y, 0.4), x, y, src, 0.45, 0.09), 36, (12, 18), ang=math.pi / 2 + s * 0.3, lmul=(2, 4))
        fill(sh, (cx - 0.11, 0.82, cx + 0.11, 0.88), lambda x, y: tex(shoe, x, y, 0.4), 16, (8, 11), ang=0.0, lmul=(2, 4), aj=0.05)
        sds += [shin, thigh, sh]
    return sds


def finish(parts, cx, n=85, ymin=None):
    sil = lambda x, y: min(p(x, y) for p in parts)
    SIL.append(sil)
    rim(sil, (cx - 0.14, 0.32, cx + 0.14, 0.70), n, sunrim, w=(3.5, 6.5), light=SUN, thresh=0.38, ymin=ymin, ymax=0.69)
    rim(sil, (cx - 0.14, 0.68, cx + 0.14, 0.88), 12, lambda x, y, s: (0.80, 0.36, 0.16), w=(3, 4.5), light=SUN, thresh=0.5, ymin=0.69)
    return sil


def seated(cx, top, pants, shoe, skin, hair, style):
    hy = 0.385
    ph = (cx, 0.588); face = (cx, hy + 0.05)
    parts = legs_and_shoes(cx, pants, shoe, ph)
    torso = lambda x, y: sd_box(x, y, cx, 0.56, 0.075, 0.108, 0.04)
    fill(torso, (cx - 0.09, 0.44, cx + 0.09, 0.68), lambda x, y: glow(tex(top, x, y), x, y, ph, 0.38, 0.05), 40, (20, 30), ang=math.pi / 2, lmul=(2, 4))
    fill(torso, (cx - 0.09, 0.44, cx + 0.09, 0.68), lambda x, y: glow(tex(top, x, y), x, y, ph, 0.38, 0.05), 50, (9, 15), ang=math.pi / 2, lmul=(3, 6), aj=0.3, base=0)
    parts.append(torso)
    if style == 'man':
        tri = [(cx - 0.022, 0.452), (cx + 0.022, 0.452), (cx, 0.545)]
        fill(lambda x, y: sd_poly(x, y, tri), (cx - 0.03, 0.45, cx + 0.03, 0.55), lambda x, y: glow((0.34, 0.37, 0.50), x, y, ph, 1.0, 0.09), 14, (7, 10), ang=math.pi / 2, lmul=(2, 4))
        line([(cx, 0.462), (cx + 0.002, 0.535)], 9, (0.50, 0.10, 0.12), 1, 0, 0.9, 0.5)
    if style == 'parent':
        fill(lambda x, y: sd_ell(x, y, cx, 0.462, 0.05, 0.022), (cx - 0.06, 0.43, cx + 0.06, 0.49), lambda x, y: glow(tex((0.42, 0.37, 0.33), x, y), x, y, ph, 0.8, 0.12), 22, (8, 12), ang=0.0, lmul=(2, 5), aj=0.3)
    for k in range(int(14 * DENS)):                                   # phone light spilling on the chest
        rr = U_(0.02, 0.065); a0 = U_(3.3, 6.1)
        add(arc(ph[0], ph[1] - 0.01, rr * 0.8, a0, a0 + U_(0.5, 1.2)), U_(9, 15), mix(top, CYAN, 0.5), 0.10, 0.4, 0.5, 0.5)
    for s in (-1, 1):
        ua = lambda x, y, s=s: sd_cap(x, y, cx + s * 0.076, 0.48, cx + s * 0.090, 0.585, 0.026)
        fa = lambda x, y, s=s: sd_cap(x, y, cx + s * 0.090, 0.585, cx + s * 0.026, 0.603, 0.021)
        hand = lambda x, y, s=s: sd_ell(x, y, cx + s * 0.018, 0.604, 0.017, 0.015)
        dk = (top[0] * 0.8, top[1] * 0.8, top[2] * 0.8)
        fill(ua, (cx - 0.13, 0.44, cx + 0.13, 0.63), lambda x, y: tex(dk, x, y), 34, (11, 16), ang=math.pi / 2 - s * 0.12, lmul=(3, 6))
        fill(fa, (cx - 0.13, 0.55, cx + 0.13, 0.64), lambda x, y: glow(tex(top, x, y), x, y, ph, 0.5, 0.05), 26, (10, 14), ang=-s * 0.25 + (math.pi if s > 0 else 0), lmul=(2, 5))
        fill(hand, (cx - 0.05, 0.58, cx + 0.05, 0.63), lambda x, y: glow(skin, x, y, ph, 0.45, 0.05), 9, (7, 9), ang=0.0, lmul=(1.5, 3))
        parts += [ua, fa]
    neck = lambda x, y: sd_cap(x, y, cx, 0.42, cx, 0.455, 0.018)
    fill(neck, (cx - 0.03, 0.40, cx + 0.03, 0.48), lambda x, y: glow((skin[0] * 0.7, skin[1] * 0.7, skin[2] * 0.7), x, y, ph, 0.25, 0.12), 8, (8, 11), ang=math.pi / 2)
    head = lambda x, y: sd_ell(x, y, cx, hy, 0.033, 0.042)
    if style == 'woman':                                               # long hair falling forward
        for s in (-1, 1):
            hh = lambda x, y, s=s: sd_cap(x, y, cx + s * 0.03, hy, cx + s * 0.04, 0.475, 0.017)
            fill(hh, (cx - 0.07, 0.36, cx + 0.07, 0.50), lambda x, y: tex(hair, x, y, 0.6), 22, (6, 9), ang=math.pi / 2, lmul=(4, 9))
            parts.append(hh)
        parts.append(lambda x, y: sd_box(x, y, cx, 0.43, 0.04, 0.045, 0.0))
    fill(head, (cx - 0.04, hy - 0.05, cx + 0.04, hy + 0.05), lambda x, y: glow(tex(skin, x, y, 0.2), x, y, (cx, hy + 0.060), 0.34, 0.026), 46, (8, 12), ang=math.pi / 2, lmul=(2, 4), aj=0.4)
    hl = {'man': 0.000, 'woman': 0.012, 'parent': 0.008}[style]
    hairsd = lambda x, y: max(sd_ell(x, y, cx, hy - 0.002, 0.036, 0.045), y - (hy + hl - 0.008 * abs(x - cx) / 0.033))
    fill(hairsd, (cx - 0.045, hy - 0.055, cx + 0.045, hy + 0.02), lambda x, y: tex(hair, x, y, 0.7), 50, (6, 9),
         ang=lambda x, y: math.atan2(y - (hy - 0.028), x - cx), flow=True, lmul=(3, 7), margin=0.2)
    parts += [neck, head, lambda x, y: sd_ell(x, y, cx, hy - 0.002, 0.036, 0.045)]
    fc = glow(skin, cx, hy + 0.03, face, 0.5, 0.05)
    line([(cx - 0.022, hy + hl + 0.007), (cx + 0.022, hy + hl + 0.007)], 6, (skin[0] * 0.38, skin[1] * 0.36, skin[2] * 0.45), 0.75)   # eyes in shadow
    line([(cx, hy + hl + 0.012), (cx, hy + 0.034)], 4, mix(fc, (0.86, 0.96, 1.0), 0.45), 0.85)                                        # nose
    if style == 'man':                                                 # glasses catching the screen
        for s in (-1, 1):
            add(arc(cx + s * 0.014, hy + 0.019, 0.0085, 0, 6.3, 10), 2.6, (0.03, 0.03, 0.05), 1, 0, .9, .9)
            line([(cx + s * 0.010, hy + 0.019), (cx + s * 0.018, hy + 0.019)], 5, (0.82, 0.97, 1.0), 0.95)
        line([(cx - 0.006, hy + 0.017), (cx + 0.006, hy + 0.017)], 2.4, (0.03, 0.03, 0.05))
    if style == 'woman':                                               # headphones
        add(arc(cx, hy - 0.004, 0.040, 3.35, 6.07, 14), 7, (0.10, 0.36, 0.40), 1, 0, .9, .9)
        for s in (-1, 1):
            cup = lambda x, y, s=s: sd_ell(x, y, cx + s * 0.037, hy + 0.008, 0.011, 0.018)
            fill(cup, (cx - 0.06, hy - 0.02, cx + 0.06, hy + 0.04), lambda x, y: (0.12, 0.42, 0.46), 6, (7, 9), ang=math.pi / 2, lmul=(1.5, 3))
            parts.append(cup)
    if style == 'parent':                                              # bun
        bun = lambda x, y: sd_ell(x, y, cx + 0.004, hy - 0.052, 0.018, 0.016)
        fill(bun, (cx - 0.03, hy - 0.08, cx + 0.03, hy - 0.03), lambda x, y: tex(hair, x, y, 0.7), 10, (6, 8),
             ang=lambda x, y: math.atan2(y - (hy - 0.052), x - cx) + 1.57, flow=True, lmul=(2, 5), margin=0.2)
        parts.append(bun)
    # the phone, seen from behind: a dark slab with a halo
    phsd = lambda x, y: sd_box(x, y, cx, 0.585, 0.0155, 0.027, 0.004)
    fill(phsd, (cx - 0.02, 0.55, cx + 0.02, 0.62), lambda x, y: (0.035, 0.035, 0.055), 9, (7, 9), ang=math.pi / 2, lmul=(2, 4), margin=0.3, aj=0.02)
    line([(cx - 0.015, 0.557), (cx + 0.015, 0.557)], 3.5, (0.88, 0.98, 1.0))
    line([(cx - 0.0175, 0.56), (cx - 0.0175, 0.60)], 2.5, (0.60, 0.88, 1.0), 0.8)
    line([(cx + 0.0175, 0.56), (cx + 0.0175, 0.60)], 2.5, (0.60, 0.88, 1.0), 0.8)
    return finish(parts, cx)


def paint_man():
    layer('passenger: the office worker')
    seated(0.15, (0.075, 0.085, 0.15), (0.065, 0.07, 0.12), (0.04, 0.035, 0.045), (0.40, 0.27, 0.22), (0.08, 0.07, 0.08), 'man')


def paint_woman():
    layer('passenger: headphones')
    seated(0.38, (0.42, 0.28, 0.07), (0.10, 0.14, 0.27), (0.42, 0.42, 0.50), (0.33, 0.21, 0.17), (0.20, 0.08, 0.06), 'woman')


def paint_parent():
    layer('passenger: the parent')
    cx = 0.84
    tx = cx + 0.117                                                    # shopping at her feet
    line([(tx - 0.012, 0.775), (tx - 0.034, 0.70)], 7, (0.20, 0.36, 0.14))
    line([(tx - 0.004, 0.775), (tx - 0.016, 0.705)], 6, (0.28, 0.44, 0.16))
    line([(tx + 0.012, 0.78), (tx + 0.03, 0.695)], 13, (0.50, 0.32, 0.15))
    line([(tx + 0.017, 0.76), (tx + 0.033, 0.70)], 4, (1.0, 0.78, 0.45), 0.9)
    tote = lambda x, y: sd_box(x, y, tx, 0.812, 0.036, 0.046, 0.006)
    fill(tote, (tx - 0.05, 0.75, tx + 0.05, 0.87), lambda x, y: tex((0.33, 0.29, 0.28), x, y, 0.3), 26, (10, 15), ang=math.pi / 2, lmul=(2, 5))
    add(arc(tx, 0.775, 0.02, 3.3, 6.1, 9), 4, (0.22, 0.19, 0.19), 1, 0, .9, .9)
    rim(tote, (tx - 0.06, 0.74, tx + 0.06, 0.88), 14, sunrim, w=(3, 5), light=SUN, thresh=0.3)
    seated(cx, (0.29, 0.11, 0.25), (0.06, 0.06, 0.09), (0.10, 0.055, 0.05), (0.40, 0.27, 0.22), (0.11, 0.08, 0.07), 'parent')


def paint_kid():
    layer('the child at the window')
    cx = 0.61
    parts = []
    jacket = (0.50, 0.11, 0.12)
    for s in (-1, 1):
        calf = lambda x, y, s=s: sd_cap(x, y, cx + s * 0.028, 0.612, cx + s * 0.031, 0.652, 0.022)
        fill(calf, (cx - 0.06, 0.58, cx + 0.06, 0.68), lambda x, y: tex((0.12, 0.17, 0.33), x, y), 12, (9, 12), ang=math.pi / 2, lmul=(1.5, 3))
        sole = lambda x, y, s=s: sd_ell(x, y, cx + s * 0.032, 0.672, 0.019, 0.027)
        fill(sole, (cx - 0.06, 0.64, cx + 0.06, 0.70), lambda x, y: tex((0.40, 0.38, 0.42), x, y, 0.2), 12, (7, 9), ang=math.pi / 2, lmul=(1.5, 3))
        for k in range(4):
            yy = 0.656 + 0.010 * k
            line([(cx + s * 0.032 - 0.012, yy), (cx + s * 0.032 + 0.012, yy)], 2.5, (0.17, 0.16, 0.20), 0.9)
        parts += [calf, sole]
    torso = lambda x, y: sd_box(x, y, cx, 0.54, 0.053, 0.083, 0.032)
    fill(torso, (cx - 0.07, 0.45, cx + 0.07, 0.63), lambda x, y: tex(jacket, x, y, 0.45), 40, (14, 22), ang=math.pi / 2, lmul=(2, 4))
    fill(torso, (cx - 0.07, 0.45, cx + 0.07, 0.63), lambda x, y: tex(jacket, x, y, 0.5), 26, (7, 11), ang=math.pi / 2, lmul=(3, 6), aj=0.3, base=0)
    hood = lambda x, y: sd_ell(x, y, cx, 0.462, 0.043, 0.022)
    fill(hood, (cx - 0.05, 0.43, cx + 0.05, 0.49), lambda x, y: tex((0.40, 0.085, 0.10), x, y), 14, (8, 11), ang=0.0, lmul=(2, 4), aj=0.3)
    la = lambda x, y: sd_cap(x, y, cx - 0.047, 0.485, cx - 0.074, 0.522, 0.017)
    ra1 = lambda x, y: sd_cap(x, y, cx + 0.047, 0.485, cx + 0.078, 0.462, 0.017)
    ra2 = lambda x, y: sd_cap(x, y, cx + 0.078, 0.462, cx + 0.090, 0.418, 0.014)
    for a_, an in ((la, 2.2), (ra1, -0.6), (ra2, -1.3)):
        fill(a_, (cx - 0.11, 0.39, cx + 0.12, 0.55), lambda x, y: tex(jacket, x, y, 0.5), 12, (8, 11), ang=an, lmul=(2, 4))
    skin = (0.50, 0.30, 0.22)
    for hx_, hy_ in ((cx - 0.079, 0.518), (cx + 0.092, 0.405)):
        hs = lambda x, y, a=hx_, b=hy_: sd_ell(x, y, a, b, 0.012, 0.013)
        fill(hs, (hx_ - 0.02, hy_ - 0.02, hx_ + 0.02, hy_ + 0.02), lambda x, y: skin, 6, (6, 8), ang=math.pi / 2, lmul=(1.5, 3))
        parts.append(hs)
    head = lambda x, y: sd_ell(x, y, cx + 0.004, 0.405, 0.031, 0.036)
    hair = (0.17, 0.085, 0.05)
    fill(head, (cx - 0.04, 0.36, cx + 0.05, 0.45), lambda x, y: tex(hair, x, y, 0.7), 46, (6, 9),
         ang=lambda x, y: math.atan2(y - 0.385, x - (cx + 0.004)), flow=True, lmul=(3, 7), margin=0.2)
    for s in (-1, 1):
        pg = lambda x, y, s=s: sd_ell(x, y, cx + 0.004 + s * 0.036, 0.418, 0.010, 0.016)
        fill(pg, (cx - 0.06, 0.39, cx + 0.07, 0.45), lambda x, y: tex(hair, x, y, 0.7), 7, (5, 7), ang=math.pi / 2, lmul=(2, 4), margin=0.2)
        dot(cx + 0.004 + s * 0.031, 0.407, 6, (0.95, 0.80, 0.25))
        parts.append(pg)
    parts += [torso, hood, la, ra1, ra2, head]
    sil = finish(parts, cx, n=120)
    rim(sil, (cx - 0.12, 0.36, cx + 0.13, 0.70), 40, lambda x, y, s: (1.0, 0.93, 0.70), w=(2.5, 4), light=SUN, thresh=0.45, inset=0.002)


def paint_courier():
    layer('passenger: the courier, asleep')
    cx = 1.06
    ph = (cx + 0.055, 0.512)
    jacket = (0.17, 0.30, 0.10); skin = (0.36, 0.23, 0.18)
    parts = legs_and_shoes(cx, (0.07, 0.07, 0.10), (0.32, 0.36, 0.16), ph)
    torso = lambda x, y: sd_box(x, y, cx, 0.545, 0.088, 0.075, 0.045)
    fill(torso, (cx - 0.1, 0.46, cx + 0.1, 0.63), lambda x, y: glow(tex(jacket, x, y), x, y, ph, 0.5, 0.07), 50, (14, 22), ang=math.pi / 2, lmul=(2, 4))
    for s in (-1, 1):
        ua = lambda x, y, s=s: sd_cap(x, y, cx + s * 0.090, 0.505, cx + s * 0.094, 0.585, 0.025)
        fill(ua, (cx - 0.13, 0.47, cx + 0.13, 0.62), lambda x, y: tex((jacket[0] * 0.8, jacket[1] * 0.8, jacket[2] * 0.8), x, y), 26, (10, 15), ang=math.pi / 2, lmul=(2, 5))
        line([(cx + s * 0.070, 0.555), (cx + s * 0.118, 0.553)], 4.5, (0.55, 0.60, 0.62), 0.9)          # hi-vis band
        parts.append(ua)
    BAG = (0.64, 0.17, 0.15)
    bag = lambda x, y: sd_box(x, y, cx, 0.592, 0.079, 0.072, 0.008)
    fill(bag, (cx - 0.09, 0.51, cx + 0.09, 0.67), lambda x, y: glow(tex(BAG, x, y, 0.3, 14), x, y, ph, 0.35, 0.06), 60, (14, 22), ang=0.0, lmul=(2, 5), aj=0.05)
    fill(bag, (cx - 0.09, 0.51, cx + 0.09, 0.67), lambda x, y: tex(BAG, x, y, 0.5, 14), 30, (7, 11), ang=math.pi / 2, lmul=(3, 7), aj=0.05, base=0)
    for k in range(int(8 * DENS)):
        x = U_(cx - 0.07, cx + 0.03)
        line([(x, 0.524), (x + U_(0.02, 0.05), 0.524)], 5, glow((0.86, 0.36, 0.28), x, 0.524, ph, 0.7, 0.06), 0.9)
    add(arc(cx, 0.598, 0.026, 0, 6.3, 22), 6, (0.80, 0.78, 0.74), 1, 0, .9, .9)
    for dx in (-0.008, 0.0, 0.008):
        line([(cx + dx, 0.584), (cx + dx, 0.598)], 3, (0.80, 0.78, 0.74))
    line([(cx, 0.598), (cx, 0.614)], 4, (0.80, 0.78, 0.74))
    line([(cx - 0.076, 0.648), (cx + 0.076, 0.648)], 5, (0.52, 0.56, 0.62), 0.95)
    for s in (-1, 1):
        fa = lambda x, y, s=s: sd_cap(x, y, cx + s * 0.094, 0.585, cx + s * 0.058, 0.566, 0.020)
        fill(fa, (cx - 0.13, 0.53, cx + 0.13, 0.62), lambda x, y: tex(jacket, x, y), 18, (9, 13), ang=0.4 * s, lmul=(2, 4))
        hand = lambda x, y, s=s: sd_ell(x, y, cx + s * 0.050, 0.562, 0.015, 0.013)
        fill(hand, (cx - 0.08, 0.54, cx + 0.08, 0.59), lambda x, y: glow(skin, x, y, ph, 0.8, 0.06), 7, (6, 8), ang=0.0, lmul=(1.5, 3))
        parts += [fa]
    hc = (cx - 0.014, 0.487)
    head = lambda x, y: sd_ell(x, y, hc[0], hc[1], 0.044, 0.034)
    fill(head, (hc[0] - 0.05, hc[1] - 0.04, hc[0] + 0.05, hc[1] + 0.04), lambda x, y: glow(tex(skin, x, y, 0.2), x, y, ph, 1.0, 0.065), 40, (8, 11), ang=0.0, lmul=(2, 4), aj=0.4)
    helm = lambda x, y: max(sd_ell(x, y, hc[0] - 0.003, hc[1] - 0.003, 0.049, 0.039), 0.8 * (x - hc[0]) + (y - hc[1]) * 0.9 - 0.004)
    fill(helm, (hc[0] - 0.06, hc[1] - 0.05, hc[0] + 0.05, hc[1] + 0.04), lambda x, y: tex((0.36, 0.38, 0.47), x, y, 0.3), 40, (7, 10),
         ang=lambda x, y: math.atan2(y - hc[1], x - hc[0]) + 1.57, flow=True, lmul=(3, 7), margin=0.2)
    for k in range(3):
        a0 = 3.5 + 0.45 * k
        line([(hc[0] + 0.018 * math.cos(a0), hc[1] + 0.014 * math.sin(a0)), (hc[0] + 0.036 * math.cos(a0), hc[1] + 0.028 * math.sin(a0))], 4, (0.10, 0.10, 0.14))
    add(arc(hc[0] + 0.019, hc[1] + 0.006, 0.006, 0.3, 2.8, 6), 2.6, (0.10, 0.06, 0.07), 1, 0, .8, .8)   # closed eye
    line([(hc[0] + 0.030, hc[1] + 0.016), (hc[0] + 0.036, hc[1] + 0.018)], 2.6, (0.16, 0.08, 0.08))
    parts += [torso, bag, head, lambda x, y: sd_ell(x, y, hc[0] - 0.003, hc[1] - 0.003, 0.049, 0.039)]
    for k in range(int(12 * DENS)):
        rr = U_(0.012, 0.05); a0 = U_(3.3, 6.0)
        add(arc(ph[0], ph[1], rr, a0, a0 + U_(0.5, 1.2)), U_(7, 12), CYAN, 0.16, 0.6, 0.5, 0.5)
    line([(ph[0] - 0.02, ph[1] + 0.003), (ph[0] + 0.02, ph[1] - 0.001)], 8, (0.05, 0.05, 0.07))
    line([(ph[0] - 0.018, ph[1] - 0.001), (ph[0] + 0.018, ph[1] - 0.005)], 4.5, (0.86, 0.98, 1.0))
    finish(parts, cx)


def paint_shafts():
    layer('light through the dusty air')
    n = 0; tries = 0
    while n < 60 and tries < 20000:
        tries += 1
        x, y = hpt(0.0, 1.5, 0.31, WB)
        if sd_win(x, y) > -0.012: continue
        if any(s(x, y) < 0.004 for s in SIL): continue
        dx, dy = x - SUN[0], y - SUN[1]; d = math.hypot(dx, dy)
        if d < 0.05 or dy / d < 0.22: continue
        dx /= d; dy /= d
        L = U_(0.25, 0.75)
        add(seg(x, y, x + dx * L, y + dy * L, 14), U_(45, 120), jit((1.0, 0.76, 0.44), 0.01, 0.02), U_(0.035, 0.07), 0.15, 0.3, 0.9, 0.02)
        n += 1
    for k in range(0):
        a = U_(0, 6.283); r0 = U_(0.02, 0.035); r1 = U_(0.06, 0.17)
        add(seg(SUN[0] + r0 * math.cos(a), SUN[1] + r0 * math.sin(a), SUN[0] + r1 * math.cos(a), SUN[1] + r1 * math.sin(a), 5), U_(3, 6),
            (1.0, 0.95, 0.75), 0.30, 0.5, 0.9, 0.05)


def paint_pole():
    layer('grab pole')
    xp = 0.495
    y = -0.03
    while y < 1.0:
        add(seg(xp + 0.0006 * g(0, 1), y, xp + 0.0006 * g(0, 1), y + 0.26, 12), 34, jit((0.085, 0.095, 0.15), 0.006, 0.03), 1.0, 0.0, 0.95, 0.95)
        y += 0.2
    y = -0.03
    while y < 1.0:
        k = math.exp(-((y + 0.06 - 0.33) / 0.30) ** 2)
        add(seg(xp + 0.0055, y, xp + 0.0055, y + 0.14, 8), 8, mix((0.55, 0.30, 0.20), (1.0, 0.88, 0.60), k), 0.95, 0.2, 0.8, 0.8)
        add(seg(xp - 0.0062, y, xp - 0.0062, y + 0.14, 8), 4, (0.28, 0.46, 0.52), 0.85, 0.2, 0.8, 0.8)
        y += 0.115


def paint_standing():
    layer('the standing passenger')
    BODY = [(1.300, 0.405), (1.36, 0.378), (1.40, 0.366), (1.47, 0.366), (1.52, 0.378), (1.585, 0.405), (1.64, 0.48), (1.64, 1.04), (1.318, 1.04),
            (1.302, 0.80), (1.288, 0.56)]
    pc = (1.193, 0.455); phw, phh, rot = 0.037, 0.071, -0.10
    body = lambda x, y: sd_poly(x, y, BODY)

    def coatc(x, y):
        c = mix((0.03, 0.035, 0.07), (0.085, 0.095, 0.16), N2(x * 5, y * 3))
        return glow(c, x, y, (1.24, 0.50), 0.25, 0.12)
    for k in range(int(26 * DENS)):                                   # screen light in the air
        rr = U_(0.05, 0.13); a0 = U_(0, 6.283)
        add(arc(pc[0], pc[1], rr, a0, a0 + U_(0.5, 1.3)), U_(16, 28), CYAN, 0.10, 0.7, 0.5, 0.5)
    fill(body, (1.27, 0.36, 1.64, 1.04), coatc, 130, (30, 54), ang=math.pi / 2, lmul=(2.5, 5), margin=0.3, aj=0.15)
    fill(body, (1.27, 0.36, 1.64, 1.04), coatc, 150, (10, 18), ang=math.pi / 2, lmul=(4, 9), margin=0.3, aj=0.2, dry=0.6, base=0)
    pack = lambda x, y: sd_box(x, y, 1.478, 0.68, 0.102, 0.20, 0.05)
    fill(pack, (1.36, 0.47, 1.60, 0.90), lambda x, y: tex((0.12, 0.15, 0.23), x, y, 0.3, 10), 80, (16, 28), ang=math.pi / 2, lmul=(2.5, 5))
    fill(pack, (1.36, 0.47, 1.60, 0.90), lambda x, y: tex((0.15, 0.18, 0.27), x, y, 0.4, 14), 60, (7, 11), ang=math.pi / 2, lmul=(4, 8), dry=0.7, base=0)
    line([(1.40, 0.56), (1.556, 0.56)], 4, (0.27, 0.31, 0.42), 0.95)                        # zip
    line([(1.40, 0.72), (1.40, 0.84), (1.556, 0.84), (1.556, 0.72)], 3.5, (0.06, 0.07, 0.11), 0.95)   # pocket
    line([(1.385, 0.60), (1.385, 0.66)], 6, (0.95, 0.60, 0.28))                               # reflective tab
    rim(pack, (1.36, 0.47, 1.60, 0.90), 26, lambda x, y, s: mix((0.40, 0.24, 0.20), (0.95, 0.55, 0.26), s), w=(3, 5), light=SUN, thresh=0.5, ln=(0.03, 0.08))
    for xs in (1.392, 1.555):
        line([(xs, 0.385), (xs + 0.006, 0.50)], 14, (0.06, 0.065, 0.10))
    neck = lambda x, y: sd_cap(x, y, 1.435, 0.31, 1.435, 0.372, 0.034)
    fill(neck, (1.39, 0.29, 1.48, 0.40), lambda x, y: tex((0.085, 0.06, 0.07), x, y), 14, (10, 14), ang=math.pi / 2, lmul=(1.5, 3))
    cheek = lambda x, y: sd_ell(x, y, 1.389, 0.298, 0.014, 0.028)
    fill(cheek, (1.37, 0.26, 1.41, 0.33), lambda x, y: glow((0.42, 0.27, 0.22), x, y, (1.36, 0.31), 0.85, 0.035), 12, (6, 8), ang=math.pi / 2, lmul=(2, 4), margin=0.2)
    head = lambda x, y: sd_ell(x, y, 1.435, 0.262, 0.056, 0.068)
    fill(head, (1.37, 0.19, 1.50, 0.34), lambda x, y: tex((0.055, 0.045, 0.06), x, y, 0.8), 60, (8, 12), ang=math.pi / 2, lmul=(3, 6), aj=0.2)
    HAT = (0.36, 0.24, 0.07)
    beanie = lambda x, y: max(sd_ell(x, y, 1.435, 0.238, 0.061, 0.056), y - 0.272)
    fill(beanie, (1.36, 0.17, 1.51, 0.28), lambda x, y: tex(HAT, x, y, 0.4), 60, (9, 13), ang=math.pi / 2, lmul=(3, 6), aj=0.1, margin=0.2)
    for k in range(int(14 * DENS)):
        x = U_(1.38, 1.49)
        line([(x, 0.250), (x + 0.001, 0.270)], 3.2, (HAT[0] * 0.6, HAT[1] * 0.6, HAT[2] * 0.6), 0.8)
    ear = lambda x, y: sd_ell(x, y, 1.381, 0.284, 0.009, 0.016)
    fill(ear, (1.36, 0.26, 1.40, 0.31), lambda x, y: glow((0.44, 0.28, 0.23), x, y, (1.37, 0.29), 0.7, 0.03), 6, (5, 7), ang=math.pi / 2, lmul=(1.5, 3), margin=0.2)
    line([(1.3815, 0.284), (1.380, 0.300)], 5, (0.90, 0.96, 1.0))
    ua = lambda x, y: sd_cap(x, y, 1.318, 0.445, 1.268, 0.612, 0.046)
    fa = lambda x, y: sd_cap(x, y, 1.268, 0.612, 1.222, 0.560, 0.034)
    fill(ua, (1.20, 0.38, 1.38, 0.68), coatc, 60, (14, 24), ang=1.86, lmul=(3, 6))
    fill(fa, (1.17, 0.50, 1.32, 0.67), coatc, 40, (12, 18), ang=0.85, lmul=(2, 4))
    # --- the phone, its screen turned toward us
    cr, sr = math.cos(rot), math.sin(rot)
    Lc = lambda u, v: (pc[0] + u * phw * cr - v * phh * sr, pc[1] + u * phw * sr + v * phh * cr)
    hand = lambda x, y: sd_ell(x, y, 1.207, 0.533, 0.026, 0.026)
    fill(hand, (1.17, 0.49, 1.25, 0.57), lambda x, y: glow((0.36, 0.23, 0.19), x, y, (1.20, 0.50), 0.8, 0.04), 16, (7, 10), ang=0.3, lmul=(1.5, 3))
    psd = lambda x, y: sd_rbox(x, y, pc[0], pc[1], phw, phh, rot, 0.008)
    fill(psd, (1.14, 0.37, 1.25, 0.54), lambda x, y: (0.025, 0.025, 0.04), 34, (9, 12), ang=math.pi / 2 + rot, lmul=(3, 6), margin=0.3, aj=0.02)
    PH = [(-0.76, (0.40, 0.24, 0.55)), (-0.45, (0.85, 0.40, 0.45)), (-0.15, (1.0, 0.62, 0.28)), (0.07, (1.0, 0.84, 0.45))]
    rows = 30
    for k in range(rows):
        v = -0.92 + 1.84 * k / (rows - 1)
        if -0.74 <= v <= 0.20:
            c = (0.11, 0.06, 0.12) if v > 0.09 else ramp(PH, v)
        else:
            c = (0.93, 0.96, 1.0)
        line([Lc(-0.86, v), Lc(0.0, v), Lc(0.86, v)], 9.5, jit(c, 0.006, 0.01), 1.0, 0.0, 0.95, 0.95)
    for u0, v0 in ((-0.5, 0.06), (-0.15, 0.03), (0.45, 0.05), (0.7, 0.02)):       # skyline in the photo
        line([Lc(u0 - 0.12, v0), Lc(u0 + 0.12, v0)], 7, (0.11, 0.06, 0.12))
    p = Lc(0.18, -0.20); dot(p[0], p[1], 13, (1.0, 0.97, 0.80))
    p = Lc(0.18, -0.20)
    add(arc(p[0], p[1], 0.008, 0, 6.3, 10), 4, (1.0, 0.88, 0.55), 0.8, 0, .8, .8)
    p = Lc(-0.70, -0.84); dot(p[0], p[1], 9, (0.55, 0.60, 0.70))
    line([Lc(-0.48, -0.84), Lc(0.1, -0.84)], 3.5, (0.45, 0.48, 0.56))
    hx_, hv = -0.66, 0.36                                                          # the like
    line([Lc(hx_ - 0.16, hv - 0.03), Lc(hx_, hv + 0.07), Lc(hx_ + 0.16, hv - 0.03)], 4.5, (0.93, 0.16, 0.22))
    for du in (-0.085, 0.085):
        p = Lc(hx_ + du, hv - 0.035); dot(p[0], p[1], 7.5, (0.93, 0.16, 0.22))
    add(arc(*Lc(-0.25, 0.36), 0.0055, 0, 6.3, 9), 2.4, (0.35, 0.38, 0.46), 1, 0, .9, .9)
    line([Lc(0.05, 0.40), Lc(0.20, 0.33), Lc(0.08, 0.36)], 2.4, (0.35, 0.38, 0.46))
    for v0, u1 in ((0.52, 0.55), (0.62, 0.15), (0.72, 0.4)):
        line([Lc(-0.78, v0), Lc(u1, v0)], 3.2, (0.50, 0.53, 0.60))
    for u0 in (-0.6, -0.2, 0.2, 0.6):
        p = Lc(u0, 0.87); dot(p[0], p[1], 5.5, (0.30, 0.32, 0.40))
    thumb = lambda x, y: sd_cap(x, y, 1.228, 0.545, 1.214, 0.508, 0.0095)
    fill(thumb, (1.19, 0.48, 1.25, 0.57), lambda x, y: glow((0.40, 0.26, 0.21), x, y, (1.21, 0.50), 0.9, 0.03), 9, (6, 8), ang=1.9, lmul=(2, 4), margin=0.2)
    # --- light on the figure
    sil = lambda x, y: min(body(x, y), neck(x, y), head(x, y), beanie(x, y), ua(x, y), fa(x, y), pack(x, y))
    rim(sil, (1.17, 0.17, 1.62, 1.02), 190, sunrim, w=(5, 9), light=SUN, thresh=0.3, ln=(0.03, 0.09))
    rim(sil, (1.17, 0.17, 1.62, 0.70), 50, lambda x, y, s: (1.0, 0.94, 0.72), w=(2.5, 4), light=SUN, thresh=0.55, inset=0.002, ln=(0.03, 0.08))
    arm = lambda x, y: min(ua(x, y), fa(x, y))
    rim(arm, (1.17, 0.45, 1.33, 0.68), 30, lambda x, y, s: mix((0.20, 0.40, 0.52), CYAN, s), w=(3, 5), light=pc, thresh=0.35, ymin=0.47)


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
    ap.add_argument('--out', default='commute'); ap.add_argument('--frames', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out + '/frames', exist_ok=True)
    paint_ground(); paint_view(); paint_wall(); paint_ceiling(); paint_bench_floor()
    paint_man(); paint_woman(); paint_kid(); paint_parent(); paint_courier()
    paint_shafts(); paint_pole(); paint_standing()
    layer('end')
    N = len(strokes)
    for k in range(len(layer_marks) - 1):
        print(f'{layer_marks[k][0]:>34}: {layer_marks[k + 1][1] - layer_marks[k][1]}')
    print('TOTAL STROKES', N)
    P = np.zeros((N, MAXP, 2), np.float32); NP = np.zeros(N, np.int64)
    WD = np.zeros(N); COL = np.zeros((N, 3)); OP = np.zeros(N); DRY = np.zeros(N); T0 = np.zeros(N); T1 = np.zeros(N); CM = np.zeros(N)
    for i, (pts, w, c, op, dr, t0, t1, cm) in enumerate(strokes):
        NP[i] = len(pts); P[i, :len(pts)] = np.array(pts, np.float32) * S
        WD[i] = w; COL[i] = c; OP[i] = op; DRY[i] = dr; T0[i] = t0; T1[i] = t1; CM[i] = cm
    can = np.empty((H, W, 3), np.float32); can[:] = np.array([0.93, 0.905, 0.85], np.float32)
    hgt = np.zeros((H, W), np.float32)
    A = np.zeros((H, W), np.float32); Ub = np.zeros((H, W), np.float32); Tb = np.zeros((H, W), np.float32)
    marks = sorted(set([0] + [int(round(N * (k / a.frames) ** 1.25)) for k in range(1, a.frames + 1)])) if a.frames else [0, N]
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
    Image.fromarray(fin).resize((1400, round(1400 * H / W)), Image.LANCZOS).save(a.out + '/small.png')


if __name__ == '__main__':
    main()
