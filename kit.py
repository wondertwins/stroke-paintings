"""Figure kit for the 'CEO steps down' paintings: local coordinate frames, a simple lit-colour shader,
and the recurring character (grey hair, rectangular glasses, black leather jacket) from three views."""
import math
import engine as E
from engine import *

pi = math.pi
SKIN = (0.92, 0.74, 0.57)
HAIR = (0.60, 0.61, 0.64)
LEATHER = (0.05, 0.05, 0.06)
TEE = (0.085, 0.085, 0.095)
FRAME = (0.035, 0.035, 0.045)

LIGHTS = []      # ('d', (lx, ly, lz), col), ('f', ...) = fill with no highlights, or ('p', (x, y, z), col, radius)
AMB = [(0.12, 0.12, 0.14)]


def set_lights(lights, amb):
    LIGHTS[:] = lights; AMB[0] = amb


def _norm(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-9
    return (v[0] / n, v[1] / n, v[2] / n)


def lit(sd, base, rad=0.03, spec=0.0, shin=24.0, wr=0.0, wf=40.0, texa=0.12, texf=30.0):
    """Colour function: base colour shaded as if the shape were softly rounded (normals from the SDF)."""
    e = 0.0012

    def f(x, y):
        d = sd(x, y)
        gx = sd(x + e, y) - sd(x - e, y); gy = sd(x, y + e) - sd(x, y - e)
        gn = math.hypot(gx, gy) + 1e-9
        k = min(max(1.0 + d / rad, 0.0), 1.0) ** 1.6 * 0.93
        nx = gx / gn * k; ny = gy / gn * k
        if wr:
            nx += wr * (N2(x * wf, y * wf, 3) - 0.5); ny += wr * (N2(x * wf + 31.0, y * wf + 7.0, 3) - 0.5)
        nz = math.sqrt(max(0.03, 1.0 - nx * nx - ny * ny))
        n = _norm((nx, ny, nz))
        b = base(x, y) if callable(base) else base
        if texa:
            t = 1 - texa / 2 + texa * N2(x * texf, y * texf, 3)
            b = (b[0] * t, b[1] * t, b[2] * t)
        r, gg, bl = b[0] * AMB[0][0], b[1] * AMB[0][1], b[2] * AMB[0][2]
        for L in LIGHTS:
            if L[0] in 'df':
                l = _norm(L[1]); att = 1.0
            else:
                v = (L[1][0] - x, L[1][1] - y, L[1][2]); dist = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                l = _norm(v); att = 1.0 / (1.0 + (dist / L[3]) ** 2)
            c = L[2]
            df = max(0.0, n[0] * l[0] + n[1] * l[1] + n[2] * l[2]) * att
            r += b[0] * df * c[0]; gg += b[1] * df * c[1]; bl += b[2] * df * c[2]
            if spec and L[0] != 'f':
                h = _norm((l[0], l[1], l[2] + 1.0))
                s = max(0.0, n[0] * h[0] + n[1] * h[1] + n[2] * h[2]) ** shin * spec * att
                r += s * c[0]; gg += s * c[1]; bl += s * c[2]
        return (r, gg, bl)
    return f


def sd_tcap(x, y, ax, ay, bx, by, ra, rb):
    vx = bx - ax; vy = by - ay; wx = x - ax; wy = y - ay
    t = min(max((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-12), 0.0), 1.0)
    return math.hypot(wx - t * vx, wy - t * vy) - (ra + (rb - ra) * t)


def smooth(pts, n=8):
    """Catmull-Rom through the points."""
    if len(pts) < 3: return list(pts)
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n; t2 = t * t; t3 = t2 * t
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in (0, 1)))
    out.append(pts[-1])
    return out[:MAXP]


class Fr:
    """A local frame: origin, scale (scene units per local unit), rotation, optional mirror."""

    def __init__(self, ox, oy, sc, rot=0.0, flip=1):
        self.ox, self.oy, self.sc, self.rot, self.flip = ox, oy, sc, rot, flip
        self.c = math.cos(rot); self.s = math.sin(rot)

    def l2w(self, lx, ly):
        x = lx * self.flip * self.sc; y = ly * self.sc
        return (self.ox + x * self.c - y * self.s, self.oy + x * self.s + y * self.c)

    def w2l(self, x, y):
        dx = x - self.ox; dy = y - self.oy
        return ((dx * self.c + dy * self.s) / self.sc * self.flip, (-dx * self.s + dy * self.c) / self.sc)

    def sub(self, lx, ly, rot=0.0, sc=1.0, flip=1):
        o = self.l2w(lx, ly)
        return Fr(o[0], o[1], self.sc * sc, self.rot + self.flip * rot, self.flip * flip)

    def ang(self, a):
        return (a if self.flip == 1 else pi - a) + self.rot

    def sd(self, f):
        return lambda x, y: self.sc * f(*self.w2l(x, y))

    def bb(self, b):
        P = [self.l2w(b[0], b[1]), self.l2w(b[2], b[1]), self.l2w(b[0], b[3]), self.l2w(b[2], b[3])]
        return (min(p[0] for p in P), min(p[1] for p in P), max(p[0] for p in P), max(p[1] for p in P))

    def px(self, w):
        return w * self.sc * E.S

    def pts(self, P):
        return [self.l2w(a, b) for a, b in P]

    def fill(self, f, b, base, n, w, ang=pi / 2, rad=0.25, spec=0.0, shin=24.0, wr=0.0, wf=5.0, col=None, lmul=(2.5, 5),
             margin=0.3, aj=0.15, flow=False, opac=0.97, block=True, jc=(0.012, 0.04), texa=0.12, dry=0.25, edge=None):
        sdw = self.sd(f)
        colfn = col if col is not None else lit(sdw, base, rad * self.sc, spec, shin, wr, wf / self.sc, texa, 5.0 / self.sc)
        if callable(ang):
            wa = lambda x, y: self.ang(ang(*self.w2l(x, y)))
            bang = self.ang(pi / 2)
        else:
            wa = self.ang(ang); bang = None
        E.fill(sdw, self.bb(b), colfn, n, (self.px(w[0]), self.px(w[1])), ang=wa, lmul=lmul, margin=margin, aj=aj, flow=flow,
               opac=opac, base=None if block else 0, bang=bang, jc=jc, dry=dry)
        if edge is None: edge = block
        if edge:                                        # redraw the contour so the silhouette is crisp
            ew = edge if isinstance(edge, tuple) else (w[0] * 0.7, w[0] * 1.1)
            bw = self.bb(b); per = 2 * ((bw[2] - bw[0]) + (bw[3] - bw[1]))
            ins = 0.5 * ew[0] * self.sc
            E.rim(sdw, bw, int(per / (0.45 * self.sc) * 3.0) + 4, lambda x, y, s: colfn(x, y), w=(self.px(ew[0]), self.px(ew[1])), inset=ins,
                  ln=(0.3 * self.sc, 0.7 * self.sc), opac=0.98, band=0.2 * self.sc, tol=max(0.0025, ins * 0.8))
        return sdw

    def line(self, P, w, col, opac=1.0, dry=0.0, t0=0.9, t1=0.9, sm=0):
        P = smooth(P, sm) if sm else P
        E.line(self.pts(P), self.px(w), col, opac, dry, t0, t1)

    def rim(self, f, b, n, col, w=(0.02, 0.04), light=None, thresh=0.2, inset=0.03, ln=(0.15, 0.5), opac=0.95, ymin=None, ymax=None):
        sdw = f if getattr(f, '_world', False) else self.sd(f)
        colfn = col if callable(col) else (lambda x, y, s: col)
        ins = inset * self.sc
        E.rim(sdw, self.bb(b), n, colfn, w=(self.px(w[0]), self.px(w[1])), light=light, thresh=thresh, inset=ins,
              ln=(ln[0] * self.sc, ln[1] * self.sc), opac=opac, band=0.25 * self.sc, tol=max(0.004, ins * 1.2), ymin=ymin, ymax=ymax)


def wfill(sd, bbox, colfn, n, w, **kw):
    E.fill(sd, bbox, colfn, n, w, **kw)


def ALL(x, y):
    return -1.0


# ----------------------------------------------------------------------------- the character
def shade_strokes(fr, P, w, col, opac):
    fr.line(P, w, col, opac, 0.2, 0.5, 0.5, sm=6)


def glasses_front(fr, y=0.115, glint=(0.75, 0.85, 1.0)):
    for s in (-1, 1):
        cx = s * 0.168; hw = 0.128; hh = 0.062
        lens = lambda a, b, cx=cx: sd_box(a, b, cx, y, hw - 0.012, hh - 0.012, 0.01)
        fr.fill(lens, (cx - hw, y - hh, cx + hw, y + hh), None, 5, (0.03, 0.05), ang=0.0, opac=0.38,
                col=lambda a, b: (0.16, 0.19, 0.24), block=False, lmul=(2, 4), aj=0.05)
        fr.line([(cx - hw, y - hh), (cx + hw, y - hh), (cx + hw, y + hh), (cx - hw, y + hh), (cx - hw, y - hh - 0.008)], 0.028, FRAME)
        fr.line([(cx - hw + 0.03, y + hh - 0.03), (cx - hw + 0.10, y - hh + 0.03)], 0.018, glint, 0.55)
        fr.line([(cx + s * hw, y - hh + 0.012), (s * 0.385, y - 0.045)], 0.024, FRAME)
    fr.line([(-0.045, y - 0.03), (0.0, y - 0.045), (0.045, y - 0.03)], 0.026, FRAME, sm=5)


def strands(fr, hair, angfn, n, box, ln=(0.10, 0.22), col=(0.93, 0.93, 0.95), opac=0.6):
    k = 0; tries = 0
    while k < int(n * E.DENS) and tries < n * 60:
        tries += 1
        x = U_(box[0], box[2]); y = U_(box[1], box[3])
        if hair(x, y) > -0.02: continue
        P = [(x, y)]; L = U_(ln[0], ln[1]); st = 0.03
        for i in range(int(L / st)):
            a = angfn(*P[-1]); q = (P[-1][0] + st * math.cos(a), P[-1][1] + st * math.sin(a))
            if hair(*q) > -0.012: break
            P.append(q)
        if len(P) < 3: continue
        fr.line(P, U_(0.012, 0.02), jit(col, 0.01, 0.05), opac * U_(0.6, 1.0), 0.3, 0.3, 0.3)
        k += 1


def skin_fn(fr, skin, shadow):
    def f(x, y):
        lx, ly = fr.w2l(x, y)
        k = min(max(shadow(lx, ly), 0.0), 1.0)
        return (skin[0] * (1 - 0.62 * k), skin[1] * (1 - 0.70 * k), skin[2] * (1 - 0.70 * k))
    return f


def bump(v, c, w):
    return math.exp(-((v - c) / w) ** 2)


def head_front_bowed(fr, toplight=1.0, skin=SKIN):
    """Seen from the front, head dropped forward: the crown and grey hair dominate, the face is foreshortened."""
    skull = lambda x, y: min(sd_ell(x, y, 0, -0.07, 0.375, 0.42), sd_ell(x, y, 0, 0.16, 0.285, 0.335))
    # brow, nose and lip throw shadows down the face
    shadow = lambda x, y: toplight * (0.50 * bump(y, 0.115, 0.075) * (1 - 0.6 * bump(x, 0, 0.04)) + 0.62 * sstep((y - 0.30) / 0.16)
                                      + 0.25 * sstep((abs(x) - 0.17) / 0.12) * sstep((y - 0.05) / 0.2))
    sb = skin_fn(fr, skin, shadow)
    for s in (-1, 1):
        ear = lambda x, y, s=s: sd_ell(x, y, s * 0.385, 0.06, 0.045, 0.095)
        fr.fill(ear, (s * 0.385 - 0.06, -0.05, s * 0.385 + 0.06, 0.17), (skin[0] * 0.85, skin[1] * 0.75, skin[2] * 0.70), 3, (0.03, 0.045), rad=0.05, edge=False)
    fr.fill(skull, (-0.4, -0.5, 0.4, 0.5), sb, 70, (0.04, 0.07), ang=0.0, rad=0.26, lmul=(1.5, 3), aj=0.12, jc=(0.006, 0.02), texa=0.05)
    dk = (skin[0] * 0.30, skin[1] * 0.22, skin[2] * 0.22)
    # nose
    fr.line([(0.010, 0.16), (0.014, 0.23), (0.010, 0.295)], 0.04, mix(skin, (1, 0.95, 0.85), 0.35), 0.7, sm=5)
    fr.line([(-0.024, 0.18), (-0.03, 0.26), (-0.05, 0.305)], 0.02, dk, 0.45, sm=5)
    fr.line([(-0.055, 0.318), (-0.02, 0.332), (0.02, 0.332), (0.055, 0.318)], 0.024, dk, 0.9, sm=5)
    # closed, lowered eyes
    for s in (-1, 1):
        fr.line([(s * 0.095, 0.118), (s * 0.165, 0.137), (s * 0.235, 0.122)], 0.02, (0.07, 0.05, 0.05), 0.95, sm=5)
        fr.line([(s * 0.07, 0.036), (s * 0.17, 0.016), (s * 0.275, 0.042)], 0.034, (0.36, 0.36, 0.38), 0.95, sm=5)   # brow
    # mouth, pressed and turned down
    fr.line([(-0.09, 0.400), (-0.05, 0.388), (0.0, 0.386), (0.05, 0.388), (0.09, 0.400)], 0.02, (0.26, 0.11, 0.10), 0.95, sm=5)
    glasses_front(fr)
    # hair: seen from above, combed back from the brow
    hl = lambda x: -0.115 + 1.75 * x * x
    hair = lambda x, y: max(sd_ell(x, y, 0, -0.09, 0.405, 0.455), (y - hl(x)) * 0.7)
    af = lambda x, y: math.atan2(y + 0.36, x - 0.03) + 0.2
    fr.fill(hair, (-0.42, -0.56, 0.42, 0.14), HAIR, 90, (0.03, 0.055), rad=0.14, lmul=(3, 7), flow=True, margin=0.22,
            ang=af, jc=(0.015, 0.07), texa=0.25, edge=False)
    strands(fr, hair, af, 36, (-0.42, -0.56, 0.42, 0.14))
    strands(fr, hair, af, 14, (-0.42, -0.56, 0.42, 0.14), col=(0.30, 0.31, 0.35))
    return lambda x, y: min(skull(x, y), hair(x, y))


_FACE = smooth([(0.10, -0.44), (0.25, -0.36), (0.325, -0.20), (0.325, -0.085), (0.385, 0.03), (0.455, 0.125), (0.44, 0.17), (0.365, 0.185),
                (0.378, 0.235), (0.352, 0.28), (0.366, 0.325), (0.34, 0.375), (0.345, 0.43), (0.28, 0.495), (0.13, 0.475), (-0.02, 0.39),
                (-0.13, 0.22), (-0.20, -0.05), (-0.10, -0.38)], 4)


def head_profile(fr, skin=SKIN, mouth=True, toplight=0.6):
    """Profile facing local +x; origin at the head centre, unit = head height."""
    skull = lambda x, y: sd_ell(x, y, -0.05, -0.10, 0.405, 0.385)
    face = lambda x, y: sd_poly(x, y, _FACE)
    head = lambda x, y: min(skull(x, y), face(x, y))
    shadow = lambda x, y: toplight * (0.45 * bump(y, -0.03, 0.07) * sstep((x - 0.12) / 0.1) + 0.5 * sstep((y - 0.30) / 0.22) * sstep((0.30 - x) / 0.3)
                                      + 0.35 * sstep((0.02 - x) / 0.2) * sstep((y + 0.1) / 0.3))
    fr.fill(head, (-0.5, -0.52, 0.5, 0.54), skin_fn(fr, skin, shadow), 80, (0.04, 0.07), ang=pi / 2 - 0.15, rad=0.13, lmul=(1.2, 2.2), aj=0.3,
            jc=(0.006, 0.02), texa=0.05, edge=(0.022, 0.034))
    dk = (skin[0] * 0.32, skin[1] * 0.24, skin[2] * 0.24)
    fr.line([(0.378, 0.168), (0.41, 0.178), (0.435, 0.162)], 0.018, dk, 0.8, sm=4)           # nostril
    if mouth:
        fr.line([(0.352, 0.281), (0.30, 0.286), (0.262, 0.305)], 0.018, (0.26, 0.11, 0.10), 0.95, sm=4)
    # ear
    ear = lambda x, y: sd_ell(x, y, -0.125, 0.035, 0.062, 0.105)
    fr.fill(ear, (-0.2, -0.08, -0.05, 0.15), (skin[0] * 0.95, skin[1] * 0.84, skin[2] * 0.78), 4, (0.03, 0.045), rad=0.06, edge=False)
    fr.line([(-0.10, -0.03), (-0.155, 0.0), (-0.15, 0.08), (-0.11, 0.10)], 0.016, dk, 0.6, sm=5)
    # lowered eye and brow
    fr.line([(0.225, -0.045), (0.265, -0.03), (0.30, -0.035)], 0.018, (0.07, 0.05, 0.05), 0.95, sm=4)
    fr.line([(0.20, -0.135), (0.27, -0.15), (0.33, -0.128)], 0.032, (0.36, 0.36, 0.38), 0.95, sm=4)
    # rectangular glasses, seen nearly edge-on
    fr.fill(lambda x, y: sd_box(x, y, 0.275, -0.04, 0.062, 0.06), (0.2, -0.11, 0.36, 0.03), None, 3, (0.03, 0.05), ang=pi / 2,
            opac=0.3, col=lambda a, b: (0.16, 0.19, 0.24), block=False, lmul=(1.5, 3))
    fr.line([(0.20, -0.105), (0.35, -0.11), (0.352, 0.025), (0.205, 0.03), (0.20, -0.113)], 0.026, FRAME)
    fr.line([(0.20, -0.085), (0.02, -0.07), (-0.115, -0.045)], 0.024, FRAME, sm=4)
    fr.line([(0.225, 0.0), (0.26, -0.085)], 0.016, (0.75, 0.85, 1.0), 0.55)
    # hair
    HP = [(0.275, -0.37), (0.20, -0.285), (0.10, -0.19), (0.03, -0.06), (0.005, 0.07), (-0.05, 0.0), (-0.08, -0.085), (-0.20, -0.07),
          (-0.235, 0.13), (-0.32, 0.255), (-0.435, 0.12), (-0.485, -0.10), (-0.43, -0.35), (-0.26, -0.515), (0.0, -0.56), (0.21, -0.50)]
    hair = lambda x, y: sd_poly(x, y, HP)
    af = lambda x, y: math.atan2(y - 0.25, x - 0.45) + pi / 2 + 0.25 * (y + 0.2)
    fr.fill(hair, (-0.5, -0.58, 0.32, 0.27), HAIR, 90, (0.03, 0.055), rad=0.12, lmul=(3, 7), flow=True, margin=0.22,
            ang=af, jc=(0.015, 0.07), texa=0.25, edge=False)
    strands(fr, hair, af, 30, (-0.5, -0.58, 0.32, 0.27))
    strands(fr, hair, af, 12, (-0.5, -0.58, 0.32, 0.27), col=(0.30, 0.31, 0.35))
    return lambda x, y: min(head(x, y), hair(x, y))


def head_back(fr, skin=SKIN):
    """From behind: grey hair, ears, the tips of the glasses' arms."""
    for s in (-1, 1):
        ear = lambda x, y, s=s: sd_ell(x, y, s * 0.385, 0.07, 0.05, 0.10)
        fr.fill(ear, (s * 0.385 - 0.06, -0.05, s * 0.385 + 0.06, 0.18), (skin[0] * 0.9, skin[1] * 0.80, skin[2] * 0.74), 3, (0.03, 0.045), rad=0.05, edge=False)
        fr.line([(s * 0.40, -0.02), (s * 0.435, 0.0), (s * 0.43, 0.06)], 0.024, FRAME, sm=4)
    head = lambda x, y: min(sd_ell(x, y, 0, -0.06, 0.38, 0.44), sd_ell(x, y, 0, 0.12, 0.30, 0.34))
    fr.fill(head, (-0.4, -0.52, 0.4, 0.48), skin, 20, (0.05, 0.085), rad=0.25, lmul=(2, 4), jc=(0.006, 0.02), texa=0.05)
    nape = lambda x: 0.30 - 0.5 * x * x
    hair = lambda x, y: max(sd_ell(x, y, 0, -0.07, 0.405, 0.46), (y - nape(x)) * 0.8)
    af = lambda x, y: math.atan2(y + 0.24, x - 0.04) + 0.3
    fr.fill(hair, (-0.42, -0.55, 0.42, 0.34), HAIR, 110, (0.03, 0.055), rad=0.14, lmul=(3, 7), flow=True, margin=0.22,
            ang=af, jc=(0.015, 0.07), texa=0.25, edge=False)
    strands(fr, hair, af, 40, (-0.42, -0.55, 0.42, 0.34))
    strands(fr, hair, af, 14, (-0.42, -0.55, 0.42, 0.34), col=(0.30, 0.31, 0.35))
    return lambda x, y: min(head(x, y), hair(x, y))


def leather(fr, f, b, n, w, ang=pi / 2, rad=0.22, wr=0.55, **kw):
    return fr.fill(f, b, LEATHER, n, w, ang=ang, rad=rad, spec=0.9, shin=26.0, wr=wr, wf=4.0, texa=0.0, jc=(0.006, 0.05), **kw)


def crease(fr, P, w=0.03, col=(0.42, 0.44, 0.50), opac=0.6):
    fr.line(P, w, col, opac, 0.3, 0.2, 0.2, sm=6)


def hand(fr, cx, cy, rot=0.0, sc=1.0, skin=SKIN, fingers=4, grip=True):
    """A fist/gripping hand about 0.5 head long; local +y is toward the fingertips."""
    h = fr.sub(cx, cy, rot, sc)
    palm = lambda x, y: sd_box(x, y, 0, 0, 0.16, 0.15, 0.07)
    h.fill(palm, (-0.2, -0.2, 0.2, 0.2), skin, 8, (0.04, 0.07), ang=pi / 2, rad=0.12, lmul=(1.5, 3))
    if grip:
        for k in range(fingers):
            fx = -0.12 + 0.24 * k / (fingers - 1)
            fg = lambda x, y, fx=fx: sd_cap(x, y, fx, 0.06, fx * 1.05, 0.21, 0.036)
            h.fill(fg, (fx - 0.05, 0.0, fx + 0.05, 0.27), skin, 2, (0.03, 0.045), ang=pi / 2, rad=0.05, lmul=(1.5, 3))
        for k in range(fingers - 1):
            fx = -0.08 + 0.24 * k / (fingers - 1)
            h.line([(fx, 0.05), (fx * 1.05, 0.22)], 0.014, (skin[0] * 0.35, skin[1] * 0.27, skin[2] * 0.25), 0.8)
    return h
