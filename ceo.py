"""Five original paintings, no reference image: a CEO steps down. Each is a different picture of shame.
The same character throughout: grey hair, rectangular glasses, black leather jacket."""
import argparse, math, random
import engine as E
from engine import *
from kit import *


def sheet():
    E.setup(2400, 900)
    set_lights([('d', (-0.3, -0.8, 0.5), (1.0, 0.95, 0.85))], (0.30, 0.30, 0.34))
    wfill(ALL, (0, 0, E.AR, 1), lambda x, y: (0.30, 0.33, 0.40), 0, (60, 80), ang=0.3)
    head_profile(Fr(0.45, 0.5, 0.7))
    head_front_bowed(Fr(1.33, 0.5, 0.7))
    head_back(Fr(2.2, 0.5, 0.7))


def spot(x, y, cx, cy, rx, ry):
    return math.exp(-(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2))


def burst(x, y, r, col=(1.0, 1.0, 1.0), rays=10):
    """A camera flash: halo, rays, hot core."""
    for k in range(int(10 * E.DENS)):
        rr = r * U_(0.5, 2.6); a0 = U_(0, 6.28)
        add(arc(x, y, rr, a0, a0 + U_(1.5, 3.5)), r * E.S * U_(0.5, 0.9), mix(col, (0.6, 0.75, 1.0), 0.5), 0.07, 0.3, 0.5, 0.5)
    for k in range(rays):
        a0 = k * 6.283 / rays + U_(-0.15, 0.15); L = r * U_(2.0, 5.5)
        add(seg(x, y, x + L * math.cos(a0), y + L * math.sin(a0), 6), r * E.S * U_(0.10, 0.22), col, 0.75, 0.2, 1.0, 0.05)
    for k in range(6):
        a0 = U_(0, 6.28)
        add(arc(x, y, r * U_(0.15, 0.6), a0, a0 + 4.5), r * E.S * 0.8, (1, 1, 1), 0.9, 0.0, 0.9, 0.9)
    dot(x, y, r * E.S * 1.5, (1, 1, 1))


def jacket_front(fb, lectern_y=1.30, hands=True):
    """Waist-up, facing us, arms down to grip something at lectern_y. fb: origin at the pit of the neck, unit = head."""
    parts = []
    tee = lambda x, y: sd_poly(x, y, [(-0.30, -0.16), (0.30, -0.16), (0.16, 1.7), (-0.16, 1.7)])
    fb.fill(tee, (-0.3, -0.2, 0.3, 1.7), TEE, 10, (0.07, 0.11), rad=0.3, texa=0.05, edge=False)
    for s in (-1, 1):
        half = lambda x, y, s=s: sd_poly(x, y, [(s * 0.30, -0.22), (s * 0.70, -0.06), (s * 1.04, 0.10), (s * 1.13, 0.5), (s * 1.02, 1.75), (s * 0.10, 1.75), (s * 0.12, 0.75)]) - 0.03
        leather(fb, half, (min(s * 0.05, s * 1.2), -0.3, max(s * 0.05, s * 1.2), 1.8), 60, (0.07, 0.13), ang=pi / 2 + s * 0.08)
        ua = lambda x, y, s=s: sd_tcap(x, y, s * 0.98, 0.26, s * 1.30, 1.02, 0.25, 0.20)
        leather(fb, ua, (min(s * 0.7, s * 1.6), 0.0, max(s * 0.7, s * 1.6), 1.25), 40, (0.06, 0.11), ang=pi / 2 - s * 0.35)
        fa = lambda x, y, s=s: sd_tcap(x, y, s * 1.30, 1.02, s * 1.02, lectern_y - 0.08, 0.20, 0.15)
        leather(fb, fa, (min(s * 0.8, s * 1.55), 0.8, max(s * 0.8, s * 1.55), lectern_y + 0.1), 30, (0.06, 0.10), ang=pi / 2 + s * 0.75)
        lap = lambda x, y, s=s: sd_poly(x, y, [(s * 0.30, -0.24), (s * 0.60, -0.10), (s * 0.50, 0.22), (s * 0.20, 0.72), (s * 0.13, 0.70), (s * 0.24, 0.1)])
        leather(fb, lap, (min(s * 0.1, s * 0.65), -0.26, max(s * 0.1, s * 0.65), 0.75), 14, (0.05, 0.08), ang=pi / 2 + s * 0.3, rad=0.08)
        parts += [half, ua, fa]
        crease(fb, [(s * 0.62, -0.08), (s * 0.52, 0.22), (s * 0.22, 0.72)], 0.022, (0.34, 0.36, 0.42), 0.75)         # lapel edge
        for k in range(7):                                                                                         # elbow creases
            t = U_(0.80, 1.12); x0 = U_(1.12, 1.22)
            crease(fb, [(s * x0, t), (s * (x0 + 0.10), t + U_(0.02, 0.07)), (s * (x0 + 0.22), t - 0.02)], U_(0.018, 0.03), opac=U_(0.25, 0.5))
        for k in range(6):                                                                                         # sleeve and chest sheen
            t = U_(0.25, 0.7)
            crease(fb, [(s * (0.93 + 0.30 * t), 0.20 + t * 0.9), (s * (0.97 + 0.30 * t), 0.32 + t * 0.9), (s * (0.99 + 0.31 * t), 0.44 + t * 0.9)], U_(0.03, 0.05), (0.30, 0.32, 0.38), U_(0.2, 0.4))
        for k in range(5):
            x0 = U_(0.35, 0.75); y0 = U_(0.35, 1.0)
            crease(fb, [(s * x0, y0), (s * (x0 + 0.04), y0 + 0.12), (s * (x0 + 0.02), y0 + 0.26)], U_(0.02, 0.035), (0.20, 0.21, 0.26), U_(0.25, 0.45))
        crease(fb, [(s * 0.80, 0.25), (s * 0.84, 0.60), (s * 0.90, 1.0)], 0.03, (0.01, 0.01, 0.015), 0.8)             # armhole seam
        if hands:
            hand(fb, s * 0.98, lectern_y - 0.02, rot=-s * 0.15, sc=1.0)
    fb.line([(0.13, 0.75), (0.11, 1.7)], 0.02, (0.55, 0.57, 0.62), 0.8)                                               # zip
    return lambda x, y: min(p(x, y) for p in parts)


def podium():
    """1. The statement: head bowed at the lectern, under one hard light, in front of the flashes."""
    E.setup(2400, 3000, dens=1.0)
    key = (0.40, -0.25, 0.30)
    set_lights([('p', key, (1.9, 1.75, 1.5), 0.75), ('f', (0.0, 0.5, 0.85), (0.34, 0.33, 0.38))], (0.05, 0.05, 0.07))
    layer('curtain')

    def cone(x, y):
        hw = 0.035 + 0.30 * (y + 0.03)
        return sstep((hw - abs(x - 0.40)) / (0.02 + 0.10 * (y + 0.03)) ) * (0.25 + 0.75 * math.exp(-max(0.0, y - 0.05) * 1.6))

    def curtain(x, y):
        fold = 0.5 + 0.5 * math.sin(x * 46 + 2.0 * N2(x * 4, y * 0.8))
        c = mix((0.020, 0.024, 0.052), (0.045, 0.055, 0.115), fold)
        k = cone(x, y)
        c = mix(c, (0.36 + 0.10 * fold, 0.35 + 0.10 * fold, 0.40 + 0.12 * fold), 0.62 * k)
        return c
    wfill(ALL, (0, 0, E.AR, 1), curtain, 320, (40, 76), ang=pi / 2, lmul=(2, 4.5), aj=0.02, jc=(0.004, 0.03), seglen=(2.0, 4.0))
    layer('haze')
    for k in range(int(26 * E.DENS)):                                   # dust in the beam
        t = U_(-0.8, 0.8); y0 = U_(0.0, 0.25)
        add(seg(0.40 + t * (0.035 + 0.30 * y0), y0, 0.40 + t * (0.035 + 0.30 * (y0 + 0.25)), y0 + 0.22, 6), U_(30, 70), (1.0, 0.95, 0.82), 0.05, 0.5, 0.4, 0.1)
    layer('figure')
    fb = Fr(0.40, 0.452, 0.168)
    neck = lambda x, y: sd_cap(x, y, 0, -0.3, 0, 0.0, 0.21)
    fb.fill(neck, (-0.25, -0.5, 0.25, 0.2), (SKIN[0] * 0.5, SKIN[1] * 0.42, SKIN[2] * 0.4), 6, (0.06, 0.1), rad=0.2, edge=False)
    body = jacket_front(fb, lectern_y=1.28, hands=False)
    head_front_bowed(fb.sub(0, -0.40), toplight=0.8)
    fb.rim(body, (-1.6, -0.3, 1.6, 1.0), 60, lambda x, y, s: mix((0.26, 0.27, 0.31), (0.72, 0.71, 0.68), s), w=(0.018, 0.04), light=(0.40, -0.6), thresh=0.6,
           inset=0.04, ln=(0.10, 0.32), opac=0.65, ymax=0.60)
    layer('lectern')
    ly = fb.l2w(0, 1.28)[1]
    top = [(0.40 - 0.185, ly), (0.40 + 0.185, ly), (0.40 + 0.215, ly + 0.055), (0.40 - 0.215, ly + 0.055)]
    front = [(0.40 - 0.215, ly + 0.055), (0.40 + 0.215, ly + 0.055), (0.40 + 0.19, 1.05), (0.40 - 0.19, 1.05)]
    wood = lambda x, y: tuple(v * (0.5 + 1.6 * spot(x, y, 0.40, ly, 0.3, 0.12)) for v in tex((0.20, 0.115, 0.07), x * 6, y * 0.6, 0.5, 30))
    wfill(lambda x, y: sd_poly(x, y, front), (0.15, ly, 0.65, 1.05), lambda x, y: tuple(v * 0.45 for v in wood(x, y)), 60, (20, 36), ang=pi / 2, lmul=(4, 9), aj=0.03)
    wfill(lambda x, y: sd_poly(x, y, top), (0.15, ly - 0.01, 0.65, ly + 0.07), wood, 30, (14, 24), ang=0.0, lmul=(4, 9), aj=0.03)
    line([top[3], top[2]], 7, (0.62, 0.42, 0.26), 0.9)
    line([front[0], front[3]], 6, (0.30, 0.19, 0.12), 0.8); line([front[1], front[2]], 6, (0.30, 0.19, 0.12), 0.8)
    for s in (-1, 1):
        hand(fb, s * 1.04, 1.30, rot=-s * 0.12, sc=1.05)
    layer('microphones')
    MC = [(0.70, 0.62, 0.12), (0.12, 0.28, 0.62), (0.80, 0.78, 0.74), (0.72, 0.36, 0.10), (0.16, 0.46, 0.36), (0.50, 0.12, 0.38)]
    for k, mx in enumerate((-0.52, -0.30, -0.10, 0.11, 0.31, 0.52)):
        tip = fb.l2w(mx * 0.62 + U_(-0.02, 0.02), 0.66 + 0.24 * abs(mx) / 0.52 + U_(-0.03, 0.03)); base = fb.l2w(mx, 1.31)
        line(smooth([base, ((base[0] + tip[0]) / 2 + 0.004 * (1 if mx > 0 else -1), (base[1] + tip[1]) / 2 + 0.012), (tip[0], tip[1] + 0.03)], 6), 8, (0.015, 0.015, 0.02))
        cube = lambda x, y, tip=tip: sd_box(x, y, tip[0], tip[1] + 0.040, 0.016, 0.014, 0.002)
        wfill(cube, (tip[0] - 0.02, tip[1] + 0.022, tip[0] + 0.02, tip[1] + 0.058), lit(cube, MC[k], 0.012), 3, (8, 12), ang=0.0, lmul=(1.5, 3), jc=(0.006, 0.03))
        hd = lambda x, y, tip=tip: sd_ell(x, y, tip[0], tip[1], 0.0165, 0.025)
        wfill(hd, (tip[0] - 0.02, tip[1] - 0.03, tip[0] + 0.02, tip[1] + 0.03), lit(hd, (0.30, 0.30, 0.33), 0.018, 0.0), 5, (9, 14), ang=pi / 2, lmul=(1.5, 3))
    layer('press')
    PR = (0.012, 0.014, 0.028)

    def person(cx, top, hw, arm=None):
        hdr = hw * 0.42
        sds = [lambda x, y: sd_ell(x, y, cx, top + hdr * 1.2, hdr, hdr * 1.25), lambda x, y: sd_box(x, y, cx, top + hdr * 2.3 + 0.2, hw, 0.2, hw * 0.5)]
        if arm:
            sds.append(lambda x, y: sd_cap(x, y, cx + arm[0] * 0.5, top + hdr * 3, arm[1], arm[2], hw * 0.2))
        sd = lambda x, y: min(f(x, y) for f in sds)
        wfill(sd, (cx - hw - 0.1, min(top, arm[2] if arm else top) - 0.02, cx + hw + 0.1, 1.02), lambda x, y: PR, 14, (22, 40), ang=pi / 2, lmul=(2, 4))
        return sd
    crowd = []
    crowd.append(person(0.06, 0.83, 0.085)); crowd.append(person(0.24, 0.875, 0.08, (-1, 0.20, 0.80)))
    crowd.append(person(0.44, 0.89, 0.085)); crowd.append(person(0.60, 0.86, 0.08, (1, 0.665, 0.79)))
    crowd.append(person(0.77, 0.84, 0.09))
    crowd.append(person(0.15, 0.935, 0.09)); crowd.append(person(0.34, 0.95, 0.085)); crowd.append(person(0.53, 0.955, 0.09)); crowd.append(person(0.70, 0.94, 0.085))
    # a camera, two phones
    cam = lambda x, y: sd_box(x, y, 0.20, 0.785, 0.035, 0.022, 0.004)
    wfill(cam, (0.15, 0.75, 0.25, 0.82), lambda x, y: (0.03, 0.03, 0.045), 4, (14, 22), ang=0.0, lmul=(1.5, 3))
    for (px_, py_, rot) in ((0.665, 0.765, 0.10), (0.365, 0.865, -0.08)):
        ph = lambda x, y, px_=px_, py_=py_, rot=rot: sd_rbox(x, y, px_, py_, 0.019, 0.034, rot, 0.003)
        wfill(ph, (px_ - 0.04, py_ - 0.05, px_ + 0.04, py_ + 0.05), lambda x, y: (0.02, 0.02, 0.03), 3, (12, 18), ang=pi / 2 + rot, lmul=(1.5, 3))
        scr = lambda x, y, px_=px_, py_=py_, rot=rot: sd_rbox(x, y, px_, py_, 0.015, 0.029, rot, 0.002)
        wfill(scr, (px_ - 0.04, py_ - 0.05, px_ + 0.04, py_ + 0.05), lambda x, y, px_=px_, py_=py_: mix((0.10, 0.13, 0.25), (0.95, 0.9, 0.8), spot(x, y, px_, py_ - 0.008, 0.012, 0.012)),
              6, (5, 8), ang=pi / 2 + rot, lmul=(1.5, 3), jc=(0.004, 0.02))
        dot(px_ + 0.001, py_ + 0.004, 12, (0.03, 0.03, 0.04)); dot(px_ + 0.0005, py_ - 0.008, 7, (0.62, 0.62, 0.65))
    allc = lambda x, y: min(f(x, y) for f in crowd)
    E.rim(allc, (0, 0.78, E.AR, 0.99), 60, lambda x, y, s: mix((0.20, 0.24, 0.36), (0.78, 0.84, 1.0), s), w=(3, 6), light=(0.40, 0.45), thresh=0.35, inset=0.003, ln=(0.02, 0.07), opac=0.7)
    burst(0.215, 0.762, 0.012); burst(0.735, 0.815, 0.008); burst(0.03, 0.80, 0.006)


def glyphs(txt, x, y, h, w, col, sp=0.72):
    """Tiny block capitals out of straight strokes."""
    G = {'C': [[(1, 0.1), (0.5, 0), (0, 0.3), (0, 0.7), (0.5, 1), (1, 0.9)]], 'E': [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.8, 0.5)]],
         'O': [[(0.5, 0), (0, 0.3), (0, 0.7), (0.5, 1), (1, 0.7), (1, 0.3), (0.5, 0)]], 'X': [[(0, 0), (1, 1)], [(1, 0), (0, 1)]],
         'I': [[(0.5, 0), (0.5, 1)]], 'T': [[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]]}
    for i, ch in enumerate(txt):
        for st in G.get(ch, []):
            line([(x + (i * sp + p[0] * 0.5) * h, y + p[1] * h) for p in st], w, col)


def city(x0, x1, ytop, yhor, seed, glow=(0.42, 0.22, 0.14)):
    """A night skyline behind glass."""
    rr = random.Random(seed)

    def sky(x, y):
        t = (y - ytop) / (yhor - ytop)
        return mix(mix((0.018, 0.026, 0.07), (0.06, 0.07, 0.16), t), glow, 0.55 * sstep((t - 0.62) / 0.38) ** 2)
    wfill(lambda x, y: sd_box(x, y, (x0 + x1) / 2, (ytop + yhor) / 2, (x1 - x0) / 2, (yhor - ytop) / 2), (x0, ytop, x1, yhor), sky, 120, (26, 50), ang=0.0,
          lmul=(3, 7), aj=0.03, jc=(0.004, 0.03))
    for depth in (0, 1):
        x = x0 - 0.02
        while x < x1:
            w = rr.uniform(0.04, 0.11); h = rr.uniform(0.07, 0.24) * (0.8 if depth == 0 else 1.15)
            if rr.random() < 0.15: h *= 1.5
            top = max(ytop + 0.03, yhor - h)
            col = (0.05, 0.06, 0.13) if depth == 0 else (0.022, 0.028, 0.065)
            b = lambda a, c, x=x, w=w, top=top: sd_box(a, c, x + w / 2, (top + yhor) / 2 + 0.01, w / 2 * 0.92, (yhor - top) / 2 + 0.01)
            wfill(b, (x, top, x + w, yhor + 0.02), lambda a, c, col=col: col, 2, (16, 28), ang=pi / 2, lmul=(3, 6), jc=(0.004, 0.03))
            for k in range(int(w * (yhor - top) * 2600)):
                wx = x + w * rr.uniform(0.12, 0.88); wy = rr.uniform(top + 0.012, yhor - 0.008)
                wc = rr.choice([(1.0, 0.78, 0.42), (1.0, 0.86, 0.6), (0.62, 0.80, 1.0), (1.0, 0.70, 0.35)])
                add(seg(wx - 0.004, wy, wx + 0.004, wy, 2), rr.uniform(3.5, 6.5), wc, rr.uniform(0.35, 0.95) * (0.6 if depth == 0 else 1.0), 0.0, 0.9, 0.9)
            x += w * rr.uniform(0.9, 1.25)


def seated_profile(fb, table_y=1.5, skin=SKIN):
    """Side view, facing local +x, slumped forward over a table. fb: origin at the base of the neck, unit = head."""
    parts = []
    # far-side legs and shoes in shadow
    TROU = (0.10, 0.105, 0.125)
    thigh = lambda x, y: sd_tcap(x, y, -0.45, 2.32, 1.10, 2.42, 0.33, 0.25)
    shin = lambda x, y: sd_tcap(x, y, 1.10, 2.42, 0.95, 3.75, 0.24, 0.16)
    shoe = lambda x, y: min(sd_box(x, y, 1.12, 3.90, 0.34, 0.10, 0.08), sd_cap(x, y, 0.95, 3.7, 0.95, 3.85, 0.15))
    fb.fill(shin, (0.6, 2.3, 1.5, 3.95), TROU, 24, (0.07, 0.12), ang=pi / 2, rad=0.2)
    fb.fill(thigh, (-0.8, 1.9, 1.5, 2.8), TROU, 30, (0.08, 0.13), ang=0.05, rad=0.25)
    fb.fill(shoe, (0.6, 3.6, 1.6, 4.05), (0.035, 0.035, 0.04), 8, (0.06, 0.09), ang=0.0, rad=0.1, spec=0.4)
    neck = lambda x, y: sd_tcap(x, y, -0.02, 0.0, 0.22, -0.42, 0.22, 0.17)
    fb.fill(neck, (-0.3, -0.6, 0.5, 0.2), (skin[0] * 0.78, skin[1] * 0.70, skin[2] * 0.66), 8, (0.05, 0.08), rad=0.18, texa=0.05, jc=(0.006, 0.02))
    tee = lambda x, y: sd_poly(x, y, [(0.05, -0.15), (0.40, 0.0), (0.56, 0.6), (0.2, 0.7)])
    fb.fill(tee, (0.0, -0.2, 0.6, 0.7), TEE, 5, (0.06, 0.09), rad=0.2, edge=False)
    torso = lambda x, y: sd_poly(x, y, [(-0.34, -0.16), (0.10, -0.12), (0.36, 0.25), (0.52, 0.85), (0.46, 1.55), (0.30, 2.42), (-0.86, 2.45), (-0.98, 1.5), (-0.84, 0.45), (-0.62, 0.0)]) - 0.04
    leather(fb, torso, (-1.1, -0.3, 0.7, 2.6), 110, (0.07, 0.13), ang=pi / 2 + 0.12)
    collar = lambda x, y: sd_poly(x, y, [(-0.36, -0.22), (0.12, -0.20), (0.32, 0.12), (0.22, 0.34), (-0.05, 0.05), (-0.40, -0.02)])
    leather(fb, collar, (-0.45, -0.3, 0.4, 0.4), 12, (0.05, 0.08), ang=0.3, rad=0.08)
    crease(fb, [(-0.40, -0.02), (-0.05, 0.05), (0.22, 0.34)], 0.022, (0.32, 0.33, 0.38), 0.7)
    crease(fb, [(0.34, 0.2), (0.50, 0.85), (0.45, 1.55), (0.32, 2.3)], 0.02, (0.50, 0.52, 0.56), 0.6)             # zip edge
    ua = lambda x, y: sd_tcap(x, y, -0.12, 0.42, 0.72, table_y - 0.24, 0.30, 0.23)
    leather(fb, ua, (-0.5, 0.1, 1.1, table_y + 0.1), 46, (0.06, 0.11), ang=0.85)
    fa = lambda x, y: sd_tcap(x, y, 0.72, table_y - 0.24, 1.58, table_y - 0.17, 0.22, 0.16)
    leather(fb, fa, (0.4, table_y - 0.6, 1.8, table_y + 0.1), 30, (0.06, 0.10), ang=0.08)
    for k in range(5):
        t = U_(0.6, 1.0); bx = -0.12 + 0.84 * t; by = 0.42 + (table_y - 0.66) * t; a0 = U_(-0.5, 0.5)
        crease(fb, [(bx - 0.14, by + 0.08 + 0.1 * a0), (bx, by + U_(-0.02, 0.05)), (bx + 0.13, by - 0.12 - 0.1 * a0)], U_(0.018, 0.03), opac=U_(0.15, 0.35))
    for k in range(3):
        bx = U_(0.75, 1.0)
        crease(fb, [(bx, table_y - 0.36), (bx + U_(0.0, 0.08), table_y - 0.22), (bx, table_y - 0.08)], U_(0.018, 0.03), opac=U_(0.15, 0.35))
    crease(fb, [(0.0, 0.22), (0.30, 0.48), (0.62, 0.86)], 0.05, (0.50, 0.50, 0.54), 0.30)
    for k in range(8):
        bx = U_(-0.8, 0.2); by = U_(0.5, 2.1)
        crease(fb, [(bx, by), (bx + 0.05, by + 0.14), (bx + 0.02, by + 0.3)], U_(0.02, 0.035), (0.20, 0.21, 0.26), U_(0.2, 0.45))
    hands = lambda x, y: min(sd_ell(x, y, 1.80, table_y - 0.17, 0.20, 0.15), sd_ell(x, y, 1.92, table_y - 0.13, 0.13, 0.12))
    fb.fill(hands, (1.55, table_y - 0.4, 2.1, table_y + 0.05), skin, 10, (0.04, 0.07), ang=0.0, rad=0.12, texa=0.05, jc=(0.006, 0.02), lmul=(1.5, 3))
    for k in range(3):
        fb.line([(1.74 + 0.07 * k, table_y - 0.27), (1.80 + 0.07 * k, table_y - 0.10)], 0.016, (skin[0] * 0.4, skin[1] * 0.3, skin[2] * 0.28), 0.7)
    parts += [torso, ua, fa, thigh, shin]
    return lambda x, y: min(p(x, y) for p in parts)


def boardroom():
    """2. After the vote: the table is empty and he is still sitting at the end of it."""
    E.setup(3000, 1875)
    lamp = (1.165, 0.155)
    set_lights([('p', (lamp[0], lamp[1], 0.16), (3.0, 2.45, 1.7), 0.42), ('f', (-0.6, -0.3, 0.6), (0.30, 0.40, 0.66))], (0.05, 0.06, 0.10))
    warm = lambda x, y: spot(x, y, 1.18, 0.60, 0.34, 0.30)
    TY0, TY1, TX0, TX1 = 0.615, 0.675, 0.03, 1.285
    layer('room')
    wfill(ALL, (0, 0, E.AR, 1), lambda x, y: (0.03, 0.033, 0.05), 0, (50, 80), ang=0.0)
    city(0, E.AR, 0.07, 0.56, 7)
    wfill(lambda x, y: sd_box(x, y, 0.8, 0.63, 0.9, 0.075), (0, 0.555, E.AR, 0.705), lambda x, y: mix((0.022, 0.028, 0.065), (0.05, 0.055, 0.10), N2(x * 9, y * 30)), 30, (20, 34),
          ang=0.0, lmul=(3, 7), jc=(0.004, 0.03))     # dark low-rise and street glow at the foot of the glass
    for k in range(70):
        xx = U_(0, E.AR); yy = U_(0.565, 0.70)
        add(seg(xx - 0.006, yy, xx + 0.006, yy, 2), U_(3.5, 7), R.choice([(1.0, 0.72, 0.38), (1.0, 0.85, 0.6), (0.9, 0.3, 0.2)]), U_(0.3, 0.8), 0.0, 0.9, 0.9)
    layer('glass')
    for k in range(0):
        a0 = U_(0, 6.28); rr = U_(0.02, 0.16)
        add(arc(lamp[0] - 0.05, lamp[1] + 0.03, rr, a0, a0 + U_(2.5, 5.0)), U_(50, 90), (1.0, 0.72, 0.40), 0.022, 0.4, 0.5, 0.5)
    for xm in (0.0, 0.32, 0.64, 0.96, 1.28, 1.595):                    # mullions
        line([(xm, 0.0), (xm, 0.705)], 26, (0.012, 0.014, 0.025))
        line([(xm + 0.007, 0.07), (xm + 0.007, 0.70)], 4, mix((0.05, 0.06, 0.10), (0.75, 0.55, 0.32), warm(xm, 0.4)), 0.8)
    wfill(lambda x, y: y - 0.075, (0, 0, E.AR, 0.075), lambda x, y: mix((0.016, 0.018, 0.03), (0.16, 0.12, 0.08), spot(x, y, lamp[0], 0.03, 0.30, 0.08)), 24, (26, 40),
          ang=0.0, lmul=(4, 9), aj=0.01, jc=(0.004, 0.03))
    layer('floor')

    def floor(x, y):
        c = mix((0.04, 0.046, 0.075), (0.065, 0.075, 0.115), N2(x * 14, y * 40))
        return mix(c, (0.40, 0.27, 0.15), 0.75 * spot(x, y, 1.23, 0.88, 0.36, 0.11))
    wfill(lambda x, y: 0.70 - y, (0, 0.70, E.AR, 1), floor, 150, (24, 50), ang=0.0, lmul=(4, 9), aj=0.02, jc=(0.005, 0.03))
    line([(0, 0.704), (E.AR, 0.704)], 7, (0.10, 0.10, 0.14), 0.8)
    layer('far chairs')

    def chair(cx, top, bot, hw, tilt, shade, near):
        back = lambda x, y: sd_rbox(x, y, cx, (top + bot) / 2, hw, (bot - top) / 2, tilt, 0.03)
        base = (0.13, 0.13, 0.17) if near else (0.09, 0.09, 0.13)
        colf = lit(back, base, 0.035, 0.25, 20, 0.0, 40, 0.15, 25)
        wfill(back, (cx - hw - 0.05, top - 0.03, cx + hw + 0.05, bot + 0.03), colf, 14, (20, 34), ang=pi / 2 + tilt, lmul=(2, 5), jc=(0.005, 0.03))
        E.rim(back, (cx - hw - 0.05, top - 0.03, cx + hw + 0.05, bot + 0.03), 10 if near else 6, lambda x, y, s: mix((0.20, 0.17, 0.15), (1.0, 0.76, 0.46), s * shade), w=(3, 6),
              light=lamp, thresh=0.3, inset=0.004, ln=(0.03, 0.09), opac=0.85)
        if near:
            line([(cx, bot - 0.01), (cx, 0.90)], 16, (0.02, 0.02, 0.03))
            for a0 in (-1, -0.35, 0.35, 1):
                line([(cx, 0.90), (cx + a0 * 0.075, 0.925 + 0.012 * (1 - abs(a0)))], 9, (0.02, 0.02, 0.03))
                dot(cx + a0 * 0.075, 0.935 + 0.012 * (1 - abs(a0)), 14, (0.015, 0.015, 0.02))
            st = lambda x, y: sd_box(x, y, cx, bot + 0.035, hw * 1.05, 0.02, 0.015)
            wfill(st, (cx - hw - 0.03, bot, cx + hw + 0.03, bot + 0.07), lambda x, y: (0.04, 0.04, 0.05), 4, (18, 26), ang=0.0, lmul=(2, 5))
    for i, cx in enumerate((0.16, 0.41, 0.66, 0.91)):
        chair(cx + U_(-0.015, 0.015), 0.455, 0.63, 0.062, U_(-0.04, 0.04), 0.25 + 0.75 * warm(cx, 0.5), False)
    layer('table')

    def top(x, y):
        c = tex((0.085, 0.05, 0.035), x * 2, y * 20, 0.5, 30)
        c = mix(c, (0.80, 0.52, 0.27), 0.85 * spot(x, y, 1.16, 0.645, 0.20, 0.06))
        return mix(c, (0.10, 0.13, 0.24), 0.35 * (1 - warm(x, y)) * N2(x * 30, 3.0))
    tp = [(TX0 + 0.03, TY0), (TX1 - 0.03, TY0), (TX1, TY1), (TX0, TY1)]
    wfill(lambda x, y: sd_poly(x, y, tp), (TX0, TY0, TX1, TY1), top, 80, (12, 22), ang=0.0, lmul=(4, 10), aj=0.01, jc=(0.006, 0.03))
    apron = lambda x, y: sd_box(x, y, (TX0 + TX1) / 2, TY1 + 0.02, (TX1 - TX0) / 2, 0.02)
    wfill(apron, (TX0, TY1, TX1, TY1 + 0.04), lambda x, y: mix((0.03, 0.022, 0.02), (0.26, 0.16, 0.09), spot(x, y, 1.20, 0.69, 0.16, 0.06)), 20, (16, 24), ang=0.0, lmul=(4, 10), aj=0.01)
    line([(TX0, TY1), (TX1, TY1)], 5, (0.30, 0.20, 0.13), 0.7)
    line([(0.72, TY1), (TX1, TY1)], 5, (0.95, 0.70, 0.40), 0.7)
    for lx in (0.20, 1.12):
        leg = lambda x, y, lx=lx: sd_box(x, y, lx, 0.80, 0.05, 0.095)
        wfill(leg, (lx - 0.06, 0.70, lx + 0.06, 0.90), lambda x, y: (0.022, 0.018, 0.02), 6, (22, 34), ang=pi / 2, lmul=(3, 6), jc=(0.003, 0.02))
    layer('his chair')
    fb = Fr(1.375, 0.452, 0.106, 0.0, -1)
    cb = lambda x, y: sd_rbox(x, y, -1.18, 0.95, 0.20, 1.55, -0.06, 0.18)
    fb.fill(cb, (-1.6, -0.8, -0.8, 2.7), (0.10, 0.085, 0.085), 30, (0.09, 0.15), rad=0.2, spec=0.25)
    cs = lambda x, y: sd_box(x, y, -0.35, 2.75, 0.95, 0.14, 0.1)
    fb.fill(cs, (-1.4, 2.5, 0.7, 2.95), (0.08, 0.07, 0.07), 10, (0.09, 0.14), ang=0.0, rad=0.12)
    fb.line([(-0.4, 2.85), (-0.4, 3.75)], 0.14, (0.02, 0.02, 0.03))
    for a0 in (-1, -0.3, 0.4, 1):
        fb.line([(-0.4, 3.72), (-0.4 + a0 * 0.75, 3.92)], 0.08, (0.02, 0.02, 0.03)); fb.line([(-0.4 + a0 * 0.75, 3.98), (-0.4 + a0 * 0.75 + 0.01, 3.98)], 0.12, (0.015, 0.015, 0.02))
    layer('the man')
    body = seated_profile(fb, table_y=1.60)
    hf = fb.sub(0.50, -0.56, 0.50)
    head_profile(hf, toplight=0.4)
    fb.rim(body, (-1.2, -0.3, 2.0, 2.0), 60, lambda x, y, s: mix((0.30, 0.27, 0.26), (1.0, 0.86, 0.62), s), w=(0.025, 0.055), light=(lamp[0], lamp[1] - 0.1), thresh=0.5,
           inset=0.045, ln=(0.10, 0.32), opac=0.8)
    layer('on the table')
    paper = [(1.065, 0.628), (1.145, 0.628), (1.162, 0.662), (1.072, 0.662)]
    wfill(lambda x, y: sd_poly(x, y, paper), (1.05, 0.62, 1.18, 0.67), lambda x, y: (0.97, 0.90, 0.76), 6, (8, 12), ang=0.0, lmul=(2, 5), jc=(0.004, 0.015), margin=0.15)
    for k in range(5):
        yy = 0.634 + 0.0045 * k
        line([(1.076 + 0.0015 * k, yy), (1.140 + 0.003 * k - (0.03 if k == 4 else 0), yy)], 1.6, (0.35, 0.30, 0.26), 0.8)
    line(smooth([(1.122, 0.657), (1.132, 0.653), (1.138, 0.658), (1.148, 0.654)], 4), 1.8, (0.10, 0.10, 0.25))
    line([(1.168, 0.640), (1.192, 0.652)], 5, (0.03, 0.03, 0.05)); line([(1.168, 0.640), (1.174, 0.643)], 5, (0.85, 0.70, 0.40))
    plate = lambda x, y: sd_poly(x, y, [(0.950, 0.604), (1.040, 0.604), (1.047, 0.641), (0.943, 0.641)])
    wfill(plate, (0.94, 0.60, 1.05, 0.645), lambda x, y: mix((0.42, 0.30, 0.13), (0.95, 0.76, 0.40), (0.641 - y) / 0.037), 5, (7, 10), ang=0.0, lmul=(2, 5), margin=0.15, jc=(0.006, 0.02))
    glyphs('CEO', 0.966, 0.611, 0.023, 4.2, (0.06, 0.04, 0.03), sp=0.82)
    layer('near chairs')
    for i, cx in enumerate((0.075, 0.325, 0.575, 0.825)):
        chair(cx + U_(-0.02, 0.02), 0.535 + U_(-0.01, 0.01), 0.80, 0.085, U_(-0.07, 0.07), 0.2 + 0.8 * warm(cx + 0.15, 0.55), True)
    layer('lamp')
    line([(lamp[0], 0.0), (lamp[0], lamp[1] - 0.05)], 5, (0.02, 0.02, 0.03))
    shade = lambda x, y: sd_poly(x, y, [(lamp[0] - 0.028, lamp[1] - 0.055), (lamp[0] + 0.028, lamp[1] - 0.055), (lamp[0] + 0.075, lamp[1]), (lamp[0] - 0.075, lamp[1])])
    wfill(shade, (lamp[0] - 0.08, lamp[1] - 0.06, lamp[0] + 0.08, lamp[1] + 0.005), lambda x, y: (0.03, 0.03, 0.04), 6, (12, 18), ang=0.0, lmul=(2, 5), margin=0.15)
    for k in range(5):
        line([(lamp[0] - 0.07 + 0.004 * k, lamp[1] + 0.003), (lamp[0] + 0.07 - 0.004 * k, lamp[1] + 0.003)], 9 - k, mix((1.0, 0.75, 0.40), (1.0, 0.98, 0.9), k / 4), 0.9)
    for k in range(int(40 * E.DENS)):                                 # the cone of light
        t = U_(-1, 1); y1 = U_(0.35, 0.62)
        add(seg(lamp[0] + t * 0.06, lamp[1] + 0.01, lamp[0] + t * (0.06 + 0.55 * (y1 - lamp[1])), y1, 8), U_(70, 130), (1.0, 0.86, 0.60), 0.02, 0.5, 0.6, 0.1)


TROU = (0.10, 0.105, 0.125)


def walking_profile(fb, skin=SKIN, box=True):
    """Side view, mid-stride, facing local +x, carrying a box. fb: origin at the base of the neck, unit = head."""
    parts = []
    hip = (-0.05, 3.0)
    for (knee, ank, toe, k) in (((-0.42, 4.55), (-1.12, 5.80), (-0.86, 6.18), 0.75), ((0.55, 4.50), (0.70, 5.98), (1.12, 6.14), 1.0)):
        col = (TROU[0] * k, TROU[1] * k, TROU[2] * k)
        th = lambda x, y, knee=knee: sd_tcap(x, y, hip[0], hip[1], knee[0], knee[1], 0.36, 0.25)
        sh = lambda x, y, knee=knee, ank=ank: sd_tcap(x, y, knee[0], knee[1], ank[0], ank[1], 0.25, 0.15)
        a1 = math.atan2(knee[1] - hip[1], knee[0] - hip[0]); a2 = math.atan2(ank[1] - knee[1], ank[0] - knee[0])
        fb.fill(sh, (-1.5, 4.2, 1.2, 6.2), col, 26, (0.07, 0.12), ang=a2, rad=0.2)
        fb.fill(th, (-0.9, 2.6, 1.0, 4.9), col, 30, (0.08, 0.13), ang=a1, rad=0.25)
        shoe = lambda x, y, ank=ank, toe=toe: sd_tcap(x, y, ank[0] - 0.05, ank[1] + 0.10, toe[0], toe[1], 0.17, 0.10)
        fb.fill(shoe, (-1.4, 5.7, 1.4, 6.35), (0.035, 0.035, 0.04), 8, (0.06, 0.09), ang=math.atan2(toe[1] - ank[1], toe[0] - ank[0]), rad=0.1, spec=0.5)
        parts += [th, sh, shoe]
    neck = lambda x, y: sd_tcap(x, y, -0.02, 0.0, 0.20, -0.42, 0.22, 0.17)
    fb.fill(neck, (-0.3, -0.6, 0.5, 0.2), (skin[0] * 0.80, skin[1] * 0.72, skin[2] * 0.68), 8, (0.05, 0.08), rad=0.18, texa=0.05, jc=(0.006, 0.02))
    tee = lambda x, y: sd_poly(x, y, [(0.05, -0.15), (0.40, 0.0), (0.56, 0.7), (0.2, 0.8)])
    fb.fill(tee, (0.0, -0.2, 0.6, 0.8), TEE, 5, (0.06, 0.09), rad=0.2, edge=False)
    torso = lambda x, y: sd_poly(x, y, [(-0.34, -0.16), (0.10, -0.12), (0.38, 0.28), (0.52, 1.0), (0.50, 2.2), (0.44, 3.12), (-0.52, 3.18), (-0.64, 2.0), (-0.70, 0.9), (-0.58, 0.15)]) - 0.04
    leather(fb, torso, (-0.8, -0.3, 0.7, 3.3), 120, (0.07, 0.13), ang=pi / 2 + 0.05)
    collar = lambda x, y: sd_poly(x, y, [(-0.36, -0.22), (0.12, -0.20), (0.32, 0.12), (0.22, 0.34), (-0.05, 0.05), (-0.40, -0.02)])
    leather(fb, collar, (-0.45, -0.3, 0.4, 0.4), 12, (0.05, 0.08), ang=0.3, rad=0.08)
    crease(fb, [(-0.40, -0.02), (-0.05, 0.05), (0.22, 0.34)], 0.022, (0.32, 0.33, 0.38), 0.7)
    crease(fb, [(0.36, 0.25), (0.52, 1.0), (0.50, 2.2), (0.45, 3.05)], 0.02, (0.55, 0.56, 0.60), 0.65)             # zip edge
    for k in range(9):
        bx = U_(-0.55, 0.25); by = U_(0.5, 2.8)
        crease(fb, [(bx, by), (bx + 0.05, by + 0.14), (bx + 0.02, by + 0.3)], U_(0.02, 0.035), (0.20, 0.21, 0.26), U_(0.2, 0.45))
    if box:
        B = [(0.50, 1.42), (1.78, 1.36), (1.80, 2.26), (0.52, 2.34)]
        # things from the desk
        fb.line(smooth([(1.45, 1.40), (1.50, 1.05), (1.42, 0.80)], 5), 0.03, (0.16, 0.30, 0.14))
        for (lx, ly, ar) in ((1.30, 0.86, -2.4), (1.58, 0.82, -0.7), (1.40, 0.66, -1.6), (1.62, 1.05, -0.2), (1.26, 1.08, -2.9)):
            leaf = lambda x, y, lx=lx, ly=ly, ar=ar: sd_rbox(x, y, lx, ly, 0.17, 0.06, ar, 0.06)
            fb.fill(leaf, (lx - 0.2, ly - 0.2, lx + 0.2, ly + 0.2), (0.20, 0.42, 0.18), 3, (0.05, 0.08), ang=ar, rad=0.06, edge=False, lmul=(1.5, 3))
        frame = lambda x, y: sd_rbox(x, y, 0.92, 1.22, 0.24, 0.30, 0.12, 0.01)
        fb.fill(frame, (0.6, 0.85, 1.25, 1.6), (0.10, 0.08, 0.07), 5, (0.05, 0.08), rad=0.05)
        inner = lambda x, y: sd_rbox(x, y, 0.92, 1.22, 0.18, 0.24, 0.12, 0.01)
        fb.fill(inner, (0.6, 0.85, 1.25, 1.6), (0.55, 0.60, 0.66), 4, (0.04, 0.06), rad=0.3, edge=False)
        bx = lambda x, y: sd_poly(x, y, B)
        fb.fill(bx, (0.45, 1.3, 1.85, 2.4), (0.62, 0.45, 0.28), 30, (0.06, 0.11), ang=0.0, rad=0.6, lmul=(3, 6), aj=0.04, texa=0.2)
        fb.line([(0.52, 1.56), (1.79, 1.50)], 0.03, (0.36, 0.25, 0.15), 0.7)
        fb.line([(1.05, 1.40), (1.06, 1.52)], 0.05, (0.80, 0.68, 0.48), 0.8)                                          # tape
        parts.append(bx)
    ua = lambda x, y: sd_tcap(x, y, -0.10, 0.42, 0.10, 1.72, 0.30, 0.24)
    leather(fb, ua, (-0.5, 0.1, 0.5, 2.0), 46, (0.06, 0.11), ang=pi / 2 - 0.15)
    fa = lambda x, y: sd_tcap(x, y, 0.10, 1.72, 1.05, 2.10, 0.23, 0.17)
    leather(fb, fa, (-0.2, 1.4, 1.3, 2.4), 30, (0.06, 0.10), ang=0.38)
    for k in range(5):
        by = U_(1.2, 1.75); a0 = U_(-0.4, 0.4)
        crease(fb, [(-0.16, by + 0.06 * a0), (0.02, by + 0.05), (0.26, by - 0.08 * a0)], U_(0.018, 0.03), opac=U_(0.15, 0.4))
    hd = lambda x, y: min(sd_ell(x, y, 1.20, 2.16, 0.19, 0.15), sd_ell(x, y, 1.30, 2.26, 0.15, 0.10))
    fb.fill(hd, (0.95, 1.95, 1.5, 2.4), skin, 10, (0.04, 0.07), ang=0.3, rad=0.12, texa=0.05, jc=(0.006, 0.02), lmul=(1.5, 3))
    for k in range(3):
        fb.line([(1.18 + 0.07 * k, 2.10), (1.22 + 0.07 * k, 2.30)], 0.016, (skin[0] * 0.4, skin[1] * 0.3, skin[2] * 0.28), 0.7)
    parts += [torso, ua, fa]
    return lambda x, y: min(p(x, y) for p in parts)


def bystander(cx, feet, h, col, hair, pose, rng, face=1):
    """A simple standing figure, seen straight on and back-lit; returns its silhouette."""
    u = h / 7.4; hy = feet - h + 0.5 * u
    P = []
    head = lambda x, y: sd_ell(x, y, cx, hy, 0.37 * u, 0.48 * u)
    torso = lambda x, y: sd_poly(x, y, [(cx - 0.74 * u, hy + 0.98 * u), (cx + 0.74 * u, hy + 0.98 * u), (cx + 0.60 * u, hy + 3.6 * u), (cx - 0.60 * u, hy + 3.6 * u)]) - 0.15 * u
    legs = [lambda x, y, s=s: sd_tcap(x, y, cx + s * 0.34 * u, hy + 3.5 * u, cx + s * 0.40 * u, feet, 0.33 * u, 0.22 * u) for s in (-1, 1)]
    arms = []
    for s in (-1, 1):
        sh = (cx + s * 0.80 * u, hy + 1.15 * u)
        if pose == 'cross':
            el = (cx + s * 0.98 * u, hy + 2.3 * u); hd = (cx - s * 0.45 * u, hy + 2.15 * u + s * 0.12 * u)
        elif pose == 'phone' and s == face:
            el = (cx + s * 1.20 * u, hy + 1.9 * u); hd = (cx + s * 1.15 * u, hy + 0.75 * u)
        elif pose == 'mouth' and s == face:
            el = (cx + s * 1.0 * u, hy + 2.2 * u); hd = (cx + s * 0.15 * u, hy + 0.45 * u)
        else:
            el = (cx + s * 0.95 * u, hy + 2.35 * u); hd = (cx + s * 0.92 * u, hy + 3.6 * u)
        arms.append(lambda x, y, sh=sh, el=el: sd_tcap(x, y, sh[0], sh[1], el[0], el[1], 0.26 * u, 0.21 * u))
        arms.append(lambda x, y, el=el, hd=hd: sd_tcap(x, y, el[0], el[1], hd[0], hd[1], 0.21 * u, 0.15 * u))
    trou = (col[0] * 0.45 + 0.01, col[1] * 0.45 + 0.012, col[2] * 0.45 + 0.02)
    bb = (cx - 1.7 * u, hy - 0.6 * u, cx + 1.7 * u, feet + 0.01)
    for lg in legs:
        wfill(lg, bb, lambda x, y: trou, 5, (h * E.S * 0.035, h * E.S * 0.05), ang=pi / 2, lmul=(2, 5), jc=(0.006, 0.03))
    wfill(lambda x, y: sd_cap(x, y, cx, hy, cx, hy + 0.8 * u, 0.2 * u), bb, lambda x, y: (0.14, 0.10, 0.08), 2, (h * E.S * 0.03, h * E.S * 0.04), ang=pi / 2)
    wfill(torso, bb, lambda x, y: tex(col, x, y, 0.2, 40), 9, (h * E.S * 0.04, h * E.S * 0.06), ang=pi / 2, lmul=(2, 5), jc=(0.006, 0.03))
    for ar in arms:
        wfill(ar, bb, lambda x, y: (col[0] * 0.8, col[1] * 0.8, col[2] * 0.8), 4, (h * E.S * 0.03, h * E.S * 0.045), ang=pi / 2, lmul=(2, 4), jc=(0.006, 0.03))
    wfill(head, bb, lambda x, y: (0.17, 0.12, 0.10), 3, (h * E.S * 0.025, h * E.S * 0.035), ang=pi / 2, lmul=(1.5, 3), jc=(0.006, 0.03))
    hr = lambda x, y: max(sd_ell(x, y, cx, hy - 0.04 * u, 0.41 * u, 0.50 * u), (y - (hy - 0.12 * u + rng.uniform(0, 0)))) if hair[1] == 0 else \
        min(max(sd_ell(x, y, cx, hy - 0.04 * u, 0.42 * u, 0.50 * u), y - (hy - 0.10 * u)), sd_box(x, y, cx, hy + 0.35 * u, 0.43 * u, 0.55 * u, 0.2 * u) if abs(x - cx) > 0.24 * u else 1.0)
    wfill(hr, bb, lambda x, y: hair[0], 4, (h * E.S * 0.02, h * E.S * 0.03), ang=pi / 2, lmul=(1.5, 3), jc=(0.006, 0.03), margin=0.2)
    if pose == 'phone':
        px_, py_ = cx + face * 1.15 * u, hy + 0.55 * u
        wfill(lambda x, y: sd_box(x, y, px_, py_, 0.17 * u, 0.30 * u, 0.02 * u), bb, lambda x, y: (0.02, 0.02, 0.03), 2, (h * E.S * 0.018, h * E.S * 0.025), ang=pi / 2, margin=0.2)
        dot(px_, py_ - 0.17 * u, h * E.S * 0.012, (0.75, 0.80, 0.85)); dot(px_ + 0.07 * u, py_ - 0.2 * u, h * E.S * 0.008, (1, 1, 1))
    P = [head, torso] + legs + arms
    return lambda x, y: min(p(x, y) for p in P)


def walkout():
    """3. The walk: one box, the whole floor watching through the glass, and a door full of daylight."""
    E.setup(3000, 1875)
    DX0, DX1, DY0, FL = 1.345, 1.565, 0.215, 0.745          # door, floor line
    sun = (1.50, 0.50)
    set_lights([('d', (0.85, -0.25, 0.45), (1.9, 1.7, 1.35)), ('f', (-0.6, -0.2, 0.7), (0.16, 0.24, 0.28))], (0.05, 0.055, 0.07))
    layer('office behind glass')
    GX1 = 1.215

    def office(x, y):
        c = mix((0.60, 0.72, 0.72), (0.82, 0.89, 0.86), sstep((y - 0.16) / 0.36))
        c = mix(c, (0.50, 0.62, 0.64), 0.8 * sstep((0.20 - y) / 0.06))                                         # ceiling
        return mix(c, (0.38, 0.49, 0.53), 0.85 * sstep((y - 0.565) / 0.02))                                      # desk partitions
    wfill(lambda x, y: sd_box(x, y, GX1 / 2, 0.43, GX1 / 2, 0.32), (0, 0.11, GX1, 0.75), office, 200, (24, 48), ang=pi / 2, lmul=(2, 5), aj=0.03, jc=(0.004, 0.018))
    for row, (yy, hw, hh, sp) in enumerate(((0.140, 0.050, 0.010, 0.150), (0.178, 0.034, 0.006, 0.105))):   # ceiling light panels
        xx = 0.04 + 0.03 * row
        while xx < GX1 - 0.03:
            pn = lambda x, y, xx=xx, yy=yy, hw=hw, hh=hh: sd_box(x, y, xx, yy, hw, hh)
            wfill(pn, (xx - hw, yy - hh, xx + hw, yy + hh), lambda x, y: (0.97, 1.0, 0.97), 2, (hh * E.S * 1.6, hh * E.S * 2.0), ang=0.0, lmul=(3, 6), margin=0.1, jc=(0.003, 0.01))
            xx += sp
    for xx in (0.20, 0.52, 0.86, 1.13):                     # columns and a glass door inside
        line([(xx, 0.20), (xx, 0.57)], 16, (0.52, 0.63, 0.64), 0.8)
    line([(0.66, 0.57), (0.66, 0.30), (0.75, 0.30), (0.75, 0.57)], 5, (0.42, 0.53, 0.55), 0.8)
    for k in range(9):                                      # monitors on the desks
        mx = 0.06 + k * 0.135 + U_(-0.02, 0.02)
        m = lambda x, y, mx=mx: sd_box(x, y, mx, 0.548, 0.026, 0.02, 0.003)
        wfill(m, (mx - 0.04, 0.52, mx + 0.04, 0.58), lambda x, y: (0.20, 0.28, 0.36), 2, (12, 18), ang=0.0, lmul=(2, 4), jc=(0.006, 0.03), margin=0.2)
    layer('the staff')
    rng = random.Random(4)
    PAL = [(0.09, 0.12, 0.18), (0.20, 0.08, 0.10), (0.11, 0.15, 0.11), (0.16, 0.15, 0.14), (0.07, 0.14, 0.17), (0.20, 0.16, 0.08), (0.14, 0.10, 0.17), (0.22, 0.22, 0.24)]
    HAIRS = [((0.05, 0.04, 0.04), 0), ((0.07, 0.05, 0.04), 1), ((0.22, 0.13, 0.07), 1), ((0.04, 0.04, 0.05), 0), ((0.30, 0.22, 0.12), 0), ((0.10, 0.07, 0.05), 1)]
    poses = ['cross', 'plain', 'phone', 'mouth', 'plain', 'cross', 'phone', 'plain', 'cross', 'mouth', 'plain']
    staff = []
    xs = [0.07, 0.175, 0.27, 0.385, 0.48, 0.60, 0.70, 0.80, 0.905, 1.00, 1.10]
    order = list(range(len(xs))); rng.shuffle(order)
    for i in sorted(order, key=lambda i: (i % 2)):          # back row first
        back = (i % 2 == 0)
        hgt = rng.uniform(0.36, 0.41) * (0.90 if back else 1.0)
        staff.append(bystander(xs[i] + rng.uniform(-0.015, 0.015), FL + (0.0 if back else 0.035), hgt, PAL[i % len(PAL)], HAIRS[i % len(HAIRS)], poses[i], rng))
    alls = lambda x, y: min(f(x, y) for f in staff)
    E.rim(alls, (0, 0.28, GX1, FL), 150, lambda x, y, s: (0.80, 0.93, 0.92), w=(2.5, 4.5), inset=0.002, ln=(0.015, 0.05), opac=0.55, ymax=FL - 0.01)
    layer('walls')
    DARK = (0.035, 0.038, 0.05)
    wfill(lambda x, y: y - 0.115, (0, 0, E.AR, 0.115), lambda x, y: mix(DARK, (0.10, 0.12, 0.13), sstep((y - 0.06) / 0.06) * (1 if x < GX1 else 0.3)), 30, (26, 44), ang=0.0,
          lmul=(4, 9), aj=0.01, jc=(0.004, 0.03))
    pier = lambda x, y: min(sd_box(x, y, (GX1 + DX0) / 2, 0.43, (DX0 - GX1) / 2, 0.33), sd_box(x, y, (DX1 + E.AR) / 2 + 0.01, 0.43, (E.AR - DX1) / 2 + 0.01, 0.33),
                            sd_box(x, y, (DX0 + DX1) / 2, 0.16, (DX1 - DX0) / 2, 0.06))
    wfill(pier, (GX1, 0.10, E.AR, 0.76), lambda x, y: mix(DARK, (0.30, 0.25, 0.18), 0.5 * bump(x, DX0 - 0.01, 0.03) + 0.5 * bump(x, DX1 + 0.01, 0.02)), 40, (24, 44),
          ang=pi / 2, lmul=(3, 7), aj=0.02, jc=(0.004, 0.03))
    for xm in (0.0, 0.305, 0.61, 0.915, GX1):               # mullions
        line([(xm, 0.105), (xm, FL + 0.005)], 20, (0.02, 0.022, 0.03))
        line([(xm + 0.006, 0.12), (xm + 0.006, FL)], 3, (0.45, 0.42, 0.36), 0.6)
    line([(0, 0.112), (GX1, 0.112)], 14, (0.02, 0.022, 0.03))
    for k in range(int(26 * E.DENS)):                       # reflections sliding across the glass
        x0 = U_(0.0, GX1 - 0.1); y0 = U_(0.14, 0.60); L = U_(0.05, 0.16)
        add(seg(x0, y0 + L, x0 + L * 0.55, y0, 4), U_(20, 46), (1.0, 0.96, 0.88), 0.045, 0.5, 0.5, 0.5)
    layer('the door')

    def outside(x, y):
        t = (y - DY0) / (FL - DY0)
        return mix((1.0, 0.99, 0.93), (1.0, 0.86, 0.60), sstep((t - 0.45) / 0.55) * 0.8)
    wfill(lambda x, y: sd_box(x, y, (DX0 + DX1) / 2, (DY0 + FL) / 2, (DX1 - DX0) / 2, (FL - DY0) / 2), (DX0, DY0, DX1, FL), outside, 40, (24, 44), ang=pi / 2, lmul=(3, 7),
          aj=0.03, jc=(0.004, 0.012))
    for xx in (DX0, (DX0 + DX1) / 2, DX1):
        line([(xx, DY0 - 0.005), (xx, FL)], 9, (0.05, 0.05, 0.06))
    line([(DX0, DY0), (DX1, DY0)], 10, (0.05, 0.05, 0.06))
    for s in (-1, 1):
        line([((DX0 + DX1) / 2 + s * 0.012, 0.47), ((DX0 + DX1) / 2 + s * 0.012, 0.56)], 5, (0.25, 0.22, 0.18))
    for k in range(int(40 * E.DENS)):                       # bloom
        xx = U_(DX0 - 0.10, DX1 + 0.04); y0 = U_(DY0 - 0.02, FL - 0.2)
        add(seg(xx, y0, xx + U_(-0.01, 0.01), y0 + U_(0.1, 0.2), 4), U_(60, 110), (1.0, 0.90, 0.66), 0.035 * bump(xx, (DX0 + DX1) / 2, 0.16), 0.4, 0.5, 0.5)
    sg = lambda x, y: sd_box(x, y, (DX0 + DX1) / 2, 0.168, 0.052, 0.022, 0.004)
    wfill(sg, (DX0, 0.14, DX1, 0.20), lambda x, y: (0.10, 0.62, 0.34), 5, (12, 18), ang=0.0, lmul=(2, 5), margin=0.2, jc=(0.006, 0.03))
    glyphs('EXIT', (DX0 + DX1) / 2 - 0.040, 0.155, 0.026, 4.5, (0.90, 1.0, 0.93), sp=0.80)
    layer('floor')

    def floor(x, y):
        c = mix((0.055, 0.06, 0.075), (0.09, 0.095, 0.11), N2(x * 5, y * 30))
        c = mix(c, (0.42, 0.56, 0.58), 0.55 * math.exp(-(y - FL) / 0.035) * (1 if x < GX1 else 0))            # the office, mirrored in the stone
        k = math.exp(-(max(0.0, 1.42 - x) / 0.80) ** 2) * (0.45 + 0.55 * sstep((y - FL) / 0.16)) * (1 - 0.5 * sstep((x - 1.50) / 0.12))   # daylight
        return mix(c, (1.0, 0.90, 0.66), 0.92 * k)
    wfill(lambda x, y: FL - y, (0, FL, E.AR, 1), floor, 260, (20, 46), ang=0.0, lmul=(4, 10), aj=0.02, jc=(0.006, 0.03))
    line([(0, FL + 0.003), (GX1, FL + 0.003)], 8, (0.02, 0.022, 0.03))
    layer('his shadow')
    fb = Fr(0.985, 0.397, 0.0815)
    fy = fb.l2w(0, 6.2)[1]
    shd = lambda x, y: min(sd_tcap(x, y, 1.065, fy + 0.006, 0.80, fy + 0.024, 0.008, 0.020), sd_tcap(x, y, 0.915, fy + 0.010, 0.80, fy + 0.024, 0.008, 0.020),
                           sd_tcap(x, y, 0.80, fy + 0.024, 0.47, fy + 0.044, 0.022, 0.036), sd_ell(x, y, 0.385, fy + 0.050, 0.055, 0.026),
                           sd_ell(x, y, 0.66, fy + 0.016, 0.075, 0.018))
    wfill(shd, (0.3, fy - 0.03, 1.1, fy + 0.09), lambda x, y: (0.035, 0.037, 0.05), 30, (12, 22), ang=-0.06, lmul=(3, 8), aj=0.03, jc=(0.004, 0.03))
    layer('the man')
    body = walking_profile(fb)
    head_profile(fb.sub(0.47, -0.58, 0.42), toplight=0.25)
    hipy = fb.l2w(0, 3.1)[1]
    fb.rim(body, (-1.3, -0.3, 2.0, 3.2), 70, lambda x, y, s: mix((0.45, 0.40, 0.34), (1.0, 0.93, 0.76), s), w=(0.025, 0.06), light=(sun[0] + 2.0, sun[1]), thresh=0.45,
           inset=0.045, ln=(0.12, 0.4), opac=0.85, ymax=hipy)
    fb.rim(body, (-1.3, 3.0, 2.0, 6.3), 26, lambda x, y, s: mix((0.30, 0.28, 0.26), (0.80, 0.72, 0.58), s), w=(0.02, 0.035), light=(sun[0] + 2.0, sun[1]), thresh=0.6,
           inset=0.04, ln=(0.10, 0.3), opac=0.55, ymin=hipy)


def jacket_back(fb, hand_y=1.28, skin=SKIN):
    """From behind, arms braced forward on a counter. fb: origin at the nape, unit = head."""
    parts = []
    for s in (-1, 1):
        hd = lambda x, y, s=s: sd_ell(x, y, s * 1.24, hand_y + 0.10, 0.17, 0.13)
        fb.fill(hd, (min(s, s * 1.5), hand_y - 0.1, max(s, s * 1.5), hand_y + 0.3), skin, 8, (0.04, 0.07), ang=0.0, rad=0.12, texa=0.05, jc=(0.006, 0.02), lmul=(1.5, 3))
        for k in range(3):
            fb.line([(s * (1.16 + 0.07 * k), hand_y + 0.12), (s * (1.15 + 0.07 * k), hand_y + 0.23)], 0.016, (skin[0] * 0.4, skin[1] * 0.3, skin[2] * 0.28), 0.7)
        fa = lambda x, y, s=s: sd_tcap(x, y, s * 1.40, 0.98, s * 1.25, hand_y - 0.06, 0.22, 0.16)
        leather(fb, fa, (min(s * 0.9, s * 1.7), 0.7, max(s * 0.9, s * 1.7), hand_y + 0.15), 26, (0.06, 0.10), ang=pi / 2 + s * 0.4)
        ua = lambda x, y, s=s: sd_tcap(x, y, s * 0.94, 0.42, s * 1.40, 0.98, 0.25, 0.23)
        leather(fb, ua, (min(s * 0.6, s * 1.75), 0.0, max(s * 0.6, s * 1.75), 1.3), 44, (0.06, 0.11), ang=pi / 2 - s * 0.5)
        parts += [fa, ua]
        for k in range(6):
            t = U_(0.72, 1.12); x0 = U_(1.18, 1.3)
            crease(fb, [(s * x0, t), (s * (x0 + 0.10), t + U_(0.02, 0.07)), (s * (x0 + 0.22), t - 0.02)], U_(0.018, 0.03), opac=U_(0.2, 0.45))
    back = lambda x, y: sd_poly(x, y, [(-0.36, -0.14), (0.36, -0.14), (0.74, -0.02), (1.06, 0.14), (1.15, 0.6), (1.04, 1.6), (1.0, 2.8), (-1.0, 2.8), (-1.04, 1.6), (-1.15, 0.6),
                                       (-1.06, 0.14), (-0.74, -0.02)]) - 0.04
    leather(fb, back, (-1.25, -0.25, 1.25, 2.9), 240, (0.08, 0.14), ang=pi / 2, wr=0.7)
    collar = lambda x, y: sd_poly(x, y, [(-0.40, -0.22), (0.40, -0.22), (0.55, 0.02), (0.30, 0.10), (-0.30, 0.10), (-0.55, 0.02)]) - 0.02
    leather(fb, collar, (-0.6, -0.26, 0.6, 0.14), 16, (0.05, 0.08), ang=0.0, rad=0.10)
    crease(fb, [(-0.55, 0.04), (-0.30, 0.12), (0.30, 0.12), (0.55, 0.04)], 0.022, (0.30, 0.32, 0.37), 0.7)
    crease(fb, [(-1.0, 0.50), (-0.5, 0.62), (0, 0.65), (0.5, 0.62), (1.0, 0.50)], 0.028, (0.008, 0.008, 0.012), 0.85)                 # yoke seam
    crease(fb, [(-0.98, 0.46), (-0.5, 0.58), (0, 0.61), (0.5, 0.58), (0.98, 0.46)], 0.016, (0.26, 0.28, 0.33), 0.5)
    crease(fb, [(0.0, 0.66), (0.01, 1.6), (0.0, 2.8)], 0.022, (0.008, 0.008, 0.012), 0.8)                                            # centre seam
    for s in (-1, 1):
        crease(fb, [(s * 0.86, 0.22), (s * 0.90, 0.7), (s * 0.96, 1.3)], 0.03, (0.008, 0.008, 0.012), 0.8)                            # armhole
        for k in range(8):
            bx = s * U_(0.12, 0.8); by = U_(0.8, 2.5)
            crease(fb, [(bx, by), (bx + s * 0.06, by + 0.14), (bx + s * 0.02, by + 0.3)], U_(0.02, 0.04), (0.17, 0.18, 0.22), U_(0.2, 0.45))
        for k in range(4):                                                                                                         # pull across the shoulder blades
            by = U_(0.75, 1.3)
            crease(fb, [(s * U_(0.2, 0.4), by), (s * 0.6, by - 0.04), (s * 0.85, by - 0.14)], U_(0.018, 0.028), (0.16, 0.17, 0.21), U_(0.15, 0.3))
    parts.append(back)
    return lambda x, y: min(p(x, y) for p in parts)


def mirror():
    """4. The washroom: braced on the counter, he cannot look up at the man in the glass."""
    E.setup(2400, 3000)
    MX0, MX1, MY0, MY1 = 0.05, 0.75, 0.10, 0.70
    CT, CE = 0.785, 0.81                                    # counter front edge, top and bottom
    tube = lambda x, y: math.exp(-max(0.0, abs(x - 0.40) - 0.28) / 0.10) * math.exp(-abs(y - 0.058) / 0.20)
    layer('tiled wall')

    def wall(x, y):
        c = mix((0.17, 0.235, 0.225), (0.24, 0.32, 0.30), N2(x * 9, y * 9))
        return mix(tuple(v * (0.55 + 0.45 * sstep((0.9 - y) / 0.8)) for v in c), (0.66, 0.78, 0.74), 0.8 * tube(x, y) ** 1.5)
    wfill(ALL, (0, 0, E.AR, 1), wall, 200, (44, 80), ang=pi / 2, lmul=(2, 4.5), aj=0.03, jc=(0.004, 0.025))
    for k in range(1, 30):
        yy = k * 0.0355
        line([(0, yy), (E.AR, yy)], 4, (0.07, 0.10, 0.10), 0.6)
    for k in range(0, 24):
        xx = 0.012 + k * 0.0355
        line([(xx, 0), (xx, 1)], 4, (0.07, 0.10, 0.10), 0.6)
    layer('in the mirror: the room behind')

    def room(x, y):
        c = mix((0.085, 0.115, 0.12), (0.125, 0.165, 0.165), N2(x * 7, y * 4))
        c = tuple(v * (0.55 + 1.25 * math.exp(-((y - 0.10) / 0.22) ** 2)) for v in c)
        return mix(c, (0.05, 0.065, 0.07), 0.8 * sstep((y - 0.52) / 0.10))
    msd = lambda x, y: sd_box(x, y, (MX0 + MX1) / 2, (MY0 + MY1) / 2, (MX1 - MX0) / 2, (MY1 - MY0) / 2)
    wfill(msd, (MX0, MY0, MX1, MY1), room, 240, (40, 74), ang=pi / 2, lmul=(2, 5), aj=0.03, jc=(0.004, 0.02), margin=0.1)
    for k in range(1, 26):                                  # far-wall tiles, smaller and dimmer
        yy = MY0 + k * 0.024
        line([(MX0, yy), (MX1, yy)], 2.5, (0.045, 0.065, 0.07), 0.4)
    for k in range(1, 30):
        xx = MX0 + k * 0.024
        line([(xx, MY0), (xx, MY1)], 2.5, (0.045, 0.065, 0.07), 0.4)
    # the door behind him, ajar on a lit corridor
    door = lambda x, y: sd_box(x, y, 0.655, 0.47, 0.075, 0.25)
    wfill(door, (0.57, 0.21, 0.74, 0.72), lambda x, y: mix((0.055, 0.06, 0.07), (0.10, 0.11, 0.12), sstep((0.45 - y) / 0.3)), 30, (30, 50), ang=pi / 2, lmul=(3, 7), aj=0.02,
          jc=(0.004, 0.02))
    slit = lambda x, y: sd_box(x, y, 0.724, 0.47, 0.007, 0.25)
    wfill(slit, (0.71, 0.21, 0.74, 0.72), lambda x, y: mix((1.0, 0.80, 0.50), (1.0, 0.92, 0.72), N2(x * 3, y * 9)), 10, (18, 26), ang=pi / 2, lmul=(4, 9), aj=0.01,
          jc=(0.004, 0.02), margin=0.1)
    for k in range(int(14 * E.DENS)):
        y0 = U_(0.22, 0.60)
        add(seg(0.722 + U_(-0.02, 0.006), y0, 0.722 + U_(-0.02, 0.006), y0 + U_(0.06, 0.12), 3), U_(40, 70), (1.0, 0.78, 0.46), 0.05, 0.4, 0.5, 0.5)
    line([(0.612, 0.50), (0.628, 0.50)], 9, (0.45, 0.47, 0.50), 0.9)                                                              # handle
    wfill(lambda x, y: sd_box(x, y, 0.40, 0.67, 0.35, 0.03), (MX0, 0.64, MX1, 0.70), lambda x, y: mix((0.30, 0.34, 0.34), (0.44, 0.49, 0.48), N2(x * 6, y * 40)), 26, (20, 34),
          ang=0.0, lmul=(4, 9), aj=0.02, jc=(0.004, 0.02), margin=0.15)                                                         # the counter, mirrored
    line([(MX0, 0.641), (MX1, 0.641)], 6, (0.60, 0.66, 0.64), 0.8)
    layer('in the mirror: the man')
    set_lights([('p', (0.25, 0.04, 0.24), (2.3, 2.4, 2.35), 0.55), ('f', (0.0, 0.4, 0.9), (0.24, 0.27, 0.28))], (0.05, 0.06, 0.065))
    fr = Fr(0.25, 0.420, 0.128)
    neck = lambda x, y: sd_cap(x, y, 0, -0.3, 0, 0.0, 0.21)
    fr.fill(neck, (-0.25, -0.5, 0.25, 0.2), (SKIN[0] * 0.5, SKIN[1] * 0.42, SKIN[2] * 0.4), 6, (0.06, 0.1), rad=0.2, edge=False)
    rbody = jacket_front(fr, lectern_y=1.72)
    head_front_bowed(fr.sub(0, -0.40), toplight=1.0)
    fr.rim(rbody, (-1.6, -0.3, 1.6, 1.0), 50, lambda x, y, s: mix((0.22, 0.25, 0.27), (0.62, 0.70, 0.69), s), w=(0.018, 0.04), light=(0.25, -0.6), thresh=0.6,
           inset=0.04, ln=(0.10, 0.32), opac=0.6, ymax=0.50)
    layer('glass')
    for k in range(int(26 * E.DENS)):                       # the silvering: cool, a little milky, with slanting sheen
        x0 = U_(MX0, MX1 - 0.1); y0 = U_(MY0 + 0.03, MY1 - 0.2); L = U_(0.08, 0.2)
        add(seg(x0 + L * 0.5, y0, x0, y0 + L, 4), U_(50, 110), (0.72, 0.90, 0.88), 0.035, 0.4, 0.5, 0.5)
    for (x0, y0, L, w) in ((0.47, 0.105, 0.20, 26), (0.515, 0.105, 0.12, 12), (0.10, 0.50, 0.12, 14)):
        add(seg(x0 + L * 0.45, y0, x0, y0 + L, 4), w, (0.85, 0.97, 0.95), 0.10, 0.5, 0.3, 0.3)
    fw = 13
    line([(MX0, MY0), (MX1, MY0), (MX1, MY1), (MX0, MY1), (MX0, MY0 - 0.002)], fw, (0.50, 0.57, 0.57))
    line([(MX0 + 0.003, MY0 + 0.003), (MX1 - 0.003, MY0 + 0.003)], 4, (0.80, 0.90, 0.88), 0.8)
    layer('strip light')
    tb = lambda x, y: sd_box(x, y, 0.40, 0.058, 0.285, 0.011, 0.01)
    wfill(lambda x, y: sd_box(x, y, 0.40, 0.058, 0.30, 0.019, 0.004), (0.09, 0.035, 0.71, 0.08), lambda x, y: (0.24, 0.28, 0.28), 10, (20, 30), ang=0.0, lmul=(4, 9), aj=0.01)
    wfill(tb, (0.11, 0.045, 0.69, 0.071), lambda x, y: (0.96, 1.0, 0.97), 12, (22, 30), ang=0.0, lmul=(4, 9), aj=0.005, jc=(0.003, 0.01), margin=0.15)
    for k in range(int(40 * E.DENS)):
        x0 = U_(0.10, 0.70); y0 = 0.058 + U_(-0.05, 0.07)
        add(seg(x0 - 0.06, y0, x0 + 0.06, y0 + U_(-0.01, 0.01), 3), U_(50, 110), (0.80, 1.0, 0.94), 0.035, 0.4, 0.5, 0.5)
    layer('counter')

    def marble(x, y):
        v = abs(math.sin(x * 22 + 7 * N2(x * 3, y * 9) + y * 6))
        c = mix((0.52, 0.57, 0.56), (0.80, 0.85, 0.83), sstep((y - 0.70) / 0.085) * 0.6 + 0.4 * N2(x * 5, y * 20))
        c = mix(c, (0.30, 0.34, 0.35), 0.45 * math.exp(-v * 9))
        return tuple(k * (0.62 + 0.38 * math.exp(-((x - 0.35) / 0.5) ** 2)) for k in c)
    wfill(lambda x, y: sd_box(x, y, 0.4, (MY1 + CT) / 2, 0.5, (CT - MY1) / 2), (0, MY1, E.AR, CT), marble, 90, (22, 40), ang=0.0, lmul=(4, 10), aj=0.03, jc=(0.004, 0.02), margin=0.1)
    wfill(lambda x, y: sd_box(x, y, 0.4, (CT + CE) / 2, 0.5, (CE - CT) / 2), (0, CT, E.AR, CE), lambda x, y: mix((0.30, 0.34, 0.34), (0.40, 0.44, 0.44), N2(x * 8, y * 30)), 20, (22, 30),
          ang=0.0, lmul=(5, 12), aj=0.01, jc=(0.004, 0.02), margin=0.1)
    line([(0, CT), (E.AR, CT)], 6, (0.90, 0.95, 0.93), 0.85)
    wfill(lambda x, y: CE - y, (0, CE, E.AR, 1), lambda x, y: mix((0.035, 0.045, 0.05), (0.06, 0.075, 0.08), N2(x * 4, y * 14) * sstep((0.95 - y) / 0.2)), 70, (40, 70),
          ang=pi / 2, lmul=(2.5, 6), aj=0.02, jc=(0.004, 0.02))
    line([(0, CE + 0.002), (E.AR, CE + 0.002)], 7, (0.015, 0.02, 0.025))
    line([(0, MY1 + 0.002), (MX0, MY1 + 0.002)], 5, (0.20, 0.24, 0.24), 0.8); line([(MX1, MY1 + 0.002), (E.AR, MY1 + 0.002)], 5, (0.20, 0.24, 0.24), 0.8)
    # basin and tap
    bas = lambda x, y: sd_ell(x, y, 0.135, 0.748, 0.088, 0.021)
    wfill(bas, (0.04, 0.72, 0.23, 0.775), lambda x, y: mix((0.34, 0.39, 0.40), (0.86, 0.92, 0.90), sstep((y - 0.735) / 0.03)), 14, (12, 20), ang=0.0, lmul=(3, 7), aj=0.05,
          jc=(0.004, 0.02), margin=0.15)
    line(arc(0.135, 0.748, 0.088, 3.3, 6.1, 20) if False else [(0.135 + 0.088 * math.cos(a), 0.748 + 0.021 * math.sin(a)) for a in [3.14 + 0.16 * i for i in range(21)]], 5,
         (0.26, 0.30, 0.31), 0.9)
    dot(0.135, 0.752, 12, (0.10, 0.12, 0.13))
    tap = smooth([(0.135, 0.716), (0.135, 0.672), (0.142, 0.655), (0.158, 0.651), (0.172, 0.662), (0.174, 0.688)], 6)
    line(tap, 17, (0.42, 0.47, 0.49)); line([(p[0] - 0.0022, p[1]) for p in tap], 6, (0.93, 0.98, 0.97), 0.9)
    line([(0.118, 0.716), (0.152, 0.716)], 13, (0.48, 0.53, 0.55)); line([(0.100, 0.708), (0.117, 0.711)], 10, (0.55, 0.60, 0.62))
    line([(0.174, 0.700), (0.1742, 0.712)], 4, (0.86, 0.96, 0.98), 0.9)                                                          # the drip
    # the lanyard and badge, taken off
    ln_ = smooth([(0.262, 0.757), (0.250, 0.741), (0.228, 0.727), (0.240, 0.716), (0.262, 0.722), (0.268, 0.740), (0.266, 0.757)], 6)
    line(ln_, 6, (0.72, 0.13, 0.12))
    bdg = lambda x, y: sd_rbox(x, y, 0.262, 0.766, 0.021, 0.0125, -0.15, 0.002)
    wfill(bdg, (0.235, 0.748, 0.29, 0.785), lambda x, y: (0.95, 0.96, 0.93), 5, (8, 12), ang=-0.15, lmul=(2, 5), jc=(0.004, 0.015), margin=0.15)
    dot(0.2495, 0.7665, 13, (0.45, 0.42, 0.40)); glyphs('CEO', 0.2585, 0.7600, 0.0105, 2.4, (0.08, 0.08, 0.12), sp=0.85)
    layer('the man')
    set_lights([('p', (0.42, 0.03, -0.10), (2.6, 2.75, 2.7), 0.60), ('d', (-0.45, -0.55, 0.7), (0.20, 0.23, 0.25)), ('f', (0.2, 0.3, 0.9), (0.10, 0.12, 0.13))], (0.05, 0.06, 0.07))
    fb = Fr(0.55, 0.565, 0.172)
    nk = lambda x, y: sd_cap(x, y, 0, -0.5, 0, 0.0, 0.215)
    fb.fill(nk, (-0.25, -0.6, 0.25, 0.1), (SKIN[0] * 0.62, SKIN[1] * 0.54, SKIN[2] * 0.5), 6, (0.06, 0.1), rad=0.2, edge=False, texa=0.05)
    back_l = list(LIGHTS)
    set_lights([('p', (0.45, 0.10, 0.10), (2.0, 2.1, 2.05), 0.60), ('f', (0.2, 0.2, 0.9), (0.42, 0.46, 0.47))], (0.07, 0.08, 0.09))
    hsd = head_back(fb.sub(0, -0.62))
    set_lights(back_l, (0.05, 0.06, 0.07))
    body = jacket_back(fb)
    fb.rim(body, (-1.8, -0.3, 1.8, 1.4), 70, lambda x, y, s: mix((0.22, 0.25, 0.27), (0.66, 0.74, 0.72), s), w=(0.018, 0.04), light=(0.42, -0.4), thresh=0.55,
           inset=0.04, ln=(0.12, 0.4), opac=0.65)


def standing_profile(fb, skin=SKIN):
    """Side view, standing still, facing local +x, arms hanging. fb: origin at the base of the neck, unit = head."""
    parts = []
    hip = (-0.05, 3.0)
    for (knee, ank, toe, k) in (((-0.12, 4.55), (-0.30, 5.92), (0.16, 6.14), 0.75), ((0.10, 4.55), (0.02, 5.95), (0.52, 6.14), 1.0)):
        col = (TROU[0] * k, TROU[1] * k, TROU[2] * k)
        th = lambda x, y, knee=knee: sd_tcap(x, y, hip[0], hip[1], knee[0], knee[1], 0.36, 0.25)
        sh = lambda x, y, knee=knee, ank=ank: sd_tcap(x, y, knee[0], knee[1], ank[0], ank[1], 0.25, 0.15)
        fb.fill(sh, (-0.7, 4.2, 0.6, 6.2), col, 22, (0.07, 0.12), ang=pi / 2, rad=0.2)
        fb.fill(th, (-0.6, 2.6, 0.6, 4.9), col, 26, (0.08, 0.13), ang=pi / 2, rad=0.25)
        shoe = lambda x, y, ank=ank, toe=toe: sd_tcap(x, y, ank[0] - 0.05, ank[1] + 0.10, toe[0], toe[1], 0.17, 0.10)
        fb.fill(shoe, (-0.7, 5.7, 0.8, 6.35), (0.035, 0.035, 0.04), 8, (0.06, 0.09), ang=0.3, rad=0.1, spec=0.5)
        parts += [th, sh, shoe]
    neck = lambda x, y: sd_tcap(x, y, -0.02, 0.0, 0.22, -0.42, 0.22, 0.17)
    fb.fill(neck, (-0.3, -0.6, 0.5, 0.2), (skin[0] * 0.80, skin[1] * 0.72, skin[2] * 0.68), 8, (0.05, 0.08), rad=0.18, texa=0.05, jc=(0.006, 0.02))
    tee = lambda x, y: sd_poly(x, y, [(0.05, -0.15), (0.40, 0.0), (0.56, 0.7), (0.2, 0.8)])
    fb.fill(tee, (0.0, -0.2, 0.6, 0.8), TEE, 5, (0.06, 0.09), rad=0.2, edge=False)
    torso = lambda x, y: sd_poly(x, y, [(-0.36, -0.16), (0.10, -0.12), (0.38, 0.28), (0.50, 1.0), (0.46, 2.2), (0.42, 3.12), (-0.52, 3.18), (-0.66, 2.0), (-0.74, 0.9), (-0.62, 0.15)]) - 0.04
    leather(fb, torso, (-0.85, -0.3, 0.7, 3.3), 120, (0.07, 0.13), ang=pi / 2 + 0.05)
    collar = lambda x, y: sd_poly(x, y, [(-0.36, -0.22), (0.12, -0.20), (0.32, 0.12), (0.22, 0.34), (-0.05, 0.05), (-0.40, -0.02)])
    leather(fb, collar, (-0.45, -0.3, 0.4, 0.4), 12, (0.05, 0.08), ang=0.3, rad=0.08)
    crease(fb, [(-0.40, -0.02), (-0.05, 0.05), (0.22, 0.34)], 0.022, (0.32, 0.33, 0.38), 0.7)
    crease(fb, [(0.36, 0.25), (0.50, 1.0), (0.46, 2.2), (0.43, 3.05)], 0.02, (0.55, 0.56, 0.60), 0.65)
    for k in range(9):
        bx = U_(-0.55, 0.25); by = U_(0.5, 2.8)
        crease(fb, [(bx, by), (bx + 0.05, by + 0.14), (bx + 0.02, by + 0.3)], U_(0.02, 0.035), (0.20, 0.21, 0.26), U_(0.2, 0.45))
    hd = lambda x, y: sd_ell(x, y, 0.13, 3.20, 0.15, 0.20)
    fb.fill(hd, (-0.1, 2.95, 0.35, 3.45), skin, 8, (0.04, 0.07), ang=pi / 2, rad=0.12, texa=0.05, jc=(0.006, 0.02), lmul=(1.5, 3))
    ua = lambda x, y: sd_tcap(x, y, -0.12, 0.42, -0.06, 1.80, 0.30, 0.24)
    leather(fb, ua, (-0.5, 0.1, 0.4, 2.1), 46, (0.06, 0.11), ang=pi / 2)
    fa = lambda x, y: sd_tcap(x, y, -0.06, 1.80, 0.10, 2.98, 0.23, 0.17)
    leather(fb, fa, (-0.4, 1.5, 0.4, 3.2), 30, (0.06, 0.10), ang=pi / 2 - 0.12)
    for k in range(5):
        by = U_(1.4, 2.0); a0 = U_(-0.4, 0.4)
        crease(fb, [(-0.30, by + 0.06 * a0), (-0.06, by + 0.05), (0.16, by - 0.08 * a0)], U_(0.018, 0.03), opac=U_(0.15, 0.4))
    parts += [torso, ua, fa]
    return lambda x, y: min(p(x, y) for p in parts)


def pdist(P, x, y):
    return min(sd_cap(x, y, P[i][0], P[i][1], P[i + 1][0], P[i + 1][1], 0.0) for i in range(len(P) - 1))


def stage():
    """5. The last keynote: a very small man under a very large chart."""
    E.setup(3000, 1875)
    SX0, SX1, SY0, SY1 = 0.13, 1.47, 0.045, 0.575
    FL, FE = 0.635, 0.905                                    # back and front edges of the stage floor
    CX = 1.215                                               # where he stands
    rng = random.Random(11)
    up = [(0.165, 0.470)]
    while up[-1][0] < 0.90:
        x = up[-1][0] + rng.uniform(0.022, 0.04); t = (x - 0.165) / 0.75
        up.append((x, 0.470 - 0.355 * t ** 1.25 + rng.uniform(-0.028, 0.028) * (0.4 + t)))
    up.append((0.925, 0.098))
    down = [(0.925, 0.098), (0.945, 0.175), (0.965, 0.150), (0.99, 0.265), (1.01, 0.240), (1.04, 0.365), (1.058, 0.345), (1.085, 0.445), (1.118, 0.492)]
    curve = up + down[1:]
    GRN, RED = (0.45, 1.0, 0.72), (1.0, 0.20, 0.13)

    def screen(x, y, soft=1.0):
        c = mix((0.018, 0.028, 0.07), (0.035, 0.05, 0.11), sstep((y - SY0) / (SY1 - SY0)))
        du = pdist(up, x, y); dd = pdist(down, x, y)
        cy = pl(curve, x) if curve[0][0] < x < curve[-1][0] else None
        if cy is not None and y > cy:                         # area under the line
            a = 0.30 * math.exp(-(y - cy) / 0.16)
            c = mix(c, GRN if x < 0.925 else RED, a if x < 0.925 else 1.5 * a)
        c = mix(c, GRN, 0.35 * math.exp(-du / (0.022 * soft)))
        c = mix(c, RED, 0.65 * math.exp(-dd / (0.05 * soft)))
        return c
    layer('hall')
    wfill(ALL, (0, 0, E.AR, 1), lambda x, y: mix((0.012, 0.012, 0.022), (0.035, 0.02, 0.03), N2(x * 3, y * 3)), 60, (60, 100), ang=pi / 2, lmul=(2, 5), jc=(0.003, 0.015))
    for s, x0, x1 in ((-1, 0.0, SX0 - 0.012), (1, SX1 + 0.012, E.AR)):       # wings: heavy curtains
        def curt(x, y, s=s):
            fold = 0.5 + 0.5 * math.sin(x * 120 + 2.0 * N2(x * 6, y * 0.8))
            edge = math.exp(-(abs(x - (SX0 if s < 0 else SX1))) / 0.06)
            return mix(mix((0.02, 0.012, 0.02), (0.07, 0.022, 0.03), fold), (0.30, 0.07, 0.07) if s > 0 else (0.10, 0.20, 0.17), 0.45 * edge * fold)
        wfill(lambda x, y, x0=x0, x1=x1: sd_box(x, y, (x0 + x1) / 2, 0.33, (x1 - x0) / 2, 0.34), (x0, 0, x1, 0.67), curt, 40, (22, 40), ang=pi / 2, lmul=(4, 10), aj=0.015,
              jc=(0.004, 0.02), margin=0.1)
    layer('screen')
    ssd = lambda x, y: sd_box(x, y, (SX0 + SX1) / 2, (SY0 + SY1) / 2, (SX1 - SX0) / 2, (SY1 - SY0) / 2)
    wfill(ssd, (SX0, SY0, SX1, SY1), screen, 420, (22, 44), ang=0.0, lmul=(2.5, 6), aj=0.06, jc=(0.004, 0.02), margin=0.1, seglen=(2.0, 4.0))
    for k in range(1, 6):
        yy = SY0 + k * (SY1 - SY0) / 6
        line([(SX0, yy), (SX1, yy)], 3, (0.12, 0.17, 0.30), 0.5)
    for k in range(1, 10):
        xx = SX0 + k * (SX1 - SX0) / 10
        line([(xx, SY0), (xx, SY1)], 3, (0.12, 0.17, 0.30), 0.4)
        line([(xx, SY1 - 0.012), (xx, SY1)], 5, (0.35, 0.42, 0.60), 0.8)
    layer('the line')
    for k in range(int(40 * E.DENS)):                        # glow
        i = rng.randrange(len(down) - 1); a, b = down[i], down[i + 1]
        add(seg(a[0], a[1], b[0], b[1], 3), U_(40, 90), RED, 0.06, 0.4, 0.5, 0.5)
    for i in range(len(up) - 1):
        line([up[i], up[i + 1]], 13, mix(GRN, (1, 1, 1), 0.25), 1.0, 0.0, 0.95, 0.95)
    for i in range(len(down) - 1):
        line([down[i], down[i + 1]], 17, mix(RED, (1.0, 0.75, 0.6), 0.25), 1.0, 0.0, 0.95, 0.95)
    for p in up[::3]:
        dot(p[0], p[1], 12, (0.92, 1.0, 0.95))
    tip = down[-1]; dx, dy = tip[0] - down[-2][0], tip[1] - down[-2][1]; dn = math.hypot(dx, dy); dx /= dn; dy /= dn
    ah = [(tip[0] + dx * 0.030, tip[1] + dy * 0.030), (tip[0] - dx * 0.022 - dy * 0.026, tip[1] - dy * 0.022 + dx * 0.026),
          (tip[0] - dx * 0.022 + dy * 0.026, tip[1] - dy * 0.022 - dx * 0.026)]
    wfill(lambda x, y: sd_poly(x, y, ah), (tip[0] - 0.05, tip[1] - 0.05, tip[0] + 0.05, tip[1] + 0.05), lambda x, y: mix(RED, (1.0, 0.75, 0.6), 0.25), 8, (9, 14),
          ang=math.atan2(dy, dx), lmul=(2, 4), margin=0.15, jc=(0.004, 0.02))
    dot(up[-1][0], up[-1][1], 26, (1, 1, 1)); line([(0.925, 0.098), (0.925, SY0 + 0.012)], 3, (0.8, 0.9, 0.85), 0.5)
    # a ticker of losses along the top
    xx = SX0 + 0.03
    while xx < SX1 - 0.05:
        tri = [(xx, SY0 + 0.016), (xx + 0.016, SY0 + 0.016), (xx + 0.008, SY0 + 0.030)]
        wfill(lambda x, y, tri=tri: sd_poly(x, y, tri), (xx - 0.002, SY0 + 0.012, xx + 0.018, SY0 + 0.034), lambda x, y: RED, 2, (6, 9), ang=0.0, lmul=(1.5, 3), margin=0.1,
              jc=(0.004, 0.02))
        L = rng.uniform(0.03, 0.07)
        line([(xx + 0.024, SY0 + 0.023), (xx + 0.024 + L, SY0 + 0.023)], 8, (0.45, 0.50, 0.62), 0.7)
        xx += L + 0.055
    line([(SX0, SY0), (SX1, SY0), (SX1, SY1), (SX0, SY1), (SX0, SY0 - 0.002)], 9, (0.01, 0.01, 0.015))
    layer('stage')
    wfill(lambda x, y: sd_box(x, y, 0.8, (SY1 + FL) / 2 + 0.004, 0.9, (FL - SY1) / 2), (0, SY1 + 0.004, E.AR, FL), lambda x, y: (0.014, 0.014, 0.022), 16, (26, 40), ang=0.0,
          lmul=(5, 12), jc=(0.003, 0.012), margin=0.1)
    pool = lambda x, y: spot(x, y, CX - 0.01, 0.862, 0.17, 0.034)

    def floor(x, y):
        t = (y - FL) / (FE - FL)
        c = mix((0.03, 0.03, 0.042), (0.055, 0.05, 0.06), N2(x * 4, y * 30))
        m = screen(min(max(x, SX0 + 0.01), SX1 - 0.01), SY1 - 0.01 - t * 0.50, soft=3.0)                       # the screen, mirrored in the black floor
        c = (c[0] + 0.55 * m[0] * (1 - 0.5 * t), c[1] + 0.55 * m[1] * (1 - 0.5 * t), c[2] + 0.55 * m[2] * (1 - 0.5 * t))
        return mix(c, (0.90, 0.86, 0.76), 0.85 * pool(x, y))
    wfill(lambda x, y: sd_box(x, y, 0.8, (FL + FE) / 2, 0.9, (FE - FL) / 2), (0, FL, E.AR, FE), floor, 300, (20, 44), ang=0.0, lmul=(4, 10), aj=0.02, jc=(0.004, 0.025), margin=0.1)
    for k in range(int(60 * E.DENS)):                        # vertical streaks of reflected red
        x0 = U_(0.93, 1.14); y0 = U_(FL, FL + 0.12)
        add(seg(x0, y0, x0 + U_(-0.004, 0.004), y0 + U_(0.03, 0.10), 3), U_(10, 26), RED, U_(0.05, 0.14), 0.4, 0.4, 0.2)
    line([(0, FL), (E.AR, FL)], 5, (0.10, 0.09, 0.11), 0.8)
    wfill(lambda x, y: sd_box(x, y, 0.8, FE + 0.012, 0.9, 0.012), (0, FE, E.AR, FE + 0.024), lambda x, y: mix((0.05, 0.045, 0.05), (0.26, 0.22, 0.20), pool(x, y - 0.05)), 14,
          (20, 28), ang=0.0, lmul=(5, 12), aj=0.01, margin=0.1)
    line([(0, FE), (E.AR, FE)], 5, (0.30, 0.26, 0.25), 0.8)
    layer('the microphone he left')
    line([(0.80, 0.800), (0.80, 0.612)], 6, (0.30, 0.31, 0.34)); line([(0.7985, 0.800), (0.7985, 0.615)], 2.5, (0.75, 0.78, 0.82), 0.8)
    line([(0.80, 0.612), (0.812, 0.598)], 7, (0.30, 0.31, 0.34)); line([(0.810, 0.600), (0.822, 0.586)], 15, (0.08, 0.08, 0.10)); dot(0.8225, 0.5855, 17, (0.42, 0.44, 0.48))
    wfill(lambda x, y: sd_ell(x, y, 0.80, 0.803, 0.035, 0.008), (0.76, 0.79, 0.84, 0.815), lambda x, y: (0.11, 0.11, 0.13), 4, (10, 14), ang=0.0, lmul=(3, 6), margin=0.15)
    wfill(lambda x, y: sd_ell(x, y, 0.775, 0.812, 0.07, 0.007), (0.70, 0.80, 0.85, 0.822), lambda x, y: (0.012, 0.012, 0.02), 4, (9, 13), ang=0.0, lmul=(3, 6), margin=0.15, opac=0.7)
    layer('spotlight')
    for k in range(int(50 * E.DENS)):
        t = U_(-1, 1); y0 = U_(-0.02, 0.5)
        hw = lambda y: 0.012 + 0.17 * (y + 0.05)
        add(seg(CX - 0.01 + t * hw(y0), y0, CX - 0.01 + t * hw(y0 + 0.33), y0 + 0.33, 6), U_(40, 90), (1.0, 0.96, 0.84), 0.035, 0.5, 0.4, 0.2)
    layer('his shadow')
    fb = Fr(CX, 0.545, 0.0515)
    fy = fb.l2w(0, 6.2)[1]
    shd = lambda x, y: sd_ell(x, y, CX - 0.03, fy + 0.008, 0.075, 0.014)
    wfill(shd, (CX - 0.12, fy - 0.01, CX + 0.06, fy + 0.03), lambda x, y: (0.05, 0.045, 0.05), 12, (10, 16), ang=0.0, lmul=(3, 7), aj=0.03, jc=(0.004, 0.02))
    layer('the man')
    set_lights([('p', (CX + 0.02, 0.18, 0.22), (2.6, 2.5, 2.3), 0.55), ('f', (-0.8, -0.25, 0.35), (0.85, 0.16, 0.12)), ('f', (0.3, 0.3, 0.9), (0.14, 0.14, 0.17))], (0.05, 0.05, 0.065))
    body = standing_profile(fb)
    L0 = list(LIGHTS); L0[1] = ('f', (-0.8, -0.25, 0.35), (0.40, 0.10, 0.08)); set_lights(L0, (0.05, 0.05, 0.065))
    head_profile(fb.sub(0.52, -0.52, 0.58), toplight=0.5)
    fb.rim(body, (-1.0, -0.3, 0.8, 6.3), 60, lambda x, y, s: mix((0.45, 0.10, 0.08), (1.0, 0.30, 0.20), s), w=(0.025, 0.05), light=(CX - 2.0, 0.55), thresh=0.55,
           inset=0.04, ln=(0.12, 0.4), opac=0.8)
    fb.rim(body, (-1.0, -0.3, 0.8, 3.2), 30, lambda x, y, s: mix((0.40, 0.38, 0.36), (1.0, 0.96, 0.86), s), w=(0.025, 0.05), light=(CX + 0.05, -0.5), thresh=0.6,
           inset=0.04, ln=(0.10, 0.3), opac=0.8)
    layer('audience')
    AUD = (0.012, 0.012, 0.02)
    crowd = []
    for row, (top, hw, n, off) in enumerate(((0.925, 0.040, 15, 0.03), (0.955, 0.052, 12, 0.09))):
        for i in range(n):
            cx = off + i * (E.AR / n) + rng.uniform(-0.012, 0.012); tp = top + rng.uniform(-0.008, 0.008); hr = hw * 0.50
            if row == 0 and i in (5, 6):                     # two seats already empty
                continue
            sd_ = lambda x, y, cx=cx, tp=tp, hr=hr, hw=hw: min(sd_ell(x, y, cx, tp + hr * 1.2, hr, hr * 1.25), sd_box(x, y, cx, tp + hr * 2.4 + 0.2, hw, 0.2, hw * 0.5))
            wfill(sd_, (cx - hw - 0.02, tp - 0.01, cx + hw + 0.02, 1.02), lambda x, y: AUD, 6, (20, 34), ang=pi / 2, lmul=(2, 4), jc=(0.002, 0.01))
            crowd.append(sd_)
    allc = lambda x, y: min(f(x, y) for f in crowd)
    E.rim(allc, (0, 0.90, E.AR, 0.995), 110, lambda x, y, s: mix((0.22, 0.06, 0.06), (0.85, 0.24, 0.18), s), w=(3, 6), light=(1.05, 0.45), thresh=0.4, inset=0.003,
          ln=(0.015, 0.05), opac=0.7)
    for (px_, py_, rot) in ((0.565, 0.905, 0.06), (1.01, 0.935, -0.08), (0.27, 0.925, 0.10)):      # phones held up
        arm = lambda x, y, px_=px_, py_=py_: sd_cap(x, y, px_, py_ + 0.02, px_ + 0.012, 1.0, 0.012)
        wfill(arm, (px_ - 0.03, py_, px_ + 0.04, 1.0), lambda x, y: AUD, 3, (16, 22), ang=pi / 2, lmul=(2, 4))
        ph = lambda x, y, px_=px_, py_=py_, rot=rot: sd_rbox(x, y, px_, py_, 0.013, 0.023, rot, 0.003)
        wfill(ph, (px_ - 0.03, py_ - 0.03, px_ + 0.03, py_ + 0.03), lambda x, y: (0.02, 0.02, 0.03), 3, (12, 18), ang=pi / 2 + rot, lmul=(1.5, 3))
        scr = lambda x, y, px_=px_, py_=py_, rot=rot: sd_rbox(x, y, px_, py_, 0.0105, 0.020, rot, 0.002)
        wfill(scr, (px_ - 0.03, py_ - 0.03, px_ + 0.03, py_ + 0.03), lambda x, y: (0.10, 0.13, 0.26), 5, (6, 9), ang=pi / 2 + rot, lmul=(1.5, 3), jc=(0.004, 0.02))
        line([(px_ - 0.006, py_ - 0.008), (px_ + 0.001, py_ - 0.012), (px_ + 0.006, py_ + 0.004)], 2.5, RED)
        dot(px_ + 0.004, py_ + 0.010, 5, (0.95, 0.9, 0.8))


SCENES = {'sheet': sheet, 'podium': podium, 'boardroom': boardroom, 'walkout': walkout, 'mirror': mirror, 'stage': stage}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('scene'); ap.add_argument('--out', default=None); ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--dens', type=float, default=1.0)
    ap.add_argument('--nolabel', action='store_true')
    a = ap.parse_args()
    _setup = E.setup
    E.setup = lambda w, h, seed=20261112, dens=1.0: _setup(w, h, seed, dens * a.dens)
    SCENES[a.scene]()
    E.finish(a.out or ('ceo_' + a.scene), a.frames, a.scale, quiet=True, label=not a.nolabel)
