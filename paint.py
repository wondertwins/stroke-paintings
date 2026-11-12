"""Stroke-based painter: builds an image from a blank canvas using N individual brush strokes.

Each stroke is a curved, tapered, bristle-textured brush mark of a single paint colour. It is placed
where the canvas is currently most wrong, follows the local flow of the subject, and is only kept if
it brings the canvas closer to the reference. Strokes go from house-painter wide to hairline fine.
"""
import argparse, math, os, time
import numpy as np, cv2
from numba import njit
from PIL import Image, ImageOps, ImageDraw, ImageFont

LOGW = 10


@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


@njit(cache=True, fastmath=True)
def paint_chunk(tgt, can, hgt, A, U, T, theta, coh, ds, i0, i1, N, wmin, wmax, log, stats):
    H, W, _ = tgt.shape
    L = theta.shape[0]
    Hd, Wd = theta.shape[1], theta.shape[2]
    MAXP = 160
    px = np.empty(MAXP); py = np.empty(MAXP)
    fx = np.empty(MAXP); fy = np.empty(MAXP)
    bx_ = np.empty(MAXP); by_ = np.empty(MAXP)
    cum = np.empty(MAXP)
    bris = np.empty(512)
    ref = np.empty(3); tc = np.empty(3); col = np.empty(3)
    K = 8

    for i in range(i0, i1):
        wbase = wmin * math.sqrt(N / (i + 1.0))
        if wbase > wmax:
            wbase = wmax
        fails = 0
        shrink = 1.0
        while True:
            stats[0] += 1
            w = wbase * math.exp(0.28 * np.random.randn()) * shrink
            if w < 1.4:
                w = 1.4
            if w > wmax * 1.25:
                w = wmax * 1.25
            r = 0.5 * w

            # ---- where: tournament on current error
            best = -1.0
            x0 = 0.0; y0 = 0.0
            pr = max(1.0, 0.5 * r)
            for k in range(K):
                cx = np.random.randint(0, W); cy = np.random.randint(0, H)
                e = 0.0
                for s in range(5):
                    sx = cx; sy = cy
                    if s == 1: sx = cx + int(pr)
                    elif s == 2: sx = cx - int(pr)
                    elif s == 3: sy = cy + int(pr)
                    elif s == 4: sy = cy - int(pr)
                    if sx < 0: sx = 0
                    if sx >= W: sx = W - 1
                    if sy < 0: sy = 0
                    if sy >= H: sy = H - 1
                    for c in range(3):
                        d = can[sy, sx, c] - tgt[sy, sx, c]
                        e += d * d
                if e > best:
                    best = e; x0 = cx + 0.5; y0 = cy + 0.5

            # reference colour at the start point
            for c in range(3):
                ref[c] = 0.0
            for s in range(5):
                sx = int(x0); sy = int(y0)
                if s == 1: sx += int(pr)
                elif s == 2: sx -= int(pr)
                elif s == 3: sy += int(pr)
                elif s == 4: sy -= int(pr)
                if sx < 0: sx = 0
                if sx >= W: sx = W - 1
                if sy < 0: sy = 0
                if sy >= H: sy = H - 1
                for c in range(3):
                    ref[c] += tgt[sy, sx, c] * 0.2

            # ---- direction field level for this brush size
            if w < 5.0: l0 = 0
            elif w < 10.0: l0 = 1
            elif w < 20.0: l0 = 2
            elif w < 40.0: l0 = 3
            else: l0 = 4
            if l0 > L - 2: l0 = L - 2

            Lmax = w * (3.5 + 6.0 * np.random.rand())
            step = r
            if step < 2.0: step = 2.0
            maxsteps = int(0.5 * Lmax / step)
            if maxsteps > 70: maxsteps = 70
            tol2 = 0.016 + 0.02 * np.random.rand()
            jit = 0.14 * np.random.randn()

            # initial direction
            ix = int(x0 / ds); iy = int(y0 / ds)
            if ix >= Wd: ix = Wd - 1
            if iy >= Hd: iy = Hd - 1
            ang = 0.0; found = False
            for l in range(l0, L):
                if coh[l, iy, ix] > 0.12:
                    ang = theta[l, iy, ix]; found = True
                    if l == L - 1:
                        jit += (np.random.rand() - 0.5) * 0.7
                    break
            ang += jit
            d0x = math.cos(ang); d0y = math.sin(ang)

            nf = 0; nb = 0
            for direction in range(2):
                sgn = 1.0 if direction == 0 else -1.0
                pdx = d0x * sgn; pdy = d0y * sgn
                qx = x0; qy = y0
                cnt = 0
                for s in range(maxsteps):
                    ix = int(qx / ds); iy = int(qy / ds)
                    if ix >= Wd: ix = Wd - 1
                    if iy >= Hd: iy = Hd - 1
                    found = False
                    for l in range(l0, L):
                        if coh[l, iy, ix] > 0.12:
                            ang = theta[l, iy, ix] + jit; found = True
                            break
                    if found:
                        dx = math.cos(ang); dy = math.sin(ang)
                        if dx * pdx + dy * pdy < 0.0:
                            dx = -dx; dy = -dy
                        dx = 0.55 * dx + 0.45 * pdx; dy = 0.55 * dy + 0.45 * pdy
                        nrm = math.sqrt(dx * dx + dy * dy)
                        if nrm < 1e-6:
                            dx = pdx; dy = pdy
                        else:
                            dx /= nrm; dy /= nrm
                    else:
                        dx = pdx; dy = pdy
                    nx = qx + dx * step; ny = qy + dy * step
                    if nx < 0 or ny < 0 or nx >= W or ny >= H:
                        break
                    sx = int(nx); sy = int(ny)
                    ox = int(-dy * pr); oy = int(dx * pr)
                    d2 = 0.0
                    for c in range(3):
                        v = tgt[sy, sx, c] * 0.5
                        ax_ = min(max(sx + ox, 0), W - 1); ay_ = min(max(sy + oy, 0), H - 1)
                        bx2 = min(max(sx - ox, 0), W - 1); by2 = min(max(sy - oy, 0), H - 1)
                        v += 0.25 * tgt[ay_, ax_, c] + 0.25 * tgt[by2, bx2, c]
                        dd = v - ref[c]
                        d2 += dd * dd
                    if d2 > tol2:
                        break
                    qx = nx; qy = ny; pdx = dx; pdy = dy
                    if direction == 0:
                        fx[cnt] = qx; fy[cnt] = qy
                    else:
                        bx_[cnt] = qx; by_[cnt] = qy
                    cnt += 1
                if direction == 0: nf = cnt
                else: nb = cnt

            M = 0
            if nf + nb == 0:
                hl = 0.7 * w
                px[0] = x0 - d0x * hl; py[0] = y0 - d0y * hl
                px[1] = x0 + d0x * hl; py[1] = y0 + d0y * hl
                M = 2
            else:
                for k in range(nb - 1, -1, -1):
                    px[M] = bx_[k]; py[M] = by_[k]; M += 1
                px[M] = x0; py[M] = y0; M += 1
                for k in range(nf):
                    px[M] = fx[k]; py[M] = fy[k]; M += 1
            if np.random.rand() < 0.5:  # random brushing direction
                for k in range(M // 2):
                    tx = px[k]; px[k] = px[M - 1 - k]; px[M - 1 - k] = tx
                    ty = py[k]; py[k] = py[M - 1 - k]; py[M - 1 - k] = ty
            cum[0] = 0.0
            for k in range(1, M):
                cum[k] = cum[k - 1] + math.sqrt((px[k] - px[k - 1]) ** 2 + (py[k] - py[k - 1]) ** 2)
            Ltot = cum[M - 1]
            if Ltot < 1e-3:
                continue

            # ---- rasterise the brush footprint (union of tapered capsules)
            gx0 = W; gy0 = H; gx1 = -1; gy1 = -1
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
                if bx1 > W - 1: bx1 = W - 1
                if by1 > H - 1: by1 = H - 1
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
                        reff = r * (0.62 + 0.38 * s1) * (0.42 + 0.58 * s2)
                        a = reff + 0.5 - d
                        if a <= 0.0:
                            continue
                        if a > 1.0: a = 1.0
                        u = d / reff
                        if u > 1.0: u = 1.0
                        if vx * fyy - vy * fxx < 0.0: u = -u
                        cur = A[y, x]
                        if a > cur + 1e-6 or (a >= cur - 1e-6 and abs(u) < abs(U[y, x])):
                            A[y, x] = a; U[y, x] = u; T[y, x] = arc
            if gx1 < 0:
                continue

            # ---- load the brush: bristle profile, opacity, dryness
            nbr = int(w * 0.55) + 4
            if nbr > 500: nbr = 500
            prev = np.random.rand()
            for k in range(nbr):
                v = np.random.rand()
                prev = 0.45 * prev + 0.55 * v
                bris[k] = prev
            mn = 1.0; mx = 0.0
            for k in range(nbr):
                if bris[k] < mn: mn = bris[k]
                if bris[k] > mx: mx = bris[k]
            for k in range(nbr):
                bris[k] = (bris[k] - mn) / (mx - mn + 1e-6)
            opac = 0.82 + 0.18 * np.random.rand()
            dry = np.random.rand() ** 1.5
            sa = 0.0
            col[0] = 0.0; col[1] = 0.0; col[2] = 0.0
            bmean = 0.0
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
                        g = (bv - thr) / 0.18 + 1.0
                        if g < 0.0: g = 0.0
                        if g < 1.0: a *= g
                    a *= opac * (0.9 + 0.1 * bv)
                    A[y, x] = a; U[y, x] = bv
                    sa += a; bmean += a * bv
                    for c in range(3):
                        col[c] += a * tgt[y, x, c]
            if sa < 0.75:
                for y in range(gy0, gy1 + 1):
                    for x in range(gx0, gx1 + 1):
                        A[y, x] = 0.0; U[y, x] = 0.0
                continue
            bmean /= sa
            cj = 0.006
            for c in range(3):
                col[c] = col[c] / sa + cj * np.random.randn()
                if col[c] < 0.0: col[c] = 0.0
                if col[c] > 1.0: col[c] = 1.0
            cmod = 0.07

            # ---- does this stroke improve the painting?
            delta = 0.0
            for y in range(gy0, gy1 + 1):
                for x in range(gx0, gx1 + 1):
                    a = A[y, x]
                    if a <= 0.0:
                        continue
                    m = 1.0 + cmod * (U[y, x] - bmean)
                    for c in range(3):
                        o = can[y, x, c]; tg = tgt[y, x, c]
                        nv = o + a * (col[c] * m - o)
                        delta += (nv - tg) * (nv - tg) - (o - tg) * (o - tg)

            forced = fails >= 4000
            if delta < 0.0 or forced:
                if forced: stats[1] += 1
                tk = math.sqrt(w / 10.0)
                if tk > 1.6: tk = 1.6
                if tk < 0.45: tk = 0.45
                for y in range(gy0, gy1 + 1):
                    for x in range(gx0, gx1 + 1):
                        a = A[y, x]
                        if a > 0.0:
                            bv = U[y, x]
                            m = 1.0 + cmod * (bv - bmean)
                            for c in range(3):
                                nv = can[y, x, c] + a * (col[c] * m - can[y, x, c])
                                if nv < 0.0: nv = 0.0
                                if nv > 1.0: nv = 1.0
                                can[y, x, c] = nv
                            hgt[y, x] += a * (tk * (0.35 + 0.65 * bv) - hgt[y, x])
                            A[y, x] = 0.0; U[y, x] = 0.0
                log[i, 0] = x0; log[i, 1] = y0; log[i, 2] = w; log[i, 3] = Ltot; log[i, 4] = M
                log[i, 5] = col[0]; log[i, 6] = col[1]; log[i, 7] = col[2]; log[i, 8] = sa; log[i, 9] = fails
                stats[2] += sa
                break
            else:
                for y in range(gy0, gy1 + 1):
                    for x in range(gx0, gx1 + 1):
                        A[y, x] = 0.0; U[y, x] = 0.0
                fails += 1
                if fails % 20 == 0:
                    shrink *= 0.88


def flow_fields(tgt, ds):
    """Edge-tangent direction + coherence at several scales (structure tensor)."""
    g = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = gx * gx, gy * gy, gx * gy
    H, W = g.shape
    out_t, out_c = [], []
    for s in [2.0, 4.0, 8.0, 16.0, 40.0]:
        a = cv2.GaussianBlur(jxx, (0, 0), s); b = cv2.GaussianBlur(jyy, (0, 0), s); c = cv2.GaussianBlur(jxy, (0, 0), s)
        a = cv2.resize(a, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        c = cv2.resize(c, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        grad_ang = 0.5 * np.arctan2(2 * c, a - b)          # dominant gradient direction
        tang = grad_ang + np.pi / 2                        # strokes run along edges
        lam = np.sqrt((a - b) ** 2 + 4 * c * c)
        cohr = lam / (a + b + 1e-5)
        cohr[(a + b) < 2e-4] = 0.0                         # flat areas: no reliable direction
        out_t.append(tang.astype(np.float32)); out_c.append(cohr.astype(np.float32))
    # global, always-defined field for flat regions (sky): broad sweep of the composition
    f = max(1, W // 96)
    sm = cv2.resize(cv2.GaussianBlur(g, (0, 0), f), (W // f, H // f), interpolation=cv2.INTER_AREA)
    sx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3); sy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    a = cv2.GaussianBlur(sx * sx, (0, 0), 5, borderType=cv2.BORDER_REFLECT); b = cv2.GaussianBlur(sy * sy, (0, 0), 5, borderType=cv2.BORDER_REFLECT)
    c = cv2.GaussianBlur(sx * sy, (0, 0), 5, borderType=cv2.BORDER_REFLECT)
    a, b, c = [cv2.resize(v, (W // ds, H // ds), interpolation=cv2.INTER_CUBIC) for v in (a, b, c)]
    out_t.append((0.5 * np.arctan2(2 * c, a - b) + np.pi / 2).astype(np.float32))
    out_c.append(np.ones_like(out_c[0]))
    return np.stack(out_t), np.stack(out_c)


def relief(can, hgt, strength=0.3):
    """Light the paint surface from the upper-left so individual strokes read as physical paint."""
    h = cv2.GaussianBlur(hgt, (0, 0), 0.8)
    hx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    hy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    sh = -(hx * 0.7 + hy * 0.7) * strength
    sh = np.clip(sh, -0.12, 0.12)
    out = can * (1.0 + sh[..., None]) + np.maximum(sh, 0)[..., None] * 0.15
    return np.clip(out, 0, 1)


def to_u8(a):
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('--out', default='out')
    ap.add_argument('--width', type=int, default=3000)
    ap.add_argument('--strokes', type=int, default=700000)
    ap.add_argument('--wmin', type=float, default=2.2)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--fexp', type=float, default=3.2)
    a = ap.parse_args()
    os.makedirs(a.out + '/frames', exist_ok=True)

    im = ImageOps.exif_transpose(Image.open(a.src)).convert('RGB')
    W = a.width; H = round(im.height * W / im.width)
    im = im.resize((W, H), Image.LANCZOS)
    tgt = np.asarray(im, dtype=np.float32) / 255.0
    N = a.strokes
    ds = 2
    theta, coh = flow_fields(tgt, ds)
    can = np.empty_like(tgt); can[:] = np.array([0.93, 0.905, 0.85], np.float32)   # primed blank canvas
    hgt = np.zeros((H, W), np.float32)
    A = np.zeros((H, W), np.float32); U = np.zeros((H, W), np.float32); T = np.zeros((H, W), np.float32)
    log = np.zeros((N, LOGW), np.float32)
    stats = np.zeros(4, np.float64)
    seed_rng(a.seed)
    wmax = 0.06 * W

    # frame schedule: slow at the start (few big strokes), accelerating
    marks = sorted(set([0] + [int(round(N * (k / a.frames) ** a.fexp)) for k in range(1, a.frames + 1)]))
    fw = 1080; fh = round(H * fw / W); fh += fh % 2
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
    except Exception:
        font = ImageFont.load_default()

    def frame(idx, n):
        small = cv2.resize(to_u8(can), (fw, fh), interpolation=cv2.INTER_AREA)
        p = Image.fromarray(small); d = ImageDraw.Draw(p)
        txt = f'{n:,} strokes'
        d.text((30, fh - 64), txt, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        p.save(f'{a.out}/frames/f{idx:04d}.png', compress_level=1)

    t0 = time.time(); frame(0, 0)
    for k in range(1, len(marks)):
        paint_chunk(tgt, can, hgt, A, U, T, theta, coh, ds, marks[k - 1], marks[k], N, a.wmin, wmax, log, stats)
        frame(k, marks[k])
        if k % 20 == 0 or k == len(marks) - 1:
            mse = float(((can - tgt) ** 2).mean())
            print(f'{marks[k]:>8,} strokes  attempts={int(stats[0]):,} forced={int(stats[1])} '
                  f'psnr={10 * math.log10(1 / mse):.2f}dB  t={time.time() - t0:.0f}s', flush=True)
    assert A.max() == 0
    np.save(a.out + '/strokes.npy', log)
    np.save(a.out + '/height.npy', hgt)
    Image.fromarray(to_u8(can)).save(a.out + '/painting_flat.png')
    fin = to_u8(relief(can, hgt))
    Image.fromarray(fin).save(a.out + '/painting.png')
    Image.fromarray(fin).save(a.out + '/painting.jpg', quality=95, subsampling=0)
    im.save(a.out + '/reference.png')
    print('strokes painted:', int((log[:, 2] > 0).sum()), 'attempts:', int(stats[0]), 'forced:', int(stats[1]),
          'paint coverage (x canvas):', round(stats[2] / (H * W), 2), 'size', W, H)
    print('width percentiles', np.percentile(log[:, 2], [0, 10, 50, 90, 100]).round(1))


if __name__ == '__main__':
    main()
