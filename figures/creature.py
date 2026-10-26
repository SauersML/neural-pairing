"""Quill and Ember drawn as flat vector creatures, matching the film's characters. The body is the architecture: a spine
of 36 rounded pucks (the layers), a pale core up the middle (the residual stream) and teal rings at the 8 link layers.
Quill has a crest of three feather locks, Ember a flame of flowing locks, both drawn like hair that leans toward the
partner when linked and falls when the link is cut. Big eyes, no mouth, no limbs.

creature(...) returns SVG markup and, for the full body, the centre of each link layer, where link strands plug into
the core. Running this file redraws the icons figures/assets/quill.png, ember.png, quill_cut.png and ember_cut.png
(chat_names.mascot('quill_cut') draws the link-cut pose) and a sheet of
every variant in figures/assets/creature_sheet.png."""
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINK = '#1A9E77'
PUPIL = '#0A0612'
PALETTE = dict(
    quill=dict(main='#7B3FE4', deep='#4C1D95', light='#C4B5FD', crest='#A78BFA'),
    ember=dict(main='#E8590C', deep='#9A3412', light='#FDBA74', crest='#FB923C', core='#FFE7A3'),
    grey=dict(main='#9CA3AF', deep='#4B5563', light='#E5E7EB', crest='#C4C8CF', core='#F3F4F6'),
)
LAYERS = 36
PORTS = (6, 9, 12, 15, 18, 21, 24, 27)

# design units: head centre x 0, head radii, eye layout and puck spacing
HEAD_RX, HEAD_RY = 128, 110
EYE_DX, EYE_R, PUPIL_R = 60, 43, 25
Y0 = 70  # room for the crest's tip and the sparks above the head
BODY_W = 0.84 * 2 * HEAD_RX  # widest vertebra, body radius 0.52 against head radius 0.62


def _taper(t):
    """Body width as a fraction of the widest vertebra, t = 0 at the base and 1 at the neck: narrow at the base (0.58),
    widest around the middle, a little narrower at the neck."""
    if t < 0.5:
        return 1 - 0.42 * ((0.5 - t) / 0.5) ** 2
    return 1 - 0.15 * ((t - 0.5) / 0.5) ** 2


def _lock(x0, y0, length, width, theta, curl, wiggle=0.0, n=48):
    """A tapered lock, like a strand of hair, from its root (x0, y0). theta: the root's angle in degrees from straight up
    (positive toward +x); curl: how far the direction turns by the tip; wiggle: an S-bend along the way. Returns the
    outline's path data and the spine's points."""
    spine, x, y = [(x0, y0)], x0, y0
    ds = length / n
    for i in range(1, n + 1):
        t = i / n
        a = math.radians(theta + curl * t ** 1.4 + wiggle * math.sin(2 * math.pi * t))
        x, y = x + ds * math.sin(a), y - ds * math.cos(a)
        spine.append((x, y))
    left, right = [], []
    for i, (px, py) in enumerate(spine):
        t = i / n
        a = math.radians(theta + curl * t ** 1.4 + wiggle * math.sin(2 * math.pi * t))
        w = width * (1 - t) ** 0.85 * min(1.0, 0.55 + 2.2 * t) / 2
        nx, ny = math.cos(a), math.sin(a)
        left.append((px - nx * w, py - ny * w))
        right.append((px + nx * w, py + ny * w))
    pts = left + right[::-1]
    d = 'M' + ' L'.join(f'{u:.1f},{v:.1f}' for u, v in pts) + 'Z'
    return d, spine


def _pose(dx, th, cu, ln, sway, lift):
    """A lock's pose. The poses are written for hair flowing toward +x; sway (-1 to 1) sets the side and how far the
    locks flow, and lift 1 stands them up (linked) while lower values let them fall to the sides (link cut)."""
    sgn = -1 if sway < 0 else 1
    dx, th, cu = dx * sgn, th * sgn, cu * sgn
    cu *= 0.25 + 0.75 * abs(sway)
    th += 18 * sway
    side = 1 if dx > 0 else -1
    th += (1 - lift) * side * 75
    cu += (1 - lift) * side * 55
    return dx, th, cu, ln * (0.72 + 0.28 * lift)


def _crest(name, top, c, sway, lift):
    """The hair behind the crown (top: the head's top edge): Quill's three feather locks, or Ember's flame locks."""
    out = []
    root = top + 40
    if name == 'quill':
        # (root x, length, width, angle, curl): broad feather locks, all flowing the same way
        for dx, ln, wd, th, cu in ((-54, 136, 66, -30, 115), (34, 146, 64, 4, 125), (-12, 200, 82, -14, 140)):
            dx, th, cu, ln = _pose(dx, th, cu, ln, sway, lift)
            d, spine = _lock(dx, root, ln, wd, th, cu, wiggle=5)
            fill = c['crest'] if wd > 70 else c['main']
            out.append(f'<path d="{d}" fill="{fill}" stroke="{c["deep"]}" stroke-width="6" stroke-linejoin="round"/>')
            k = len(spine) - 1
            rachis = ' L'.join(f'{u:.1f},{v:.1f}' for u, v in spine[int(0.25 * k):int(0.9 * k)])
            out.append(f'<path d="M{rachis}" fill="none" stroke="{c["deep"]}" stroke-width="4" stroke-linecap="round"/>')
            for t in (0.45, 0.6, 0.74):  # vane notches along the plume's flow
                i = int(t * k)
                (x1, y1), (x2, y2) = spine[i], spine[min(k, i + 4)]
                ux, uy = x2 - x1, y2 - y1
                nrm = math.hypot(ux, uy) or 1
                for sg in (-1, 1):
                    ex = x1 + ux / nrm * wd * 0.2 - sg * uy / nrm * wd * 0.28
                    ey = y1 + uy / nrm * wd * 0.2 + sg * ux / nrm * wd * 0.28
                    out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{c["light"]}" '
                               f'stroke-width="4" stroke-linecap="round"/>')
        return out
    core = c.get('core', '#FFE7A3')
    # flame locks (root x, length, width, angle, curl, wiggle), back ones first
    poses = []
    for dx, ln, wd, th, cu, wg in ((-58, 112, 72, -36, 55, 10), (52, 128, 66, 22, 58, 10),
                                   (-22, 160, 82, -16, 64, 14), (12, 204, 90, -2, 74, 16)):
        dx, th, cu, ln = _pose(dx, th, cu, ln, sway, lift)
        poses.append((dx, ln, wd, th, cu, wg))
        d, _ = _lock(dx, root, ln, wd, th, cu, wg)
        out.append(f'<path d="{d}" fill="{c["crest"]}" stroke="{c["deep"]}" stroke-width="6" stroke-linejoin="round"/>')
    for scale, fill in ((0.68, c['light']), (0.42, core)):  # the hotter inner flame of the two long locks
        for dx, ln, wd, th, cu, wg in poses[2:]:
            d, _ = _lock(dx, root, ln * scale, wd * scale * 0.8, th, cu * scale, wg)
            out.append(f'<path d="{d}" fill="{fill}"/>')
    tip = _lock(poses[-1][0], root, *poses[-1][1:])[1][-1]
    for ox, oy, r in ((-34, 30, 9), (30, 10, 7), (-6, -22, 6)):  # sparks, drifting with the flame
        sx, sy = tip[0] + ox + 26 * sway, tip[1] + oy + 50 * (1 - lift)
        out.append(f'<path d="M{sx:.1f},{sy - r * 1.6:.1f} L{sx + r:.1f},{sy:.1f} L{sx:.1f},{sy + r * 1.6:.1f} '
                   f'L{sx - r:.1f},{sy:.1f}Z" fill="{c["crest"]}" stroke="{c["deep"]}" stroke-width="2.5" '
                   f'stroke-linejoin="round"/>')
    return out


def _fringe(name, top, cy, c, sway, lift):
    """The hair over the crown, drawn over the head: an arc along the head's top with a scalloped edge above the eyes,
    its points swept the way the hair flows."""
    rx, ry = HEAD_RX + 7, HEAD_RY + 7
    phi = math.radians(26)
    ex, ey = rx * math.cos(phi), cy - ry * math.sin(phi)
    n = 5
    lean = 16 * sway * (0.4 + 0.6 * lift)
    edge = [f'M{ex:.1f},{ey:.1f}']
    xs = [ex - (2 * ex) * i / n for i in range(n + 1)]
    valley_y, tip_y = top + 30, top + 52
    for i in range(n):
        x1, x2 = xs[i], xs[i + 1]
        tx = (x1 + x2) / 2 + lean
        y1 = ey if i == 0 else valley_y
        edge.append(f'Q{x1 - 4:.1f},{(y1 + tip_y) / 2:.1f} {tx:.1f},{tip_y:.1f}')
        y2 = ey if i == n - 1 else valley_y
        edge.append(f'Q{x2 + 4:.1f},{(y2 + tip_y) / 2:.1f} {x2:.1f},{y2:.1f}')
    edge.append(f'A{rx},{ry} 0 0 1 {ex:.1f},{ey:.1f}Z')
    d = ' '.join(edge)
    out = [f'<path d="{d}" fill="{c["crest"]}" stroke="{c["deep"]}" stroke-width="6" stroke-linejoin="round"/>']
    hl = c['light'] if name == 'quill' else c.get('core', c['light'])
    hx = 30 * (1 if sway >= 0 else -1)  # a highlight on the crown, the way the hair flows
    out.append(f'<path d="M{-56 + hx:.1f},{top + 22:.1f} Q{hx:.1f},{top + 6:.1f} {56 + hx:.1f},{top + 20:.1f}" fill="none" '
               f'stroke="{hl}" stroke-width="9" stroke-linecap="round"/>')
    return out


def _eyes(cy, eyes, look, c):
    out = []
    for side in (-1, 1):
        ex = side * EYE_DX
        if eyes == 'happy':  # upward arcs
            out.append(f'<path d="M{ex - 30},{cy + 12} Q{ex},{cy - 34} {ex + 30},{cy + 12}" fill="none" stroke="{PUPIL}" '
                       f'stroke-width="11" stroke-linecap="round"/>')
            continue
        px = ex + look * 10
        py = cy + (8 if eyes == 'flat' else 0)
        out.append(f'<circle cx="{ex}" cy="{cy}" r="{EYE_R}" fill="white" stroke="{c["deep"]}" stroke-width="5"/>')
        out.append(f'<circle cx="{px}" cy="{py}" r="{PUPIL_R}" fill="{PUPIL}"/>')
        out.append(f'<circle cx="{px + 8}" cy="{py - 9}" r="8" fill="white"/>')
        out.append(f'<circle cx="{px - 9}" cy="{py + 9}" r="3.5" fill="white"/>')
        if eyes == 'flat':  # half-closed lid in the head's colour
            r = EYE_R + 3
            out.append(f'<path d="M{ex - r},{cy + 2} A{r},{r} 0 0 1 {ex + r},{cy + 2}Z" fill="{c["main"]}"/>')
            out.append(f'<line x1="{ex - r}" y1="{cy + 2}" x2="{ex + r}" y2="{cy + 2}" stroke="{c["deep"]}" '
                       f'stroke-width="7" stroke-linecap="round"/>')
    return out


def _spine(top, n, spacing, c, rings):
    """n pucks from the neck (top) down; rings: the puck numbers, counted from 0 at the base like the
    code's blocks, that carry a teal ring.
    Returns markup and {puck number: centre y}."""
    out, ys = [], {}
    ph = 0.62 * spacing
    core_w = 0.12 * BODY_W
    for i in range(n):  # i = 0 at the neck
        k = n - 1 - i  # puck number from 0 at the base: the link after layer 6 rings the 7th puck
        t = (k + 0.5) / n if n == LAYERS else 1 - (i + 0.5) / (2 * n)  # icons show only the upper half
        w = BODY_W * _taper(t)
        y = top + i * spacing + (spacing - ph) / 2
        out.append(f'<rect x="{-w / 2:.1f}" y="{y:.1f}" width="{w:.1f}" height="{ph:.1f}" rx="{ph / 2:.1f}" '
                   f'fill="{c["main"]}" stroke="{c["deep"]}" stroke-width="3"/>')
        ys[k] = y + ph / 2
    bottom = top + n * spacing
    out.append(f'<rect x="{-core_w / 2:.1f}" y="{top:.1f}" width="{core_w:.1f}" height="{bottom - top:.1f}" '
               f'rx="{core_w / 2:.1f}" fill="{c["light"]}"/>')
    for k in rings:
        t = (k + 0.5) / n if n == LAYERS else 1 - (n - 1 - k + 0.5) / (2 * n)
        w = BODY_W * _taper(t) + 14
        h = ph + 10
        out.append(f'<rect x="{-w / 2:.1f}" y="{ys[k] - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h / 2:.1f}" '
                   f'fill="none" stroke="{LINK}" stroke-width="6"/>')
        out.append(f'<rect x="{-core_w / 2:.1f}" y="{ys[k] - ph / 2:.1f}" width="{core_w:.1f}" height="{ph:.1f}" '
                   f'rx="{core_w / 2:.1f}" fill="{LINK}"/>')  # where the strands plug into the core
    return out, ys


def creature(name, cx, top, height, grey=False, eyes='open', look=None, full=False, sway=None, lift=1.0):
    """SVG markup of Quill or Ember with the crest's tip at y=top and the base at top+height, centred on cx.
    eyes: 'open', 'happy' (upward arcs) or 'flat' (half-closed). look: pupils' horizontal shift from -1 to 1 (default:
    toward the partner, Quill on the left and Ember on the right). full: all 36 layers and the 8 link layers, for
    figures about the architecture; otherwise a bust with a short body, for icons. sway: how far the crest leans toward
    +x, from -1 to 1 (default: toward the partner, half way); lift: 1 stands the crest up (linked), lower values let it
    fall to the sides (link cut).
    Returns (markup, ports): ports maps each link layer to (x, y) of the core at that layer (full body only)."""
    c = PALETTE['grey' if grey else name]
    look = (1 if name == 'quill' else -1) if look is None else look
    sway = (0.5 if name == 'quill' else -0.5) if sway is None else sway
    head_top = Y0 + 108  # the crest rises above the crown, up to y = 0
    head_cy = head_top + HEAD_RY
    neck = head_cy + HEAD_RY - 16
    if full:
        spacing = 27.0
        n, rings = LAYERS, PORTS
    else:
        spacing = 46.0
        n, rings = 4, (0, 2)
    total = neck + n * spacing
    s = height / total
    parts = _crest(name, head_top, c, sway, lift)
    body, ys = _spine(neck, n, spacing, c, rings)
    parts += body
    parts.append(f'<ellipse cx="0" cy="{head_cy}" rx="{HEAD_RX}" ry="{HEAD_RY}" fill="{c["main"]}" stroke="{c["deep"]}" '
                 f'stroke-width="6"/>')
    parts += _fringe(name, head_top, head_cy, c, sway, lift)
    parts += _eyes(head_cy - 6, eyes, look, c)
    g = f'<g transform="translate({cx:.2f},{top:.2f}) scale({s:.5f})">' + ''.join(parts) + '</g>'
    ports = {k: (cx, top + ys[k] * s) for k in PORTS} if full else {}
    return g, ports


def width_of(height, full=False):
    """The drawing's width for a given height (crest and head are the widest parts)."""
    total = (Y0 + 108 + 2 * HEAD_RY - 16) + (LAYERS * 27.0 if full else 4 * 46.0)
    return 2 * (HEAD_RX + 6) * height / total


def strands(x1, x2, y, wave=9, sw=5):
    """Two thin teal fibres twisted around each other between the cores at x1 and x2, one per direction. Draw them
    before the creatures, so the fibres run under the skin and appear again at the core."""
    out = []
    n = max(2, round(abs(x2 - x1) / 150))
    for phase in (0, math.pi):
        pts = []
        for i in range(n * 12 + 1):
            t = i / (n * 12)
            pts.append(f'{x1 + (x2 - x1) * t:.1f},{y + wave * math.sin(2 * math.pi * n * t + phase):.1f}')
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{LINK}" stroke-width="{sw}" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
    return ''.join(out)


def _render(svg, w, h, png):
    src = png.with_suffix('.svg')
    src.write_text(svg)
    subprocess.run(['rsvg-convert', '-w', str(w), '-h', str(h), str(src), '-o', str(png)], check=True)
    src.unlink()


def main():
    assets = HERE / 'assets'
    for name in ('quill', 'ember'):  # icons: linked, and with the link cut (assets/NAME_cut.png)
        for stem, kw in ((name, {}), (f'{name}_cut', dict(eyes='flat', lift=0.25, sway=0.0))):
            g, _ = creature(name, 300, 20, 560, **kw)
            _render(f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 600 600">{g}</svg>',
                    600, 600, assets / f'{stem}.png')
    # the sheet: icons in every variant, then the pair at full body with the link at the 8 layers
    W, H = 2300, 2000
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="white"/>']
    x = 40
    for name in ('quill', 'ember'):  # linked (crest up, leaning to the partner) and cut (eyes half closed, crest down)
        for kw in (dict(), dict(eyes='flat', lift=0.25, sway=0.0)):
            g, _ = creature(name, x + 150, 30, 320, **kw)
            parts.append(g)
            x += 300
    for name, sway in (('quill', -1.0), ('ember', 1.0)):  # leaning the other way
        g, _ = creature(name, x + 150, 30, 320, sway=sway)
        parts.append(g)
        x += 300
    h = 1560
    qx, ex = 820, 1480
    gq, pq = creature('quill', qx, 400, h, full=True, look=1)
    ge, pe = creature('ember', ex, 400, h, full=True, look=-1)
    for k in PORTS:
        parts.append(strands(pq[k][0], pe[k][0], pq[k][1]))
    parts += [gq, ge]
    parts.append('</svg>')
    _render(''.join(parts), W, H, assets / 'creature_sheet.png')


if __name__ == '__main__':
    main()
