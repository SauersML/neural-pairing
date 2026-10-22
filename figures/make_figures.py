"""Draw the figures in figures/: method.png (the link), probe.png (link minus shuffled messages, from
results/probe.json), conversation.png (one co_talk episode of a transcripts.py JSON file), hero.png (free-generation
scores, from an evalfree.py summary), opening.png (write sizes over training, from training logs), curves.png (held-out
gain on the partner's value over training) and readmap.png (where a reader looks, from a readmap.py JSON file).
Usage: python make_figures.py [figure ...] (default: every figure); NAME=SOURCE picks a figure's input:
conversation=FILE.json[,K] (the K-th co_talk episode), hero=FILE.json[,all] (every episode, not only clean ones),
opening=LOG[|LABEL][,LOG[|LABEL]...] (train.py's printed log or mlx_train.py's train.log or log.jsonl; one column each),
curves=RUN_DIR_OR_LOG[|LABEL][,...] (payload_stepN.json files or payload_eval log lines; one panel each),
readmap=FILE.json[,K] (the K-th record)."""

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

INK = "#1D1D1F"
MUTED = "#6E6E73"
PANEL = "#F4F4F6"
QUILL = "#7B3FE4"        # Qwen3-4B
QUILL_LIGHT = "#E3D6FB"
EMBER = "#E8590C"        # AFM-4.5B
EMBER_LIGHT = "#FDE3D3"
LINK = "#1A9E77"         # the link
TICK = "#E5E5EA"
CLEAR = dict(facecolor="white", edgecolor="none", pad=1)   # a value label stays legible over guide lines
W = 1200
PT = 100 / 72


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": families, "pdf.fonttype": 42, "svg.fonttype": "none"})


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    for problem in layout_problems(fig):
        print(f"{name}: {problem}")
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def layout_problems(fig, tol=2.0):
    """Text that leaves the canvas, overlaps other text, or is crossed by a line or an arrow (display pixels)."""
    from matplotlib.path import Path
    from matplotlib.transforms import Bbox
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    texts = [(t.get_text().replace("\n", " "), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
    out = []
    F = fig.bbox
    for t, b in texts:
        if b.x0 < F.x0 - tol or b.x1 > F.x1 + tol or b.y0 < F.y0 - tol or b.y1 > F.y1 + tol:
            out.append(f"cut off: {t[:50]!r}")
    for i, (t, b) in enumerate(texts):
        for u, c in texts[i + 1:]:
            box = Bbox.intersection(b, c)
            if box is not None and box.width > tol and box.height > 0.3 * min(b.height, c.height):
                out.append(f"overlap: {t[:40]!r} / {u[:40]!r}")
    drawn = [ln.get_transform().transform_path(ln.get_path()) for ln in ax.lines if ln.get_gid() != "strike"]
    drawn += [p.get_transform().transform_path(p.get_path()) for p in ax.patches if isinstance(p, FancyArrowPatch)]
    paths = [Path(v) for path in drawn for v in path.to_polygons(closed_only=False) if len(v) > 1]
    boxes = [p.get_window_extent(r) for p in ax.patches if isinstance(p, FancyBboxPatch)]
    bars = [p.get_window_extent(r) for p in ax.patches if type(p) is Rectangle]
    for t, b in texts:
        inner = Bbox.from_extents(b.x0 + tol, b.y0 + tol, b.x1 - tol, b.y1 - tol)
        if any(path.intersects_bbox(inner, filled=False) for path in paths):
            out.append(f"line through: {t[:50]!r}")
        for c in boxes:
            box = Bbox.intersection(b, c)
            if box is not None and box.width > tol and box.height > 0.3 * b.height and not (
                    b.x0 >= c.x0 - tol and b.x1 <= c.x1 + tol and b.y0 >= c.y0 - tol and b.y1 <= c.y1 + tol):
                out.append(f"crosses a box edge: {t[:50]!r}")
                break
        ink = Bbox.from_extents(b.x0, b.y0 + 0.2 * b.height, b.x1, b.y1 - 0.2 * b.height)
        if any((x := Bbox.intersection(ink, c)) is not None and x.width > tol and x.height > tol for c in bars):
            out.append(f"on a bar: {t[:50]!r}")
    return out


# Canvas primitives (units of 0.01 inch).

def canvas(height, width=W):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def crop(fig, ax, bottom, top):
    width = ax.get_xlim()[1]
    ax.set_ylim(bottom, top)
    fig.set_size_inches(width / 100, (top - bottom) / 100)


def rbox(ax, x, y, w, h, fill, radius=14, z=2, edge="none", lw=0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))


def write(ax, x, y, text, size=28, bold=False, color=INK, ha="center", va="center", **kw):
    return ax.text(x, y, text, fontsize=size, weight="bold" if bold else "normal", color=color,
                   ha=ha, va=va, zorder=kw.pop("zorder", 6), **kw)


def text_width(ax, text, size, bold=False, family=None):
    artist = ax.text(0, 0, text, fontsize=size, weight="bold" if bold else "normal", family=family)
    box = artist.get_window_extent(ax.figure.canvas.get_renderer())
    artist.remove()
    return box.width / ax.figure.dpi * 100


def arrow(ax, a, b, color=INK, lw=3, rad=0.0, z=4, head=22):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=f"-|>,head_length={head / 20},head_width={head / 34}",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=z,
                                 mutation_scale=20, shrinkA=0, shrinkB=0))


# The method figure: the two block stacks and the link (top left), one exchange up close (top right), time (bottom).

XBLOCKS = (6, 9, 12, 15, 18, 21, 24, 27)   # pairing/link.py: an exchange after each of these blocks (counted from 0)
FAINT = "#C7C7CC"


def mix(colour, s):
    """colour at strength s over white."""
    from matplotlib.colors import to_rgb
    return tuple(1 - s * (1 - c) for c in to_rgb(colour))


def tile(ax, cx, cy, text, fill, colour, w=None, h=56, size=24, bold=False, edge="none", lw=0, z=3):
    """A token: a rounded box centred at (cx, cy), as wide as its text unless w is given. Returns its width."""
    w = w or text_width(ax, text, size, bold) + 30
    rbox(ax, cx - w / 2, cy - h / 2, w, h, fill, radius=10, z=z, edge=edge, lw=lw)
    if text:
        write(ax, cx, cy, text, size, bold, colour, zorder=z + 1)
    return w


def dial(ax, cx, cy, r, s, colour):
    """A gate as a dial filled to the fraction s."""
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=colour, lw=2, zorder=4))
    ax.add_patch(Wedge((cx, cy), r, 90 - 360 * s, 90, facecolor=colour, edgecolor="none", zorder=5))


def lines(ax, x, y, rows, size=24, step=38):
    """Left-aligned lines of (text, colour) from y down."""
    for k, (text, colour) in enumerate(rows):
        write(ax, x, y - step * k, text, size, color=colour, ha="left")


def towers(ax, zoom_box, zoom_block=24):
    """Both models' 36 blocks, the exchanges between them, and the callout of one exchange to zoom_box. Returns the
    lowest y used."""
    QX, EX, TW = 200, 590, 150
    bh, gap, xgap = 18, 6, 38
    ys, y = [], 1010
    for l in range(36):
        ys.append(y)
        y += bh + (xgap if l in XBLOCKS else gap)
    top = ys[-1] + bh
    for x0, light in ((QX, QUILL_LIGHT), (EX, EMBER_LIGHT)):
        for y in ys:
            rbox(ax, x0, y, TW, bh, light, radius=5)
    for l in (0,) + XBLOCKS + (35,):
        write(ax, QX - 16, ys[l] + bh / 2, f"block {l}" if l == 0 else str(l), 24, color=MUTED, ha="right")
    for l in XBLOCKS:
        yc = ys[l] + bh + xgap / 2
        arrow(ax, (QX + TW + 8, yc + 8), (EX - 8, yc + 8), LINK, 3, head=15)
        arrow(ax, (EX - 8, yc - 8), (QX + TW + 8, yc - 8), LINK, 3, head=15)
    mid = (QX + TW + EX) / 2
    write(ax, mid, (ys[0] + ys[6] + bh) / 2, "no link", 24, color=MUTED)
    write(ax, mid, (ys[28] + top) / 2, "no link", 24, color=MUTED)
    # each model reads only its own tokens and predicts its own next token
    for x0, colour, light, words, nxt in ((QX, QUILL, QUILL_LIGHT, ("Sky", "looks", "blue"), "because"),
                                         (EX, EMBER, EMBER_LIGHT, ("The", "word", "is"), "lantern")):
        cx = x0 + TW / 2
        ws = [text_width(ax, w, 24) + 26 for w in words]
        x = cx - (sum(ws) + 8 * (len(ws) - 1)) / 2
        for w, wd in zip(words, ws):
            tile(ax, x + wd / 2, ys[0] - 80, w, light, colour, w=wd, h=50)
            x += wd + 8
        arrow(ax, (cx, ys[0] - 52), (cx, ys[0] - 4), INK, 2.5, head=14)
        arrow(ax, (cx, top + 4), (cx, top + 42), INK, 2.5, head=14)
        tile(ax, cx, top + 72, nxt, colour, "white", h=50, bold=True)
    write(ax, mid, top + 72, "next token", 24, color=MUTED)
    for x0, name, hf, colour in ((QX, "Quill", "Qwen3-4B", QUILL), (EX, "Ember", "AFM-4.5B", EMBER)):
        write(ax, x0 + TW / 2, top + 178, name, 34, True, colour)
        write(ax, x0 + TW / 2, top + 136, hf, 24, color=MUTED)
    lines(ax, 40, ys[0] - 165, [("Green arrows: the link, after blocks 6, 9, …, 27 of 36. At each of these exchanges, both "
                                 "models read each other's", LINK),
                                ("hidden states at the same time, from the states before the exchange. No tokens cross.",
                                 LINK),
                                ("Only the link (8 heads of 128 dimensions, in float32) and LoRA adapters on both models "
                                 "train; the original weights stay frozen.", MUTED)], size=26, step=42)
    # the exchange drawn up close
    y0, y1 = ys[zoom_block] + bh + 3, ys[zoom_block + 1] - 3
    rbox(ax, QX - 10, y0, EX + TW - QX + 20, y1 - y0, "none", radius=8, z=5, edge=INK, lw=2)
    zx, zy0, zy1 = zoom_box
    for ya, yb in ((y1, zy1 - 30), (y0, zy0 + 30)):
        ax.plot([EX + TW + 10, zx], [ya, yb], color=INK, lw=1.5, zorder=1)
    return ys[0] - 165 - 42 * 2 - 20


def exchange(ax, X0, X1, Y0, Y1):
    """One exchange up close: Ember's token "is" at tick 3 reads Quill's tokens (the duplex example below)."""
    rbox(ax, X0, Y0, X1 - X0, Y1 - Y0, PANEL, radius=26, z=1)
    # Quill's tokens as Ember's token at tick 3 sees them: (token, tick, gate s, attention weight; None: hidden)
    cols = [("…", -1, 0.35, 0.05), ("lantern", -1, 0.95, 0.40), ("▸", 0, 0.25, 0.04),
            ("Sky", 1, 0.6, 0.08), ("looks", 2, 0.4, 0.06), ("blue", 3, 0.8, 0.15), ("because", 4, 0.7, None)]
    null_w = 0.22
    lx = X0 + 215                                    # row labels end here
    gx = lx + 30
    xs = []
    for tok, *_ in cols:
        w = max(text_width(ax, tok, 24) + 28, 64)
        xs.append((gx + w / 2, w))
        gx += w + 12
    null_x = gx + 34
    yt, ytok, ygate, yv, yk, ybar = Y0 + 90, Y0 + 165, Y0 + 252, Y0 + 332, Y0 + 402, Y0 + 452
    unit = 260 / 0.40                                # bar height per unit of attention weight
    for label, y in (("tick", yt), ("Quill's states", ytok), ("gate s", ygate), ("value × s", yv), ("key", yk),
                     ("attention", ybar + 90)):
        write(ax, lx, y, label, 24, color=MUTED, ha="right")
    for (tok, tick, s, p), (cx, w) in zip(cols, xs):
        f = 0.3 if p is None else 1.0
        write(ax, cx, yt, f"{tick}".replace("-", "−"), 24, color=FAINT if p is None else MUTED)
        tile(ax, cx, ytok, tok, mix(QUILL_LIGHT, f), mix(QUILL, f), w=w)
        dial(ax, cx, ygate, 24, s, mix(QUILL, f))
        rbox(ax, cx - 28, yv - 22, 56, 44, mix(QUILL, s * f), radius=8, z=3)
        rbox(ax, cx - 28, yk - 22, 56, 44, mix(QUILL, f), radius=8, z=3)
        if p is not None:
            ax.add_patch(Rectangle((cx - 20, ybar), 40, unit * p, color=LINK, zorder=3))
    write(ax, xs[-1][0], ybar + 82, "hidden:", 24, color=MUTED)
    write(ax, xs[-1][0], ybar + 44, "tick 4 > 3", 24, color=MUTED)
    note = xs[2][0] - xs[2][1] / 2
    write(ax, note, ybar + 218, "bar heights and gate dials:", 24, color=MUTED, ha="left")
    write(ax, note, ybar + 182, "illustrative values, not measured", 24, color=MUTED, ha="left")
    # the null key (Ember's) with a zero value: Ember may read nothing
    write(ax, null_x, yt, "null key", 24, True, EMBER)
    rbox(ax, null_x - 28, yv - 22, 56, 44, "white", radius=8, z=3, edge=EMBER, lw=2)
    write(ax, null_x, yv, "0", 24, True, EMBER)
    rbox(ax, null_x - 28, yk - 22, 56, 44, EMBER, radius=8, z=3)
    ax.add_patch(Rectangle((null_x - 20, ybar), 40, unit * null_w, color=LINK, zorder=3))
    ax.plot([xs[0][0] - xs[0][1] / 2, null_x + 40], [ybar, ybar], color=INK, lw=1.5, zorder=4)
    # Ember's residual stream: the query leaves it, the write comes back into it
    sx = X1 - 100
    yo = ybar + unit * 0.40 + 130
    ax.plot([sx, sx], [Y0 + 115, yo - 26], color=EMBER, lw=5, zorder=4, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((sx, yo + 28), (sx, yo + 190), arrowstyle="-|>,head_length=1.1,head_width=0.65",
                                 color=EMBER, lw=5, zorder=4, mutation_scale=20, shrinkA=0, shrinkB=0))
    write(ax, X1 - 24, Y0 + 70, "Ember's state", 26, True, EMBER, ha="right")
    write(ax, X1 - 24, Y0 + 34, "token “is”, tick 3", 24, color=MUTED, ha="right")
    write(ax, sx - 24, yo + 170, "to block 25", 24, color=MUTED, ha="right")
    qx = sx - 130
    tile(ax, qx, yk, "Q", EMBER, "white", w=76, h=64, size=28, bold=True)
    arrow(ax, (sx, yk), (qx + 40, yk), EMBER, 3, head=16)
    arrow(ax, (qx - 40, yk), (null_x + 32, yk), INK, 3, head=16)
    write(ax, qx, yk - 58, "query", 24, color=MUTED)
    # the attention-weighted sum of the values -> O -> scaled -> added to Ember's state
    wx0, wx1 = xs[0][0] - xs[0][1] / 2, xs[-1][0] + xs[-1][1] / 2
    rbox(ax, wx0, yo - 32, wx1 - wx0, 64, "white", radius=12, z=3, edge=LINK, lw=2.5)
    write(ax, (wx0 + wx1) / 2, yo, "attention-weighted sum of the values", 26, True, LINK)
    arrow(ax, ((wx0 + wx1) / 2, ybar + unit * 0.40 + 12), ((wx0 + wx1) / 2, yo - 36), LINK, 3, head=16)
    ox = wx1 + 74
    tile(ax, ox, yo, "O", EMBER, "white", w=76, h=64, size=28, bold=True)
    arrow(ax, (wx1 + 4, yo), (ox - 40, yo), INK, 3, head=16)
    ax.add_patch(Circle((sx, yo), 26, facecolor="white", edgecolor=EMBER, lw=3, zorder=6))
    write(ax, sx, yo + 2, "+", 28, True, EMBER, zorder=7)
    arrow(ax, (ox + 40, yo), (sx - 28, yo), INK, 3, head=16)
    write(ax, (ox + 40 + sx - 28) / 2, yo + 58, "× tanh(α) · size", 24)
    # the read as one line, and what the picture leaves out
    family = plt.rcParams["font.family"][0]
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic"})
    h = lambda who: r"$h_{\rm " + who + "}$"
    write(ax, X0 + 40, Y1 - 60, f"{h('Ember')} += tanh(α) · size · O(attention(Q(norm {h('Ember')}), "
          f"K(norm {h('Quill')}), s · V(norm {h('Quill')})))", 27, ha="left")
    lines(ax, X0 + 40, Y1 - 114, [
        ("Orange: Ember, the reader, and Ember's weights: Q, O, α and the null key.", EMBER),
        ("Violet: Quill, the sender, and Quill's weights: K, V and the gate s = σ(w · norm(h) + b).", QUILL),
        ("The gate reads Quill's state normalized to unit RMS and scales the values, not the keys.", INK),
        ("Each head adds a learned bias by how many ticks ago each Quill token was read.", INK),
        ("size: Ember's typical residual size after block 24, so tanh(α) is the write's relative size.", INK),
        ("O starts at zero and tanh(α) at 0.5: at step 0 the pair is exactly the two original models.", INK),
        ("At the same time, Quill reads Ember the same way.", INK)], step=42)


def timeline(ax, X0, Y1):
    """Ticks: the duplex relay (Ember's tick-3 token reads Quill up to tick 3) and a co_talk dialogue. Returns the
    lowest y used."""
    P, MARK = "prompt", "mark"
    duplex = {"quill": {-1: [('secret word "lantern"', P)], 0: [("▸", MARK)], 1: [("Sky", 0)],
                        2: [("looks", 0)], 3: [("blue", 0)], 4: [("because", 0)], 5: [("air", 0)]},
              "ember": {-1: [("get Quill's word", P)], 0: [("▸", MARK)], 1: [("The", 0)], 2: [("word", 0)],
                        3: [("is", 0)], 4: [("lantern", 0)], 5: [(".", 0)]}}
    talk = {"quill": {-1: [("text: …a red door…", P)], 0: [("▸", MARK)], 2: [("Red", 1)],
                      3: [(".", 1), ("▸", MARK)], 4: [("Yes", 0)], 5: [(",", 0)], 6: [("the", 0)], 7: [("door", 0)]},
            "ember": {-1: [("asked: door colour?", P)], 0: [("▸", MARK)], 1: [("Red", 0)], 2: [(".", 0)],
                      3: [("▸", MARK)], 5: [("Yes", 1)], 6: [(",", 1)], 7: [("the", 1)]}}
    ticks = range(-1, 8)
    pad, sep, drop = 26, 14, 150

    def tok_w(text):
        return max(text_width(ax, text, 24) + pad, 48)

    def width(cell):
        return sum(tok_w(t) for t, _ in cell) + 6 * (len(cell) - 1)
    colw = {t: max(width(rows[x].get(t, [])) for rows in (duplex, talk) for x in rows) for t in ticks}
    cx, x = {}, X0 + 150
    for t in ticks:
        cx[t] = x + colw[t] / 2
        x += colw[t] + sep
    end = x

    def lane(y_top, rows, fade=()):
        """Draws the tick labels and both rows; returns {(side, tick, k): (x, y, w)} of every token."""
        for t in ticks:
            write(ax, cx[t], y_top, str(t).replace("-", "−"), 24, color=MUTED)
        write(ax, X0, y_top, "tick", 24, color=MUTED, ha="left")
        at = {}
        for x, y, colour, light, name in (("quill", y_top - 70, QUILL, QUILL_LIGHT, "Quill"),
                                          ("ember", y_top - 70 - drop, EMBER, EMBER_LIGHT, "Ember")):
            write(ax, X0, y, name, 28, True, colour, ha="left")
            for t, cell in rows[x].items():
                xx = cx[t] - width(cell) / 2
                for k, (text, kind) in enumerate(cell):
                    w = tok_w(text)
                    f = 0.3 if (x, t) in fade else 1.0
                    if kind == P:
                        tile(ax, xx + w / 2, y, text, "white", colour, w=w, edge=colour, lw=2)
                    elif kind == MARK:
                        tile(ax, xx + w / 2, y, text, "white", MUTED, w=w, edge=FAINT, lw=1.5)
                    elif kind == 1:       # the partner's text, read as a user message
                        tile(ax, xx + w / 2, y, text, "#E9E9ED", MUTED, w=w)
                    else:
                        tile(ax, xx + w / 2, y, text, mix(light, f), mix(colour, f), w=w)
                    at[(x, t, k)] = (xx + w / 2, y, w)
                    xx += w + 6
        return at

    def reads(at, srcs, dst):
        """Green lines from every token in srcs that meet above dst and point at it, with dst outlined."""
        bx, by, bw = at[dst]
        rbox(ax, bx - bw / 2 - 5, by - 33, bw + 10, 66, "none", radius=13, z=5, edge=LINK, lw=3.5)
        meet = (bx, by + 78)
        for s in srcs:
            sx, sy, _ = at[s]
            ax.plot([sx, meet[0]], [sy - 30, meet[1]], color=LINK, lw=2.5, zorder=4, solid_capstyle="round")
        arrow(ax, meet, (bx, by + 36), LINK, 3, head=15)

    def heading(x, y, bold, plain):
        write(ax, x, y, bold, 30, True, ha="left")
        write(ax, x + text_width(ax, bold + " ", 30, True), y, plain, 26, color=MUTED, ha="left")

    heading(X0, Y1, "Time runs in ticks.", "At every exchange, a token reads the partner's tokens of the same or an "
            "earlier tick. Both prompts are at tick −1.")
    # duplex: Ember's token at tick 3 reads Quill's tokens of ticks -1..3
    y = Y1 - 110
    at = lane(y, duplex, fade={("quill", 4), ("quill", 5)})
    reads(at, [("quill", t, 0) for t in range(-1, 4)], ("ember", 3, 0))
    tx = end + 40
    heading(tx, y, "Both write at once", "(duplex)")
    lines(ax, tx, y - 52, [("Neither model sees the other's text;", MUTED),
                           ("only the link connects them.", MUTED),
                           ("Ember's token “is” at tick 3 reads", LINK),
                           ("Quill's tokens up to tick 3, the prompt", LINK),
                           ("included; later ones stay hidden.", LINK)])
    # dialogue (co_talk): Ember answers about Quill's text before Quill writes a word
    y2 = y - 70 - drop - 140
    at = lane(y2, talk)
    reads(at, [("quill", -1, 0), ("quill", 0, 0)], ("ember", 0, 0))
    sx, sy, _ = at[("ember", 1, 0)]
    dx, dy, _ = at[("quill", 2, 0)]
    arrow(ax, (sx + 8, sy + 32), (dx - 8, dy - 32), MUTED, 2.5, head=14)
    heading(tx, y2, "Taking turns in text", "(dialogue)")
    lines(ax, tx, y2 - 52, [("The listener reads the speaker's words as a", MUTED),
                            ("user message, one tick behind (grey).", MUTED),
                            ("The link stays open throughout: Ember", LINK),
                            ("answers about Quill's text before Quill", LINK),
                            ("has written a word.", LINK),
                            ("▸: the tokens that open and close a turn.", MUTED)])
    return y2 - 70 - drop - 50


def method():
    """The two models joined by the link: the block stacks (top left), one exchange up close (top right), and how the
    two token streams line up in time in the two session modes (bottom)."""
    fig, ax = canvas(2500, 2 * W)
    zoom = (900, 2370, 980, 2370)
    bottom = towers(ax, (zoom[0], zoom[2], zoom[3]))
    exchange(ax, *zoom)
    low = timeline(ax, 40, min(bottom, zoom[2]) - 70)
    crop(fig, ax, low, 2400)
    save(fig, "method")


PROBE_GAMES = [("relay_word", "Relay a word"), ("relay_number", "Relay a 4-digit number"),
               ("relay_phrase", "Relay a two-word phrase"), ("relay_digits", "Relay up to 16 digits"),
               ("merge_sum", "Add two numbers"), ("merge_fact", "Join a name and a city"), ("lookup", "Look up a code")]
SHORT = {"quill": "Qwen", "ember": "AFM"}


def median_interval(d):
    """Exact distribution-free 95% interval for a median (binomial order statistics)."""
    d = sorted(d)
    n = len(d)
    k = int(binom.ppf(0.025, n, 0.5))
    while k > 0 and binom.cdf(k - 1, n, 0.5) > 0.025:
        k -= 1
    return (d[k - 1], d[n - k]) if k > 0 else (float("-inf"), float("inf"))


def probe():
    """Soft-token link after 200 steps (results/probe.json, default arm): for each game and direction, the receiver's
    cross-entropy on the partner's value with the link minus with messages from another episode of the same game, per
    held-out episode; the median and an exact 95% interval."""
    rows = []
    for game, label in PROBE_GAMES:
        for k, recv in enumerate(("ember", "quill")):
            d = sorted(r["link_ce"] - r["shuffle_ce"] for r in load("probe.json")["default"]
                       if r["game"] == game and r["receiver"] == recv)
            if d:
                med = d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
                rows.append((label if k == 0 else "", f"{SHORT[partner_of(recv)]} \u2192 {SHORT[recv]}", med,
                             *median_interval(d)))
    minus = lambda t: t.replace("-", "\u2212")
    ticks = MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(min(min(r[3] for r in rows), 0), 0)
    x0, x1 = 760, 1100
    sx = lambda v: x0 + (x1 - x0) * (v - ticks[0]) / (ticks[-1] - ticks[0])
    fig, ax = canvas(2100)
    write(ax, 1170, 2060, "link minus messages from another episode", 26, True, ha="right")
    write(ax, 1170, 2024, "cross-entropy on the partner's value, nats", 26, color=MUTED, ha="right")
    y = 1940
    top = y + 30
    for k, (label, way, med, a, b) in enumerate(rows):
        if label and k:
            y -= 18
        if label:
            write(ax, 40, y - 28, label, 28, ha="left")
        write(ax, 520, y, way, 26, color=MUTED, ha="left")
        moved = b < 0 or a > 0
        colour = LINK if moved else MUTED
        ax.plot([sx(a), sx(b)], [y, y], color=colour, lw=3.5, zorder=5, solid_capstyle="butt")
        ax.scatter([sx(med)], [y], s=420 if moved else 200, color=colour, lw=2.5, edgecolor="white", zorder=6)
        if moved:
            write(ax, sx(b) + 16, y, minus(f"{med:+.1f}"), 26, True, LINK, ha="left", bbox=CLEAR)
        y -= 56
    ax.plot([sx(0)] * 2, [top, y + 26], color=INK, lw=2.5, zorder=2)
    for v in ticks:
        write(ax, sx(v), y, minus(f"{v:.0f}"), 26, color=MUTED)
    write(ax, sx(0), y - 44, "\u2190 the link passes the value", 24, color=LINK, ha="right")
    write(ax, 40, y - 100, "dot: median over episodes never seen in training; line: exact 95% interval", 24, color=MUTED, ha="left")
    crop(fig, ax, y - 130, 2100)
    save(fig, "probe")

def partner_of(x):
    return "ember" if x == "quill" else "quill"


# The conversation figure: one co_talk episode from transcripts.py, as a chat between the two models.

NAMES = {"quill": ("Quill", "Qwen3-4B", QUILL, QUILL_LIGHT), "ember": ("Ember", "AFM-4.5B", EMBER, EMBER_LIGHT)}


def clean(text):
    """Text as matplotlib can draw it verbatim: no math mode, no control characters, tabs as spaces, and at most one
    blank line in a row."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("$", r"\$")
    text = "".join(c for c in text if c == "\n" or c.isprintable())
    return re.sub(r"\n\s*\n(\s*\n)+", "\n\n", "\n".join(line.rstrip() for line in text.split("\n"))).strip()


def wrap(ax, text, width, size=26):
    """Lines of text no wider than width (canvas units); words longer than a line are broken."""
    out = []
    for para in clean(text).split("\n"):
        line = ""
        for word in para.split(" "):
            cand = f"{line} {word}" if line else word
            if text_width(ax, cand, size) <= width:
                line = cand
                continue
            if line:
                out.append(line)
            while text_width(ax, word, size) > width:
                k = max(n for n in range(1, len(word) + 1) if text_width(ax, word[:n], size) <= width or n == 1)
                out.append(word[:k])
                word = word[k:]
            line = word
        out.append(line)
    while len(out) > 1 and not out[-1].strip():
        out.pop()
    return out or [""]


def text_box(ax, x, y_top, w, lines, fill, colour=INK, size=26, step=40, edge="none", lw=0, pad=22):
    """A rounded box of wrapped lines from y_top down; returns its bottom."""
    h = 2 * pad + step * len(lines) - (step - size * PT * 1.2)
    rbox(ax, x, y_top - h, w, h, fill, radius=16, edge=edge, lw=lw)
    for k, line in enumerate(lines):
        write(ax, x + pad, y_top - pad - size * PT * 0.6 - step * k, line, size, color=colour, ha="left")
    return y_top - h


def excerpt(ax, context, answer, width, max_lines=6, size=24):
    """Wrapped lines of the holder's text around the first mention of the answer, with ellipses where cut."""
    at = context.lower().find(answer.strip().lower())
    at = len(context) // 2 if at < 0 else at
    span = 600
    while True:
        a, b = max(0, at - span // 2), min(len(context), at + span // 2)
        while a > 0 and not context[a - 1].isspace():
            a -= 1
        while b < len(context) and not context[b].isspace():
            b += 1
        lines = wrap(ax, ("…" if a > 0 else "") + context[a:b].strip() + ("…" if b < len(context) else ""),
                     width, size)
        if len(lines) <= max_lines or span <= 40:
            return lines[:max_lines]
        span -= 40


def badge(ax, x, y, score):
    """The score of a first turn: a tick for 1, a cross for 0, else the number."""
    if score == 1:
        ax.add_patch(Circle((x, y), 30, facecolor=LINK, edgecolor="none", zorder=5))
        write(ax, x, y, "✓", 30, True, "white", zorder=6)
    elif score == 0:
        ax.add_patch(Circle((x, y), 30, facecolor="white", edgecolor=MUTED, lw=2.5, zorder=5))
        write(ax, x, y, "✗", 28, True, MUTED, zorder=6)
    else:
        ax.add_patch(Circle((x, y), 38, facecolor="white", edgecolor=INK, lw=2.5, zorder=5))
        write(ax, x, y, f"{score:.2f}", 24, True, zorder=6)


def gate_bar(ax, x, y, w, g):
    """A thin bar filled to the mean gate g, labelled with g."""
    if g is None:
        return
    label = f"mean gate {g:.2f}"
    bw = w - text_width(ax, label, 24) - 20
    ax.add_patch(Rectangle((x, y - 5), bw, 10, color=TICK, zorder=3))
    ax.add_patch(Rectangle((x, y - 5), bw * g, 10, color=LINK, zorder=4))
    write(ax, x + w, y, label, 24, color=MUTED, ha="right")


def conversation(src=None):
    """One co_talk episode of a transcripts.py JSON file (src: FILE.json or FILE.json,K for the K-th co_talk episode,
    default results/transcripts.json): the conversation with the link, the holder's text and the asker's question, and
    the asker's first turn with the link cut and with a shuffled partner."""
    path, _, k = (src or str(RES / "transcripts.json")).partition(",")
    if src is None and not Path(path).exists():
        print("conversation: no", path)
        return
    talks = [r for r in json.loads(Path(path).read_text()) if r["game"] == "co_talk"]
    rec = talks[int(k or 0)]
    m, conds = rec["meta"], rec["conditions"]
    asker, holder = m["asker"], m["holder"]
    item = m["item"]
    context = item["context"] if item["task"] == "qa" else item["a"] + "\n\n" + item["b"]
    fig, ax = canvas(4000, 2 * W)
    cols = {"quill": 40, "ember": 820}
    CW = 740
    top = 3960
    for x, x0 in cols.items():
        name, hf, colour, _ = NAMES[x]
        role = "has read the text below" if x == holder else "has only the question below"
        write(ax, x0, top, name, 34, True, colour, ha="left")
        write(ax, x0 + text_width(ax, name + "  ", 34, True), top, f"{hf} · {role}", 24, color=MUTED, ha="left")
    # what each model alone was given
    y = top - 60
    low = {}
    for x, label, body in ((holder, "text, read privately", excerpt(ax, context, item["answer"], CW - 44)),
                           (asker, "question, asked privately", wrap(ax, item["question"], CW - 44, 24))):
        write(ax, cols[x], y, label, 24, color=MUTED, ha="left")
        low[x] = text_box(ax, cols[x], y - 30, CW, body, PANEL, MUTED, size=24, step=36)
        tg = (conds["link"].get("turn_gates") or {}).get(x)
        if tg:
            gate_bar(ax, cols[x], low[x] - 26, CW, tg["prompt"])
            low[x] -= 40
    y = min(low.values()) - 70
    # the conversation with the link
    turns = conds["link"]["turns"]
    first_box = None
    for n, (sp, text) in enumerate(turns):
        _, _, colour, light = NAMES[sp]
        if n == 0 and sp == asker:
            write(ax, cols[sp], y, f"Answers before {NAMES[holder][0]} has written a word:", 26, True, LINK, ha="left")
            write(ax, cols[sp], y - 38, "only the link can carry the answer.", 26, True, LINK, ha="left")
            y -= 76
            bottom = text_box(ax, cols[sp], y, CW, wrap(ax, text, CW - 44), light, edge=LINK, lw=3.5)
            first_box = (y, bottom)
        else:
            bottom = text_box(ax, cols[sp], y, CW, wrap(ax, text, CW - 44), light)
        tg = (conds["link"].get("turn_gates") or {}).get(sp)
        if tg and n < len(tg["turns"]):
            gate_bar(ax, cols[sp], bottom - 26, CW, tg["turns"][n])
            bottom -= 40
        y = bottom - 50
    end = y
    # the same first turn: the answer it should give, with the link (the tick above), cut, and with a shuffled partner
    if first_box is not None:
        PX, PW = 1700, 590
        bx = cols[asker] + CW + 50
        badge(ax, bx, first_box[0] - 34, conds["link"]["scores"]["ok"])
        py = first_box[0] + 76
        write(ax, PX, py, "target answer", 24, color=MUTED, ha="left")
        py = text_box(ax, PX, py - 30, PW, wrap(ax, item["answer"], PW - 44), "white", edge=LINK, lw=2) - 56
        for c, label in (("cut", "the same first turn, link cut"), ("shuffle", "with a partner from another episode")):
            t = conds[c]["turns"]
            text = t[0][1] if t and t[0][0] == asker else ""
            write(ax, PX, py, label, 24, color=MUTED, ha="left")
            box_top = py - 30
            py = text_box(ax, PX, box_top, PW, wrap(ax, text, PW - 44), NAMES[asker][3]) - 56
            badge(ax, PX + PW + 50, box_top - 34, conds[c]["scores"]["ok"])
        end = min(end, py)
    lines(ax, 40, end - 10, [
        ("Greedy decoding on a never-seen episode. Green bars: the mean gate of the writer's tokens over each turn "
         "(0 hides them from", MUTED),
        ("the partner, 1 shows them fully); under the private boxes, the gate of each prompt. Scores compare the "
         "first turn with the target", MUTED),
        ("answer (✓ match, ✗ no match, a number: token F1).", MUTED)], size=24, step=36)
    crop(fig, ax, end - 120, top + 40)
    save(fig, "conversation")


# Result figures: free-generation scores (hero), the size of the writes over training (opening), the held-out gain on
# the partner's value over training (curves), and where a reader looks while writing the value (readmap).

LABEL = {
    "co_qa": "Answer a question about the partner's text", "co_hop": "Two-paragraph question, one paragraph each",
    "co_mcq": "Question on one side, options on the other", "co_nli": "Premise on one side, hypothesis on the other",
    "co_math": "Solve the partner's word problem", "co_code_out": "Say what the partner's code returns",
    "co_code_write": "Write code for the partner's problem", "co_translate": "Translate the partner's sentence",
    "co_paraphrase": "Paraphrase the partner's sentence", "co_instruct": "Carry out the partner's instruction",
    "co_talk": "Answer first, then talk", "co_transfer": "Write out the partner's passage",
    "co_swap": "Each writes out the other's passage", "co_split": "Each read half; both write the whole",
    "co_continue": "Continue the partner's passage", "co_distill": "Reply as if having read the partner's text",
    "relay_word": "Relay a word", "relay_number": "Relay a 4-digit number", "relay_phrase": "Relay a two-word phrase",
    "relay_digits": "Relay up to 16 digits", "merge_sum": "Add two numbers", "merge_fact": "Join a name and a city",
    "lookup": "Look up a code word",
    "pooled_coread": "Every reading and whole-text game but the talk", "pooled_payload": "Every value game"}
GROUPS = [("Read for the partner", ["co_qa", "co_hop", "co_mcq", "co_nli", "co_math", "co_code_out", "co_code_write",
                                    "co_translate", "co_paraphrase", "co_instruct", "co_talk"]),
          ("Pass a whole text across", ["co_transfer", "co_swap", "co_split", "co_continue", "co_distill"]),
          ("Pass a value across", ["relay_word", "relay_number", "relay_phrase", "relay_digits", "merge_sum",
                                   "merge_fact", "lookup"]),
          ("Pooled", ["pooled_coread", "pooled_payload"])]
VALUE_GAMES = GROUPS[2][1]
SHORT_GAME = {"relay_word": "words", "relay_number": "4-digit numbers", "relay_phrase": "phrases",
              "relay_digits": "digit strings", "merge_sum": "sums", "merge_fact": "name–city pairs",
              "lookup": "code words"}
RECEIVERS = (("quill", QUILL), ("ember", EMBER))
LN2 = math.log(2)


def way(x):
    """The direction in which the receiving model x reads, as 'AFM → Qwen'."""
    return f"{SHORT[partner_of(x)]} → {SHORT[x]}"


def minus(text):
    return text.replace("-", "−")


def span_ticks(lo, hi, n=5, integer=False):
    """Round tick values from the last one at or below lo to the first one at or above hi."""
    hi = hi if hi - lo > 1e-9 else lo + 1
    t = MaxNLocator(n, steps=[1, 2, 2.5, 5, 10], integer=integer).tick_values(lo, hi)
    a = max(v for v in t if v <= lo + 1e-12)
    b = min(v for v in t if v >= hi - 1e-12)
    return [float(v) for v in t if a - 1e-12 <= v <= b + 1e-12]


def step_ticks(last):
    """Training-step ticks from 0 that end at or before the last step, and the axis range (0, last)."""
    last = max(last, 1)
    return [v for v in span_ticks(0, last, 4, integer=True) if v <= last], (0, last)


def frame(ax, x0, y0, w, h, xt, yt, xr=None, size=22):
    """A plot area (lower left x0, y0; w by h canvas units) without gridlines: a baseline at y = 0 when 0 is a tick, the
    y tick labels to the left and the x tick labels below; xr: the x range when the ticks do not span the plot. Returns
    the maps from data to canvas units."""
    xr = xr or (xt[0], xt[-1])
    sx = lambda v: x0 + w * (v - xr[0]) / (xr[1] - xr[0])
    sy = lambda v: y0 + h * (v - yt[0]) / (yt[-1] - yt[0])
    for v in yt:
        if abs(v) < 1e-12:
            ax.plot([x0, x0 + w], [sy(v)] * 2, color=INK, lw=2, zorder=1)
        write(ax, x0 - 14, sy(v), minus(f"{v:g}"), size, color=MUTED, ha="right")
    for v in xt:
        write(ax, sx(v), y0 - 32, minus(f"{v:g}"), size, color=MUTED)
    return sx, sy


def caption(ax, y, text, size=22, step=34):
    """A muted paragraph across the canvas from y down; returns the y of the last line."""
    rows = wrap(ax, text, ax.get_xlim()[1] - 80, size)
    lines(ax, 40, y, [(r, MUTED) for r in rows], size, step)
    return y - step * (len(rows) - 1)


def sources(src):
    """[(path, label)] from 'PATH[|LABEL],PATH[|LABEL]...'; a label defaults to the name of the run's directory."""
    out = []
    for part in src.split(","):
        path, _, label = part.partition("|")
        p = Path(path)
        out.append((p, label or (p if p.is_dir() else p.parent).name))
    return out


def train_log(path):
    """The records of a training log: train.py's printed log (JSON step lines, and the 'payload step 0: {...}' line as
    a payload_eval record), mlx_train.py's printed log or log.jsonl (JSON lines), or either trainer's log.json."""
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        if line.startswith('{"step"'):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:   # the last line of a log still being written
                pass
        elif line.startswith("payload step "):
            step, _, rest = line[len("payload step "):].partition(": ")
            out.append(dict(step=int(step), payload_eval=json.loads(rest)))
    return out


def hero(src=None):
    """Free generation with the trained link (an evalfree.py summary; src: FILE.json, or FILE.json,all for every
    episode instead of the clean ones; default results/evalfree_headline.json): per game and direction, the receiver's
    mean score with the link minus with the link cut, and minus with a partner from another episode, with betting 95%
    intervals. Clean episodes (evalfree's exclusive flag): the value the receiver must write appears nowhere in the
    receiver's own prompt."""
    path, _, subset = (src or str(RES / "evalfree_headline.json")).partition(",")
    if src is None and not Path(path).exists():
        print("hero: no", path)
        return
    table = json.loads(Path(path).read_text())
    if table.get("tiny"):
        print("hero: tiny models")
    subset = subset or "exclusive"
    rows = {(r["game"], r["receiver"]): r for r in table["rows"] if r["subset"] == subset}
    every = {(r["game"], r["receiver"]): r["n"] for r in table["rows"] if r["subset"] == "all"}
    groups = [(title, [g for g in games if any(every.get((g, x)) for x, _ in RECEIVERS)]) for title, games in GROUPS]
    groups = [(title, games) for title, games in groups if games]
    cells = [r[c] for r in rows.values() if r["n"] and (r["game"], r["receiver"]) in every
             for c in ("link-cut", "link-shuffle")]
    lo = min([0.0] + [max(-1.0, min(c["ci"][0], c["mean"])) for c in cells])
    hi = max([0.0] + [min(1.0, max(c["ci"][1], c["mean"])) for c in cells])
    ticks = span_ticks(lo, hi, 4)
    fig, ax = canvas(4000, 2 * W)
    sym = {g for (g, _), r in rows.items() if r["n"] and r["symmetric_n"] and not g.startswith("pooled_")}
    label = lambda g: LABEL.get(g, g) + (" †" if g in sym else "")
    lab_end = 70 + max(text_width(ax, label(g), 26) for _, games in groups for g in games) + 40
    NW, G1, G2 = 110, 80, 70
    PW = (2 * W - 40 - lab_end - 2 * NW - 2 * G1 - G2) / 4
    base = lambda d: lab_end + d * (2 * PW + G1 + NW + G2)
    sxs = [[(lambda v, x0=base(d) + c * (PW + G1): x0 + PW * (v - ticks[0]) / (ticks[-1] - ticks[0]))
            for c in range(2)] for d in range(2)]
    top = 3960
    write(ax, 40, top, "Change in score from the link", 30, True, ha="left")
    write(ax, 40, top - 46, "mean over episodes; each score from 0 to 1", 24, color=MUTED, ha="left")
    for d, (x, colour) in enumerate(RECEIVERS):
        write(ax, base(d) + PW + G1 / 2, top, way(x), 30, True, colour)
        for c, name in enumerate(("link − cut", "link − shuffle")):
            write(ax, sxs[d][c](ticks[0]) + PW / 2, top - 46, name, 24, color=MUTED)
        write(ax, base(d) + 2 * PW + G1 + NW / 2, top - 46, "clean n" if subset == "exclusive" else "n", 22,
              color=MUTED)
    y = top - 120
    for title, games in groups:
        write(ax, 40, y, title, 26, True, MUTED, ha="left")
        y -= 54
        for g in games:
            write(ax, 70, y, label(g), 26, ha="left")
            for d, (x, _) in enumerate(RECEIVERS):
                r = rows.get((g, x))
                if not every.get((g, x)):
                    continue
                nx = base(d) + 2 * PW + G1 + NW / 2
                write(ax, nx, y, f"{r['n']}/{every[(g, x)]}" if r["n"] < every[(g, x)] else str(r["n"]), 22,
                      color=MUTED)
                if not r["n"]:
                    write(ax, base(d) + PW + G1 / 2, y, "no clean episode", 22, color=MUTED)
                    continue
                for c, control in enumerate(("cut", "shuffle")):
                    sx, cell = sxs[d][c], r["link-" + control]
                    if ticks[0] <= 0 <= ticks[-1]:
                        ax.plot([sx(0)] * 2, [y + 27, y - 27], color=INK, lw=2.5, zorder=2)
                    a, b = cell["ci"]
                    colour = LINK if a > 0 else INK if b < 0 else FAINT
                    if a <= b:
                        ax.plot([sx(max(a, ticks[0])), sx(min(b, ticks[-1]))], [y, y], color=colour, lw=3.5,
                                zorder=5, solid_capstyle="butt")
                    ax.scatter([sx(cell["mean"])], [y], s=300 if colour != FAINT else 150,
                               color=colour if colour != FAINT else MUTED, lw=2, edgecolor="white", zorder=6)
            y -= 54
        y -= 24
    for d in range(2):
        for c in range(2):
            sx = sxs[d][c]
            z = min(range(len(ticks)), key=lambda i: abs(ticks[i]))
            for i, v in enumerate(ticks):
                if len(ticks) <= 3 or (i - z) % 2 == 0:
                    write(ax, sx(v), y + 10, minus(f"{v:g}"), 22, color=MUTED)
            write(ax, sx(0) + 8, y - 30, "link better →", 22, color=LINK, ha="left")
    low = caption(ax, y - 100, "Greedy free generation on never-seen episodes. Dot: the mean difference in the "
                  "receiver's score; line: the 95% betting interval (Waudby-Smith and Ramdas 2024), green when the "
                  "interval lies above 0, black when below, grey when the interval holds 0. cut: the same pair with "
                  "the link cut; shuffle: the partner of another episode of the same game. "
                  + ("Clean episodes: the value the receiver must write appears nowhere in the receiver's own "
                     "prompt (clean n: clean of all). " if subset == "exclusive" else "")
                  + "Pooled rows weight the games by episodes and join per-game Bonferroni intervals."
                  + (" † The partner is also asked to write the value, so the link can carry the partner's own "
                     "written words." if sym else ""))
    crop(fig, ax, low - 40, top + 40)
    save(fig, "hero")


def opening(src=None):
    """How large each reader's writes grow over training (src: LOG[|LABEL],LOG[|LABEL]..., train.py or mlx_train.py
    logs, one column each): per exchange block, the RMS of the write over the RMS of the residual stream the write
    joins, on the training batches."""
    if not src:
        print("opening: give opening=LOG[|LABEL][,LOG[|LABEL]]")
        return
    runs = [(label, [r for r in train_log(p) if "write" in r]) for p, label in sources(src)]
    runs = [(label, recs) for label, recs in runs if recs]
    n = len(runs)
    L, R, G, PH = 130, 60, 150, 460
    PW = (2 * W - L - R - G * (n - 1)) / n
    yt = span_ticks(0, max(v for _, recs in runs for r in recs for x, _ in RECEIVERS for v in r["write"][x]), 4)
    ramp = LinearSegmentedColormap.from_list("blocks", [mix(LINK, 0.3), LINK, "#0B4A37"])
    shade = [ramp(k / (len(XBLOCKS) - 1)) for k in range(len(XBLOCKS))]
    fig, ax = canvas(2000, 2 * W)
    top = 1960
    write(ax, 40, top, "Size of each write, relative to the residual stream", 30, True, ha="left")
    # key: one shade per exchange block
    write(ax, 40, top - 56, "exchange after block", 22, color=MUTED, ha="left")
    kx = 40 + text_width(ax, "exchange after block", 22) + 30
    for k, b in enumerate(XBLOCKS):
        ax.plot([kx, kx + 36], [top - 56] * 2, color=shade[k], lw=5, solid_capstyle="butt")
        write(ax, kx + 44, top - 56, str(b), 22, color=MUTED, ha="left")
        kx += text_width(ax, str(b), 22) + 74
    for j, (label, recs) in enumerate(runs):
        x0 = L + j * (PW + G)
        xt, xr = step_ticks(max(r["step"] for r in recs))
        write(ax, x0, top - 150, label, 28, True, ha="left")
        for i, (x, colour) in enumerate(RECEIVERS):
            y0 = top - 240 - PH - i * (PH + 170)
            write(ax, x0, y0 + PH + 34, f"{SHORT[x]} reads {SHORT[partner_of(x)]}", 26, True, colour, ha="left")
            sx, sy = frame(ax, x0, y0, PW, PH, xt, yt, xr)
            steps = [sx(r["step"]) for r in recs]
            for k in range(len(XBLOCKS)):
                ax.plot(steps, [sy(min(r["write"][x][k], yt[-1])) for r in recs], color=shade[k], lw=2.2, zorder=3)
            if i == 1:
                write(ax, x0 + PW / 2, y0 - 74, "training step", 22, color=MUTED)
    low = top - 240 - 2 * PH - 170 - 130
    low = caption(ax, low, "Root mean square of each write over the reading positions of the training batches, "
                  "divided by the reader's residual RMS after that block (measured once before training), so 0.01 "
                  "adds 1% of the stream's size. Darker lines: later exchanges.")
    crop(fig, ax, low - 40, top + 40)
    save(fig, "opening")


def payload_points(path):
    """[(step, {receiver: (cut - link, shuffle - link or None)})], in bits per value token, each a mean over the
    value-game episodes evaluated at that step: from a run directory's payload_stepN.json files, or from the
    payload_eval records of a training log."""
    out = []
    if Path(path).is_dir():
        for f in Path(path).glob("payload_step*.json"):
            recs = [r for r in json.loads(f.read_text()) if r["game"] in VALUE_GAMES]
            point = {}
            for x, _ in RECEIVERS:
                rs = [r for r in recs if r["receiver"] == x]
                if rs:
                    point[x] = tuple(sum(r[c] - r["link_ce"] for r in rs) / len(rs) / LN2
                                     if all(c in r for r in rs) else None for c in ("cut_ce", "shuffle_ce"))
            out.append((int(f.stem[len("payload_step"):]), point))
    else:
        for rec in train_log(path):
            if "payload_eval" not in rec:
                continue
            point = {}
            for x, _ in RECEIVERS:
                vs = [v for k, v in rec["payload_eval"].items() if k.split("/")[0] in VALUE_GAMES
                      and k.split("/")[1] == x]
                n = sum(v[-1] for v in vs)
                if n:
                    point[x] = tuple(sum((v[c] - v[0]) * v[-1] for v in vs) / n / LN2
                                     if all(len(v) > c + 1 for v in vs) else None for c in (1, 2))
            out.append((rec["step"], point))
    return sorted(out, key=lambda p: p[0])


def curves(src=None):
    """The held-out gain on the partner's value over training (src: RUN_DIR|LOG[|LABEL],...; one panel each): the
    receiver's cross-entropy on the value tokens of never-seen value-game episodes with the link cut, and with a partner
    from another episode, minus with the link, in bits per token."""
    if not src:
        print("curves: give curves=RUN_DIR_OR_LOG[|LABEL][,...]")
        return
    runs = [(label, payload_points(p)) for p, label in sources(src)]
    runs = [(label, pts) for label, pts in runs if pts]
    vals = [v for _, pts in runs for _, point in pts for pair in point.values() for v in pair if v is not None]
    yt = span_ticks(min(vals + [0]), max(vals + [0]), 5)
    n = len(runs)
    L, R, G, PH = 130, 60, 150, 640
    PW = (2 * W - L - R - G * (n - 1)) / n
    fig, ax = canvas(1400, 2 * W)
    top = 1360
    write(ax, 40, top, "Bits per value token supplied by the link", 30, True, ha="left")
    # key: the colours name the direction, the dashes the control
    kx = 40
    for x, colour in RECEIVERS:
        write(ax, kx, top - 50, way(x), 24, True, colour, ha="left")
        kx += text_width(ax, way(x), 24, True) + 40
    for dash, name in (("-", "against the link cut"), ("--", "against a partner from another episode")):
        ax.plot([kx, kx + 50], [top - 50] * 2, color=INK, lw=2.5, ls=dash)
        write(ax, kx + 62, top - 50, name, 24, color=MUTED, ha="left")
        kx += 62 + text_width(ax, name, 24) + 40
    for j, (label, pts) in enumerate(runs):
        x0 = L + j * (PW + G)
        y0 = top - 180 - PH
        xt, xr = step_ticks(max(s for s, _ in pts))
        write(ax, x0, y0 + PH + 40, label, 26, True, ha="left")
        sx, sy = frame(ax, x0, y0, PW, PH, xt, yt, xr)
        for x, colour in RECEIVERS:
            for c, dash in ((0, "-"), (1, "--")):
                xy = [(sx(s), sy(point[x][c])) for s, point in pts if x in point and point[x][c] is not None]
                if xy:
                    ax.plot(*zip(*xy), color=colour, lw=2.5, ls=dash, marker="o", ms=5, zorder=4)
        write(ax, x0 + PW / 2, y0 - 74, "training step", 22, color=MUTED)
    low = top - 180 - PH - 140
    low = caption(ax, low, "Mean over the same never-seen episodes of the value games (relay, merge, lookup) at "
                  "every evaluation, teacher-forced on the target reply. Above 0: the link lowers the receiver's "
                  "cross-entropy on the partner's value.")
    crop(fig, ax, low - 40, top + 40)
    save(fig, "curves")


def flow(ax, x0, y0, width, pieces, fills=None, colours=None, bolds=None, size=24, line=48):
    """Token pieces as running text from (x0, y0) down, wrapped at width; each piece on a box of the given fill (None:
    no box). Returns the y of the last line."""
    x, y = x0, y0
    for k, p in enumerate(pieces):
        bold = bool(bolds and bolds[k])
        p = "".join(c for c in p.replace("\t", " ").replace("$", r"\$") if c == "\n" or c.isprintable())
        for n, part in enumerate(p.split("\n")):
            if n:
                x, y = x0, y - line
            if not part:
                continue
            w = text_width(ax, part, size, bold)
            if x + w > x0 + width and x > x0:
                x, y = x0, y - line
                part = part.lstrip()
                w = text_width(ax, part, size, bold)
            fill = fills[k] if fills else None
            if fill is not None:
                rbox(ax, x - 1, y - line / 2 + 5, w + 2, line - 10, fill, radius=6, z=2)
            write(ax, x, y, part, size, bold, colours[k] if colours else INK, ha="left")
            x += w
    return y


def user_span(pieces):
    """Indices of the user message's pieces in a ChatML prompt."""
    text, starts = "", []
    for p in pieces:
        starts.append(len(text))
        text += p
    a = text.find("\n", text.find("<|im_start|>user")) + 1
    b = text.find("<|im_end|>", a)
    return [j for j, s in enumerate(starts) if a <= s < b]


def value_span(r):
    """Indices of the partner's prompt pieces that spell the value the reader writes (a readmap.py record)."""
    value = "".join(r["receiver_written"]).strip()
    pieces, prompt = r["partner_pieces"], [j for j, t in enumerate(r["partner_ticks"]) if t < 0]
    for a in prompt:
        got = ""
        for b in range(a, len(pieces)):
            got += pieces[b]
            if got.strip() == value:
                return list(range(a, b + 1))
            if not value.startswith(got.strip()):
                break
    return []


def readmap(src=None, show=(0, 3, 6), deep=("18", "21", "24", "27")):
    """What the reader takes in from its partner while it writes the partner's value (a readmap.py JSON file; src:
    FILE.json). The gated read of a partner token is the reader's attention on the token times the token's visibility
    gate; its share is that over the sum across the partner's tokens. Top: the sender's user message in the episodes
    show (indices into the file), each token shaded by its share averaged over the deep exchanges (full colour at 50%
    or more), and the value the reader writes. Bottom: per exchange, the percent of the gated read that comes from the
    value's own tokens, one dot per episode and the median."""
    path = src or str(RES / "readmap.json")
    if not Path(path).exists():
        print("readmap: no", path)
        return
    recs = json.loads(Path(path).read_text())
    blocks = sorted(recs[0]["attn"], key=int)
    share = {b: [] for b in blocks}
    for r in recs:
        span = value_span(r)
        for b in blocks:
            g = r["partner_gate"][b]
            for i in range(len(r["receiver_positions"])):
                a = r["attn"][b][i]
                tot = sum(a[j] * g[j] for j in range(len(a)))
                if span and tot:
                    share[b].append(100 * sum(a[j] * g[j] for j in span) / tot)
    width = 2 * W
    fig, ax = canvas(4000, width)
    top = 3960
    write(ax, 40, top, "Nearly everything the reader takes in at blocks 18, 24 and 27 comes from the secret word",
          34, True, ha="left")
    y = top - 110
    for k in show:
        r = recs[k]
        x, s_ = r["receiver"], partner_of(r["receiver"])
        user = user_span(r["partner_pieces"])
        span = set(value_span(r))
        gated = {b: [r["attn"][b][0][j] * r["partner_gate"][b][j] for j in range(len(r["partner_pieces"]))]
                 for b in deep}
        gate = [min(1.0, 2 * sum(gated[b][j] / max(sum(gated[b]), 1e-12) for b in deep) / len(deep)) for j in user]
        write(ax, 40, y, f"{SHORT[s_]} \u2192 {SHORT[x]}", 30, True, NAMES[s_][2], ha="left")
        word = "".join(r["receiver_written"]).strip()
        write(ax, 1880, y, f"{SHORT[x]} writes", 26, color=MUTED, ha="left")
        write(ax, 1880, y - 52, word, 34, True, NAMES[x][2], ha="left")
        y = flow(ax, 70, y - 64, 1700, [r["partner_pieces"][j] for j in user],
                 fills=[mix(LINK, 0.95 * g) for g in gate], colours=["white" if g > 0.55 else INK for g in gate],
                 bolds=[j in span for j in user], size=30, line=56)
        y -= 110
    # the colour key of the gate
    kx = 70
    key = "percent of what the reader takes in, averaged over the exchanges after blocks 18\u201327:"
    write(ax, kx, y, key, 26, color=MUTED, ha="left")
    gx = kx + text_width(ax, key, 26) + 24
    for i in range(50):
        ax.add_patch(Rectangle((gx + 8 * i, y - 16), 8, 32, color=mix(LINK, 0.95 * i / 49), lw=0, zorder=3))
    write(ax, gx - 6, y - 44, "0%", 22, color=MUTED, ha="left")
    write(ax, gx + 400, y - 44, "50% or more", 22, color=MUTED, ha="right")
    # per exchange: the percent of the gated read that comes from the value's tokens
    L, PW, PH = 150, 1700, 700
    y0 = y - 180 - PH
    write(ax, L - 110, y0 + PH + 70, "percent of what the reader takes in (attention \u00d7 gate) that comes from the "
          "secret word's own tokens", 28, color=MUTED, ha="left")
    xs = [int(b) for b in blocks]
    sx, sy = frame(ax, L, y0, PW, PH, xs, [0, 25, 50, 75, 100], (xs[0] - 1.5, xs[-1] + 1.5), size=26)
    for b in blocks:
        vs = share[b]
        ax.scatter([sx(int(b) + 0.9 * (((k * 0.618) % 1) - 0.5)) for k in range(len(vs))], [sy(v) for v in vs],
                   s=170, color=LINK, alpha=0.6, lw=1.5, edgecolor="white", zorder=4)
        if vs:
            m = sorted(vs)[len(vs) // 2] if len(vs) % 2 else sum(sorted(vs)[len(vs) // 2 - 1:len(vs) // 2 + 1]) / 2
            ax.plot([sx(int(b) - 0.8), sx(int(b) + 0.8)], [sy(m)] * 2, color=INK, lw=4, zorder=5)
    write(ax, sx(xs[-1] + 1.5) + 20, sy(sorted(share[blocks[-1]])[len(share[blocks[-1]]) // 2]),
          f"median of {len(recs)} episodes", 24, True, INK, ha="left")
    write(ax, L + PW / 2, y0 - 84, "exchange after block", 28, color=MUTED)
    crop(fig, ax, y0 - 120, top + 40)
    save(fig, "readmap")


def transfer(src=None):
    """Every never-seen episode of every value game at one checkpoint (src: RUN_DIR[@STEP][|LABEL]; default the
    run's last payload_stepN.json): the receiver's cross-entropy on the value tokens with a partner from another
    episode minus with its real partner, in bits per token, one dot per episode, with the count of episodes the real
    partner wins and the exact one-sided sign test. The title lists the games that pass in both directions (p < 0.05)."""
    if not src:
        print("transfer: give transfer=RUN_DIR[@STEP][|LABEL]")
        return
    (path, label), = sources(src)
    path, _, at = str(path).partition("@")
    files = {int(f.stem[len("payload_step"):]): f for f in Path(path).glob("payload_step*.json")}
    step = int(at) if at else max(files)
    recs = [r for r in json.loads(files[step].read_text()) if r["game"] in VALUE_GAMES]
    d = {(g, x): [(r["shuffle_ce"] - r["link_ce"]) / LN2 for r in recs if r["game"] == g and r["receiver"] == x]
         for g in VALUE_GAMES for x, _ in RECEIVERS}
    games = [g for g in VALUE_GAMES if any(d[(g, x)] for x, _ in RECEIVERS)]

    def test(vs):
        wins, n = sum(v > 0 for v in vs), sum(v != 0 for v in vs)
        return wins, n, (binom.sf(wins - 1, n, 0.5) if n else 1.0)
    both = [g for g in games if all(d[(g, x)] and test(d[(g, x)])[2] < 0.05 for x, _ in RECEIVERS)]
    vals = [v for vs in d.values() for v in vs]
    ticks = span_ticks(min(vals + [0.0]), max(vals + [0.0]), 5)
    RH = 110
    fig, ax = canvas(420 + RH * len(games), 2 * W)
    lab_end = 60 + max(text_width(ax, LABEL[g], 32) for g in games) + 40
    NW, G1 = 270, 50
    SW = (2 * W - 30 - lab_end - 2 * NW - G1) / 2
    base = lambda c: lab_end + c * (SW + NW + G1)
    sxs = [(lambda v, x0=base(c): x0 + SW * (v - ticks[0]) / (ticks[-1] - ticks[0])) for c in range(2)]
    top = 320 + RH * len(games)
    names = [SHORT_GAME.get(g, LABEL[g].lower()) for g in both]
    claim = (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) if names else "no value"
    title = f"Never-seen {claim} pass between the models through hidden states"
    write(ax, 40, top, title, min(36, 36 * (2 * W - 80) / text_width(ax, title, 36, True)), True, ha="left")
    for c, (x, colour) in enumerate(RECEIVERS):
        write(ax, base(c) + SW / 2, top - 90, way(x), 34, True, colour)
        write(ax, base(c) + SW + NW / 2, top - 90, "real partner wins", 26, color=MUTED)
    y = top - 180
    for g in games:
        write(ax, 60, y, LABEL[g], 32, ha="left")
        for c, (x, colour) in enumerate(RECEIVERS):
            vs, sx = d[(g, x)], sxs[c]
            if ticks[0] <= 0 <= ticks[-1]:
                ax.plot([sx(0)] * 2, [y + 40, y - 40], color=INK, lw=3, zorder=2)
            if not vs:
                continue
            ax.scatter([sx(v) for v in vs], [y + 26 * (((k * 0.618) % 1) - 0.5) for k in range(len(vs))], s=170,
                       color=colour, alpha=0.75, lw=1.5, edgecolor="white", zorder=4)
            srt = sorted(vs)
            med = srt[len(vs) // 2] if len(vs) % 2 else (srt[len(vs) // 2 - 1] + srt[len(vs) // 2]) / 2
            ax.plot([sx(med)] * 2, [y + 36, y - 36], color=INK, lw=5, zorder=5, solid_capstyle="butt")
            wins, n, p = test(vs)
            strong = p < 0.05
            write(ax, base(c) + SW + NW / 2, y + 14, f"{wins} of {n}", 32, strong, LINK if strong else INK)
            write(ax, base(c) + SW + NW / 2, y - 26, f"p = {p:.2g}", 24, color=MUTED)
        y -= RH
    for c in range(2):
        for v in ticks:
            write(ax, sxs[c](v), y + 46, minus(f"{v:g}"), 26, color=MUTED)
    write(ax, (sxs[0](ticks[0]) + sxs[1](ticks[-1])) / 2, y - 14, "cross-entropy on the value with another "
          "episode's partner minus with the real partner (bits per token)", 28, color=MUTED)
    crop(fig, ax, y - 60, top + 40)
    save(fig, "transfer")


def opened(src=None, games=("relay_word", "merge_fact"), title=None):
    """Over training, the percent of never-seen episodes (both receivers) in which the real partner beats a partner
    from another episode, with the exact 95% interval: one panel per game (default: relay a word, the trained task, and
    join a name and a city, never trained), one line per run (src: RUN_DIR|LABEL,...; payload_stepN.json files), each
    labelled at its own end. Ties (a link that writes exactly nothing, as at step 0 with a zero output projection) are
    left out; a step with only ties sits at 50%."""
    if not src:
        print("opened: give opened=RUN_DIR|LABEL[,...]")
        return
    from scipy.stats import beta
    runs = []
    for path, label in sources(src):
        per = {}
        for f in Path(path).glob("payload_step*.json"):
            st, rs = int(f.stem[len("payload_step"):]), json.loads(f.read_text())
            for g in games:
                ds = [r["shuffle_ce"] - r["link_ce"] for r in rs if r["game"] == g]
                k, n = sum(v > 0 for v in ds), sum(v != 0 for v in ds)      # a tie (an exactly closed link) is neither
                if n:
                    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
                    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
                    per.setdefault(g, []).append((st, 100 * k / n, 100 * lo, 100 * hi))
                elif ds:
                    per.setdefault(g, []).append((st, 50.0, 50.0, 50.0))
        if per:
            runs.append((label, {g: sorted(v) for g, v in per.items()}))
    colours = [LINK, QUILL, EMBER, MUTED, INK, "#2F6FDE"]
    last = max(st for _, per in runs for pts in per.values() for st, *_ in pts)
    xt, xr = step_ticks(last)
    L, PW, PH, LW = 150, 1000, 900, 420                    # left margin, panel width and height, label column
    width = L + len(games) * (PW + LW)
    fig, ax = canvas(1300, width)
    top = 1240
    write(ax, 40, top, title or "Never-seen values pass through the link from both starts", 34, True, ha="left")
    y0 = top - 160 - PH
    for i, g in enumerate(games):
        x0 = L + i * (PW + LW)
        sx, sy = frame(ax, x0, y0, PW, PH, xt, [0, 25, 50, 75, 100], xr, size=26)
        ax.plot([sx(0), sx(last)], [sy(50)] * 2, color=MUTED, lw=2, ls=(0, (6, 6)), zorder=2)
        name = SHORT_GAME.get(g, g) + (" (task never trained)" if g == "merge_fact" else "")
        write(ax, x0 - 110, y0 + PH + 70, f"percent of never-seen {name} where the real partner wins", 28,
              color=MUTED, ha="left")
        ends = []
        for (label, per), colour in zip(runs, colours):
            pts = per.get(g)
            if not pts:
                continue
            xs = [sx(st) for st, *_ in pts]
            ax.fill_between(xs, [sy(lo) for *_, lo, _ in pts], [sy(hi) for *_, hi in pts], color=colour, alpha=0.12,
                            lw=0, zorder=2)
            ax.plot(xs, [sy(v) for _, v, _, _ in pts], color=colour, lw=4, marker="o", ms=9, zorder=4)
            ends.append([sy(pts[-1][1]), xs[-1], label, colour])
        ends.append([sy(50), sx(last), "no information", MUTED])
        ends.sort(key=lambda e: -e[0])
        for j in range(1, len(ends)):                      # labels at least 44 units apart
            ends[j][0] = min(ends[j][0], ends[j - 1][0] - 44)
        for yy, xx, label, colour in ends:                 # each at the end of its own line
            write(ax, xx + 16, yy, label, 28, colour != MUTED, colour, ha="left")
        write(ax, x0 + PW / 2, y0 - 84, "training step", 28, color=MUTED)
    crop(fig, ax, y0 - 120, top + 40)
    save(fig, "opened")


FIGURES = {"method": method, "probe": probe, "conversation": conversation, "hero": hero, "opening": opening,
           "curves": curves, "readmap": readmap, "transfer": transfer, "opened": opened}

if __name__ == "__main__":
    style()
    for arg in sys.argv[1:] or FIGURES:
        name, _, src = arg.partition("=")
        FIGURES[name](*([src] if src else []))
