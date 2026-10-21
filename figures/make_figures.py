"""Draw the figures in figures/: method.png (the link) and probe.png (link minus shuffled messages, from
results/probe.json). Usage: python make_figures.py [figure ...] (default: every figure)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
    for v in ticks:
        ax.plot([sx(v)] * 2, [top, y + 26], color=INK if v == 0 else TICK, lw=2.5 if v == 0 else 1.5,
                zorder=2 if v == 0 else 1)
        write(ax, sx(v), y, minus(f"{v:.0f}"), 26, color=MUTED)
    write(ax, sx(0), y - 44, "\u2190 the link carries the value", 24, color=LINK, ha="right")
    write(ax, 40, y - 100, "dot: median over held-out episodes; line: exact 95% interval", 24, color=MUTED, ha="left")
    crop(fig, ax, y - 130, 2100)
    save(fig, "probe")

def partner_of(x):
    return "ember" if x == "quill" else "quill"


FIGURES = {"method": method, "probe": probe}

if __name__ == "__main__":
    style()
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
