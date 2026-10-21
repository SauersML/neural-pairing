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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
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


# Text with bold and highlighted words, wrapped to a width.

def strip(ax, x, y, w, h, light, dark, marks=(), n=36):
    """A token's residual stream as a column of n layer lines (layer 0 at the bottom); marks: (layer, colour)."""
    rbox(ax, x, y, w, h, light, radius=12, z=2)
    for k in range(1, n):
        yy = y + h * k / n
        ax.plot([x + 10, x + w - 10], [yy, yy], color="white", lw=1.2, zorder=3)
    for layer, colour in marks:
        yy = y + h * (layer + 0.5) / n
        rbox(ax, x + 4, yy - h / n * 0.9, w - 8, 2 * h / n * 0.9, colour, radius=4, z=4)


def lane_token(ax, x, y, text, fill, colour, w=86, h=58, size=24, bold=False):
    rbox(ax, x, y - h / 2, w, h, fill, radius=12, z=3)
    write(ax, x + w / 2, y, text, size, bold, colour)


def method():
    """How a message becomes a slot in the partner's input (left), and the two ways the pair runs (right)."""
    fig, ax = canvas(1300, 2 * W)
    # left: one message from Quill's token t to a slot in Ember's input before Ember's token t + 1
    top = 1250
    write(ax, 40, top - 20, "Quill · Qwen3-4B", 34, True, QUILL, ha="left")
    sw, sh = 120, 420
    qx, qy = 120, top - 540
    strip(ax, qx, qy, sw, sh, QUILL_LIGHT, QUILL, marks=((35, QUILL), (24, QUILL)))
    write(ax, qx + sw / 2, qy - 36, "token t", 26, color=MUTED)
    ly = lambda y0, layer: y0 + sh * (layer + 0.5) / 36
    write(ax, qx - 16, ly(qy, 35), "36", 24, True, QUILL, ha="right")
    write(ax, qx - 16, ly(qy, 24), "24", 24, True, QUILL, ha="right")
    # last layer -> next-token probabilities (the text part)
    y0 = ly(qy, 35)
    px = qx + sw + 70
    arrow(ax, (qx + sw, y0), (px - 8, y0), INK, 3)
    for k, hgt in enumerate((52, 18, 34, 10, 24, 8, 14, 6)):
        ax.add_patch(Rectangle((px + k * 26, y0 - 26), 20, hgt, color=QUILL, zorder=4))
    write(ax, px + 104, y0 + 54, "next-token probabilities", 26, True)
    # layer 24 -> send head -> message -> gate (the learned part)
    y1 = ly(qy, 24)
    bx = qx + sw + 250
    rbox(ax, bx, y1 - 45, 210, 90, QUILL, radius=16, z=3)
    write(ax, bx + 105, y1, "send head", 28, True, "white")
    arrow(ax, (qx + sw, y1), (bx, y1), INK, 3)
    mx = bx + 260
    for k in range(16):
        c = plt.cm.PuOr(0.15 + 0.7 * ((k * 7) % 16) / 15)
        ax.add_patch(Rectangle((mx + k * 14, y1 - 28), 12, 56, color=c, zorder=4))
    write(ax, mx + 112, y1 - 62, "256 numbers", 26, True)
    arrow(ax, (bx + 210, y1), (mx - 6, y1), INK, 3)
    gx = mx + 260
    ax.add_patch(Circle((gx + 34, y1), 34, color=LINK, zorder=4))
    write(ax, gx + 34, y1, "on", 24, True, "white")
    write(ax, gx + 34, y1 + 62, "gate", 26, True)
    arrow(ax, (mx + 228, y1), (gx, y1), INK, 3)
    # both parts -> one slot vector
    sx0, sy = 500, top - 800
    sbw = text_width(ax, "slot", 30, True) + 80
    rbox(ax, sx0, sy - 45, sbw, 90, LINK, radius=16, z=3)
    write(ax, sx0 + sbw / 2, sy, "slot", 30, True, "white")
    ax.add_patch(FancyArrowPatch((px + 104, y0 - 30), (sx0 + 30, sy + 45),
                                 arrowstyle="-|>,head_length=1.1,head_width=0.65", connectionstyle="arc3,rad=0.1",
                                 color=INK, lw=3, zorder=4, mutation_scale=20))
    ax.add_patch(FancyArrowPatch((gx + 34, y1 - 34), (sx0 + sbw - 30, sy + 45),
                                 arrowstyle="-|>,head_length=1.1,head_width=0.65", connectionstyle="arc3,rad=-0.2",
                                 color=INK, lw=3, zorder=4, mutation_scale=20))
    write(ax, sx0 + sbw + 24, sy + 18, "mix of the embeddings of", 24, color=MUTED, ha="left")
    write(ax, sx0 + sbw + 24, sy - 16, "tokens in both vocabularies", 24, color=MUTED, ha="left")
    write(ax, sx0 + sbw + 24, sy - 50, "+ learned vector", 24, color=MUTED, ha="left")
    # Ember's input: the slot is an extra position before Ember's token t + 1
    ry = top - 990
    seq = [("The", False), ("slot", True), ("secret", False), ("slot", True), ("word", False)]
    x = 40
    xs = []
    for text, slot in seq:
        w = text_width(ax, text, 26, slot) + 44
        lane_token(ax, x, ry, text, LINK if slot else EMBER_LIGHT, "white" if slot else EMBER, w=w, h=62, size=26,
                   bold=slot)
        xs.append((x, w))
        x += w + 16
    arrow(ax, (sx0 + sbw / 2, sy - 45), (xs[3][0] + xs[3][1] / 2, ry + 31), INK, 3)
    write(ax, 40, ry - 80, "Ember · AFM-4.5B input", 34, True, EMBER, ha="left")
    write(ax, 40, ry - 130, "later Ember tokens attend to each slot", 24, color=MUTED, ha="left")
    # right: duplex and dialogue lanes
    ox = W + 60
    write(ax, ox, top - 20, "Both write at once", 34, True, ha="left")
    write(ax, ox, top - 64, "neither reads the other's text", 26, color=MUTED, ha="left")
    q = ["Sky", " looks", " blue", " because", " air", " scatters"]
    e = ["The", " secret", " word", " is", " lan", "tern"]
    lx0, step = ox + 140, 160
    qy2, ey2 = top - 170, top - 380
    for k in range(6):
        lane_token(ax, lx0 + k * step, qy2, q[k].strip(), QUILL_LIGHT, QUILL, w=130)
        lane_token(ax, lx0 + k * step, ey2, e[k].strip(), EMBER_LIGHT, EMBER, w=130)
    for k in range(5):
        on = k in (0, 1, 3)
        c = LINK if on else "#D8D8DC"
        arrow(ax, (lx0 + k * step + 100, ey2 + 30), (lx0 + (k + 1) * step + 30, qy2 - 30), c, 3 if on else 2, head=16)
        on2 = k in (1, 2)
        c2 = LINK if on2 else "#D8D8DC"
        arrow(ax, (lx0 + k * step + 100, qy2 - 30), (lx0 + (k + 1) * step + 30, ey2 + 30), c2, 3 if on2 else 2, head=16)
    write(ax, ox, qy2, "Quill", 28, True, QUILL, ha="left")
    write(ax, ox, ey2, "Ember", 28, True, EMBER, ha="left")
    write(ax, ox, top - 560, "Taking turns in text", 34, True, ha="left")
    write(ax, ox, top - 604, "the listener reads the speaker's text as it streams", 26, color=MUTED, ha="left")
    qy3, ey3 = top - 710, top - 920
    spk = ["Do", " you", " like", " rain", "?", ""]
    lis = ["Do", " you", " like", " rain", "?", "I"]
    for k in range(6):
        if k < 5:
            lane_token(ax, lx0 + k * step, qy3, spk[k].strip(), QUILL_LIGHT, QUILL, w=130)
            lane_token(ax, lx0 + k * step, ey3, lis[k].strip(), PANEL, MUTED, w=130)
        else:
            lane_token(ax, lx0 + k * step, ey3, "I", EMBER_LIGHT, EMBER, w=130)
    for k in range(5):
        arrow(ax, (lx0 + k * step + 100, qy3 - 30), (lx0 + (k + 1) * step + 30, ey3 + 30),
              LINK if k in (0, 2, 3) else "#D8D8DC", 3 if k in (0, 2, 3) else 2, head=16)
        arrow(ax, (lx0 + k * step + 100, ey3 + 30), (lx0 + (k + 1) * step + 30, qy3 - 30),
              LINK if k in (1, 3) else "#D8D8DC", 3 if k in (1, 3) else 2, head=16)
    write(ax, ox, qy3, "Quill", 28, True, QUILL, ha="left")
    write(ax, ox, ey3, "Ember", 28, True, EMBER, ha="left")
    legend_y = top - 1060
    x = ox
    for label, colour in (("message sent (gate on)", LINK), ("gate off", "#D8D8DC")):
        ax.plot([x, x + 50], [legend_y, legend_y], color=colour, lw=4)
        write(ax, x + 64, legend_y, label, 26, True, ha="left")
        x += 64 + text_width(ax, label, 26, True) + 60
    crop(fig, ax, top - 1160, top + 20)
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
