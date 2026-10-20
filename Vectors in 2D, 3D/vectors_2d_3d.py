"""
ZMathLab · Linear Algebra · Vectors in 2-Space, 3-Space and n-Space
=====================================================================
A complete, visual companion to Anton (Elementary Linear Algebra), Section 3.1.

Run:      python vectors_2d_3d.py
Needs:    numpy, matplotlib            (pip install numpy matplotlib)
Output:   a folder  "vector_figures/"  with one clear, high-resolution PNG per idea,
          plus every Exercise Set 3.1 answer printed and checked in the terminal.

Contents
  0. Helper functions (drawing arrows, grids, 3-D axes)
  1. The story: "50 steps" versus "50 steps north-east"
  2. Geometric vectors: tail, tip, equivalent vectors
  3. Addition: triangle rule, parallelogram rule, commutativity
  4. Subtraction and the negative of a vector
  5. Scalar multiplication and parallel vectors
  6. Sums of three vectors, and the box diagonal in 3-D
  7. Components in 2-D and 3-D
  8. Vector between two points (end minus start), Anton Example 1
  9. Componentwise operations, Anton Example 3
 10. Theorem 3.1.1 checked numerically
 11. Linear combinations
 12. The RGB colour cube
 13. Misconceptions, drawn
 14. Exercise Set 3.1, fully computed (same numbers as the book)
"""
import os
from fractions import Fraction as F

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

# ---------------------------------------------------------------- style
OUT = "vector_figures"
os.makedirs(OUT, exist_ok=True)
NAVY, GOLD = "#0B1832", "#E0A82E"
BLUE, ORANGE, GREEN, RED, PURPLE, TEAL = "#2F7FD8", "#F07A2E", "#2EA866", "#D64545", "#8A5CD6", "#1FA59A"
GRID = "#C9D6EA"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "figure.dpi": 110, "savefig.dpi": 200, "font.family": "DejaVu Sans"})


# ================================================================ 0. helpers
def arrow(ax, start, vec, color, label=None, lw=2.6, lpos=0.55, off=(0.12, 0.12), ls="-", fs=14):
    """Draw vector `vec` starting at `start` in a 2-D axis, with an optional label."""
    s = np.asarray(start, float); v = np.asarray(vec, float)
    ax.annotate("", xy=s + v, xytext=s,
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.4", color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0))
    if label:
        p = s + lpos * v + np.asarray(off)
        ax.text(p[0], p[1], label, color=color, fontsize=fs, fontweight="bold", ha="center", va="center")


def grid2d(ax, xr, yr, title=None, axes=True):
    """Square grid with light lines and optional x/y axes."""
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal")
    ax.set_xticks(range(int(np.ceil(xr[0])), int(xr[1]) + 1))
    ax.set_yticks(range(int(np.ceil(yr[0])), int(yr[1]) + 1))
    ax.grid(color=GRID, lw=0.8); ax.tick_params(labelsize=9, colors="#5B6B85")
    for s in ax.spines.values():
        s.set_visible(False)
    if axes:
        ax.axhline(0, color="#5B6B85", lw=1.1); ax.axvline(0, color="#5B6B85", lw=1.1)
    if title:
        ax.set_title(title, color=NAVY)


def arrow3d(ax, start, vec, color, label=None, lw=3):
    s = np.asarray(start, float); v = np.asarray(vec, float)
    ax.quiver(*s, *v, color=color, lw=lw, arrow_length_ratio=0.12)
    if label:
        p = s + 1.05 * v
        ax.text(*p, label, color=color, fontsize=13, fontweight="bold")


def axes3d(ax, lim=4, title=None):
    for i, (c, n) in enumerate(zip(["#5B6B85"] * 3, "xyz")):
        e = np.zeros(3); e[i] = lim
        ax.quiver(0, 0, 0, *e, color=c, lw=1, arrow_length_ratio=0.05)
        ax.text(*(e * 1.08), n, fontsize=13, color=c)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_zlim(0, lim)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    if title:
        ax.set_title(title, color=NAVY)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  saved", path)


# ================================================================ 1. story
def fig_story():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    t = np.linspace(0, 2 * np.pi, 400)
    for ax in axs:
        grid2d(ax, (-6, 6), (-6, 6), axes=False)
        ax.plot(0, 0, "o", color=NAVY, ms=9); ax.text(0, -0.7, "Bilal", ha="center", fontsize=12, fontweight="bold")
    ax = axs[0]
    ax.plot(5 * np.cos(t), 5 * np.sin(t), "--", color=RED, lw=2)
    for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot(5 * np.cos(a), 5 * np.sin(a), "o", color=RED, ms=6)
    ax.set_title('"She is 50 steps away"\nA number only: she could be ANYWHERE on the circle', color=RED)
    ax = axs[1]
    ax.plot(5 * np.cos(t), 5 * np.sin(t), "--", color="#bbb", lw=1)
    v = 5 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    arrow(ax, (0, 0), v, GOLD, lw=4)
    ax.plot(*v, "o", color=GREEN, ms=12); ax.text(v[0] + 0.3, v[1] + 0.3, "Hania", color=GREEN, fontsize=14, fontweight="bold")
    ax.set_title('"50 steps, NORTH-EAST"\nNumber + direction = exactly ONE place', color=GREEN)
    fig.suptitle("Why vectors?  A scalar says HOW MUCH. A vector says how much AND WHICH WAY.", fontsize=15, color=NAVY, fontweight="bold")
    save(fig, "01_story_scalar_vs_vector.png")


# ================================================================ 2. geometric vectors
def fig_geometric():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axs[0]; grid2d(ax, (-1, 6), (-1, 4), "An arrow: tail (initial point) and tip (terminal point)", axes=False)
    arrow(ax, (0.5, 0.5), (4, 2.4), BLUE, r"$\mathbf{v}=\overrightarrow{AB}$", lw=4, off=(-0.3, 0.45))
    ax.plot(0.5, 0.5, "o", color=NAVY, ms=9); ax.text(0.5, 0.0, "A  (initial point)", fontsize=12, ha="left")
    ax.text(4.5, 3.3, "B  (terminal point)", fontsize=12, ha="center")
    ax = axs[1]; grid2d(ax, (0, 10), (0, 5), "Equivalent vectors: same length + same direction = SAME vector", axes=False)
    for s in [(0.5, 0.5), (3, 2), (5.5, 0.4), (7, 2.6)]:
        arrow(ax, s, (2, 1.3), BLUE, lw=3)
    save(fig, "02_geometric_and_equivalent_vectors.png")


# ================================================================ 3. addition
def fig_addition():
    v, w = np.array([4, 1]), np.array([1, 3])
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    ax = axs[0]; grid2d(ax, (-0.5, 6), (-0.5, 5), "(a) Triangle rule: tip to tail")
    arrow(ax, (0, 0), v, BLUE, "v", off=(0, -0.4)); arrow(ax, v, w, ORANGE, "w", off=(0.35, 0))
    arrow(ax, (0, 0), v + w, GOLD, "v + w", lw=3.5, off=(-0.6, 0.35))
    ax = axs[1]; grid2d(ax, (-0.5, 6), (-0.5, 5), "(b) Parallelogram rule: common tail")
    arrow(ax, (0, 0), v, BLUE, "v", off=(0, -0.4)); arrow(ax, (0, 0), w, ORANGE, "w", off=(-0.4, 0))
    ax.plot([w[0], (v + w)[0]], [w[1], (v + w)[1]], "--", color=BLUE, alpha=0.6)
    ax.plot([v[0], (v + w)[0]], [v[1], (v + w)[1]], "--", color=ORANGE, alpha=0.6)
    arrow(ax, (0, 0), v + w, GOLD, "v + w", lw=3.5, off=(-0.6, 0.35))
    ax = axs[2]; grid2d(ax, (-0.5, 6), (-0.5, 5), "(c) v + w = w + v  (both roads, one corner)")
    arrow(ax, (0, 0), v, BLUE, "v", off=(0, -0.4)); arrow(ax, v, w, ORANGE, "w", off=(0.35, 0))
    arrow(ax, (0, 0), w, ORANGE, "w", off=(-0.4, 0)); arrow(ax, w, v, BLUE, "v", off=(0, 0.35))
    arrow(ax, (0, 0), v + w, GOLD, lw=3.5)
    save(fig, "03_addition_triangle_parallelogram.png")


# ================================================================ 4. subtraction
def fig_subtraction():
    v, w = np.array([4, 1]), np.array([1, 3])
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    ax = axs[0]; grid2d(ax, (-5, 5), (-4, 4), "(a) v and its negative -v")
    arrow(ax, (0, 0), v, BLUE, "v", off=(0, 0.4)); arrow(ax, (0, 0), -v, RED, "-v", off=(0, -0.4))
    ax = axs[1]; grid2d(ax, (-5, 5), (-4, 4), "(b) w - v = w + (-v)")
    arrow(ax, (0, 0), w, ORANGE, "w", off=(0.4, 0)); arrow(ax, (0, 0), -v, RED, "-v", off=(0, -0.4))
    ax.plot([w[0], (w - v)[0]], [w[1], (w - v)[1]], "--", color=RED, alpha=0.6)
    ax.plot([-v[0], (w - v)[0]], [-v[1], (w - v)[1]], "--", color=ORANGE, alpha=0.6)
    arrow(ax, (0, 0), w - v, PURPLE, "w - v", lw=3.5, off=(-0.6, 0.2))
    ax = axs[2]; grid2d(ax, (-1, 6), (-1, 5), "(c) Quick way: from tip of v to tip of w")
    arrow(ax, (0, 0), v, BLUE, "v", off=(0, -0.4)); arrow(ax, (0, 0), w, ORANGE, "w", off=(-0.4, 0))
    arrow(ax, v, w - v, PURPLE, "w - v", lw=3.5, off=(0.6, 0.3))
    save(fig, "04_subtraction.png")


# ================================================================ 5. scalar multiples
def fig_scalar():
    v = np.array([2, 1])
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    ax = axs[0]; grid2d(ax, (-7, 7), (-4, 4), "Scalar multiples kv: |k| stretches, a minus sign reverses")
    for k, c, oy in [(1, BLUE, 1.6), (0.5, GREEN, 2.4), (2, ORANGE, 0.8), (-1, RED, -0.8), (-3, PURPLE, -1.6)]:
        arrow(ax, (0, oy), k * v, c, f"{k}v", off=(0, 0.35))
    ax = axs[1]; grid2d(ax, (-1, 9), (-1, 5), "Parallel = collinear: one is a multiple of the other")
    arrow(ax, (0, 0), v, BLUE, "v", off=(0, 0.4)); arrow(ax, (0, -0.4), 2.5 * v, ORANGE, "2.5v", off=(0.8, -0.2))
    arrow(ax, (3, 2.5), 2.5 * v, ORANGE, "2.5v (moved)", off=(0, 0.5))
    save(fig, "05_scalar_multiples_and_parallel.png")


# ================================================================ 6. three vectors
def fig_three():
    u, v, w = np.array([3, 1]), np.array([1, 2.5]), np.array([-2, 1.5])
    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_subplot(1, 2, 1); grid2d(ax, (-1, 5), (-0.5, 6), "u + v + w: tip to tail, first tail to last tip")
    arrow(ax, (0, 0), u, BLUE, "u", off=(0, -0.4)); arrow(ax, u, v, ORANGE, "v", off=(0.4, 0))
    arrow(ax, u + v, w, GREEN, "w", off=(0.2, 0.35)); arrow(ax, (0, 0), u + v + w, GOLD, "u+v+w", lw=3.5, off=(-0.8, 0))
    ax = fig.add_subplot(1, 2, 2, projection="3d"); axes3d(ax, 4, "In 3-D: u+v+w is the diagonal of the box")
    a, b, c = np.array([3, 0, 0]), np.array([0, 3, 0]), np.array([0, 0, 2.5])
    corners = [np.zeros(3), a, b, c, a + b, a + c, b + c, a + b + c]
    for p in corners:
        for q in corners:
            d = q - p
            if np.count_nonzero(d) == 1 and d.sum() > 0:
                ax.plot(*zip(p, q), color="#bbb", lw=1)
    arrow3d(ax, (0, 0, 0), a, BLUE, "u"); arrow3d(ax, (0, 0, 0), b, ORANGE, "v"); arrow3d(ax, (0, 0, 0), c, GREEN, "w")
    arrow3d(ax, (0, 0, 0), a + b + c, GOLD, "u+v+w", lw=4)
    save(fig, "06_sum_of_three_vectors_box.png")


# ================================================================ 7. components
def fig_components():
    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_subplot(1, 2, 1); grid2d(ax, (-1, 5), (-1, 4), "2-D: v = (v1, v2) = (3, 2)")
    arrow(ax, (0, 0), (3, 2), BLUE, lw=3.5)
    ax.plot([3, 3], [0, 2], "--", color=ORANGE); ax.plot([0, 3], [0, 0], color=GREEN, lw=4, alpha=0.6)
    ax.text(1.5, -0.45, "v1 = 3", color=GREEN, fontsize=13, ha="center", fontweight="bold")
    ax.text(3.2, 1.0, "v2 = 2", color=ORANGE, fontsize=13, fontweight="bold")
    ax.text(3.1, 2.25, "(3, 2)", color=BLUE, fontsize=14, fontweight="bold")
    ax = fig.add_subplot(1, 2, 2, projection="3d"); axes3d(ax, 4, "3-D: v = (v1, v2, v3) = (2, 3, 2.5)")
    p = np.array([2, 3, 2.5])
    ax.plot([2, 2, 0], [0, 3, 3], [0, 0, 0], "--", color="#999"); ax.plot([2, 2], [3, 3], [0, 2.5], "--", color="#999")
    arrow3d(ax, (0, 0, 0), p, BLUE, "(2, 3, 2.5)", lw=4)
    save(fig, "07_components_2d_3d.png")


# ================================================================ 8. end minus start
def fig_end_minus_start():
    P1, P2 = np.array([1, 1]), np.array([5, 4])
    fig, ax = plt.subplots(figsize=(8, 6.5)); grid2d(ax, (-0.5, 6), (-0.5, 5), "Vector from P1 to P2 = END minus START")
    arrow(ax, (0, 0), P1, GREEN, "OP1", off=(0.35, -0.25)); arrow(ax, (0, 0), P2, ORANGE, "OP2", off=(0.2, 0.4))
    arrow(ax, P1, P2 - P1, BLUE, "P1P2 = OP2 - OP1", lw=4, off=(-1.2, 0.5))
    ax.text(1, 0.6, "P1(1,1)", fontsize=12); ax.text(5.05, 4.15, "P2(5,4)", fontsize=12)
    ax.text(0.2, 4.4, f"P1P2 = (5-1, 4-1) = {tuple(int(x) for x in P2 - P1)}", fontsize=14, color=BLUE, fontweight="bold",
            bbox=dict(fc="white", ec=BLUE))
    save(fig, "08_end_minus_start.png")
    # Anton Example 1
    P1, P2 = np.array([2, -1, 4]), np.array([7, 5, -8])
    print("\nAnton Example 1: P1(2,-1,4), P2(7,5,-8)")
    for n, a, b in zip("xyz", P1, P2):
        print(f"   {n}: {b} - ({a}) = {b - a}")
    print("   v =", tuple(int(x) for x in P2 - P1), "   (book: (5, 6, -12))")


# ================================================================ 9. componentwise
def fig_componentwise():
    v, w = np.array([4, 1.5]), np.array([1.5, 3]); k = 2.5; vv = np.array([2, 1.5])
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
    ax = axs[0]; grid2d(ax, (-0.5, 6.5), (-0.5, 5.5), "Add matching components")
    arrow(ax, (0, 0), v, BLUE, "(v1, v2)", off=(0.3, -0.4)); arrow(ax, (0, 0), w, ORANGE, "(w1, w2)", off=(-0.9, 0))
    arrow(ax, (0, 0), v + w, GOLD, "(v1+w1, v2+w2)", lw=3.5, off=(-0.6, 0.6))
    ax = axs[1]; grid2d(ax, (-0.5, 6.5), (-0.5, 5.5), "Multiply every component by k")
    arrow(ax, (0, 0), k * vv, ORANGE, "(kv1, kv2)", lw=3.5, off=(-0.3, 0.6)); arrow(ax, (0, 0), vv, BLUE, "(v1, v2)", off=(0.6, -0.3))
    save(fig, "09_componentwise_operations.png")
    v, w = np.array([1, -3, 2]), np.array([4, 2, 1])
    print("\nAnton Example 3: v=(1,-3,2), w=(4,2,1)")
    print("   v + w =", tuple(v + w), " 2v =", tuple(2 * v), " -w =", tuple(-w), " v - w =", tuple(v - w))


# ================================================================ 10. Theorem 3.1.1
def check_theorem():
    rng = np.random.default_rng(1)
    u, v, w = rng.integers(-9, 10, (3, 5)); k, m = 3, -2
    z = np.zeros(5, int)
    rules = {
        "(a) u+v = v+u": (u + v, v + u), "(b) (u+v)+w = u+(v+w)": ((u + v) + w, u + (v + w)),
        "(c) u+0 = u": (u + z, u), "(d) u+(-u) = 0": (u + (-u), z),
        "(e) k(u+v) = ku+kv": (k * (u + v), k * u + k * v), "(f) (k+m)u = ku+mu": ((k + m) * u, k * u + m * u),
        "(g) k(mu) = (km)u": (k * (m * u), (k * m) * u), "(h) 1u = u": (1 * u, u)}
    print("\nTheorem 3.1.1 on random vectors in R^5:", "u =", u, "v =", v, "w =", w)
    for name, (a, b) in rules.items():
        print(f"   {name:26s} {'TRUE' if np.array_equal(a, b) else 'FALSE'}")


# ================================================================ 11. linear combinations
def fig_lincomb():
    v1, v2 = np.array([1, 2]), np.array([3, 0]); k1, k2 = 2, 5 / 3
    fig, ax = plt.subplots(figsize=(9, 6.5)); grid2d(ax, (-0.5, 8), (-0.5, 5.5), "Linear combination: (7,4) = 2(1,2) + 5/3 (3,0)")
    arrow(ax, (0, 0), v1, BLUE, "v1", off=(-0.35, 0)); arrow(ax, (0, 0), v2, ORANGE, "v2", off=(0, -0.4))
    arrow(ax, (0, 0), k1 * v1, BLUE, "2 v1", lw=1.6, ls="--", off=(-0.5, 0.3))
    arrow(ax, k1 * v1, k2 * v2, ORANGE, "5/3 v2", off=(0, 0.35))
    arrow(ax, (0, 0), k1 * v1 + k2 * v2, GOLD, "(7, 4)", lw=4, off=(0.5, -0.4))
    save(fig, "11_linear_combination.png")


# ================================================================ 12. RGB cube
def fig_rgb():
    fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(projection="3d")
    n = 7; g = np.linspace(0, 1, n)
    R, G, B = np.meshgrid(g, g, g); pts = np.c_[R.ravel(), G.ravel(), B.ravel()]
    surf = pts[(pts == 0).any(1) | (pts == 1).any(1)]
    ax.scatter(surf[:, 0], surf[:, 1], surf[:, 2], c=surf, s=70, depthshade=False)
    ax.plot([0, 1], [0, 1], [0, 1], "--", color="gray", lw=2)
    names = {(0, 0, 0): "Black", (1, 0, 0): "Red", (0, 1, 0): "Green", (0, 0, 1): "Blue", (1, 1, 0): "Yellow",
             (1, 0, 1): "Magenta", (0, 1, 1): "Cyan", (1, 1, 1): "White"}
    for p, nm in names.items():
        ax.text(*(np.array(p) * 1.12 - 0.06), f"{nm} {p}", fontsize=10, fontweight="bold")
    ax.set_xlabel("red k1"); ax.set_ylabel("green k2"); ax.set_zlabel("blue k3"); ax.view_init(22, 35)
    ax.set_title("RGB colour cube: c = k1 r + k2 g + k3 b,  0 <= ki <= 1\n(dashed diagonal = greys)", color=NAVY)
    save(fig, "12_rgb_colour_cube.png")


# ================================================================ 13. misconceptions
def fig_misconceptions():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    ax = axs[0]; grid2d(ax, (-1, 5), (-1, 9), "Mistake: 'start minus end'")
    P1, P2 = np.array([3, 5]), np.array([2, 8])
    arrow(ax, P1, P2 - P1, GREEN, "right: (-1, 3)", off=(-1.6, 0))
    arrow(ax, P2, P1 - P2, RED, "wrong: (1, -3)", lw=1.8, ls="--", off=(1.8, 0.3))
    ax.text(3.1, 4.6, "P1(3,5)", fontsize=11); ax.text(2.1, 8.2, "P2(2,8)", fontsize=11)
    ax = axs[1]; grid2d(ax, (-1, 5), (-1, 5), "Mistake: 3 east + 4 north = 7 away")
    arrow(ax, (0, 0), (3, 0), BLUE, "3", off=(0, -0.4)); arrow(ax, (3, 0), (0, 4), ORANGE, "4", off=(0.35, 0))
    arrow(ax, (0, 0), (3, 4), GOLD, "length 5, not 7", lw=3.5, off=(-1.4, 0.4))
    ax = axs[2]; grid2d(ax, (-5, 5), (-3, 3), "Mistake: '-2v is shorter than v'")
    arrow(ax, (0, 0.5), (2, 1), BLUE, "v", off=(0, 0.4)); arrow(ax, (0, -0.5), (-4, -2), RED, "-2v: twice as long", off=(0, -0.5))
    save(fig, "13_misconceptions.png")


# ================================================================ 14. exercises
def fr(v):
    return "(" + ", ".join(str(F(x).limit_denominator()) for x in v) + ")"


def exercises():
    A = lambda *x: np.array(x, dtype=object)  # exact integer / Fraction arithmetic
    print("\n================ Anton, Exercise Set 3.1 ================")
    print("1a", fr(A(4, 1) - A(1, 5)), "  1b", fr(A(0, 0, 4) - A(2, 3, 0)))
    print("2a", fr(A(-3, 3) - A(2, 3)), "  2b", fr(A(0, 4, 4) - A(3, 0, 4)))
    print("3a", fr(A(2, 8) - A(3, 5)), "  3b", fr(A(2, 4, 2) - A(5, -2, 1)))
    print("4a", fr(A(-4, -1) - A(-6, 2)), "  4b", fr(A(-1, 6, 1) - A(0, 0, 0)))
    print("5a terminal", fr(A(1, 1) + A(1, 2)), "  5b initial", fr(A(-1, -1, 2) - A(1, 1, 3)))
    print("6a initial", fr(A(2, 0) - A(1, 2)), "  6b terminal", fr(A(0, 2, 0) + A(1, 1, 3)))
    Q, v = A(3, 0, -5), A(4, -2, -1); print("7a P = Q - v", fr(Q - v), "  7b P = Q + v", fr(Q + v), "  (k = 1)")
    P, v = A(-1, 3, -5), A(6, 7, -3); print("8a Q = P + v", fr(P + v), "  8b Q = P - v", fr(P - v), "  (k = 1)")
    u, v, w = A(4, -1), A(0, 5), A(-3, -3)
    print("9 ", fr(u + w), fr(v - 3 * u), fr(2 * (u - 5 * w)), fr(3 * v - 2 * (u + 2 * w)))
    u, v, w = A(-3, 1, 2), A(4, 0, -8), A(6, -1, -4)
    print("10", fr(v - w), fr(6 * u + 2 * v), fr(-3 * (v - 8 * w)), fr((2 * u - 7 * w) - (8 * v + u)))
    u, v, w = A(-3, 2, 1, 0), A(4, 7, -3, 2), A(5, -2, 8, 1)
    print("11", fr(v - w), fr(-u + (v - 4 * w)), fr(6 * (u - 3 * v)), fr((6 * v - w) - (4 * u + v)))
    print("13 x = (3u + v - 4w)/3 =", fr((3 * u + v - 4 * w) * F(1, 3)))
    u, v, w = A(1, 2, -3, 5, 0), A(0, 4, -1, 1, 2), A(7, 1, -4, -2, 3)
    print("12", fr(v + w), fr(3 * (2 * u - v)), fr((3 * u - v) - (2 * u + 4 * w)), fr(F(1, 2) * (w - 5 * v + 2 * u) + v))
    print("14 x = (2u - v - w)/6 =", fr((2 * u - v - w) * F(1, 6)))

    def parallel(a, b):  # is a parallel to b?  (rank of [a b] <= 1)
        return np.linalg.matrix_rank(np.array([a, b], float)) <= 1
    u = [-2, 1, 0, 3, 5, 1]
    print("15 a:", parallel([4, 2, 0, 6, 10, 2], u), " b:", parallel([4, -2, 0, -6, -10, -2], u), " c:", parallel([0] * 6, u))
    print("16 a: t = 1   b: t = 0 (zero vector)   c: t^2 = -1/4 -> no real t")
    sol = lambda M, b: np.linalg.lstsq(np.array(M, float).T, np.array(b, float), rcond=None)
    for name, M, b in [("17", [[1, -1, 3, 5], [2, 1, 0, -3]], [1, -4, 9, 18]),
                       ("18", [[2, 1, 0, 1, -1], [-2, 3, 1, 0, 2]], [-8, 8, 3, -1, 7]),
                       ("19", [[1, -1, 0], [3, 2, 1], [0, 1, 4]], [-1, 1, 19]),
                       ("20", [[-1, 0, 2], [2, 2, -2], [1, -2, 1]], [-6, 12, 4]),
                       ("21", [[-2, 9, 6], [-3, 2, 1], [1, 7, 5]], [0, 5, 4]),
                       ("22", [[1, 0, 1, 0], [1, 0, -2, 1], [2, 0, 1, 2]], [1, -2, 2, 3])]:
        x, res, *_ = sol(M, b)
        ok = np.allclose(np.array(M, float).T @ x, b)
        print(f"{name} coefficients = {np.round(x, 4)}   exact solution exists: {ok}")
    P, Q = A(2, 3, -2), A(7, -4, 1)
    print("23a midpoint", fr((P + Q) * F(1, 2)), "  23b", fr(P + F(3, 4) * (Q - P)))
    print("24 terminal point = midpoint of P1P2")
    for nm, (u, v, w) in {"a": (A(5, -5), A(-10, 2), A(3, 8)), "b": (A(10, -7), A(-3, 8), A(-4, -9))}.items():
        print(f"25{nm} u+v+w = {fr(u + v + w)}    26{nm} u-v+w = {fr(u - v + w)}")
    print("27 Q = 2M - P =", fr(2 * A(4, 0, -6) - A(1, 3, 7)))
    print("28 yes: w = -(u+v) lies in the plane of u and v")
    ang = np.deg2rad(np.arange(0, 360, 60)); r = np.c_[np.cos(ang), np.sin(ang)]
    print("29 sum of hexagon radii =", np.round(r.sum(0), 12), "  halved =", np.round((0.5 * r).sum(0), 12))


def fig_hexagon():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ang = np.deg2rad(np.arange(0, 360, 60)); cols = [BLUE, ORANGE, GREEN] * 2
    for ax, s, t in [(axs[0], 1, "Ex 29(a): six radial vectors, sum = 0"), (axs[1], 0.5, "Ex 29(b): each halved, sum still 0")]:
        grid2d(ax, (-1.4, 1.4), (-1.4, 1.4), t, axes=False)
        ax.fill(np.cos(ang), np.sin(ang), fc="#f3f6fb", ec="gray")
        for a, c in zip(ang, cols):
            arrow(ax, (0, 0), s * np.array([np.cos(a), np.sin(a)]), c, lw=3)
    save(fig, "14_exercise29_hexagon.png")


def fig_ex25():
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
    data = {"a": ((0, 0), [(5, -5), (-10, 2), (3, 8)]), "b": ((-5, 4), [(10, -7), (-3, 8), (-4, -9)])}
    for ax, (nm, (s, vs)) in zip(axs, data.items()):
        grid2d(ax, (-5, 5), (-5, 5), f"Ex 25({nm}): u + v + w = {fr(np.sum(vs, 0))}")
        p = np.array(s, float)
        for v, c, l in zip(vs, [BLUE, ORANGE, GREEN], "uvw"):
            arrow(ax, p, v, c, l, off=(0.4, 0.4)); p = p + v
        arrow(ax, s, np.sum(vs, 0), GOLD, lw=3.5, ls="--")
    save(fig, "15_exercise25.png")


if __name__ == "__main__":
    print("Drawing figures into", OUT)
    fig_story(); fig_geometric(); fig_addition(); fig_subtraction(); fig_scalar(); fig_three()
    fig_components(); fig_end_minus_start(); fig_componentwise(); check_theorem(); fig_lincomb(); fig_rgb()
    fig_misconceptions(); fig_hexagon(); fig_ex25(); exercises()
    print("\nDone. Open the folder", OUT)
