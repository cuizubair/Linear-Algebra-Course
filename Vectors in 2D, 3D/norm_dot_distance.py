"""
ZMathLab · Linear Algebra · Norm, Dot Product and Distance (Anton, Section 3.2)
==============================================================================
Run:      python norm_dot_distance.py
Needs:    numpy, matplotlib            (pip install numpy matplotlib)
Output:   folder "norm_dot_figures/" with one clear PNG per idea, and every
          Exercise Set 3.2 answer computed and checked in the terminal.

Contents
  1. Story: taste vectors and cosine similarity
  2. Norm in 2D (Pythagoras) and 3D (Pythagoras twice)
  3. Unit vectors, normalizing, i, j, k
  4. Distance between points; circle / disc / outside (Ex. 19)
  5. Dot product: acute / right / obtuse, and the "shadow" picture
  6. Angle of a cube diagonal (Example 7) and Ex. 24
  7. Triangle inequality and parallelogram equation
  8. Cauchy-Schwarz checked on 10,000 random vectors in R^7
  9. ISBN check digit
 10. Exercise Set 3.2 answers (same numbers as the book)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = "norm_dot_figures"
os.makedirs(OUT, exist_ok=True)
NAVY, GOLD = "#0B1832", "#E0A82E"
BLUE, ORANGE, GREEN, RED, PURPLE, MUTED = "#2F7FD8", "#F07A2E", "#2EA866", "#D64545", "#8A5CD6", "#5B6B85"
GRID = "#C9D6EA"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "savefig.dpi": 200})


def arrow(ax, s, v, c, label=None, lw=2.6, off=(0.15, 0.15), ls="-"):
    s = np.asarray(s, float); v = np.asarray(v, float)
    ax.annotate("", xy=s + v, xytext=s, arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.4", color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
    if label:
        p = s + 0.55 * v + np.asarray(off)
        ax.text(p[0], p[1], label, color=c, fontsize=14, fontweight="bold", ha="center", va="center")


def grid(ax, xr, yr, title=None, axes=True):
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.8)
    ax.tick_params(labelsize=9, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    if axes:
        ax.axhline(0, color=MUTED, lw=1.1); ax.axvline(0, color=MUTED, lw=1.1)
    if title:
        ax.set_title(title, color=NAVY)


def save(fig, name):
    p = os.path.join(OUT, name); fig.tight_layout(); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); print("  saved", p)


def norm(v):
    return float(np.sqrt(np.dot(v, v)))


def cosang(u, v):
    return float(np.dot(u, v) / (norm(u) * norm(v)))


# ------------------------------------------------------------------ 1
def fig_story():
    fig, ax = plt.subplots(figsize=(9, 6.5)); grid(ax, (-0.5, 7.5), (-0.5, 4.8), "Taste vectors: a small angle means similar music")
    zara = np.array([5, 3]); songs = {"mother's song": ([5.6, 3.6], GREEN), "party song": ([6.5, 0.6], RED), "lullaby": ([0.8, 3.9], MUTED)}
    arrow(ax, (0, 0), zara, BLUE, "Zara", lw=3.5, off=(0.2, 0.35))
    for n, (v, c) in songs.items():
        arrow(ax, (0, 0), v, c, lw=2.2)
        ax.text(v[0] + 0.1, v[1] + 0.15, f"{n}\ncos = {cosang(zara, v):.3f}", color=c, fontsize=11, fontweight="bold")
    ax.set_xlabel("energy"); ax.set_ylabel("calmness")
    save(fig, "01_story_cosine_similarity.png")


# ------------------------------------------------------------------ 2
def fig_norm():
    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_subplot(1, 2, 1); grid(ax, (-0.5, 5), (-0.5, 4.8), "2D: ||(3,4)|| = sqrt(9+16) = 5")
    ax.fill([0, 3, 3], [0, 0, 4], color=GOLD, alpha=0.25)
    arrow(ax, (0, 0), (3, 4), BLUE, "||v|| = 5", lw=3.5, off=(-0.6, 0.2))
    ax.plot([0, 3], [0, 0], color=GREEN, lw=4); ax.plot([3, 3], [0, 4], color=ORANGE, lw=4)
    ax.text(1.5, -0.35, "3", color=GREEN, fontsize=14, ha="center", fontweight="bold"); ax.text(3.2, 2, "4", color=ORANGE, fontsize=14, fontweight="bold")
    ax = fig.add_subplot(1, 2, 2, projection="3d"); ax.set_title("3D: Pythagoras twice", color=NAVY)
    p = np.array([3, 3.5, 2.5])
    ax.plot([0, 3], [0, 0], [0, 0], color=GREEN, lw=3); ax.plot([3, 3], [0, 3.5], [0, 0], color=ORANGE, lw=3)
    ax.plot([0, 3], [0, 3.5], [0, 0], "--", color=PURPLE, lw=2); ax.plot([3, 3], [3.5, 3.5], [0, 2.5], color=RED, lw=3)
    ax.quiver(0, 0, 0, *p, color=BLUE, lw=3.5, arrow_length_ratio=0.1)
    ax.text(*(p * 1.05), f"||v|| = {norm(p):.3f}", color=BLUE, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_zlim(0, 3); ax.view_init(22, -55)
    save(fig, "02_norm_2d_3d.png")
    print("\nAnton Example 1:", norm([-3, 2, 1]), "= sqrt(14) =", np.sqrt(14), "|", norm([2, -1, 3, -5]), "= sqrt(39)")


# ------------------------------------------------------------------ 3
def fig_unit():
    v = np.array([4.8, 2.4]); u = v / norm(v)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]; grid(ax, (-1.4, 5.5), (-1.4, 3.2), "Normalizing: u = v / ||v||  (length 1)")
    t = np.linspace(0, 2 * np.pi, 300); ax.plot(np.cos(t), np.sin(t), "--", color=MUTED)
    arrow(ax, (0, 0), v, BLUE, "v", lw=3); arrow(ax, (0, 0), u, GOLD, "u", lw=4, off=(0, 0.3))
    ax = axs[1]; grid(ax, (-0.5, 4.5), (-0.5, 3.5), "v = (3,2) = 3i + 2j")
    arrow(ax, (0, 0), (1, 0), BLUE, "i", off=(0, -0.3)); arrow(ax, (0, 0), (0, 1), ORANGE, "j", off=(-0.3, 0))
    arrow(ax, (0, 0), (3, 0), BLUE, "3i", lw=1.5, ls="--", off=(0, -0.3)); arrow(ax, (3, 0), (0, 2), ORANGE, "2j", lw=1.5, ls="--", off=(0.35, 0))
    arrow(ax, (0, 0), (3, 2), GOLD, "v", lw=3.5, off=(-0.3, 0.3))
    save(fig, "03_unit_vectors.png")
    print("Anton Example 2: u =", np.array([2, 2, -1]) / 3, " ||u|| =", norm(np.array([2, 2, -1]) / 3))


# ------------------------------------------------------------------ 4
def fig_distance():
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
    ax = axs[0]; grid(ax, (-0.5, 7), (-0.5, 5.5), "d = ||P2 - P1||")
    P1, P2 = np.array([1, 1]), np.array([6, 4.5])
    ax.fill([1, 6, 6], [1, 1, 4.5], color=GOLD, alpha=0.2); arrow(ax, P1, P2 - P1, BLUE, f"d = {norm(P2 - P1):.2f}", lw=3, off=(-0.8, 0.4))
    t = np.linspace(0, 2 * np.pi, 300)
    for ax, mode in zip(axs[1:], ["= 1", "<= 1", "> 1"]):
        grid(ax, (-2, 2), (-2, 2), "||r - r0|| " + mode, axes=False)
        if mode == "> 1":
            ax.fill([-2, 2, 2, -2], [-2, -2, 2, 2], color=BLUE, alpha=0.15); ax.fill(np.cos(t), np.sin(t), color="white"); ax.plot(np.cos(t), np.sin(t), "--", color=BLUE)
        elif mode == "<= 1":
            ax.fill(np.cos(t), np.sin(t), color=BLUE, alpha=0.3); ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=2)
        else:
            ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=3)
        ax.plot(0, 0, "ko"); ax.text(0.08, -0.25, "r0")
    save(fig, "04_distance_and_circles.png")
    u, v = np.array([1, 3, -2, 7]), np.array([0, 7, 2, 2]); print("Anton Example 4: d =", norm(u - v), "= sqrt(58)")


# ------------------------------------------------------------------ 5
def fig_dot():
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.5))
    u = np.array([2.6, 0])
    for ax, v, t in zip(axs[:3], [[2, 1.2], [0, 1.9], [-1.6, 1.4]], ["acute", "right angle", "obtuse"]):
        v = np.array(v); grid(ax, (-2, 3.2), (-0.8, 2.4), f"{t}: u.v = {np.dot(u, v):.2f}", axes=False)
        arrow(ax, (0, 0), u, BLUE, "u", off=(0.2, -0.25)); arrow(ax, (0, 0), v, ORANGE, "v", off=(0.2, 0.25))
    ax = axs[3]; grid(ax, (-0.3, 5.5), (-0.8, 3.2), "u.v = ||u|| x (shadow of v on u)", axes=False)
    u2, v2 = np.array([5, 0]), np.array([3, 2.4])
    arrow(ax, (0, 0), u2, BLUE, "u", off=(0, -0.3)); arrow(ax, (0, 0), v2, ORANGE, "v", off=(-0.3, 0.2))
    sh = np.dot(u2, v2) / norm(u2)
    ax.plot([0, sh], [0, 0], color=GOLD, lw=7, alpha=0.8); ax.plot([3, 3], [0, 2.4], "--", color=MUTED)
    ax.text(sh / 2, -0.45, f"shadow = {sh:.1f}", color=GOLD, ha="center", fontweight="bold")
    save(fig, "05_dot_product_sign_and_shadow.png")
    print("Anton Example 6:", np.dot([0, 0, 1], [0, 2, 2]), "|", np.dot([-1, 3, 5, 7], [-3, -4, 1, 0]))


# ------------------------------------------------------------------ 6
def fig_cube():
    fig = plt.figure(figsize=(14, 6.5))
    for k, (title, vecs) in enumerate([("Example 7: diagonal vs edge = 54.74 deg", [((0, 0, 0), (1, 0, 0), BLUE, "u1"), ((0, 0, 0), (1, 1, 1), GOLD, "d")]),
                                         ("Ex. 24: d.u -> 35 deg, d.v = 0 -> 90 deg", [((0, 0, 0), (1, 1, 0), BLUE, "u"), ((0, 0, 0), (1, 1, 1), GOLD, "d"), ((1, 1, 0), (-1, 0, 1), ORANGE, "v")])]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d"); ax.set_title(title, color=NAVY)
        cs = [np.array(c) for c in np.ndindex(2, 2, 2)]
        for p in cs:
            for q in cs:
                if np.abs(p - q).sum() == 1 and (q - p).sum() == 1:
                    ax.plot(*zip(p, q), color="#bbb")
        for s, v, c, l in vecs:
            ax.quiver(*s, *v, color=c, lw=3, arrow_length_ratio=0.12); ax.text(*(np.array(s) + np.array(v) * 1.05), l, color=c, fontsize=13, fontweight="bold")
        ax.view_init(22, 35)
    save(fig, "06_cube_angles.png")
    print("Example 7 angle:", np.degrees(np.arccos(1 / np.sqrt(3))))


# ------------------------------------------------------------------ 7
def fig_triangle():
    u, v = np.array([3, 0.6]), np.array([0.8, 2.0])
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]; grid(ax, (-0.5, 4.5), (-0.5, 3.2), f"||u+v|| = {norm(u + v):.2f} <= {norm(u) + norm(v):.2f} = ||u||+||v||", axes=False)
    arrow(ax, (0, 0), u, BLUE, "u", off=(0, -0.3)); arrow(ax, u, v, ORANGE, "v", off=(0.3, 0)); arrow(ax, (0, 0), u + v, GOLD, "u+v", lw=3.5, off=(-0.5, 0.3))
    ax = axs[1]; a, b = np.array([4, 0]), np.array([1.2, 2])
    lhs = norm(a + b) ** 2 + norm(a - b) ** 2; rhs = 2 * (norm(a) ** 2 + norm(b) ** 2)
    grid(ax, (-0.5, 5.8), (-0.6, 2.8), f"parallelogram: {lhs:.1f} = {rhs:.1f}", axes=False)
    arrow(ax, (0, 0), a, BLUE, "u", off=(0, -0.3)); arrow(ax, (0, 0), b, ORANGE, "v", off=(-0.3, 0))
    ax.plot([4, 5.2, 1.2], [0, 2, 2], color=MUTED); arrow(ax, (0, 0), a + b, GOLD, "u+v", lw=3); arrow(ax, b, a - b, PURPLE, "u-v", lw=3, off=(0.4, 0.3))
    save(fig, "07_triangle_and_parallelogram.png")


# ------------------------------------------------------------------ 8
def check_cauchy_schwarz():
    rng = np.random.default_rng(0); ok = 0; worst = 0
    for _ in range(10000):
        u, v = rng.normal(size=7), rng.normal(size=7)
        r = abs(np.dot(u, v)) / (norm(u) * norm(v)); worst = max(worst, r); ok += r <= 1 + 1e-12
    print(f"\nCauchy-Schwarz on 10,000 random pairs in R^7: holds in {ok} cases; largest |u.v|/(||u|| ||v||) = {worst:.4f}")


# ------------------------------------------------------------------ 9
def isbn(first9):
    s = int(np.dot(np.arange(1, 10), first9)); c = s % 11
    return s, ("X" if c == 10 else str(c))


# ------------------------------------------------------------------ 10
def exercises():
    A = np.array
    print("\n================ Anton, Exercise Set 3.2 ================")
    for n, v in [("1a", A([2, 2, 2])), ("1b", A([1, 0, 2, 1, 3])), ("2a", A([1, -1, 2])), ("2b", A([-2, 3, 3, -1]))]:
        print(f"{n}: ||v|| = {norm(v):.4f}, opposite unit vector = {np.round(-v / norm(v), 4)}")
    u, v, w = A([2, -2, 3]), A([1, -3, 4]), A([3, 6, -4])
    print("3:", [round(x, 4) for x in (norm(u + v), norm(u) + norm(v), norm(-2 * u + 2 * v), norm(3 * u - 5 * v + w))])
    print("4:", [round(x, 4) for x in (norm(u + v + w), norm(u - v), norm(3 * v) - 3 * norm(v), norm(u) - norm(v))])
    u, v, w = A([-2, -1, 4, 5]), A([3, 1, -5, 7]), A([-6, 2, 1, 1])
    print("5:", [round(x, 4) for x in (norm(3 * u - 5 * v + w), norm(3 * u) - 5 * norm(v) + norm(w), norm(-norm(u) * v))])
    print("6:", [round(x, 4) for x in (norm(u) + norm(-2 * v) + norm(-3 * w), norm(norm(u - v) * w))])
    print("7: k = +/-", 5 / norm(A([-2, 3, 0, 6])), "   8: k = +/-", 4 / norm(A([1, 1, 2, -3, 1])))
    for n, (a, b) in {"9a": (A([3, 1, 4]), A([2, 2, -4])), "9b": (A([1, 1, 4, 6]), A([2, -2, 3, -2])),
                      "10a": (A([1, 1, -2, 3]), A([-1, 0, 5, 1])), "10b": (A([2, -1, 1, 0, -2]), A([1, 2, 2, 2, 1]))}.items():
        print(f"{n}: u.v={a @ b}, u.u={a @ a}, v.v={b @ b}")
    for n, (a, b) in {"11a": (A([3, 3, 3]), A([1, 0, 4])), "11b": (A([0, -2, -1, 1]), A([-3, 2, 4, 4])),
                      "12a": (A([1, 2, -3, 0]), A([5, 1, 2, -2])), "12b": (A([0, 1, 1, 1, 2]), A([2, 1, 0, -1, 3]))}.items():
        c = cosang(a, b); kind = "acute" if c > 1e-12 else ("obtuse" if c < -1e-12 else "90 deg")
        print(f"{n}: d = {norm(a - b):.4f}, cos = {c:.4f} -> {kind}")
    a = 9 * A([np.cos(np.radians(120)), np.sin(np.radians(120))]); print("13: a.b =", round(a @ A([0, 5]), 4), "= 45*sqrt(3)/2")
    print("14: angle 47+43 = 90 deg -> a.b = 0")
    for n, (a, b) in {"17a": (A([-3, 1, 0]), A([2, -1, 3])), "17b": (A([0, 2, 2, 1]), A([1, 1, 1, 1])),
                      "18a": (A([4, 1, 1]), A([1, 2, 3])), "18b": (A([1, 2, 1, 2, 3]), A([0, 1, 1, 5, -2]))}.items():
        print(f"{n}: |u.v| = {abs(a @ b)} <= {norm(a) * norm(b):.4f}")
    d = A([1, 1, 1]); print("24a:", round(np.degrees(np.arccos(cosang(d, A([1, 1, 0])))), 2), "deg   24b: d.v =", d @ A([-1, 0, 1]))
    D = A([10, 15, 25]); print("25:", [round(np.degrees(np.arccos(x / norm(D))), 1) for x in D], "deg")
    th = np.linspace(0, np.pi, 7); print("26: ||v-w|| over angles:", np.round(np.sqrt(13 - 12 * np.cos(th)), 3), "-> min 1, max 5")
    print("ISBN 0-471-15307-?:", isbn([0, 4, 7, 1, 1, 5, 3, 0, 7]))


if __name__ == "__main__":
    print("Drawing figures into", OUT)
    fig_story(); fig_norm(); fig_unit(); fig_distance(); fig_dot(); fig_cube(); fig_triangle()
    check_cauchy_schwarz(); exercises()
    print("\nDone. Open the folder", OUT)
