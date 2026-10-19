"""
Linear Algebra - Lecture 8: MATRIX TRANSFORMATIONS
Muhammad Zubair

Companion Python file for the lecture notes, the slides and the video.
Every demo draws a picture that you can show in class. Most demos are
ANIMATED (the plane moves from "before" to "after") or have SLIDERS
(drag the angle / factor and watch the picture change live).

  1. What a matrix does: the grid moves, e1 and e2 become the columns
  2. Linear or not? (grid lines stay straight / a translation moves the origin)
  3. Reflection in 2D  (x-axis, y-axis, y=x, y=-x, origin, any line - slider)
  4. Reflection in 3D  (xy-, xz-, yz-plane)
  5. Projection in 2D  (onto the axes, onto any line - slider)
  6. Projection in 3D  (onto the coordinate planes) + drawing a cube on a flat screen
  7. Rotation in 2D    (animation + angle slider)
  8. Rotation in 3D    (about the x-, y- and z-axis, animated)
  9. Scaling           (sliders for a and b, circle -> ellipse, area = det)
 10. Shearing          (slider for k, deck of cards, area stays the same)
 11. Dilation and contraction (slider for k)
 12. Translation and homogeneous coordinates (2D slide = 3D shear)
 13. Composition: order matters, and rotation about a point
 14. The puzzle: three shears make a rotation (animated)
 15. Application: the fractal fern (iterated function system)
 16. Answers to the textbook exercises (checked by the computer)

Requirements:  pip install numpy matplotlib
Run:           python matrix_transformations.py          (menu)
               python matrix_transformations.py 7        (one demo)
               python matrix_transformations.py all --save   (save every picture as PNG, no windows)

Books used for the examples:
  Anton & Rorres, Elementary Linear Algebra (Sections 1.8, 1.9)
  G. Williams, Linear Algebra with Applications (Section 2.6)
  DeFranza & Gagliardi, Introduction to Linear Algebra with Applications (Section 4.6)
"""

import os
import sys

import numpy as np
import matplotlib

SAVE_ONLY = "--save" in sys.argv
if SAVE_ONLY:
    matplotlib.use("Agg")          # no windows, just PNG files

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pictures")

# ----------------------------------------------------------------------------
# Colours (same as the notes): blue = original, orange = image
# ----------------------------------------------------------------------------
BLUE, ORANGE, GREEN, RED, PURPLE, GOLD, NAVY, GRAY = (
    "#3478c8", "#e2622e", "#1e965a", "#d23c3c", "#7d46be", "#d69a14", "#0b1832", "#9aa3b5")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": NAVY,
    "font.size": 10,
})

# The letter F: it has no symmetry, so every flip, turn or slant is visible.
F = np.array([[0, 0], [0, 3], [2, 3], [2, 2.4], [0.6, 2.4], [0.6, 1.8],
              [1.6, 1.8], [1.6, 1.2], [0.6, 1.2], [0.6, 0]], dtype=float).T
SQUARE = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)


# ============================================================================
# Small helpers
# ============================================================================
def title(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def mat_str(A, digits=3):
    """Pretty string of a matrix (rounded)."""
    A = np.atleast_2d(np.round(np.asarray(A, dtype=float), digits)) + 0.0
    rows = ["  ".join(f"{v:>7g}" for v in row) for row in A]
    return "\n".join("[ " + r + " ]" for r in rows)


def show_or_save(fig, name):
    if SAVE_ONLY:
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, name + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  saved", path)
    else:
        plt.show()


def setup(ax, lim=4, title_text=""):
    """A clean square grid with axes."""
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks(range(-lim, lim + 1))
    ax.set_yticks(range(-lim, lim + 1))
    ax.grid(color="#dfe4ee", lw=0.8)
    ax.axhline(0, color=GRAY, lw=1.2)
    ax.axvline(0, color=GRAY, lw=1.2)
    ax.tick_params(labelsize=7, colors=GRAY)
    for s in ax.spines.values():
        s.set_visible(False)
    if title_text:
        ax.set_title(title_text)


def draw_shape(ax, P, color, alpha=0.35, lw=2, label=None):
    poly = Polygon(P.T, closed=True, fc=color, ec=color, alpha=alpha, lw=lw, label=label)
    ax.add_patch(poly)
    ax.add_patch(Polygon(P.T, closed=True, fill=False, ec=color, lw=lw))
    return poly


def draw_grid_image(ax, A, lim=4, color=BLUE, alpha=0.25):
    """Draw the image of the square grid under A (shows how the WHOLE plane moves)."""
    t = np.linspace(-lim, lim, 2)
    for c in range(-lim, lim + 1):
        v = A @ np.vstack([np.full(2, c), t])       # vertical line x = c
        h = A @ np.vstack([t, np.full(2, c)])       # horizontal line y = c
        ax.plot(v[0], v[1], color=color, alpha=alpha, lw=1)
        ax.plot(h[0], h[1], color=color, alpha=alpha, lw=1)


def arrow(ax, v, color, label=None, start=(0, 0)):
    ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6, mutation_scale=16))
    if label:
        ax.text(start[0] + v[0] * 1.08, start[1] + v[1] * 1.08 + 0.15, label, color=color,
                fontsize=10, fontweight="bold")


def matrix_box(ax, A, x=0.02, y=0.98, extra=""):
    """Write the matrix, its determinant and an explanation on the axes."""
    A = np.asarray(A, dtype=float)
    txt = "A =\n" + mat_str(A, 2)
    if A.shape == (2, 2):
        txt += f"\n\ndet A = {round(float(np.linalg.det(A)), 3) + 0.0:g}"
    if extra:
        txt += "\n" + extra
    ax.text(x, y, txt, transform=ax.transAxes, va="top", ha="left", family="monospace",
            fontsize=8.5, color=NAVY,
            bbox=dict(boxstyle="round", fc="#fff8e6", ec=GOLD))


def before_after(A, title_text, lim=4, shape=F, mirror=None, extra=""):
    """Static picture: original (blue) and image (orange), e1/e2 and their images."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    setup(axs[0], lim, "BEFORE")
    setup(axs[1], lim, "AFTER:  " + title_text)
    draw_shape(axs[0], shape, BLUE)
    arrow(axs[0], (1, 0), GREEN, "e1")
    arrow(axs[0], (0, 1), RED, "e2")
    draw_grid_image(axs[1], A, lim)
    draw_shape(axs[1], shape, BLUE, alpha=0.12, lw=1)
    draw_shape(axs[1], A @ shape, ORANGE)
    arrow(axs[1], A[:, 0], GREEN, "A e1 = column 1")
    arrow(axs[1], A[:, 1], RED, "A e2 = column 2")
    if mirror is not None:
        axs[1].plot(*mirror, "--", color=PURPLE, lw=2)
    matrix_box(axs[1], A, extra=extra)
    fig.tight_layout()
    return fig


def animate_matrix(A, title_text, lim=4, shape=F, frames=60, name="anim"):
    """Smoothly move the plane from I to A (the most important classroom picture)."""
    A = np.asarray(A, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    if SAVE_ONLY:                      # in save mode just store the final frame
        setup(ax, lim, title_text)
        draw_grid_image(ax, A, lim)
        draw_shape(ax, shape, BLUE, alpha=0.12, lw=1)
        draw_shape(ax, A @ shape, ORANGE)
        arrow(ax, A[:, 0], GREEN, "Ae1")
        arrow(ax, A[:, 1], RED, "Ae2")
        matrix_box(ax, A)
        show_or_save(fig, name)
        return

    def update(i):
        ax.clear()
        s = min(1, i / (frames - 15))            # hold the last frames
        s = s * s * (3 - 2 * s)                  # smooth start and stop
        M = (1 - s) * np.eye(2) + s * A
        setup(ax, lim, title_text)
        draw_grid_image(ax, M, lim)
        draw_shape(ax, shape, BLUE, alpha=0.10, lw=1)
        draw_shape(ax, M @ shape, ORANGE)
        arrow(ax, M[:, 0], GREEN, "Ae1")
        arrow(ax, M[:, 1], RED, "Ae2")
        matrix_box(ax, A)
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=40, repeat=True)
    plt.show()
    return anim


def slider_demo(make_matrix, label, vmin, vmax, vinit, title_text, lim=4, shape=F,
                extra_draw=None, name="slider"):
    """Interactive demo: drag the slider and watch the picture."""
    fig, ax = plt.subplots(figsize=(6.6, 7.0))
    plt.subplots_adjust(bottom=0.16)
    sax = fig.add_axes([0.18, 0.05, 0.64, 0.035])
    sl = Slider(sax, label, vmin, vmax, valinit=vinit, color=GOLD)

    def redraw(val):
        ax.clear()
        A = make_matrix(val)
        setup(ax, lim, title_text)
        draw_grid_image(ax, A, lim)
        draw_shape(ax, shape, BLUE, alpha=0.12, lw=1)
        draw_shape(ax, A @ shape, ORANGE)
        arrow(ax, A[:, 0], GREEN, "Ae1")
        arrow(ax, A[:, 1], RED, "Ae2")
        if extra_draw:
            extra_draw(ax, val, A)
        matrix_box(ax, A)
        fig.canvas.draw_idle()

    sl.on_changed(redraw)
    redraw(vinit)
    fig._slider = sl                       # keep a reference so it stays alive
    show_or_save(fig, name)


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# ---------------------------------------------------------------- 3D helpers
def house3d():
    """A small 3D house (no symmetry in x) made of faces."""
    V = np.array([[0, 0, 0], [2, 0, 0], [2, 1.4, 0], [0, 1.4, 0],
                  [0, 0, 1.4], [2, 0, 1.4], [2, 1.4, 1.4], [0, 1.4, 1.4],
                  [0, 0.7, 2.2], [2, 0.7, 2.2],
                  [2.0, 0.2, 1.4], [2.0, 0.2, 2.1], [2.0, 0.5, 2.1], [2.0, 0.5, 1.4]], dtype=float)
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5],
             [4, 5, 9, 8], [7, 6, 9, 8], [4, 8, 7], [5, 9, 6]]
    return V.T, faces


def draw3d(ax, V, faces, color, alpha=0.35):
    polys = [[V[:, i] for i in f] for f in faces]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor=color,
                                         alpha=alpha, linewidths=1.2))


def setup3d(ax, lim=3, title_text=""):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    for v, c in [((lim, 0, 0), RED), ((0, lim, 0), GREEN), ((0, 0, lim), BLUE)]:
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, lw=2)
    ax.set_title(title_text, fontsize=10)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


# ============================================================================
# DEMOS
# ============================================================================
def demo1():
    title("1. WHAT A MATRIX DOES: the plane moves, e1 and e2 become the columns")
    A = np.array([[2, 1], [1, 1.5]])
    print("A =\n" + mat_str(A))
    print("A e1 =", A @ [1, 0], "= first column,   A e2 =", A @ [0, 1], "= second column")
    print("Grid lines stay straight and parallel, and the origin does not move.")
    print("|det A| =", np.linalg.det(A), "= factor by which every AREA is multiplied.")
    show_or_save(before_after(A, "A = [[2,1],[1,1.5]]", extra="columns = A e1, A e2"), "01_columns")
    animate_matrix(A, "Watch the whole plane move", name="01_animation")

    # Example: Anton 1.8, Example 7 -- the matrix from two other images
    print("\nAnton 1.8 Example 7: T(-1,1)=(-5,5), T(2,-1)=(7,-6). Find A.")
    U = np.array([[-1, 2], [1, -1]]); W = np.array([[-5, 7], [5, -6]])
    print("A = W U^(-1) =\n" + mat_str(W @ np.linalg.inv(U)))


def demo2():
    title("2. LINEAR OR NOT?")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    t = np.linspace(-3, 3, 61)
    # (a) linear: shear
    setup(axs[0], 3, "LINEAR: grid lines stay straight")
    draw_grid_image(axs[0], np.array([[1, 0.7], [0.2, 1]]), 3, BLUE, 0.6)
    axs[0].plot(0, 0, "o", color=GREEN, ms=9)
    # (b) not linear: bends lines
    setup(axs[1], 3, "NOT linear: T(x,y) = (x, y + 0.25x²)")
    for c in range(-3, 4):
        axs[1].plot(t, c + 0.25 * t ** 2, color=RED, alpha=0.6, lw=1)
        axs[1].plot(np.full_like(t, c), t + 0.25 * c ** 2, color=RED, alpha=0.6, lw=1)
    # (c) translation: moves the origin
    setup(axs[2], 3, "NOT linear: translation moves 0")
    draw_shape(axs[2], F * 0.7, BLUE)
    draw_shape(axs[2], F * 0.7 + np.array([[1.2], [-1.0]]), ORANGE)
    axs[2].plot(0, 0, "o", color=BLUE, ms=9)
    axs[2].plot(1.2, -1.0, "o", color=ORANGE, ms=9)
    axs[2].annotate("", xy=(1.2, -1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2))
    axs[2].text(1.3, -1.4, "T(0) = b ≠ 0", color=ORANGE)
    fig.tight_layout()
    print("Quick test: if T(0) is not 0, T is NOT linear.")
    print("Williams 2.6 Example 2: T(x,y,z)=(xy,z) is not linear: T(2u) ≠ 2T(u) for u=(1,1,0):",
          (2 * 2, 0), "vs", (2 * 1, 0))
    show_or_save(fig, "02_linear_or_not")


def demo3():
    title("3. REFLECTION IN 2D (the mirror)")
    cases = [("x-axis", [[1, 0], [0, -1]], ([-4, 4], [0, 0])),
             ("y-axis", [[-1, 0], [0, 1]], ([0, 0], [-4, 4])),
             ("line y = x", [[0, 1], [1, 0]], ([-4, 4], [-4, 4])),
             ("line y = -x", [[0, -1], [-1, 0]], ([-4, 4], [4, -4])),
             ("the origin", [[-1, 0], [0, -1]], None)]
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.2))
    for ax, (name, A, m) in zip(axs, cases):
        A = np.array(A, dtype=float)
        setup(ax, 4, "reflect about " + name)
        draw_shape(ax, F, BLUE)
        draw_shape(ax, A @ F, ORANGE)
        if m:
            ax.plot(*m, "--", color=PURPLE, lw=2)
        ax.text(0.03, 0.03, mat_str(A, 0) + f"\ndet = {np.linalg.det(A):.0f}", transform=ax.transAxes,
                family="monospace", fontsize=8, color=NAVY,
                bbox=dict(boxstyle="round", fc="#fff8e6", ec=GOLD))
        print(f"reflection about {name:12s}: matrix {A.tolist()}, det = {np.linalg.det(A):.0f}")
    fig.tight_layout()
    print("det = -1: the F reads backwards (orientation is flipped). Reflect twice -> identity.")
    show_or_save(fig, "03_reflections_2d")

    # any line through the origin: slider for the angle
    def H(deg):
        t = np.radians(deg)
        return np.array([[np.cos(2 * t), np.sin(2 * t)], [np.sin(2 * t), -np.cos(2 * t)]])

    def mirror(ax, deg, A):
        t = np.radians(deg)
        ax.plot([-5 * np.cos(t), 5 * np.cos(t)], [-5 * np.sin(t), 5 * np.sin(t)], "--", color=PURPLE, lw=2)

    print("Mirror line at angle theta:  H = [[cos2θ, sin2θ], [sin2θ, -cos2θ]]")
    print("Example: line y = 2x  ->  H =\n" + mat_str(H(np.degrees(np.arctan(2)))) + "   (= 1/5 [[-3,4],[4,3]])")
    slider_demo(H, "mirror angle (deg)", 0, 180, 30, "Reflection about a line at angle θ",
                extra_draw=mirror, name="03_reflection_any_line")


def demo4():
    title("4. REFLECTION IN 3D (mirror = a coordinate plane)")
    V, faces = house3d()
    mats = [("xy-plane (z -> -z)", np.diag([1, 1, -1])),
            ("xz-plane (y -> -y)", np.diag([1, -1, 1])),
            ("yz-plane (x -> -x)", np.diag([-1, 1, 1]))]
    fig = plt.figure(figsize=(15, 5))
    for k, (name, M) in enumerate(mats):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        setup3d(ax, 3, "reflect in the " + name)
        draw3d(ax, V, faces, BLUE)
        draw3d(ax, M @ V, faces, ORANGE)
        print(f"{name}: matrix diag{np.diag(M).tolist()}, image of (1,2,3) = {(M @ [1, 2, 3]).tolist()}")
    fig.tight_layout()
    show_or_save(fig, "04_reflections_3d")


def demo5():
    title("5. PROJECTION IN 2D (the shadow)")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (name, P) in zip(axs, [("onto the x-axis", np.array([[1, 0], [0, 0]])),
                                   ("onto the y-axis", np.array([[0, 0], [0, 1]])),
                                   ("onto y = x", 0.5 * np.array([[1, 1], [1, 1]]))]):
        setup(ax, 4, "project " + name)
        draw_shape(ax, F, BLUE)
        img = P @ F
        for a, b in zip(F.T, img.T):
            ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=GRAY)
        ax.plot(img[0], img[1], color=ORANGE, lw=5, solid_capstyle="round")
        ax.text(0.03, 0.03, mat_str(P, 2) + "\ndet = 0", transform=ax.transAxes, family="monospace",
                fontsize=8, color=NAVY, bbox=dict(boxstyle="round", fc="#fff8e6", ec=GOLD))
    fig.tight_layout()
    P = 0.5 * np.array([[1, 1], [1, 1]])
    print("Williams 2.6 No.18: projection of (4,2) onto y = x:", P @ [4, 2])
    print("P @ P == P ?", np.allclose(P @ P, P), "  det P =", round(np.linalg.det(P), 10), "-> not invertible")
    show_or_save(fig, "05_projections_2d")

    def Pmat(deg):
        u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        return np.outer(u, u)

    def extra(ax, deg, A):
        t = np.radians(deg)
        ax.plot([-5 * np.cos(t), 5 * np.cos(t)], [-5 * np.sin(t), 5 * np.sin(t)], color=PURPLE, lw=2)
        for a, b in zip(F.T, (A @ F).T):
            ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=GRAY)

    print("Projection onto the line at angle θ:  P = u u^T,  u = (cosθ, sinθ)")
    slider_demo(Pmat, "line angle (deg)", 0, 180, 30, "Projection onto a line (shadow)",
                extra_draw=extra, name="05_projection_any_line")


def demo6():
    title("6. PROJECTION IN 3D + drawing a cube on a flat screen")
    V, faces = house3d()
    fig = plt.figure(figsize=(16, 5))
    for k, (name, M) in enumerate([("xy-plane", np.diag([1, 1, 0])), ("xz-plane", np.diag([1, 0, 1])),
                                   ("yz-plane", np.diag([0, 1, 1]))]):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        setup3d(ax, 3, "project onto the " + name)
        draw3d(ax, V, faces, BLUE, 0.25)
        draw3d(ax, M @ V, faces, ORANGE, 0.5)
        print(f"onto the {name}: (1,2,3) -> {(M @ [1, 2, 3]).tolist()}")
    # DeFranza & Gagliardi 4.6 Example 8: parallel projection of the unit cube
    ax = fig.add_subplot(1, 4, 4)
    vd = np.array([0.433, 0.25, -1.0])
    Pm = np.array([[1, 0, -vd[0] / vd[2]], [0, 1, -vd[1] / vd[2]]])
    C = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]], float).T
    p = Pm @ C
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(p[0, [i, j]], p[1, [i, j]], color=NAVY, lw=2)
    ax.plot(p[0], p[1], "o", color=ORANGE)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("cube on a flat screen\n(DeFranza 4.6, Example 8)")
    print("Parallel projection matrix (R3 -> R2):\n" + mat_str(Pm))
    for v, q in zip(C.T, p.T):
        print(f"   {tuple(v.astype(int).tolist())} -> ({q[0]:.3f}, {q[1]:.3f})")
    fig.tight_layout()
    show_or_save(fig, "06_projections_3d")


def demo7():
    title("7. ROTATION IN 2D (the turn)")
    t = np.radians(30)
    R = rot(t)
    print("R(30°) =\n" + mat_str(R))
    x = np.array([1, 1])
    print("Anton 1.8 Example 8: R(30°)(1,1) =", np.round(R @ x, 4), " length before/after:",
          round(np.linalg.norm(x), 4), round(np.linalg.norm(R @ x), 4))
    print("R(a)R(b) = R(a+b)?", np.allclose(rot(0.3) @ rot(0.5), rot(0.8)),
          "   R^-1 = R^T?", np.allclose(np.linalg.inv(R), R.T), "   det R =", round(np.linalg.det(R), 10))
    # picture of e1, e2 on the unit circle
    fig, ax = plt.subplots(figsize=(6, 6))
    setup(ax, 2, "Where do e1 and e2 go?")
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=GRAY, ls="--"))
    arrow(ax, (1, 0), GREEN, "e1"); arrow(ax, (0, 1), RED, "e2")
    arrow(ax, R[:, 0], GREEN, "(cosθ, sinθ)"); arrow(ax, R[:, 1], RED, "(-sinθ, cosθ)")
    show_or_save(fig, "07_rotation_basis")
    animate_matrix(rot(np.radians(90)), "Rotation by 90°", name="07_rotation_animation")
    slider_demo(lambda d: rot(np.radians(d)), "angle θ (deg)", -180, 180, 30, "Rotation by θ",
                name="07_rotation_slider")


def demo8():
    title("8. ROTATION IN 3D (about an axis)")
    V, faces = house3d()

    def Rx(t): c, s = np.cos(t), np.sin(t); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(t): c, s = np.cos(t), np.sin(t); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(t): c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    for name, M in [("x", Rx), ("y", Ry), ("z", Rz)]:
        print(f"R{name}(90°) =\n" + mat_str(M(np.pi / 2)))
    print("Rz(90°)(2,3,5) =", np.round(Rz(np.pi / 2) @ [2, 3, 5], 6))

    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(1, 3, k + 1, projection="3d") for k in range(3)]
    makers = [Rx, Ry, Rz]
    names = ["about the x-axis", "about the y-axis", "about the z-axis"]

    def update(i):
        ang = np.radians(min(i, 90) * 1.0)
        for ax, M, n in zip(axs, makers, names):
            ax.clear()
            setup3d(ax, 3, f"rotate {n}: {np.degrees(ang):.0f}°")
            draw3d(ax, V, faces, BLUE, 0.15)
            draw3d(ax, M(ang) @ V, faces, ORANGE, 0.45)
        return []

    if SAVE_ONLY:
        update(90)
        show_or_save(fig, "08_rotation_3d")
    else:
        anim = FuncAnimation(fig, update, frames=110, interval=40, repeat=True)
        plt.show()
        return anim


def demo9():
    title("9. SCALING (stretch / squash)")
    print("DeFranza 4.6 Example 1: triangle (0,1),(2,1),(1,3)")
    T = np.array([[0, 2, 1], [1, 1, 3]], float)
    for name, S in [("horizontal stretch by 2", np.diag([2, 1])), ("vertical contraction by 3", np.diag([1, 1 / 3])),
                    ("both", np.diag([2, 1 / 3]))]:
        print(f"  {name:28s}: vertices -> {np.round(S @ T, 3).T.tolist()}")
    print("Williams 2.6 No.21: unit circle under scaling (4,3) -> ellipse x²/16 + y²/9 = 1")

    def extra(ax, val, A):
        ax.add_patch(Circle((0, 0), 1, fill=False, ec=BLUE, lw=1.5, ls="--"))
        ax.add_patch(Ellipse((0, 0), 2 * A[0, 0], 2 * A[1, 1], fill=False, ec=ORANGE, lw=2))
        ax.text(0.98, 0.02, f"area of circle π → {np.pi * A[0, 0] * A[1, 1]:.2f}\n(= π · det A)",
                transform=ax.transAxes, ha="right", fontsize=9, color=ORANGE)

    fig, ax = plt.subplots(figsize=(6.6, 7.2))
    plt.subplots_adjust(bottom=0.2)
    sa = Slider(fig.add_axes([0.18, 0.09, 0.64, 0.03]), "a (x-factor)", 0.2, 3, valinit=2, color=GOLD)
    sb = Slider(fig.add_axes([0.18, 0.04, 0.64, 0.03]), "b (y-factor)", 0.2, 3, valinit=1, color=GOLD)

    def redraw(_):
        ax.clear()
        A = np.diag([sa.val, sb.val])
        setup(ax, 4, "Scaling  S = [[a, 0], [0, b]]")
        draw_grid_image(ax, A, 4)
        draw_shape(ax, F, BLUE, alpha=0.12, lw=1)
        draw_shape(ax, A @ F, ORANGE)
        extra(ax, None, A)
        matrix_box(ax, A)
        fig.canvas.draw_idle()

    sa.on_changed(redraw); sb.on_changed(redraw); redraw(None)
    fig._s = (sa, sb)
    show_or_save(fig, "09_scaling")


def demo10():
    title("10. SHEARING (slant, like a deck of cards)")
    print("x-shear: (x, y) -> (x + k y, y)   matrix [[1, k], [0, 1]],  det = 1 (area unchanged)")
    print("Williams 2.6 No.22: unit square, k = 2 ->", (np.array([[1, 2], [0, 1]]) @ SQUARE).T.tolist())
    print("Williams 2.6 No.24: line y = 3x under x-shear 5 -> y = 3x/16")
    print("DeFranza 4.6 Example 2: vertical shear k=2 of (1,1),(2,1),(1.5,3) ->",
          (np.array([[1, 0], [2, 1]]) @ np.array([[1, 2, 1.5], [1, 1, 3]])).T.tolist())

    def extra(ax, k, A):
        for y in np.arange(0, 3.01, 0.3):              # the cards
            ax.plot([0 + k * y, 2 + k * y], [y, y], color=GOLD, lw=1)
        ax.text(0.98, 0.02, "same base, same height\n→ same area (det = 1)", transform=ax.transAxes,
                ha="right", fontsize=9, color=NAVY)

    slider_demo(lambda k: np.array([[1, k], [0, 1]]), "k", -2, 2, 1, "Shear in the x-direction",
                lim=5, extra_draw=extra, name="10_shear")


def demo11():
    title("11. DILATION AND CONTRACTION (zoom)")
    print("T(x) = k x, matrix k I.  k > 1: dilation,  0 <= k < 1: contraction.")
    print("Williams 2.6 No.29: 3 I moves (1,-2,4) to", (3 * np.eye(3) @ [1, -2, 4]).tolist(), "; volume x 27")
    slider_demo(lambda k: k * np.eye(2), "k", 0, 2, 1.5, "Dilation / contraction  kI",
                lim=6, name="11_dilation")


def demo12():
    title("12. TRANSLATION and HOMOGENEOUS COORDINATES")
    T = np.array([[1, 0, 3], [0, 1, 1], [0, 0, 1]], float)
    print("Translation by (3,1) in homogeneous coordinates:\n" + mat_str(T))
    print("DeFranza 4.6: translate (3,2) by (1,-2):", (np.array([[1, 0, 1], [0, 1, -2], [0, 0, 1]]) @ [3, 2, 1])[:2])
    Fh = np.vstack([F, np.ones(F.shape[1])])

    fig = plt.figure(figsize=(14, 5.6))
    ax1 = fig.add_subplot(1, 2, 1)
    setup(ax1, 5, "In 2D: a slide (NOT linear, moves 0)")
    draw_shape(ax1, F, BLUE)
    draw_shape(ax1, (T @ Fh)[:2], ORANGE)
    matrix_box(ax1, T)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title("In 3D: the plane w = 1 is SHEARED (linear!)")
    xx, yy = np.meshgrid([-1, 6], [-1, 5])
    ax2.plot_surface(xx, yy, np.ones_like(xx), alpha=0.12, color=GRAY)
    ax2.add_collection3d(Poly3DCollection([Fh.T], facecolor=BLUE, alpha=0.5, edgecolor=BLUE))
    ax2.add_collection3d(Poly3DCollection([(T @ Fh).T], facecolor=ORANGE, alpha=0.6, edgecolor=ORANGE))
    for p in [(0, 0), (2, 3), (0, 3)]:
        ax2.plot([0, p[0]], [0, p[1]], [0, 1], color=GRAY, lw=0.8, ls=":")
    ax2.set_xlim(-1, 6); ax2.set_ylim(-1, 5); ax2.set_zlim(0, 1.6)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("w")
    fig.tight_layout()
    show_or_save(fig, "12_translation_homogeneous")

    # 3D translation with a 4x4 matrix
    M = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], float)
    print("Rotate 90° about z, then translate by (1,2,3):  4x4 matrix\n" + mat_str(M))
    print("  (1,0,0) ->", (M @ [1, 0, 0, 1])[:3], "   (1,1,1) ->", (M @ [1, 1, 1, 1])[:3])


def demo13():
    title("13. COMPOSITION: order matters + rotation about a point")
    R = rot(np.pi / 2)
    b = np.array([[3], [0]])
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    setup(axs[0], 5, "rotate 90°, THEN slide (3,0)")
    draw_shape(axs[0], F, BLUE); draw_shape(axs[0], R @ F + b, ORANGE)
    setup(axs[1], 5, "slide (3,0), THEN rotate 90°")
    draw_shape(axs[1], F, BLUE); draw_shape(axs[1], R @ (F + b), GREEN)
    print("Different results: the order of the matrices matters (BA ≠ AB).")
    # Williams 2.6 Example 5: rotation by 90° about P(5,4)
    h, k, t = 5, 4, np.pi / 2
    T1 = np.array([[1, 0, -h], [0, 1, -k], [0, 0, 1]]); T2 = np.array([[1, 0, h], [0, 1, k], [0, 0, 1]])
    R3 = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    M = np.round(T2 @ R3 @ T1, 10)
    tri = np.array([[1, 2, 3], [2, 8, 2], [1, 1, 1]], float)
    print("Williams 2.6 Example 5: rotation about (5,4):\n" + mat_str(M))
    print("  A(1,2), B(2,8), C(3,2) ->", np.round(M @ tri, 6)[:2].T.tolist())
    ax = axs[2]
    setup(ax, 9, "rotate 90° about P(5,4)")
    ax.set_xlim(-1, 9); ax.set_ylim(-1, 9)
    draw_shape(ax, tri[:2], BLUE); draw_shape(ax, (M @ tri)[:2], ORANGE)
    ax.plot(5, 4, "*", color=GOLD, ms=16); ax.text(5.2, 4.2, "P", color=GOLD, fontsize=12)
    fig.tight_layout()
    show_or_save(fig, "13_composition")
    # DeFranza 4.6 Example 5
    P = np.array([[1, 0, -5], [0, 1, 3], [0, 0, 1]]); Q = np.diag([1.5, 1, 1]); Rf = np.diag([1, -1, 1])
    print("DeFranza 4.6 Example 5: R Q P =\n" + mat_str(Rf @ Q @ P))


def demo14():
    title("14. THE PUZZLE: three shears make a rotation")
    S1 = np.array([[1, -1], [0, 1]]); S2 = np.array([[1, 0], [1, 1]]); S3 = S1
    print("S3 S2 S1 =\n" + mat_str(S3 @ S2 @ S1), "\n= rotation by 90°:", np.allclose(S3 @ S2 @ S1, rot(np.pi / 2)))
    th = np.radians(37)
    a, b = -np.tan(th / 2), np.sin(th)
    P = np.array([[1, a], [0, 1]]) @ np.array([[1, 0], [b, 1]]) @ np.array([[1, a], [0, 1]])
    print("General: R(θ) = Sx(-tan(θ/2)) Sy(sinθ) Sx(-tan(θ/2)).  Check for θ = 37°:", np.allclose(P, rot(th)))
    stages = [np.eye(2), S1, S2 @ S1, S3 @ S2 @ S1]
    names = ["start", "1st shear (sideways)", "2nd shear (up)", "3rd shear (sideways) = turned 90°!"]
    if SAVE_ONLY:
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
        for ax, M, n in zip(axs, stages, names):
            setup(ax, 4, n); draw_grid_image(ax, M, 4); draw_shape(ax, M @ F, ORANGE)
        show_or_save(fig, "14_three_shears")
        return
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    def update(i):
        ax.clear()
        step, s = divmod(i, 50)
        step = min(step, 2); s = 1 if i >= 150 else min(1, s / 40)
        s = s * s * (3 - 2 * s)
        M = (1 - s) * stages[step] + s * stages[step + 1]
        setup(ax, 4, names[step + 1] if s > 0.5 else names[step])
        draw_grid_image(ax, M, 4)
        draw_shape(ax, F, BLUE, alpha=0.1, lw=1)
        draw_shape(ax, M @ F, ORANGE)
        return []

    anim = FuncAnimation(fig, update, frames=190, interval=35, repeat=True)
    plt.show()
    return anim


def demo15():
    title("15. APPLICATION: the fractal fern (Williams, Section 2.6)")
    maps = [(np.array([[0.86, 0.03], [-0.03, 0.86]]), np.array([0, 1.5]), 0.83),
            (np.array([[0.2, -0.25], [0.21, 0.23]]), np.array([0, 1.5]), 0.08),
            (np.array([[-0.15, 0.27], [0.25, 0.26]]), np.array([0, 0.45]), 0.08),
            (np.array([[0, 0], [0, 0.17]]), np.array([0, 0]), 0.01)]
    p = np.array([m[2] for m in maps]); p = p / p.sum()
    rng = np.random.default_rng(0)
    n = 60000
    choice = rng.choice(4, size=n, p=p)
    pts = np.zeros((n, 2)); v = np.zeros(2)
    for i, c in enumerate(choice):
        A, b, _ = maps[c]
        v = A @ v + b
        pts[i] = v
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    for ax, m in zip(axs, [1000, 10000, n]):
        ax.scatter(pts[:m, 0], pts[:m, 1], s=0.2 if m > 5000 else 1, color=GREEN)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"{m:,} points")
    print("Four affine maps T_i(x) = A_i x + b_i, picked at random, applied again and again.")
    show_or_save(fig, "15_fern")


def demo16():
    title("16. ANSWERS TO THE TEXTBOOK EXERCISES (checked numerically)")
    r = lambda d: rot(np.radians(d))
    checks = [
        ("Anton 1.8 No.29  reflect (-1,2): x-axis, y-axis, y=x",
         [np.diag([1, -1]) @ [-1, 2], np.diag([-1, 1]) @ [-1, 2], np.array([[0, 1], [1, 0]]) @ [-1, 2]]),
        ("Anton 1.8 No.31  reflect (2,-5,3) in xy, xz, yz",
         [np.diag([1, 1, -1]) @ [2, -5, 3], np.diag([1, -1, 1]) @ [2, -5, 3], np.diag([-1, 1, 1]) @ [2, -5, 3]]),
        ("Anton 1.8 No.33  project (2,-5) onto x-, y-axis", [np.diag([1, 0]) @ [2, -5], np.diag([0, 1]) @ [2, -5]]),
        ("Anton 1.8 No.35  project (-2,1,3) onto xy, xz, yz",
         [np.diag([1, 1, 0]) @ [-2, 1, 3], np.diag([1, 0, 1]) @ [-2, 1, 3], np.diag([0, 1, 1]) @ [-2, 1, 3]]),
        ("Anton 1.8 No.37  rotate (3,-4) by 30, -60, 45, 90 deg", [r(d) @ [3, -4] for d in (30, -60, 45, 90)]),
        ("Williams 2.6 No.15  (y,x) applied to (-3,5)", [np.array([[0, 1], [1, 0]]) @ [-3, 5]]),
        ("Williams 2.6 No.18  project (4,2) onto y=x", [0.5 * np.array([[1, 1], [1, 1]]) @ [4, 2]]),
        ("Williams 2.6 No.25(a,b,c)  images of (3,2)",
         [np.array([[1, 2], [0, 1]]) @ (3 * np.eye(2)) @ [3, 2],
          np.array([[0, 1], [1, 0]]) @ np.diag([3, 2]) @ [3, 2],
          r(90) @ np.array([[1, 3], [0, 1]]) @ (2 * np.eye(2)) @ [3, 2]]),
        ("Williams 2.6 No.26  A with (1,2)->(7,3), (-1,1)->(2,3)",
         [np.array([[7, 2], [3, 3]]) @ np.linalg.inv([[1, -1], [2, 1]])]),
        ("Williams 2.6 No.30  rotation 90° about (5,1)",
         [np.array([[1, 0, 5], [0, 1, 1], [0, 0, 1]]) @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
          @ np.array([[1, 0, -5], [0, 1, -1], [0, 0, 1]])]),
        ("Williams 2.6 No.33  S R T applied to A(1,6), B(3,0), C(4,6)",
         [(np.diag([3, 5, 1]) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) @ np.array([[1, 0, 4], [0, 1, -3], [0, 0, 1]])
           @ np.array([[1, 3, 4], [6, 0, 6], [1, 1, 1]]))[:2].T]),
        ("Williams Review No.24  3·R(30°)", [3 * r(30)]),
        ("Williams Review No.29  shear y 3, scale x 2, reflect y-axis",
         [np.diag([-1, 1]) @ np.diag([2, 1]) @ np.array([[1, 0], [3, 1]])]),
        ("DeFranza 4.6 No.3  [T] and image of (1,0),(3,0),(2,2)",
         [np.diag([1, -1]) @ np.diag([1, 0.5]) @ np.diag([3, 1]),
          (np.diag([3, -0.5]) @ np.array([[1, 3, 2], [0, 0, 2]])).T]),
        ("DeFranza 4.6 No.4  [T] and rectangle image",
         [np.array([[1, 3], [0, 1]]) @ np.diag([-1, 1]),
          (np.array([[-1, 3], [0, 1]]) @ np.array([[1, 2, 2, 1], [0, 0, 3, 3]])).T]),
        ("DeFranza 4.6 No.7  translate (1,1) then rotate 30°: parallelogram",
         [(np.block([[r(30), np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]])
           @ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]) @ np.array([[0, 2, 3, 1], [0, 0, 1, 1], [1, 1, 1, 1]]))[:2].T]),
        ("DeFranza 4.6 No.8  translate (-4,2) then reflect y-axis: trapezoid",
         [(np.diag([-1, 1, 1]) @ np.array([[1, 0, -4], [0, 1, 2], [0, 0, 1]])
           @ np.array([[0, 3, 2, 1], [0, 0, 1, 1], [1, 1, 1, 1]]))[:2].T]),
    ]
    for name, results in checks:
        print("\n" + name)
        for res in results:
            print(mat_str(res, 4))


DEMOS = {i + 1: f for i, f in enumerate([demo1, demo2, demo3, demo4, demo5, demo6, demo7, demo8,
                                         demo9, demo10, demo11, demo12, demo13, demo14, demo15, demo16])}


def menu():
    print(__doc__.split("Requirements:")[0])
    while True:
        c = input("Choose a demo number (1-16), 'a' for all, or 'q' to quit: ").strip().lower()
        if c == "q":
            break
        if c == "a":
            for f in DEMOS.values():
                f()
        elif c.isdigit() and int(c) in DEMOS:
            DEMOS[int(c)]()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--save"]
    if not args:
        if SAVE_ONLY:
            args = ["all"]
        else:
            menu()
    for a in args:
        if a == "all":
            for f in DEMOS.values():
                f()
        elif a.isdigit() and int(a) in DEMOS:
            DEMOS[int(a)]()
