"""
Linear Algebra - Lecture 3: Matrices and Matrix Operations
ZMathLab | Muhammad Zubair

What this file does
-------------------
1. Walks through every idea of the lecture with numpy, printing each step
   (run it and read the console like a second set of notes).
2. Checks every number used in the LaTeX notes, so nothing in the notes is wrong.
3. Draws all the figures used in the notes into ./figs (PDF, vector quality).

Run:   python lecture3_matrices.py
Needs: numpy, matplotlib

Colour language used in every figure (same as the video series):
    BLUE   = the original object (a matrix, a vector, a shape before)
    ORANGE = the result (after an operation)
    GOLD   = the thing we are focusing on (a row, a column, the diagonal)
    TEAL   = a second input (the other matrix / vector)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

NAVY = "#0F1B3D"
BLUE = "#2F6FDE"
ORANGE = "#F07C1E"
GOLD = "#F2B705"
TEAL = "#12A39A"
GREY = "#8A93A6"
LIGHT = "#EEF3FB"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": NAVY,
    "axes.labelcolor": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def say(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def check(name, got, expected):
    got = np.array(got); expected = np.array(expected)
    ok = got.shape == expected.shape and np.allclose(got, expected)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print("      got:\n", got, "\n      expected:\n", expected)
        raise SystemExit(f"Check failed: {name}")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure saved -> figs/{name}")


def draw_matrix(ax, M, x0=0, y0=0, cell=1.0, face=LIGHT, edge=NAVY, highlight=None,
                fmt="{:g}", fontsize=13, text_color=NAVY, label=None, label_color=NAVY):
    """Draw a matrix as a grid of cells. highlight = dict {(i,j): colour}."""
    M = np.array(M)
    m, n = M.shape
    for i in range(m):
        for j in range(n):
            fc = face
            if highlight and (i, j) in highlight:
                fc = highlight[(i, j)]
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell,
                                   facecolor=fc, edgecolor="white", lw=2))
            ax.text(x0 + (j + .5) * cell, y0 - (i + .5) * cell, fmt.format(M[i, j]),
                    ha="center", va="center", fontsize=fontsize, color=text_color, fontweight="bold")
    # brackets
    h = m * cell
    for side, xs in (("l", x0 - .08), ("r", x0 + n * cell + .08)):
        d = .18 if side == "l" else -.18
        ax.plot([xs + d, xs, xs, xs + d], [y0, y0, y0 - h, y0 - h], color=edge, lw=2.2)
    if label:
        ax.text(x0 + n * cell / 2, y0 + .35, label, ha="center", va="bottom", fontsize=13,
                color=label_color, fontweight="bold")


# ======================================================================
# PART 1 - WHAT A MATRIX IS
# ======================================================================
say("1. A matrix is a table of numbers with the labels removed")

hours = np.array([[2, 3, 2, 4, 1, 4, 2],
                  [0, 3, 1, 4, 3, 2, 2],
                  [4, 1, 3, 1, 0, 0, 2]])
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
subjects = ["Math", "History", "Language"]
print("Study-hours table (Anton, Section 1.3):\n", hours)
print("size =", hours.shape, "-> a 3 x 7 matrix (3 rows, 7 columns)")
print("entry a_{2,4} (History, Thursday) =", hours[1, 3])
check("a24 = 4", hours[1, 3], 4)
print("Total hours per subject (row sums):", hours.sum(axis=1))
check("row sums", hours.sum(axis=1), [18, 15, 11])
print("Total hours per day (column sums):", hours.sum(axis=0))
check("column sums", hours.sum(axis=0), [6, 7, 6, 9, 4, 6, 6])

fig, ax = plt.subplots(figsize=(8.4, 2.6))
im = ax.imshow(hours, cmap="Blues", vmin=-1, vmax=5)
for i in range(3):
    for j in range(7):
        ax.text(j, i, hours[i, j], ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if hours[i, j] >= 3 else NAVY)
ax.set_xticks(range(7), days)
ax.set_yticks(range(3), subjects)
ax.add_patch(Rectangle((2.5, 0.5), 1, 1, fill=False, edgecolor=ORANGE, lw=3))
ax.set_xlabel("orange box:  $a_{24}=4$   (row 2 = History, column 4 = Thursday)", color=ORANGE,
              fontsize=11, fontweight="bold", labelpad=8)
ax.set_title("A week of study hours = a 3 × 7 matrix", pad=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "study_hours.pdf")

# ----------------------------------------------------------------------
say("2. A photo on your phone is a matrix of brightness numbers")
smile = np.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 2, 2, 2, 2, 1, 0],
    [1, 2, 9, 2, 2, 9, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 9, 2, 2, 9, 2, 1],
    [1, 2, 2, 9, 9, 2, 2, 1],
    [0, 1, 2, 2, 2, 2, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0]])
img = np.where(smile == 0, 255, np.where(smile == 1, 40, np.where(smile == 2, 235, 20)))
print("8 x 8 grey image, 0 = black, 255 = white:\n", img)
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
axs[0].imshow(img, cmap="gray", vmin=0, vmax=255)
axs[0].set_title("What you see")
axs[1].imshow(img, cmap="gray", vmin=0, vmax=255, alpha=.18)
for i in range(8):
    for j in range(8):
        axs[1].text(j, i, img[i, j], ha="center", va="center", fontsize=8.5,
                    color=BLUE if img[i, j] > 128 else ORANGE, fontweight="bold")
axs[1].set_title("What the phone stores: an 8 × 8 matrix")
for a in axs:
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(NAVY)
save(fig, "photo_matrix.pdf")
# brighten = add a scalar to every entry ; contrast = scalar multiple
brighter = np.clip(img + 60, 0, 255)
check("brightening keeps the size", brighter.shape, (8, 8))

# ======================================================================
# PART 2 - DIAGONALS
# ======================================================================
say("3. Square matrix of order n and its diagonals")
A4 = np.arange(1, 17).reshape(4, 4)
print("A (order 4):\n", A4)
print("main diagonal  a11,a22,a33,a44 =", np.diag(A4))
print("anti-diagonal  a14,a23,a32,a41 =", np.fliplr(A4).diagonal())
print("super-diagonal a12,a23,a34    =", np.diag(A4, 1))
print("sub-diagonal   a21,a32,a43    =", np.diag(A4, -1))
check("main diagonal", np.diag(A4), [1, 6, 11, 16])
check("anti diagonal", np.fliplr(A4).diagonal(), [4, 7, 10, 13])
check("trace", np.trace(A4), 34)

fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
specs = [("Main diagonal  (i = j)", {(k, k): GOLD for k in range(4)}),
         ("Anti-diagonal  (i + j = n + 1)", {(k, 3 - k): ORANGE for k in range(4)}),
         ("Super-diagonal  (j = i + 1)", {(k, k + 1): TEAL for k in range(3)}),
         ("Sub-diagonal  (i = j + 1)", {(k + 1, k): BLUE for k in range(3)})]
for a, (t, hl) in zip(axs, specs):
    draw_matrix(a, A4, highlight=hl, fontsize=12)
    a.set_xlim(-.4, 4.4); a.set_ylim(-4.4, .5); a.set_aspect("equal"); a.axis("off")
    a.set_title(t, fontsize=10.5)
save(fig, "diagonals.pdf")

# ======================================================================
# PART 3 - ADDITION, SUBTRACTION, SCALAR MULTIPLE (Anton Ex. 3, 4)
# ======================================================================
say("4. Sum, difference and scalar multiple")
A = np.array([[2, 1, 0, 3], [-1, 0, 2, 4], [4, -2, 7, 0]])
B = np.array([[-4, 3, 5, 1], [2, 2, 0, -1], [3, 2, -4, 5]])
print("A + B =\n", A + B)
print("A - B =\n", A - B)
check("Anton Ex3 A+B", A + B, [[-2, 4, 5, 4], [1, 2, 2, 3], [7, 0, 3, 5]])
check("Anton Ex3 A-B", A - B, [[6, -2, -5, 2], [-3, -2, 2, 5], [1, -4, 11, -5]])
A2 = np.array([[2, 3, 4], [1, 3, 1]])
check("Anton Ex4 2A", 2 * A2, [[4, 6, 8], [2, 6, 2]])

# shop example (used as a story in the notes)
jan = np.array([[12, 5, 8], [20, 9, 4]])     # rows: shop 1, shop 2 ; cols: rice, sugar, oil
feb = np.array([[10, 7, 6], [18, 11, 5]])
print("Jan + Feb sales =\n", jan + feb)
check("shop jan+feb", jan + feb, [[22, 12, 14], [38, 20, 9]])
check("shop feb-jan", feb - jan, [[-2, 2, -2], [-2, 2, 1]])
check("10% price rise 1.1*price", 1.1 * np.array([200, 150, 400]), [220, 165, 440])

fig, axs = plt.subplots(1, 1, figsize=(10, 2.5))
ax = axs
draw_matrix(ax, jan, x0=0, label="January", label_color=BLUE, face="#DCE7FB")
ax.text(3.45, -1, "+", fontsize=26, ha="center", va="center", color=NAVY, fontweight="bold")
draw_matrix(ax, feb, x0=3.9, label="February", label_color=TEAL, face="#D5F1EE")
ax.text(7.35, -1, "=", fontsize=26, ha="center", va="center", color=NAVY, fontweight="bold")
draw_matrix(ax, jan + feb, x0=7.8, label="Two months", label_color=ORANGE, face="#FDE5CF")
ax.text(-.5, -.5, "Shop 1", ha="right", va="center", fontsize=10, color=GREY)
ax.text(-.5, -1.5, "Shop 2", ha="right", va="center", fontsize=10, color=GREY)
for j, item in enumerate(["rice", "sugar", "oil"]):
    ax.text(j + .5, -2.35, item, ha="center", va="top", fontsize=9.5, color=GREY)
ax.set_xlim(-1.6, 11.2); ax.set_ylim(-2.8, 1.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, "shop_sum.pdf")

# ======================================================================
# PART 4 - PRODUCT AB (Anton Example 5)
# ======================================================================
say("5. Matrix product, row-column rule (Anton Example 5)")
A5 = np.array([[1, 2, 4], [2, 6, 0]])
B5 = np.array([[4, 1, 4, 3], [0, -1, 3, 1], [2, 7, 5, 2]])
AB5 = A5 @ B5
print("A (2x3) times B (3x4) = AB (2x4):\n", AB5)
check("Anton Ex5 AB", AB5, [[12, 27, 30, 13], [8, -4, 26, 12]])
for i in range(2):
    for j in range(4):
        terms = " + ".join(f"({A5[i,k]})({B5[k,j]})" for k in range(3))
        print(f"  (AB)_{i+1}{j+1} = {terms} = {AB5[i,j]}")

fig, ax = plt.subplots(figsize=(10.5, 3.6))
hlA = {(1, k): GOLD for k in range(3)}
hlB = {(k, 2): GOLD for k in range(3)}
draw_matrix(ax, A5, x0=0, y0=0, highlight=hlA, label="A   (2 × 3)", label_color=BLUE, face="#DCE7FB")
draw_matrix(ax, B5, x0=3.6, y0=0.5, highlight=hlB, label="B   (3 × 4)", label_color=TEAL, face="#D5F1EE")
ax.text(8.15, -1, "=", fontsize=26, ha="center", va="center", color=NAVY, fontweight="bold")
out = np.full((2, 4), np.nan)
hl = {(1, 2): ORANGE}
draw_matrix(ax, np.where(np.isnan(out), 0, out), x0=8.6, y0=0, highlight=hl, fmt="",
            label="AB   (2 × 4)", label_color=ORANGE, face="#FDE5CF")
ax.text(8.6 + 2.5, -1.5, "26", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
ax.text(6.4, -3.15, "row 2 of A  ·  column 3 of B  =  (2)(4) + (6)(3) + (0)(5)  =  26",
        ha="center", fontsize=12.5, color=NAVY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=.4", fc="#FFF6D6", ec=GOLD))
ax.set_xlim(-.6, 13.2); ax.set_ylim(-3.6, 1.4); ax.set_aspect("equal"); ax.axis("off")
save(fig, "row_column_rule.pdf")

# ----------------------------------------------------------------------
say("6. Why AB is not BA (picture: two moves on a shape)")
house = np.array([[0, 0], [2, 0], [2, 1.5], [1, 2.5], [0, 1.5], [0, 0]]).T
R = np.array([[0, -1], [1, 0]])        # rotate 90 degrees
S = np.array([[1, 1], [0, 1]])         # shear right
print("R (rotate 90):\n", R, "\nS (shear):\n", S)
print("RS =\n", R @ S, "\nSR =\n", S @ R)
check("RS != SR", not np.allclose(R @ S, S @ R), True)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for a, (M, t) in zip(axs, [(R @ S, "First shear S, then rotate R:   (RS)x"),
                           (S @ R, "First rotate R, then shear S:   (SR)x")]):
    a.fill(*house, color=BLUE, alpha=.25); a.plot(*house, color=BLUE, lw=2.5, label="original house")
    h2 = M @ house
    a.fill(*h2, color=ORANGE, alpha=.25); a.plot(*h2, color=ORANGE, lw=2.5, label="result")
    a.axhline(0, color=GREY, lw=.8); a.axvline(0, color=GREY, lw=.8)
    a.set_xlim(-4, 4); a.set_ylim(-1, 4.2); a.set_aspect("equal"); a.grid(alpha=.25)
    a.set_title(t, fontsize=11)
    a.legend(loc="upper right", fontsize=9, frameon=False)
fig.suptitle("Same two moves, different order, different house:  RS ≠ SR",
             color=NAVY, fontweight="bold", y=1.0)
save(fig, "not_commutative.pdf")

# ----------------------------------------------------------------------
say("7. Columns of AB are A times columns of B (Anton Example 7)")
check("2nd column of AB", A5 @ B5[:, 1], [27, -4])
check("1st row of AB", A5[0] @ B5, [12, 27, 30, 13])

say("8. Ax is a linear combination of the columns of A (Anton Example 8)")
A8 = np.array([[-1, 3, 2], [1, 2, -3], [2, 1, -2]])
x8 = np.array([2, -1, 3])
lc = 2 * A8[:, 0] - 1 * A8[:, 1] + 3 * A8[:, 2]
print("Ax =", A8 @ x8, "  2c1 - c2 + 3c3 =", lc)
check("Anton Ex8", A8 @ x8, [1, -9, -3])
check("Anton Ex8 as combination", lc, [1, -9, -3])

# 2-D picture of a linear combination
a1 = np.array([2, 1]); a2 = np.array([-1, 2]); xs = np.array([2, 1.5])
target = xs[0] * a1 + xs[1] * a2
print("A = [a1 a2] =", np.column_stack([a1, a2]).tolist(), " x =", xs, " Ax =", target)
check("2D Ax", np.column_stack([a1, a2]) @ xs, target)
fig, ax = plt.subplots(figsize=(6.4, 5.6))
def arrow(p, q, c, lw=3, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, color=c, lw=lw, linestyle=ls))
arrow((0, 0), a1, BLUE); ax.text(*(a1 * .55 + [0.1, -.45]), "$\\mathbf{a}_1$", color=BLUE, fontsize=15)
arrow((0, 0), a2, TEAL); ax.text(*(a2 * .55 + [-.55, .0]), "$\\mathbf{a}_2$", color=TEAL, fontsize=15)
arrow((0, 0), xs[0] * a1, BLUE, lw=2, ls="--")
arrow(xs[0] * a1, target, TEAL, lw=2, ls="--")
arrow((0, 0), target, ORANGE, lw=3.5)
ax.text(*(xs[0] * a1 + [0.15, -.5]), "$2\\,\\mathbf{a}_1$", color=BLUE, fontsize=13)
ax.text(*(target + [0.15, .1]), "$A\\mathbf{x}=2\\mathbf{a}_1+1.5\\,\\mathbf{a}_2=(2.5,\\,5)$", color=ORANGE, fontsize=12.5)
ax.set_xlim(-2, 6); ax.set_ylim(-1, 6.2); ax.set_aspect("equal"); ax.grid(alpha=.3)
ax.axhline(0, color=GREY, lw=.8); ax.axvline(0, color=GREY, lw=.8)
ax.set_title("Ax walks x1 steps along a1, then x2 steps along a2")
save(fig, "linear_combination.pdf")

# ----------------------------------------------------------------------
say("9. Column-row expansion (Anton Example 10)")
A10 = np.array([[1, 3], [2, -1]]); B10 = np.array([[2, 0, 4], [-3, 5, 1]])
P1 = np.outer(A10[:, 0], B10[0]); P2 = np.outer(A10[:, 1], B10[1])
print("c1 r1 =\n", P1, "\nc2 r2 =\n", P2, "\nsum =\n", P1 + P2, "\nAB =\n", A10 @ B10)
check("c1r1", P1, [[2, 0, 4], [4, 0, 8]])
check("c2r2", P2, [[-9, 15, 3], [3, -5, -1]])
check("Anton Ex10", P1 + P2, [[-7, 15, 7], [7, -5, 7]])
check("Anton Ex10 = AB", A10 @ B10, P1 + P2)
fig, ax = plt.subplots(figsize=(10.5, 2.4))
draw_matrix(ax, P1, x0=0, label="$\\mathbf{c}_1\\mathbf{r}_1$", label_color=BLUE, face="#DCE7FB")
ax.text(3.45, -1, "+", fontsize=26, ha="center", va="center", color=NAVY, fontweight="bold")
draw_matrix(ax, P2, x0=3.9, label="$\\mathbf{c}_2\\mathbf{r}_2$", label_color=TEAL, face="#D5F1EE")
ax.text(7.35, -1, "=", fontsize=26, ha="center", va="center", color=NAVY, fontweight="bold")
draw_matrix(ax, P1 + P2, x0=7.8, label="$AB$", label_color=ORANGE, face="#FDE5CF")
ax.set_xlim(-.5, 11.2); ax.set_ylim(-2.4, 1.0); ax.set_aspect("equal"); ax.axis("off")
save(fig, "column_row_expansion.pdf")

# ----------------------------------------------------------------------
say("10. Partitioned (block) multiplication (Williams Example 5)")
AW = np.array([[1, -1], [3, 0], [2, 4]]); BW = np.array([[1, 2, -1], [1, 3, 1]])
top = AW[:2, :1] @ BW[:1] + AW[:2, 1:] @ BW[1:]
bot = AW[2:, :1] @ BW[:1] + AW[2:, 1:] @ BW[1:]
print("top block =\n", top, "\nbottom block =\n", bot)
check("Williams Ex5 blocks", np.vstack([top, bot]), AW @ BW)
check("Williams Ex5 AB", AW @ BW, [[0, -1, -2], [3, 6, -3], [6, 16, 2]])

# block picture: big matrix with a zero block -> less work
fig, ax = plt.subplots(figsize=(6.6, 4.4))
n = 8
for i in range(n):
    for j in range(n):
        if i < 4 and j < 4: c = "#DCE7FB"
        elif i >= 4 and j >= 4: c = "#D5F1EE"
        elif i < 4 and j >= 4: c = "#FFF6D6"
        else: c = "#F1F1F1"
        ax.add_patch(Rectangle((j, -i - 1), 1, 1, facecolor=c, edgecolor="white", lw=1.5))
ax.plot([4, 4], [0, -8], color=NAVY, lw=2.5); ax.plot([0, 8], [-4, -4], color=NAVY, lw=2.5)
for (x, y, t, c) in [(2, -2, "$A_{11}$", BLUE), (6, -2, "$A_{12}$", GOLD),
                     (2, -6, "$O$\n(all zeros)", GREY), (6, -6, "$A_{22}$", TEAL)]:
    ax.text(x, y, t, ha="center", va="center", fontsize=17, color=c if c != GOLD else "#9A7300", fontweight="bold")
ax.set_xlim(-.2, 8.2); ax.set_ylim(-8.2, .6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("An 8 × 8 matrix seen as a 2 × 2 matrix of 4 × 4 blocks")
save(fig, "blocks.pdf")

# ======================================================================
# PART 5 - TRANSPOSE
# ======================================================================
say("11. Transpose: rows become columns (Anton (15))")
At = np.array([[1, -2, 4], [3, 7, 0], [-5, 8, 6]])
print("A =\n", At, "\nA^T =\n", At.T)
check("Anton (15) transpose", At.T, [[1, 3, -5], [-2, 7, 8], [4, 0, 6]])
Mr = np.array([[2, 3], [1, 4], [5, 6]])
check("(A^T)^T = A", Mr.T.T, Mr)
X = np.array([[1, 2], [3, 4]]); Y = np.array([[0, 1], [5, -2]])
check("(XY)^T = Y^T X^T", (X @ Y).T, Y.T @ X.T)
print("(XY)^T =\n", (X @ Y).T, "\nX^T Y^T =\n", X.T @ Y.T, "  <- NOT equal")
check("(XY)^T != X^T Y^T", not np.allclose((X @ Y).T, X.T @ Y.T), True)

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
cols = {0: BLUE, 1: TEAL, 2: ORANGE}
hlA = {(i, j): ["#DCE7FB", "#D5F1EE", "#FDE5CF"][i] for i in range(3) for j in range(3)}
hlT = {(i, j): ["#DCE7FB", "#D5F1EE", "#FDE5CF"][j] for i in range(3) for j in range(3)}
for k in range(3):
    hlA[(k, k)] = GOLD; hlT[(k, k)] = GOLD
draw_matrix(axs[0], At, highlight=hlA, label="A  (rows coloured)")
draw_matrix(axs[1], At.T, highlight=hlT, label="$A^T$  (same colours, now columns)")
for a in axs:
    a.set_xlim(-.4, 3.4); a.set_ylim(-3.4, .9); a.set_aspect("equal"); a.axis("off")
axs[0].plot([0, 3], [0, -3], color=NAVY, lw=1.5, ls="--")
axs[0].text(3.1, -3.1, "mirror\n(main diagonal)", fontsize=9, color=NAVY, va="top")
save(fig, "transpose.pdf")

# transpose of a picture = reflect in the line y = -x (image rows <-> columns)
Fimg = np.zeros((9, 9))
Fimg[1:8, 2] = 1; Fimg[1, 2:7] = 1; Fimg[4, 2:6] = 1   # the letter F
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6))
axs[0].imshow(Fimg, cmap="Blues", vmin=0, vmax=1.3); axs[0].set_title("image matrix F")
axs[1].imshow(Fimg.T, cmap="Oranges", vmin=0, vmax=1.3); axs[1].set_title("$F^T$: flipped over the diagonal")
for a in axs:
    a.set_xticks([]); a.set_yticks([]); a.plot([-.5, 8.5], [-.5, 8.5], color=GOLD, lw=2, ls="--")
save(fig, "transpose_picture.pdf")

# ======================================================================
# PART 6 - HOMOGENEOUS SYSTEMS
# ======================================================================
say("12. Homogeneous system Ax = 0: trivial vs nontrivial")
H1 = np.array([[1, 1], [1, -1]])
H2 = np.array([[1, 2], [2, 4]])
print("det H1 =", round(np.linalg.det(H1), 6), "-> only x = 0")
print("det H2 =", round(np.linalg.det(H2), 6), "-> a whole line of solutions, e.g. (2,-1)")
check("H2 (2,-1) solves", H2 @ np.array([2, -1]), [0, 0])
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
t = np.linspace(-3, 3, 50)
axs[0].plot(t, -t, color=BLUE, lw=2.5, label="$x+y=0$")
axs[0].plot(t, t, color=TEAL, lw=2.5, label="$x-y=0$")
axs[0].plot(0, 0, "o", color=ORANGE, ms=11)
axs[0].set_title("Only the trivial solution (0, 0)")
axs[1].plot(t, -t / 2, color=BLUE, lw=6, alpha=.5, label="$x+2y=0$")
axs[1].plot(t, -t / 2, color=TEAL, lw=2, ls="--", label="$2x+4y=0$")
axs[1].plot([2], [-1], "o", color=ORANGE, ms=10); axs[1].text(2.1, -.75, "(2, -1)", color=ORANGE, fontsize=11)
axs[1].plot(0, 0, "o", color=NAVY, ms=7)
axs[1].set_title("Same line: infinitely many (nontrivial) solutions")
for a in axs:
    a.axhline(0, color=GREY, lw=.8); a.axvline(0, color=GREY, lw=.8)
    a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal"); a.grid(alpha=.25)
    a.legend(loc="upper left", fontsize=10, frameon=False)
save(fig, "homogeneous.pdf")

# ======================================================================
# PART 7 - TRACE
# ======================================================================
say("13. Trace = sum of the main diagonal (Anton Example 12)")
B12 = np.array([[-1, 2, 7, 0], [3, 5, -8, 4], [1, 2, 7, -3], [4, -2, 1, 0]])
print("tr(B) =", np.trace(B12))
check("Anton Ex12 tr(B)", np.trace(B12), 11)
P = np.array([[1, 2], [3, 4]]); Q = np.array([[0, 1], [1, 0]])
print("tr(PQ) =", np.trace(P @ Q), " tr(P)tr(Q) =", np.trace(P) * np.trace(Q))
check("tr(PQ)=tr(QP)", np.trace(P @ Q), np.trace(Q @ P))
check("tr(PQ) != tr(P)tr(Q)", np.trace(P @ Q) != np.trace(P) * np.trace(Q), True)
check("tr(A^T)=tr(A)", np.trace(B12.T), np.trace(B12))

# trace story: who-played-whom matrix, diagonal = self-practice
fig, ax = plt.subplots(figsize=(4.6, 4.2))
hl = {(k, k): GOLD for k in range(4)}
draw_matrix(ax, B12, highlight=hl, fontsize=13)
ax.text(2, -4.6, "tr(B) = -1 + 5 + 7 + 0 = 11", ha="center", fontsize=13, color=NAVY, fontweight="bold")
ax.set_xlim(-.4, 4.4); ax.set_ylim(-5, .5); ax.set_aspect("equal"); ax.axis("off")
save(fig, "trace.pdf")

# ======================================================================
# PART 8 - COMMON MISTAKES
# ======================================================================
say("14. Mistakes students make (each one shown false with numbers)")
C = np.array([[1, 2], [-3, 0]]); D = np.array([[4, -1], [2, 3]])      # Williams Example 2
check("Williams Ex2 CD", C @ D, [[8, 5], [-12, 3]])
check("Williams Ex2 DC", D @ C, [[7, 8], [-7, 4]])
print("CD =\n", C @ D, "\nDC =\n", D @ C)
print("entrywise C*D =\n", C * D, "  <- this is NOT the matrix product")
Az = np.array([[0, 1], [0, 2]]); Bz = np.array([[1, 1], [3, 4]]); Cz = np.array([[2, 5], [3, 4]])
check("AB = AC but B != C", Az @ Bz, Az @ Cz)
check("AA = 0 with A != 0", np.array([[0, 1], [0, 0]]) @ np.array([[0, 1], [0, 0]]), [[0, 0], [0, 0]])
check("(A+B)^2 != A^2+2AB+B^2", not np.allclose((C + D) @ (C + D), C @ C + 2 * C @ D + D @ D), True)

# ======================================================================
# PART 9 - APPLICATION: prices x quantities (Anton Ex. 33)
# ======================================================================
say("15. Shopping bill as a product (Anton Exercise 33)")
Q33 = np.array([[3, 4, 3], [5, 6, 0], [2, 9, 4], [1, 1, 7]]); p33 = np.array([1, 2, 3])
print("monthly spend =", Q33 @ p33)
check("Anton Ex33", Q33 @ p33, [20, 17, 32, 24])

print("\nAll checks passed. Figures are in ./figs")
