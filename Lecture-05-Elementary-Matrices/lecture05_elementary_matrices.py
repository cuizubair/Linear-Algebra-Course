"""
Linear Algebra - Lecture 5: ELEMENTARY MATRICES
Muhammad Zubair  (ZMathLab)

Companion Python file for the lecture notes, the slides and the video.
Every section prints a step-by-step explanation in the console AND draws
a clear picture that is saved in the folder  pictures/  next to this file.

  1. Building elementary matrices from I   (swap, scale, add)
  2. E·A  =  the row operation              (Anton Example 2)
  3. The inverse of an elementary matrix    (E·E^-1 = I)
  4. The inversion algorithm [A | I] -> [I | A^-1]   (exact fractions)
       Anton Example 4,  Anton Exercise Set 1.5 No.23,  Anton Example 5 (not invertible)
  5. LU decomposition from elementary matrices
  6. Application: cryptography (Hill cipher)  "BUY IBM STOCK",  Williams 37 / 39
  7. Application: colour models  RGB <-> YIQ
  8. Geometry in 2D: swap, scale, shear act on the letter F;  three shears = 90° turn
  9. Geometry in 3D: the unit cube under elementary matrices, volume = |det|

Requirements:  pip install numpy matplotlib
Run:           python elementary_matrices.py            (all sections, save every picture)
               python elementary_matrices.py 4 6        (only sections 4 and 6)
               python elementary_matrices.py --show     (also open the windows)

Books used for the examples:
  Anton & Rorres, Elementary Linear Algebra (Section 1.5)
  G. Williams, Linear Algebra with Applications (Section 2.4 / applications)
"""

import os
import sys
from fractions import Fraction

import numpy as np
import matplotlib

SHOW = "--show" in sys.argv
if not SHOW:
    matplotlib.use("Agg")          # no windows, just PNG files

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pictures")

# ----------------------------------------------------------------------------
# Colours (same as the notes): navy + gold, blue = original, orange = image
# ----------------------------------------------------------------------------
BLUE, ORANGE, GREEN, RED, PURPLE, GOLD, NAVY, GRAY = (
    "#3478c8", "#e2622e", "#1e965a", "#d23c3c", "#7d46be", "#d69a14", "#0b1832", "#9aa3b5")
CELL, CELL_ON, CELL_HL, CELL_BAD = "#ffffff", "#e3e9f5", "#fde7a8", "#f7c6c6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": NAVY,
    "font.size": 10,
})

# The letter F: it has no symmetry, so every flip, stretch or slant is visible.
F = np.array([[0, 0], [0, 3], [2, 3], [2, 2.4], [0.6, 2.4], [0.6, 1.8],
              [1.6, 1.8], [1.6, 1.2], [0.6, 1.2], [0.6, 0]], dtype=float).T
SQUARE = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)


# ============================================================================
# Small helpers: printing
# ============================================================================
def title(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def sub(text):
    print("\n--- " + text + " " + "-" * max(0, 66 - len(text)))


def fstr(v, digits=3):
    """Nice string for one entry: Fraction -> 'a/b', float -> rounded."""
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = round(float(v), digits) + 0.0
    return f"{v:g}"


def mat_str(A, digits=3, divider=None, indent="  "):
    """Pretty string of a matrix (fractions or numbers), columns aligned."""
    A = [[fstr(v, digits) for v in row] for row in np.atleast_2d(np.asarray(A, dtype=object))]
    w = max(len(s) for row in A for s in row)
    lines = []
    for row in A:
        cells = [s.rjust(w) for s in row]
        if divider is not None:
            cells.insert(divider, "|")
        lines.append(indent + "[ " + "  ".join(cells) + " ]")
    return "\n".join(lines)


def frac_matrix(A):
    return np.array([[Fraction(v) for v in row] for row in A], dtype=object)


def identity(n):
    return frac_matrix(np.eye(n, dtype=int))


def is_identity(M):
    n = len(M)
    return all(M[i][j] == (1 if i == j else 0) for i in range(n) for j in range(n))


def show_or_save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("  [picture saved]", os.path.relpath(path, HERE))
    if SHOW:
        plt.show()
    plt.close(fig)


# ============================================================================
# Row operations and elementary matrices
#   ("swap", i, j)        Ri <-> Rj
#   ("scale", i, c)       Ri <- c Ri          (c != 0)
#   ("add", i, j, c)      Ri <- Ri + c Rj
#   rows are numbered 0,1,2,... inside the code and R1,R2,R3,... on screen
# ============================================================================
def op_name(op):
    if op[0] == "swap":
        return f"R{op[1] + 1} <-> R{op[2] + 1}"
    if op[0] == "scale":
        c = Fraction(op[2])
        return f"({fstr(c)})·R{op[1] + 1}" if c.denominator != 1 or c < 0 else f"{fstr(c)}·R{op[1] + 1}"
    i, j, c = op[1], op[2], Fraction(op[3])
    sign = "+" if c >= 0 else "-"
    a = abs(c)
    coef = "" if a == 1 else (f"({fstr(a)})" if a.denominator != 1 else fstr(a))
    return f"R{i + 1} {sign} {coef}R{j + 1}".replace(" )", ")")


def op_pretty(op):
    """Same as op_name but with nice symbols for the pictures."""
    return op_name(op).replace("<->", "↔").replace(" - ", " − ")


def inverse_op(op):
    if op[0] == "swap":
        return op
    if op[0] == "scale":
        return ("scale", op[1], 1 / Fraction(op[2]))
    return ("add", op[1], op[2], -Fraction(op[3]))


def apply_op(M, op):
    M = M.copy()
    if op[0] == "swap":
        M[[op[1], op[2]]] = M[[op[2], op[1]]]
    elif op[0] == "scale":
        M[op[1]] = M[op[1]] * Fraction(op[2])
    else:
        M[op[1]] = M[op[1]] + Fraction(op[3]) * M[op[2]]
    return M


def elementary(n, op):
    """The elementary matrix of an operation = the operation done to I."""
    return apply_op(identity(n), op)


def changed_rows(op):
    return {op[1], op[2]} if op[0] == "swap" else {op[1]}


# ============================================================================
# Helpers: drawing matrices as annotated grids (on one big "board")
#   A board is a list of rows; each row is a list of items:
#     ("M", matrix, options)   or   ("T", "text", options)
#   Sizes are in "cell units", so all matrices on a board have the same cells.
# ============================================================================
CH = 0.78              # cell height  (cell units)


def cell_width(M):
    longest = max(len(fstr(v)) for row in M for v in row)
    return max(1.0, 0.26 * longest + 0.45)


def item_size(item):
    kind, obj, opt = item
    if kind == "M":
        M = np.atleast_2d(np.asarray(obj, dtype=object))
        cw = opt.get("cw", cell_width(M))
        extra = 0.25 if opt.get("divider") else 0
        return M.shape[1] * cw + extra + 0.5, M.shape[0] * CH
    return opt.get("w", 0.32 * len(obj) + 0.4), CH


def draw_matrix(ax, M, x, ytop, opt, fs):
    """Draw M with its top-left corner at (x, ytop). Returns nothing."""
    M = np.atleast_2d(np.asarray(M, dtype=object))
    m, n = M.shape
    cw = opt.get("cw", cell_width(M))
    div = opt.get("divider")
    rows_hl = opt.get("rows", {})        # {row: colour}
    cells_hl = opt.get("cells", {})      # {(i, j): colour}
    x0 = x + 0.25
    for i in range(m):
        for j in range(n):
            cx = x0 + j * cw + (0.25 if div and j >= div else 0)
            cy = ytop - (i + 1) * CH
            v = M[i, j]
            fc = CELL_ON if (v != 0 and opt.get("shade", True)) else CELL
            bold = False
            if i in rows_hl:
                fc, bold = rows_hl[i], True
            if (i, j) in cells_hl:
                fc, bold = cells_hl[(i, j)], True
            ax.add_patch(Rectangle((cx + 0.04, cy + 0.04), cw - 0.08, CH - 0.08, fc=fc,
                                   ec="#c9d1e3", lw=0.8))
            ax.text(cx + cw / 2, cy + CH / 2, fstr(v), ha="center", va="center", fontsize=fs,
                    color=NAVY, fontweight="bold" if bold else "normal")
    w = n * cw + (0.25 if div else 0)
    # square brackets
    for xb, d in [(x0 - 0.06, 0.16), (x0 + w + 0.06, -0.16)]:
        ax.plot([xb + d, xb, xb, xb + d], [ytop, ytop, ytop - m * CH, ytop - m * CH],
                color=NAVY, lw=1.8, solid_capstyle="butt")
    if div:
        xd = x0 + div * cw + 0.125
        ax.plot([xd, xd], [ytop - m * CH, ytop], color=NAVY, lw=1.4, ls="--")
    if opt.get("label"):
        ax.text(x0 + w / 2, ytop + 0.12, opt["label"], ha="center", va="bottom",
                fontsize=opt.get("lfs", fs), color=opt.get("lcolor", NAVY), fontweight="bold")


def board(rows, name, heading="", fs=11, unit=0.52, row_gap=1.25, col_gap=0.25, row_titles=None):
    """Lay out rows of matrices / symbols and save the picture."""
    sizes = [[item_size(it) for it in row] for row in rows]
    widths = [sum(w for w, _ in r) + col_gap * (len(r) - 1) for r in sizes]
    heights = [max(h for _, h in r) for r in sizes]
    has_label = [any(it[0] == "M" and it[2].get("label") for it in row) for row in rows]
    title_h = [0.75 if (row_titles and row_titles[k]) else 0 for k in range(len(rows))]
    W = max(widths) + 0.6
    H = sum(heights) + sum(0.7 for h in has_label if h) + sum(title_h) + row_gap * (len(rows) - 1) + 0.5
    head_h = 0.9 if heading else 0
    fig = plt.figure(figsize=(W * unit, (H + head_h) * unit))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(-(H + head_h), 0)
    ax.axis("off")
    y = -head_h - 0.2
    if heading:
        ax.text(W / 2, -0.45, heading, ha="center", va="center", fontsize=fs + 3, color=NAVY,
                fontweight="bold")
    for k, row in enumerate(rows):
        if title_h[k]:
            ax.text(0.3, y - 0.3, row_titles[k], ha="left", va="center", fontsize=fs + 1,
                    color=GOLD, fontweight="bold")
            y -= title_h[k]
        if has_label[k]:
            y -= 0.7
        x = 0.3 + (max(widths) - widths[k]) / 2
        hrow = heights[k]
        for it, (w, h) in zip(row, sizes[k]):
            kind, obj, opt = it
            ytop = y - (hrow - h) / 2
            if kind == "M":
                draw_matrix(ax, obj, x, ytop, opt, fs)
            else:
                ax.text(x + w / 2, ytop - CH / 2, obj, ha="center", va="center",
                        fontsize=opt.get("fs", fs + 4), color=opt.get("color", NAVY),
                        fontweight=opt.get("weight", "bold"))
            x += w + col_gap
        y -= hrow + row_gap
    show_or_save(fig, name)


def Mi(M, **opt):
    return ("M", M, opt)


def Tx(t, **opt):
    return ("T", t, opt)


# ============================================================================
# Helpers: plane pictures (same style as Lecture 8)
# ============================================================================
def setup(ax, lim=4, title_text="", lo=None):
    lo = -lim if lo is None else lo
    ax.set_xlim(lo, lim)
    ax.set_ylim(lo, lim)
    ax.set_aspect("equal")
    ax.set_xticks(range(int(lo), lim + 1))
    ax.set_yticks(range(int(lo), lim + 1))
    ax.grid(color="#dfe4ee", lw=0.8)
    ax.axhline(0, color=GRAY, lw=1.2)
    ax.axvline(0, color=GRAY, lw=1.2)
    ax.tick_params(labelsize=7, colors=GRAY)
    for s in ax.spines.values():
        s.set_visible(False)
    if title_text:
        ax.set_title(title_text)


def draw_shape(ax, P, color, alpha=0.35, lw=2, label=None):
    ax.add_patch(Polygon(P.T, closed=True, fc=color, ec=color, alpha=alpha, lw=lw, label=label))
    ax.add_patch(Polygon(P.T, closed=True, fill=False, ec=color, lw=lw))


def draw_grid_image(ax, A, lim=4, color=BLUE, alpha=0.25):
    t = np.linspace(-lim - 6, lim + 6, 2)
    for c in range(-lim - 6, lim + 7):
        v = A @ np.vstack([np.full(2, c), t])
        h = A @ np.vstack([t, np.full(2, c)])
        ax.plot(v[0], v[1], color=color, alpha=alpha, lw=1)
        ax.plot(h[0], h[1], color=color, alpha=alpha, lw=1)


def info_box(ax, text, x=0.03, y=0.97, ha="left", va="top"):
    ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, family="monospace",
            fontsize=8.5, color=NAVY, bbox=dict(boxstyle="round", fc="#fff8e6", ec=GOLD))


# ============================================================================
# 1. Building elementary matrices from I
# ============================================================================
def section1():
    title("1. ELEMENTARY MATRICES: do ONE row operation to the identity I")
    print("An n x n matrix E is ELEMENTARY if it comes from I_n by a single row operation.")
    print("There are exactly three kinds:")
    I3 = identity(3)
    cases = [("swap two rows", ("swap", 0, 1)),
             ("multiply a row by c ≠ 0", ("scale", 1, 3)),
             ("add a multiple of one row to another", ("add", 2, 0, 3))]
    rows, titles = [], []
    for k, (kind, op) in enumerate(cases, 1):
        E = elementary(3, op)
        print(f"\n  Type {k}: {kind:38s}  operation  {op_name(op)}")
        print("  I =\n" + mat_str(I3, indent="    "))
        print(f"  E{k} =\n" + mat_str(E, indent="    "))
        hl = {r: CELL_HL for r in changed_rows(op)}
        rows.append([Mi(I3, label="I₃"), Tx("──  " + op_pretty(op) + "  ──>", fs=13, w=6.4),
                     Mi(E, label=f"E{k}", rows=hl)])
        titles.append(f"Type {k}: {kind}")
    print("\nNOT elementary: [[1,0],[0,1]] -> [[2,0],[0,3]] (two operations),"
          " [[0,1,0],[0,0,1],[1,0,0]] (two swaps).")
    print("The gold rows in the picture are the rows that changed.")
    board(rows, "01_building_elementary_matrices",
          "Elementary matrix = identity after ONE row operation", row_titles=titles)


# ============================================================================
# 2. E·A = the row operation
# ============================================================================
def section2():
    title("2. E·A  =  doing the row operation on A   (Anton Section 1.5, Example 2)")
    A = frac_matrix([[1, 0, 2, 3], [2, -1, 3, 6], [1, 4, 4, 0]])
    op = ("add", 2, 0, 3)
    E = elementary(3, op)
    EA = E @ A
    print("A =\n" + mat_str(A))
    print(f"E = I with the operation {op_name(op)} (add 3 times row 1 to row 3) =\n" + mat_str(E))
    print("\nEach entry of EA = (row of E) · (column of A):")
    for i in range(3):
        for j in range(4):
            terms = " + ".join(f"({fstr(E[i, k])})({fstr(A[k, j])})" for k in range(3))
            print(f"  (EA){i + 1}{j + 1} = {terms} = {fstr(EA[i, j])}")
    print("\nEA =\n" + mat_str(EA))
    direct = apply_op(A, op)
    print(f"Doing {op_name(op)} directly on A gives\n" + mat_str(direct))
    print("Same matrix?", bool((direct == EA).all()), " -> multiplying on the LEFT by E = doing the row operation.")
    print("Rows 1 and 2 of E are rows of I, so rows 1 and 2 of A are copied unchanged.")
    board([[Mi(E, label="E   (R3 + 3R1 on I)", rows={2: CELL_HL}), Tx("×"),
            Mi(A, label="A", rows={0: "#d6e6fa"}), Tx("="),
            Mi(EA, label="E·A", rows={2: CELL_HL})],
           [Mi(A, label="A"), Tx("──  R3 + 3R1  ──>", fs=13, w=5.6),
            Mi(direct, label="same result!", rows={2: CELL_HL})]],
          "02_E_times_A_is_a_row_operation", "Multiplying by E on the left  =  the row operation",
          row_titles=["Matrix multiplication:  row 3 of EA = 3·(row 1 of A) + (row 3 of A)",
                      "Row operation done directly on A"])


# ============================================================================
# 3. Inverse of an elementary matrix
# ============================================================================
def section3():
    title("3. EVERY ELEMENTARY MATRIX IS INVERTIBLE: undo the operation")
    print("Rule: E^-1 is the elementary matrix of the OPPOSITE operation.")
    print("   swap Ri<->Rj   is undone by   the same swap")
    print("   c·Ri           is undone by   (1/c)·Ri")
    print("   Ri + c Rj      is undone by   Ri - c Rj")
    cases = [("swap", 0, 1), ("scale", 1, 3), ("add", 2, 0, 3)]
    rows, titles = [], []
    for k, op in enumerate(cases, 1):
        E, Einv = elementary(3, op), elementary(3, inverse_op(op))
        P = E @ Einv
        print(f"\n  E{k}: {op_name(op):12s}   E{k}^-1: {op_name(inverse_op(op))}")
        print(mat_str(E, indent="    ") + "\n    times\n" + mat_str(Einv, indent="    ")
              + "\n    =\n" + mat_str(P, indent="    "))
        print(f"    E{k}·E{k}^-1 = I ?", is_identity(P), f"    E{k}^-1·E{k} = I ?", is_identity(Einv @ E))
        rows.append([Mi(E, label=f"E{k}: {op_pretty(op)}", rows={r: CELL_HL for r in changed_rows(op)}),
                     Tx("×"),
                     Mi(Einv, label=f"E{k}⁻¹: {op_pretty(inverse_op(op))}",
                        rows={r: "#d6f0de" for r in changed_rows(op)}),
                     Tx("="), Mi(P, label="I₃")])
        titles.append(["swap is its own inverse", "scale by 3  is undone by  scale by 1/3",
                       "add 3R1  is undone by  subtract 3R1"][k - 1])
    board(rows, "03_inverse_of_elementary_matrices", "E · E⁻¹ = I : the inverse operation undoes E",
          row_titles=titles)


# ============================================================================
# 4. The inversion algorithm
# ============================================================================
def gauss_jordan(A, ops=None):
    """Row reduce [A | I].  Returns (history, invertible).
    history = list of (operation or None, augmented matrix after it).
    If ops is given, that sequence of operations is used (as in the textbook);
    otherwise: forward phase (zeros below the pivots, pivots -> 1),
    then backward phase (zeros above the pivots, from the bottom up)."""
    n = len(A)
    M = np.hstack([frac_matrix(A), identity(n)])
    hist = [(None, M)]

    def do(op):
        nonlocal M
        M = apply_op(M, op)
        hist.append((op, M))
        return any(all(M[i, j] == 0 for j in range(n)) for i in range(n))   # zero row on the left?

    if ops is not None:
        for op in ops:
            if do(op):
                return hist, False
        return hist, True
    for c in range(n):                                   # forward phase
        if M[c, c] == 0:
            below = [r for r in range(c + 1, n) if M[r, c] != 0]
            if not below:
                return hist, False
            do(("swap", c, below[0]))
        if M[c, c] != 1:
            do(("scale", c, 1 / M[c, c]))
        for r in range(c + 1, n):
            if M[r, c] != 0 and do(("add", r, c, -M[r, c])):
                return hist, False
    for c in range(n - 1, 0, -1):                        # backward phase
        for r in range(c - 1, -1, -1):
            if M[r, c] != 0:
                do(("add", r, c, -M[r, c]))
    return hist, True


def print_history(hist, n):
    for k, (op, M) in enumerate(hist):
        if op is None:
            print("  Start with [A | I]:")
        else:
            print(f"  Step {k}: {op_name(op):16s}  (E{k} = I after this operation)")
        print(mat_str(M, divider=n, indent="      "))


def steps_board(hist, n, name, heading, per_row, failed=False, fs=11):
    rows, row = [], []
    cw = max(cell_width(M) for _, M in hist)          # same cell size in every panel
    for k, (op, M) in enumerate(hist):
        if op is None:
            label, opt = "Start: [ A | I ]", {}
        else:
            label = f"Step {k}:  {op_pretty(op)}"
            opt = {"rows": {r: CELL_HL for r in changed_rows(op)}}
        if failed and k == len(hist) - 1:
            zero = [i for i in range(n) if all(M[i, j] == 0 for j in range(n))]
            opt = {"cells": {(i, j): CELL_BAD for i in zero for j in range(n)}}
            label += "   → ZERO ROW!"
        row.append(Mi(M, label=label, divider=n, cw=cw, **opt))
        if sum(it[0] == "M" for it in row) == per_row or k == len(hist) - 1:
            rows.append(row)
            row = []
        else:
            row.append(Tx("→", w=0.9))
    board(rows, name, heading, fs=fs, row_gap=1.0)


def section4():
    title("4. THE INVERSION ALGORITHM   [ A | I ]  ->  [ I | A^-1 ]")
    print("Idea: if E_k ··· E_2 E_1 A = I, then A^-1 = E_k ··· E_2 E_1 = (the same operations done to I).")
    print("So we write I next to A and do every row operation to BOTH halves at once.")
    print("All arithmetic below uses exact fractions (fractions.Fraction), no rounding.")

    examples = [("Anton Example 4", [[1, 2, 3], [2, 5, 3], [1, 0, 8]], None, "04a_inversion_anton_example4", 2),
                ("Anton Exercise Set 1.5 No.23", [[-3, 1], [2, 2]],
                 [("swap", 0, 1), ("scale", 0, Fraction(1, 2)), ("add", 1, 0, 3),
                  ("scale", 1, Fraction(1, 4)), ("add", 0, 1, -1)], "04c_inversion_exercise23", 2)]
    for label, A, ops, name, per_row in examples:
        A = frac_matrix(A)
        n = len(A)
        sub(f"{label}:  A = {[[int(v) for v in r] for r in A]}")
        hist, ok = gauss_jordan(A, ops)
        print_history(hist, n)
        Ainv = hist[-1][1][:, n:]
        print("  The left half is I, so A is invertible and\n  A^-1 =\n" + mat_str(Ainv, indent="      "))
        print("  Check  A·A^-1 = I ?", is_identity(A @ Ainv), "   A^-1·A = I ?", is_identity(Ainv @ A))
        steps_board(hist, n, name, f"{label}:  [ A | I ]  →  [ I | A⁻¹ ]", per_row)

        # the elementary matrices and the two factorizations
        ops_used = [op for op, _ in hist[1:]]
        Es = [elementary(n, op) for op in ops_used]
        prod = identity(n)
        for E in Es:
            prod = E @ prod                        # E_k ... E_2 E_1
        back = identity(n)
        for op in ops_used:
            back = back @ elementary(n, inverse_op(op))   # E_1^-1 E_2^-1 ... E_k^-1
        k = len(Es)
        print(f"\n  The {k} elementary matrices used:")
        for i, (op, E) in enumerate(zip(ops_used, Es), 1):
            print(f"    E{i}  ({op_name(op)}):\n" + mat_str(E, indent="      "))
        print(f"  E{k}···E2·E1 =\n" + mat_str(prod, indent="      "),
              "\n  equal to A^-1 ?", bool((prod == Ainv).all()))
        print(f"  E1^-1·E2^-1···E{k}^-1 =\n" + mat_str(back, indent="      "),
              "\n  equal to A ?", bool((back == A).all()),
              "  -> A is a product of elementary matrices")

        r1 = []
        for i in range(k, 0, -1):
            r1.append(Mi(Es[i - 1], label=f"E{i}", rows={r: CELL_HL for r in changed_rows(ops_used[i - 1])}))
            r1.append(Tx("·", w=0.5, fs=20))
        r1[-1] = Tx("=", w=0.7)
        r1.append(Mi(prod, label="A⁻¹", shade=False, rows={i: "#d6f0de" for i in range(n)}))
        r2 = []
        for i in range(1, k + 1):
            r2.append(Mi(elementary(n, inverse_op(ops_used[i - 1])), label=f"E{i}⁻¹",
                         rows={r: "#d6e6fa" for r in changed_rows(ops_used[i - 1])}))
            r2.append(Tx("·", w=0.5, fs=20))
        r2[-1] = Tx("=", w=0.7)
        r2.append(Mi(back, label="A", shade=False, rows={i: "#d6f0de" for i in range(n)}))
        fname = name.replace("04a_inversion", "04b_factors").replace("04c_inversion", "04d_factors")
        board([r1, r2], fname, f"{label}:   A⁻¹ = E{k} ··· E1    and    A = E1⁻¹ ··· E{k}⁻¹",
              fs=10 if n == 3 else 11, col_gap=0.15,
              row_titles=[f"A⁻¹ = E{k} ··· E2 E1   (the operations in reverse order of writing)",
                          f"A = E1⁻¹ E2⁻¹ ··· E{k}⁻¹   (undo every step)"])

    sub("Anton Example 5:  A = [[1,6,4],[2,4,-1],[-1,2,5]]  (NOT invertible)")
    A = frac_matrix([[1, 6, 4], [2, 4, -1], [-1, 2, 5]])
    hist, ok = gauss_jordan(A, [("add", 1, 0, -2), ("add", 2, 0, 1), ("add", 2, 1, 1)])
    print_history(hist, 3)
    print("  A ZERO ROW appeared on the left side -> A can never become I.")
    print("  Conclusion: A is NOT invertible (singular).  numpy det A =", round(np.linalg.det(A.astype(float)), 10))
    steps_board(hist, 3, "04e_not_invertible_example5",
                "Anton Example 5: a zero row appears  →  A is NOT invertible", 2, failed=True)


# ============================================================================
# 5. LU decomposition
# ============================================================================
def section5():
    title("5. LU DECOMPOSITION from elementary matrices")
    A = frac_matrix([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    print("A =\n" + mat_str(A))
    print("Reduce A to an upper triangular U using only 'add a multiple of a row BELOW the pivot':")
    ops = [("add", 1, 0, -2), ("add", 2, 0, 1), ("add", 2, 1, 1)]
    M, Es = A.copy(), []
    for k, op in enumerate(ops, 1):
        M = apply_op(M, op)
        Es.append(elementary(3, op))
        print(f"  E{k}: {op_name(op):10s} ->\n" + mat_str(M, indent="      "))
    U = M
    L = identity(3)
    for op in ops:
        L = L @ elementary(3, inverse_op(op))
    print("E3 E2 E1 A = U =\n" + mat_str(U))
    print("So A = (E3 E2 E1)^-1 U = E1^-1 E2^-1 E3^-1 U = L U with")
    print("L = E1^-1 E2^-1 E3^-1 =\n" + mat_str(L))
    print("Notice: L is lower triangular with 1's on the diagonal, and below the diagonal")
    print("it stores the MULTIPLIERS with the sign flipped (2, -1, -1).")
    print("Check L·U = A ?", bool((L @ U == A).all()))
    gold = {(1, 0): CELL_HL, (2, 0): CELL_HL, (2, 1): CELL_HL}
    board([[Mi(Es[2], label="E3: R3 + R2", rows={2: CELL_HL}), Tx("·", w=0.5, fs=22),
            Mi(Es[1], label="E2: R3 + R1", rows={2: CELL_HL}), Tx("·", w=0.5, fs=22),
            Mi(Es[0], label="E1: R2 − 2R1", rows={1: CELL_HL}), Tx("·", w=0.5, fs=22),
            Mi(A, label="A"), Tx("=", w=0.7), Mi(U, label="U (upper)")],
           [Mi(elementary(3, inverse_op(ops[0])), label="E1⁻¹: R2 + 2R1", rows={1: "#d6e6fa"}),
            Tx("·", w=0.5, fs=22),
            Mi(elementary(3, inverse_op(ops[1])), label="E2⁻¹: R3 − R1", rows={2: "#d6e6fa"}),
            Tx("·", w=0.5, fs=22),
            Mi(elementary(3, inverse_op(ops[2])), label="E3⁻¹: R3 − R2", rows={2: "#d6e6fa"}),
            Tx("=", w=0.7), Mi(L, label="L (lower)", cells=gold)],
           [Mi(A, label="A"), Tx("=", w=0.7), Mi(L, label="L", cells=gold), Tx("·", w=0.5, fs=22),
            Mi(U, label="U", cells={(i, i): "#d6f0de" for i in range(3)})]],
          "05_LU_decomposition", "LU decomposition: A = L U",
          row_titles=["Step 1: elementary matrices turn A into U",
                      "Step 2: L = product of the inverses (multipliers with flipped sign)",
                      "Result"])


# ============================================================================
# 6. Cryptography
# ============================================================================
def to_numbers(text):
    return [27 if ch == " " else ord(ch) - 64 for ch in text.upper()]


def to_text(nums):
    return "".join(" " if v == 27 else chr(int(v) + 64) for v in nums)


def encode(text, K):
    n = len(K)
    nums = to_numbers(text)
    while len(nums) % n:
        nums.append(27)                                 # pad with spaces
    P = np.array(nums, dtype=int).reshape(-1, n).T      # columns = blocks of n letters
    C = K @ P
    return nums, P, C


def decode(numbers, K):
    n = len(K)
    Kinv = np.rint(np.linalg.inv(K)).astype(int)        # K has det ±1 -> integer inverse
    C = np.array(numbers, dtype=int).reshape(-1, n).T
    P = Kinv @ C
    return Kinv, P


def crypto_figure(axs, letters, plain, coded, heading, coded_first=False):
    x = np.arange(len(plain))
    lab = ["␣" if ch == " " else ch for ch in letters]
    order = [(coded, "encoded numbers  (sent over the wire)", ORANGE),
             (plain, "message as numbers  (A=1 … Z=26, space=27)", BLUE)]
    if not coded_first:
        order = order[::-1]
    else:
        order[1] = (plain, "decoded with K⁻¹  (K⁻¹ · each block)  →  letters", GREEN)
    for ax, (vals, ttl, col) in zip(axs, order):
        bars = ax.bar(x, vals, color=[col if v >= 0 else RED for v in vals], width=0.72)
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks(x)
        if vals is plain:
            ax.set_xticklabels(lab, fontsize=12, fontweight="bold", color=NAVY)
        else:
            ax.set_xticklabels([f"{i + 1}" for i in x], fontsize=8, color=GRAY)
        lo, hi = min(0, min(vals)), max(0, max(vals))
        pad = 0.18 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.03 * (hi - lo) if v >= 0 else -0.03 * (hi - lo)),
                    str(int(v)), ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5,
                    color=NAVY, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, colors=GRAY)
    axs[0].set_title(heading + "\n" + axs[0].get_title(), fontsize=12)


def section6():
    title("6. APPLICATION: CRYPTOGRAPHY (Hill cipher)")
    print("Code: A=1, B=2, ..., Z=26, space=27.  Cut the message into blocks of n numbers,")
    print("write each block as a column and multiply by an invertible n x n matrix K.")
    print("K is a product of elementary matrices with det = ±1, so K^-1 has integer entries.")
    K = np.array([[-3, -3, -4], [0, 1, 1], [4, 3, 4]])
    msg = "BUY IBM STOCK"
    nums, P, C = encode(msg, K)
    sub(f'Encode "{msg}" with K')
    print("K =\n" + mat_str(K), "   det K =", round(np.linalg.det(K)))
    print("message ->", nums, "(padded with spaces to a multiple of 3)")
    print("message matrix P (columns = blocks):\n" + mat_str(P))
    for j in range(P.shape[1]):
        col = P[:, j]
        print(f"  block {j + 1}: {to_text(col)!r:7s} {col.tolist()} -> K·{col.tolist()} = {C[:, j].tolist()}")
    coded = C.T.flatten().tolist()
    print("ENCODED message:", ", ".join(map(str, coded)))
    Kinv, P2 = decode(coded, K)
    sub("Decode with K^-1")
    print("K^-1 =\n" + mat_str(Kinv))
    print("K^-1 · C =\n" + mat_str(P2))
    print("DECODED message:", repr(to_text(P2.T.flatten()).rstrip()))

    fig, axs = plt.subplots(3, 1, figsize=(11, 9.6))
    letters = to_text(nums)
    crypto_figure(axs[:2], letters, nums, coded, f'Encoding "{msg}"  with  K = [[-3,-3,-4],[0,1,1],[4,3,4]]')
    dec = P2.T.flatten().tolist()
    ax = axs[2]
    x = np.arange(len(dec))
    ax.bar(x, dec, color=GREEN, width=0.72)
    for xi, v in zip(x, dec):
        ax.text(xi, v + 0.8, str(v), ha="center", va="bottom", fontsize=8.5, color=NAVY, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["␣" if ch == " " else ch for ch in to_text(dec)], fontsize=12,
                       fontweight="bold", color=NAVY)
    ax.set_ylim(0, 33)
    ax.set_title("decoded with K⁻¹  →  the original message comes back", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, colors=GRAY)
    fig.tight_layout(h_pad=2.2)
    show_or_save(fig, "06a_cryptography_buy_ibm_stock")

    # Williams 37 and 39
    K2 = np.array([[4, -3], [3, -2]])
    sub("Williams No.37: encode RETREAT with K = [[4,-3],[3,-2]]")
    nums2, P2_, C2 = encode("RETREAT", K2)
    coded2 = C2.T.flatten().tolist()
    for j in range(P2_.shape[1]):
        print(f"  {to_text(P2_[:, j])!r:5s} {P2_[:, j].tolist()} -> {C2[:, j].tolist()}")
    print("  ENCODED:", ", ".join(map(str, coded2)))
    sub("Williams No.39: decode 49, 38, -5, -3, -61, -39")
    c3 = [49, 38, -5, -3, -61, -39]
    K2inv, P3 = decode(c3, K2)
    print("K^-1 =\n" + mat_str(K2inv))
    for j in range(P3.shape[1]):
        print(f"  {c3[2 * j:2 * j + 2]} -> K^-1·block = {P3[:, j].tolist()} = {to_text(P3[:, j])!r}")
    plain3 = P3.T.flatten().tolist()
    print("  DECODED:", repr(to_text(plain3).rstrip()))

    fig, axs = plt.subplots(2, 2, figsize=(13, 7.4))
    crypto_figure(axs[:, 0], to_text(nums2), nums2, coded2, "Williams 37: encode RETREAT")
    crypto_figure(axs[:, 1], to_text(plain3), plain3, c3, "Williams 39: decode → PEACE", coded_first=True)
    fig.suptitle("Hill cipher with K = [[4, −3], [3, −2]],   K⁻¹ = [[−2, 3], [−3, 4]]",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(h_pad=2.0, w_pad=3)
    show_or_save(fig, "06b_cryptography_williams")


# ============================================================================
# 7. Colour model RGB <-> YIQ
# ============================================================================
def section7():
    title("7. APPLICATION: COLOUR MODELS  RGB <-> YIQ  (colour TV)")
    T = np.array([[.299, .587, .114], [.596, -.275, -.321], [.212, -.523, .311]])
    Tinv = np.linalg.inv(T)
    print("Y = brightness (what a black & white TV shows),  I and Q = colour information.")
    print("[Y, I, Q]^T = T [R, G, B]^T  with  T =\n" + mat_str(T))
    rgb = np.array([213, 77, 187])
    yiq = T @ rgb
    print("\nPurple  (R,G,B) = (213, 77, 187):")
    for name, row in zip("YIQ", T):
        terms = " + ".join(f"({c:g})({v})" for c, v in zip(row, rgb))
        print(f"  {name} = {terms} = {row @ rgb:.3f}")
    print("  (Y, I, Q) = (%g, %g, %g)" % tuple(np.round(yiq, 3)))
    print("\nT^-1 =\n" + mat_str(Tinv))
    back = Tinv @ yiq
    print("T^-1 (Y, I, Q) = (%g, %g, %g)" % tuple(np.round(back, 3) + 0.0), " -> the original purple comes back")
    print("\nGrey levels: T(1,1,1) = (%g, %g, %g)" % tuple(np.round(T @ [1, 1, 1], 6) + 0.0),
          "-> so T^-1 (s, 0, 0) = (s, s, s): no colour info = grey.")

    fig = plt.figure(figsize=(13, 8.4))
    ax = fig.add_axes([0.02, 0.36, 0.96, 0.58])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.75)
    ax.axis("off")
    ax.set_title("RGB → YIQ → RGB  for purple (213, 77, 187)", fontsize=14)
    col = rgb / 255
    ax.add_patch(FancyBboxPatch((0.3, 1.9), 2.4, 2.4, boxstyle="round,pad=0.05", fc=col, ec=NAVY, lw=1.5))
    ax.text(1.5, 1.45, "RGB\n(213, 77, 187)", ha="center", va="top", fontsize=11, color=NAVY, fontweight="bold")
    ax.annotate("", xy=(4.3, 3.1), xytext=(2.95, 3.1), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=GOLD))
    ax.text(3.62, 3.35, "T", ha="center", fontsize=14, color=GOLD, fontweight="bold")
    vals = [("Y", yiq[0], "brightness"), ("I", yiq[1], "orange–blue"), ("Q", yiq[2], "purple–green")]
    for k, (nm, v, d) in enumerate(vals):
        y0 = 3.9 - k * 1.0
        ax.add_patch(Rectangle((4.5, y0 - 0.38), 3.6, 0.76, fc="#fff8e6", ec=GOLD, lw=1.3))
        ax.text(4.7, y0, f"{nm} = {v:8.3f}", va="center", fontsize=12, family="monospace", color=NAVY,
                fontweight="bold")
        ax.text(8.2, y0, d, va="center", fontsize=10, color="#4a5570")
    ax.annotate("", xy=(10.25, 3.1), xytext=(9.4, 3.1), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=GOLD))
    ax.text(9.82, 3.35, "T⁻¹", ha="center", fontsize=14, color=GOLD, fontweight="bold")
    ax.add_patch(FancyBboxPatch((10.4, 1.9), 2.4, 2.4, boxstyle="round,pad=0.05", fc=np.clip(back / 255, 0, 1),
                                ec=NAVY, lw=1.5))
    ax.text(11.6, 1.45, "back to RGB\n(%g, %g, %g)" % tuple(np.round(back, 3)), ha="center", va="top",
            fontsize=11, color=NAVY, fontweight="bold")
    # what a black-and-white TV shows
    ax.add_patch(Rectangle((4.5, 0.15), 1.1, 0.8, fc=[yiq[0] / 255] * 3, ec=NAVY))
    ax.text(5.8, 0.55, "Y alone (black & white TV) = grey (130, 130, 130)", va="center", fontsize=10, color=NAVY)

    ax2 = fig.add_axes([0.04, 0.03, 0.92, 0.27])
    levels = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    ax2.set_xlim(0, len(levels))
    ax2.set_ylim(0, 1.6)
    ax2.axis("off")
    ax2.set_title("Grey strip:  T⁻¹ (s, 0, 0) = (s, s, s)   — no I, no Q  →  pure grey", fontsize=12)
    for k, s in enumerate(levels):
        g = (Tinv @ [s, 0, 0]) / 255
        ax2.add_patch(Rectangle((k + 0.06, 0.35), 0.88, 0.85, fc=np.clip(g, 0, 1), ec=NAVY, lw=1))
        ax2.text(k + 0.5, 1.3, f"YIQ ({s:.0f}, 0, 0)", ha="center", fontsize=8.5, color=NAVY)
        ax2.text(k + 0.5, 0.2, "RGB ({0:.0f}, {0:.0f}, {0:.0f})".format(s), ha="center", va="top",
                 fontsize=8.5, color="#4a5570")
    show_or_save(fig, "07_colour_model_rgb_yiq")


# ============================================================================
# 8. 2D geometry
# ============================================================================
def section8():
    title("8. GEOMETRY IN 2D: what elementary matrices do to the plane")
    cases = [("swap R1↔R2: reflect in y = x", np.array([[0, 1], [1, 0]])),
             ("scale R1 by 3: stretch x by 3", np.array([[3, 0], [0, 1]])),
             ("R1 + 2R2: shear in x", np.array([[1, 2], [0, 1]]))]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.6))
    lim = 9

    def frame(ax, ttl):                       # x from -1 to 9, y from -1 to 5
        setup(ax, lim, ttl, lo=-1)
        ax.set_ylim(-1, 5)
        ax.set_yticks(range(-1, 6))

    frame(axs[0], "ORIGINAL: letter F + unit square")
    draw_shape(axs[0], F, BLUE)
    draw_shape(axs[0], SQUARE, GREEN, alpha=0.3)
    area_F = 0.5 * abs(np.dot(F[0], np.roll(F[1], 1)) - np.dot(F[1], np.roll(F[0], 1)))   # shoelace
    info_box(axs[0], f"area(F) = {area_F:.2f}\narea(square) = 1", x=0.97, ha="right")
    print(f"Area of the letter F = {area_F:.2f},  area of the unit square = 1")
    for ax, (name, A) in zip(axs[1:], cases):
        d = np.linalg.det(A)
        frame(ax, name)
        draw_grid_image(ax, A, lim)
        draw_shape(ax, F, BLUE, alpha=0.10, lw=1)
        draw_shape(ax, A @ F, ORANGE)
        draw_shape(ax, A @ SQUARE, GREEN, alpha=0.35)
        if name.startswith("swap"):
            ax.plot([-1, 5], [-1, 5], "--", color=PURPLE, lw=2)
            ax.text(3.75, 3.2, "y = x", color=PURPLE, fontsize=10, fontweight="bold")
        info_box(ax, f"A = [[{A[0, 0]}, {A[0, 1]}],\n     [{A[1, 0]}, {A[1, 1]}]]\n"
                     f"det A = {d:.0f}\narea(F) = {area_F * abs(d):.2f}\narea(square) = {abs(d):.0f}",
                 x=0.97, ha="right")
        print(f"  {name:34s} A = {A.tolist()}  det = {d:5.1f}  -> every area is multiplied by |det| = {abs(d):.0f}"
              + ("   (det < 0: the F is flipped)" if d < 0 else ""))
    fig.tight_layout()
    show_or_save(fig, "08a_elementary_2d_letter_F")

    sub("Puzzle: three shears make a rotation by 90°")
    S1 = np.array([[1, -1], [0, 1]])
    S2 = np.array([[1, 0], [1, 1]])
    S3 = S1
    R = S3 @ S2 @ S1
    print("S1 = [[1,-1],[0,1]]  (R1 - R2),  S2 = [[1,0],[1,1]]  (R2 + R1),  S3 = S1")
    print("S3 S2 S1 =\n" + mat_str(R) + "\n = rotation by 90° [[0,-1],[1,0]] ?",
          np.allclose(R, [[0, -1], [1, 0]]))
    print("Every shear is an elementary matrix (det = 1), so the rotation keeps area: det R =",
          round(np.linalg.det(R)))
    stages = [np.eye(2), S1, S2 @ S1, R]
    names = ["start", "after S1 = [[1,−1],[0,1]]", "after S2 S1  (S2 = [[1,0],[1,1]])",
             "after S3 S2 S1 = turned 90°!"]
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, M, nme in zip(axs, stages, names):
        setup(ax, 4, nme)
        draw_grid_image(ax, M, 4)
        draw_shape(ax, F, BLUE, alpha=0.08, lw=1)
        draw_shape(ax, M @ F, ORANGE)
        info_box(ax, mat_str(M, 0, indent="").replace("[ ", "[").replace(" ]", "]")
                 + f"\ndet = {np.linalg.det(M):.0f}", x=0.03, y=0.03, va="bottom")
    fig.tight_layout()
    show_or_save(fig, "08b_three_shears_rotation")


# ============================================================================
# 9. 3D geometry
# ============================================================================
CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float).T
CUBE_F = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]


def draw3d(ax, V, faces, color, alpha=0.35, lw=1.2, edge=None):
    polys = [[V[:, i] for i in f] for f in faces]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor=edge or color, alpha=alpha,
                                         linewidths=lw))
    if edge:                                   # solid outline so the shape reads clearly
        for f in faces:
            idx = f + [f[0]]
            ax.plot(V[0, idx], V[1, idx], V[2, idx], color=edge, lw=lw)


def setup3d(ax, title_text, lim=2.2):
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_zlim(0, lim)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2]); ax.set_zticks([0, 1, 2])
    ax.set_xlabel("x", color=RED, fontweight="bold")
    ax.set_ylabel("y", color=GREEN, fontweight="bold")
    ax.set_zlabel("z", color=BLUE, fontweight="bold")
    ax.tick_params(labelsize=7, colors=GRAY)
    for v, c in [((lim, 0, 0), RED), ((0, lim, 0), GREEN), ((0, 0, lim), BLUE)]:
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, lw=2.2)
    ax.set_title(title_text, fontsize=11, pad=4)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=14, azim=-74)


def section9():
    title("9. GEOMETRY IN 3D: the unit cube, volume = |det E|")
    cases = [("swap R1 ↔ R3  (x ↔ z)", np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])),
             ("scale R3 by 2  (z × 2)", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])),
             ("R1 + R3  (shear: x → x + z)", np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]]))]
    # a marked corner so the swap is visible
    mark = np.array([1.0, 0.0, 0.0])
    mark2 = np.array([0.0, 0.0, 1.0])
    fig = plt.figure(figsize=(19, 5.8))
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    setup3d(ax, "ORIGINAL unit cube\nvolume = 1")
    draw3d(ax, CUBE_V, CUBE_F, BLUE, 0.25, lw=1.4, edge=BLUE)
    ax.scatter(*mark, color=RED, s=60, depthshade=False)
    ax.text(*(mark + [0.05, 0, 0.12]), "(1,0,0)", color=RED, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.scatter(*mark2, color=GOLD, s=60, depthshade=False)
    ax.text(*(mark2 + [0.05, 0, 0.12]), "(0,0,1)", color=GOLD, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    print("The marked corners (1,0,0) (red) and (0,0,1) (gold) show where things go.")
    for k, (name, E) in enumerate(cases, 2):
        d = np.linalg.det(E)
        ax = fig.add_subplot(1, 4, k, projection="3d")
        setup3d(ax, f"{name}\ndet = {d:.0f},  volume = |det| = {abs(d):.0f}")
        draw3d(ax, CUBE_V, CUBE_F, GRAY, 0.06, lw=0.6)
        draw3d(ax, E @ CUBE_V, CUBE_F, ORANGE, 0.30, lw=1.4, edge="#b8461c")
        q = E @ mark2
        ax.scatter(*q, color=GOLD, s=60, depthshade=False)
        ax.text(*(q + [0.05, 0, 0.12]), "(%g,%g,%g)" % tuple(q), color=GOLD, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        p = E @ mark
        ax.scatter(*p, color=RED, s=60, depthshade=False)
        ax.text(*(p + [0.05, 0, 0.12]), "(%g,%g,%g)" % tuple(p), color=RED, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        print(f"\n  {name}:  E =\n" + mat_str(E, indent="    "))
        print(f"    det E = {d:.0f}  ->  volume of the image of the unit cube = |det E| = {abs(d):.0f}")
        print(f"    corner (1,0,0) -> {tuple(p.astype(int).tolist())},   corner (1,1,1) -> {tuple((E @ [1, 1, 1]).astype(int).tolist())}")
    fig.tight_layout()
    show_or_save(fig, "09_elementary_3d_cube")
    print("\nSummary: swap -> det -1 (orientation flips), scale by c -> det c, add-multiple -> det 1.")


SECTIONS = {i + 1: f for i, f in enumerate([section1, section2, section3, section4, section5,
                                            section6, section7, section8, section9])}

if __name__ == "__main__":
    print(__doc__.split("Requirements:")[0])
    args = [a for a in sys.argv[1:] if a.isdigit() and int(a) in SECTIONS]
    for k in (map(int, args) if args else SECTIONS):
        SECTIONS[k]()
    print("\nDone. All pictures are in", OUT)
