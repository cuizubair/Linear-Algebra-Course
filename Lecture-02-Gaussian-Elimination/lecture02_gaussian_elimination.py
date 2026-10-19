"""
Linear Algebra - Lecture 2: GAUSSIAN ELIMINATION (echelon form) and
                            GAUSS-JORDAN ELIMINATION (reduced echelon form)
Muhammad Zubair  (ZMathLab)

Companion Python file for the lecture notes, the solved exercises and the video.
Every section prints a step-by-step explanation in the console (exact fractions,
every row operation written out) AND draws a picture that is saved in the
folder  pictures/  next to this file.

  1. The augmented matrix and the three elementary row operations
  2. Is it REF? RREF? both? neither?  (with reasons)    Anton 1.2 No.1-2, Williams No.1
  3. Gaussian elimination + back-substitution            Anton No.5   -> (3, 1, 2)
  4. Gauss-Jordan elimination (two versions)             Anton No.9, Williams Example 1
  5. One / none / infinitely many; pivot columns; free variables
                                                         Williams 5(b), 5(d), 6(b)
  6. Homogeneous systems: trivial and nontrivial         Anton No.13, Williams 7(d)
  7. Picture: two lines before and after row operations  (the meeting point never moves)
  8. Picture: three planes - one point / no point / a whole line
  9. Picture: staircase and pivot positions              Anton Example 9
 10. REF is not unique, RREF is unique                   Anton No.31

Requirements:  pip install numpy matplotlib
Run:           python gaussian_elimination.py            (all sections, save every picture)
               python gaussian_elimination.py 3 5        (only sections 3 and 5)
               python gaussian_elimination.py --show     (also open the windows)

Books used for the examples:
  Anton & Rorres, Elementary Linear Algebra (Section 1.2, Gaussian Elimination)
  G. Williams, Linear Algebra with Applications (Section 1.2, Gauss-Jordan Elimination)
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
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pictures")

# ----------------------------------------------------------------------------
# Colours (same as the notes): navy + gold
# ----------------------------------------------------------------------------
BLUE, ORANGE, GREEN, RED, PURPLE, GOLD, NAVY, GRAY = (
    "#3478c8", "#e2622e", "#1e965a", "#d23c3c", "#7d46be", "#d69a14", "#0b1832", "#9aa3b5")
GOLD_LIGHT = "#facb63"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": NAVY,
    "font.size": 10,
})


# ============================================================================
# Small helpers: printing
# ============================================================================
def title(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def sub(text):
    print("\n--- " + text + " " + "-" * max(0, 66 - len(text)))


def fstr(v):
    """Nice string for one entry: Fraction -> 'a/b'."""
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def mat_str(M, bar=None, indent="    ", mark_rows=()):
    """Pretty string of a matrix of fractions; bar = column index of the '|'.
    Rows in mark_rows get a '<-' so the student sees what changed."""
    S = [[fstr(v) for v in row] for row in M]
    w = max(len(s) for row in S for s in row)
    lines = []
    for i, row in enumerate(S):
        cells = [s.rjust(w) for s in row]
        if bar is not None:
            cells.insert(bar, "|")
        lines.append(indent + "[ " + "  ".join(cells) + " ]" + ("   <- changed" if i in mark_rows else ""))
    return "\n".join(lines)


def frac_matrix(rows):
    return [[Fraction(v) for v in r] for r in rows]


def copy(M):
    return [r[:] for r in M]


def show_or_save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("  [picture saved]", os.path.relpath(path, HERE))
    if SHOW:
        plt.show()
    plt.close(fig)


# ============================================================================
# Row operations
#   ("swap", i, j)        Ri <-> Rj
#   ("scale", i, c)       Ri -> c Ri          (c != 0)
#   ("add", i, j, c)      Ri -> Ri + c Rj
#   rows are numbered 0,1,2,... inside the code and R1,R2,R3,... on screen
# ============================================================================
def op_name(op):
    if op[0] == "swap":
        return f"R{op[1] + 1} <-> R{op[2] + 1}"
    if op[0] == "scale":
        c = Fraction(op[2])
        return f"R{op[1] + 1} -> ({fstr(c)})R{op[1] + 1}" if c.denominator != 1 or c < 0 else f"R{op[1] + 1} -> {fstr(c)}R{op[1] + 1}"
    i, j, c = op[1], op[2], Fraction(op[3])
    sign = "+" if c >= 0 else "-"
    a = abs(c)
    coef = "" if a == 1 else (f"({fstr(a)})" if a.denominator != 1 else fstr(a))
    return f"R{i + 1} -> R{i + 1} {sign} {coef}R{j + 1}"


def apply_op(M, op):
    M = copy(M)
    if op[0] == "swap":
        M[op[1]], M[op[2]] = M[op[2]], M[op[1]]
    elif op[0] == "scale":
        c = Fraction(op[2])
        assert c != 0, "scaling by 0 is NOT a row operation"
        M[op[1]] = [c * v for v in M[op[1]]]
    else:
        i, j, c = op[1], op[2], Fraction(op[3])
        M[i] = [a + c * b for a, b in zip(M[i], M[j])]
    return M


def changed_rows(op):
    return {op[1], op[2]} if op[0] == "swap" else {op[1]}


def do(M, op, bar=None, why=""):
    """apply one operation and PRINT it (operation, reason, new matrix)"""
    M2 = apply_op(M, op)
    print(f"\n  {op_name(op)}" + (f"      ({why})" if why else ""))
    print(mat_str(M2, bar, mark_rows=changed_rows(op)))
    return M2


# ============================================================================
# Echelon forms: leading entries, tests with reasons
# ============================================================================
def leading(M):
    """list of (row, column) of the leading entry of every nonzero row"""
    out = []
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            if v != 0:
                out.append((i, j))
                break
    return out


def echelon_report(M):
    """returns (is_ref, is_rref, list of reasons)"""
    lead = leading(M)
    m = len(M)
    zero_rows = [i for i in range(m) if all(v == 0 for v in M[i])]
    reasons = []
    for (i, c) in lead:
        if M[i][c] != 1:
            reasons.append(f"(1) fails: the leading entry of row {i + 1} is {fstr(M[i][c])}, not 1")
    for z in zero_rows:
        if any(i > z for (i, _) in lead):
            reasons.append(f"(2) fails: row {z + 1} is a zero row but a nonzero row is below it")
    for (i1, c1), (i2, c2) in zip(lead, lead[1:]):
        if c2 <= c1:
            reasons.append(f"(3) fails: leading entry of row {i2 + 1} (col {c2 + 1}) is not right of row {i1 + 1} (col {c1 + 1})")
    is_ref = not reasons
    r4 = []
    for (i, c) in lead:
        for k in range(m):
            if k != i and M[k][c] != 0:
                r4.append(f"(4) fails: column {c + 1} has a leading 1 (row {i + 1}) and also {fstr(M[k][c])} in row {k + 1}")
    is_rref = is_ref and not r4
    if is_ref:
        reasons += r4
    return is_ref, is_rref, reasons


# ============================================================================
# The algorithms (they PRINT every step)
# ============================================================================
def pick_pivot_row(M, r, c):
    rows = [i for i in range(r, len(M)) if M[i][c] != 0]
    if not rows:
        return None
    if M[r][c] != 0:
        return r
    for i in rows:                      # prefer a row that already has 1 or -1
        if abs(M[i][c]) == 1:
            return i
    return rows[0]


def forward_phase(M, ncoef, bar=None, verbose=True):
    """Gaussian elimination (Anton steps 1-5). Returns (REF, list of operations)."""
    M = copy(M)
    ops = []
    m = len(M)
    r = 0
    for c in range(len(M[0])):
        if r >= m:
            break
        p = pick_pivot_row(M, r, c)
        if p is None:
            if verbose and c < ncoef:
                print(f"\n  Column {c + 1}: no nonzero entry in rows {r + 1}..{m}  -> skip this column")
            continue
        if verbose:
            print(f"\n  Column {c + 1} (working on row {r + 1}):")
        if p != r:
            op = ("swap", r, p)
            M = do(M, op, bar, f"top entry is 0, bring a nonzero entry up") if verbose else apply_op(M, op)
            ops.append(op)
        if M[r][c] != 1:
            op = ("scale", r, 1 / M[r][c])
            M = do(M, op, bar, f"make a leading 1: divide the row by {fstr(M[r][c])}") if verbose else apply_op(M, op)
            ops.append(op)
        for i in range(r + 1, m):
            if M[i][c] != 0:
                op = ("add", i, r, -M[i][c])
                M = do(M, op, bar, f"make the {fstr(M[i][c])} below the leading 1 zero") if verbose else apply_op(M, op)
                ops.append(op)
        r += 1
    return M, ops


def backward_phase(M, bar=None, verbose=True):
    """Step 6: from the last leading 1 upward, zeros ABOVE every leading 1."""
    M = copy(M)
    ops = []
    for (r, c) in reversed(leading(M)):
        if bar is not None and c >= bar:
            continue
        for i in range(r):
            if M[i][c] != 0:
                op = ("add", i, r, -M[i][c])
                M = do(M, op, bar, f"make the {fstr(M[i][c])} above the leading 1 of row {r + 1} zero") if verbose else apply_op(M, op)
                ops.append(op)
    return M, ops


def gauss_jordan_williams(M, bar=None, verbose=True):
    """Williams: column by column, leading 1 then zeros above AND below."""
    M = copy(M)
    m = len(M)
    r = 0
    for c in range(len(M[0])):
        if r >= m:
            break
        p = pick_pivot_row(M, r, c)
        if p is None:
            continue
        if verbose:
            print(f"\n  Column {c + 1}: pivot in row {r + 1}")
        if p != r:
            M = do(M, ("swap", r, p), bar, "bring a nonzero pivot up") if verbose else apply_op(M, ("swap", r, p))
        if M[r][c] != 1:
            M = do(M, ("scale", r, 1 / M[r][c]), bar, "pivot -> 1") if verbose else apply_op(M, ("scale", r, 1 / M[r][c]))
        for i in range(m):
            if i != r and M[i][c] != 0:
                M = do(M, ("add", i, r, -M[i][c]), bar, "zero above/below the pivot") if verbose else apply_op(M, ("add", i, r, -M[i][c]))
        r += 1
    return M


def rref(M):
    """quiet reduced row echelon form (exact)"""
    E, _ = forward_phase(M, len(M[0]), verbose=False)
    E, _ = backward_phase(E, verbose=False)
    return E


def analyse(E, n, names):
    """E = REF/RREF of [A|b] with n unknowns. Print pivots, free variables, verdict."""
    lead = leading(E)
    pcols = [c for (_, c) in lead if c < n]
    free = [j for j in range(n) if j not in pcols]
    print("\n  Pivot positions (row, column):", [(i + 1, c + 1) for (i, c) in lead])
    print("  Pivot columns:", [c + 1 for c in pcols], " -> leading variables:", [names[c] for c in pcols])
    if any(c == n for (_, c) in lead):
        i = [i for (i, c) in lead if c == n][0]
        print(f"  Row {i + 1} is [0 ... 0 | {fstr(E[i][n])}]  i.e.  0 = {fstr(E[i][n])}   ->  NO SOLUTION (inconsistent)")
        return "none", pcols, free
    if free:
        print("  Non-pivot columns:", [j + 1 for j in free], " -> FREE variables:", [names[j] for j in free])
        print("  Rule: free variables = columns WITHOUT a leading 1. Never make a pivot variable free.")
        print(f"  number of free variables = n - r = {n} - {len(pcols)} = {len(free)}  ->  INFINITELY MANY solutions")
        return "many", pcols, free
    print("  Every column of unknowns has a leading 1: no free variable  ->  EXACTLY ONE solution")
    return "one", pcols, free


def back_substitution(E, n, names):
    """Gaussian elimination, last part. Works for unique solutions (exact)."""
    x = [None] * n
    for (i, c) in reversed(leading(E)):
        rhs = E[i][n]
        terms = [(j, E[i][j]) for j in range(c + 1, n) if E[i][j] != 0]
        val = rhs - sum(cf * x[j] for j, cf in terms)
        shown = fstr(rhs) + "".join(f" - ({fstr(cf)})({fstr(x[j])})" for j, cf in terms)
        x[c] = val / E[i][c]
        print(f"  Row {i + 1}:  {names[c]} = {shown} = {fstr(x[c])}")
    return x


def general_solution(E, n, names, params=("r", "s", "t", "u", "v")):
    """read a general solution from an RREF: leading variable = constant - (free terms)"""
    lead = leading(E)
    pcols = [c for (_, c) in lead if c < n]
    free = [j for j in range(n) if j not in pcols]
    pn = ["t"] if len(free) == 1 else list(params[:len(free)])
    out = {}
    for j, p in zip(free, pn):
        out[j] = {"const": Fraction(0), p: Fraction(1)}
    for (i, c) in lead:
        d = {"const": E[i][n]}
        for j, p in zip(free, pn):
            if E[i][j] != 0:
                d[p] = -E[i][j]
        out[c] = d
    for j in range(n):
        d = out[j]
        s = fstr(d["const"]) if d["const"] != 0 or len(d) == 1 else ""
        for k, v in d.items():
            if k == "const":
                continue
            sign = "-" if v < 0 else "+"
            a = abs(v)
            term = (fstr(a) if a != 1 else "") + k
            s = (("-" if v < 0 else "") + term) if s == "" else s + f" {sign} {term}"
        print(f"    {names[j]} = {s}")
    return out, free, pn


def check(A, b, x, names):
    print("  Check (put the solution back into every equation):")
    for row, bi in zip(A, b):
        lhs = sum(Fraction(a) * xi for a, xi in zip(row, x))
        terms = " + ".join(f"({a})({fstr(xi)})" for a, xi in zip(row, x) if a != 0)
        print(f"    {terms} = {fstr(lhs)}   (should be {bi})", "OK" if lhs == bi else "WRONG")


# ============================================================================
# Drawing helper: a matrix as a grid, with leading entries and the staircase
# ============================================================================
def draw_matrix(ax, M, x0=0, y0=0, bar=None, label="", stair=True, pivot_box=None):
    M = [[Fraction(v) for v in r] for r in M]
    m, n = len(M), len(M[0])
    is_ref, is_rref, _ = echelon_report(M)
    lead = leading(M)
    if stair and is_ref and lead:           # shade the zero region under the staircase
        poly = [(0, -m), (0, 0), (lead[0][1], 0)]
        for k, (i, c) in enumerate(lead):
            nxt = lead[k + 1][1] if k + 1 < len(lead) else n
            poly += [(c, -i), (c, -i - 1), (nxt, -i - 1)]
        poly += [(n, -m)]
        xs, ys = zip(*poly)
        ax.fill([x0 + v for v in xs], [y0 + v for v in ys], color="#eceff4", zorder=0)
    for i in range(m):
        for j in range(n):
            if pivot_box and (i, j) in pivot_box:
                ax.add_patch(Rectangle((x0 + j + 0.06, y0 - i - 0.94), 0.88, 0.88, fc="#cfe0f5", ec=BLUE, lw=1.5))
            if (i, j) in lead and stair:
                good = M[i][j] == 1
                ax.add_patch(Circle((x0 + j + 0.5, y0 - i - 0.5), 0.36, fc=GOLD_LIGHT if good else "#f7c6c6",
                                    ec=GOLD if good else RED, lw=1.5, zorder=2))
            ax.text(x0 + j + 0.5, y0 - i - 0.5, fstr(M[i][j]), ha="center", va="center",
                    fontsize=11, color=NAVY, zorder=3)
    for xb, d in [(x0 - 0.05, 0.2), (x0 + n + 0.05, -0.2)]:
        ax.plot([xb + d, xb, xb, xb + d], [y0, y0, y0 - m, y0 - m], color=NAVY, lw=2)
    if bar is not None:
        ax.plot([x0 + bar] * 2, [y0, y0 - m], color=NAVY, lw=1.2, ls="--")
    if stair and lead:
        p = []
        for (i, c) in lead:
            p += [(c, -i), (c, -i - 1)]
        p += [(n, -lead[-1][0] - 1)]
        xs, ys = zip(*p)
        ax.plot([x0 + v for v in xs], [y0 + v for v in ys], color=GREEN if is_ref else RED,
                lw=2.5, ls="-" if is_ref else "--", zorder=1)
    if label:
        ax.text(x0 + n / 2, y0 - m - 0.45, label, ha="center", va="top", fontsize=10, color=NAVY, fontweight="bold")


# ============================================================================
# 1. Augmented matrix and the three row operations
# ============================================================================
def section1():
    title("1. THE AUGMENTED MATRIX AND THE THREE ROW OPERATIONS")
    print("Solving with equations we rewrite x1, x2, x3, +, = again and again; only the NUMBERS change.")
    print("So we keep only the numbers: the augmented matrix [A | b]. Row i = equation i, column j = x_j.")
    print("\n  x1 +  x2 + 2x3 =  8\n -x1 - 2x2 + 3x3 =  1\n 3x1 - 7x2 + 4x3 = 10     (Anton 1.2 No.5)")
    M = frac_matrix([[1, 1, 2, 8], [-1, -2, 3, 1], [3, -7, 4, 10]])
    print("\nAugmented matrix:\n" + mat_str(M, 3))
    sub("The three elementary row operations (they never change the solutions)")
    T = frac_matrix([[0, 2, 4], [1, 1, 3]])
    print("Start:\n" + mat_str(T, 2))
    T = do(T, ("swap", 0, 1), 2, "1. SWAP two rows")
    T = do(T, ("scale", 1, Fraction(1, 2)), 2, "2. SCALE a row by c != 0")
    T = do(T, ("add", 0, 1, -1), 2, "3. ADD a multiple of another row")
    print("\nThe last matrix says x1 = 1, x2 = 2.")


# ============================================================================
# 2. REF / RREF test with reasons, and a picture
# ============================================================================
def section2():
    title("2. IS IT IN ROW ECHELON FORM (REF)? REDUCED ROW ECHELON FORM (RREF)?")
    print("(1) first nonzero entry of each nonzero row is 1 (a LEADING 1)")
    print("(2) zero rows at the bottom")
    print("(3) each leading 1 is to the RIGHT of the one above      (1)-(3) => REF")
    print("(4) a column with a leading 1 has zeros everywhere else   (1)-(4) => RREF")
    cases = [("Anton No.1(e)", [[1, 2, 0, 3, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]),
             ("Anton No.1(g)", [[1, -7, 5, 5], [0, 1, 3, 2]]),
             ("Anton No.2(b)", [[1, 0, 0], [0, 1, 0], [0, 2, 0]]),
             ("Anton No.2(e)", [[1, 2, 3], [0, 0, 0], [0, 0, 1]]),
             ("Williams No.1(c)", [[1, 2, 5, 6], [0, 1, 3, -7]]),
             ("Williams No.1(d)", [[1, 4, 0, 5], [0, 0, 2, 9]]),
             ("Williams No.1(f)", [[1, 5, 0], [0, 0, 1], [0, 0, 0]]),
             ("Williams No.1(h)", [[1, 0, 0, 3, 2], [0, 2, 0, 6, 1], [0, 0, 1, 2, 3]])]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, (name, M) in zip(axes.flat, cases):
        M = frac_matrix(M)
        ref, rr, reasons = echelon_report(M)
        verdict = "RREF (and REF)" if rr else ("REF, not RREF" if ref else "neither")
        sub(name + "  ->  " + verdict)
        print(mat_str(M))
        for r in reasons:
            print("   ", r)
        if rr:
            print("    all four properties hold")
        draw_matrix(ax, M)
        ax.set_xlim(-0.6, max(len(M[0]), 5) + 0.6)
        ax.set_ylim(-len(M) - 1.2, 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{name}\n{verdict}", color=GREEN if rr else (BLUE if ref else RED), fontsize=11)
    fig.suptitle("Leading entries circled (gold = leading 1, red = not 1);  green staircase = property (3) holds",
                 color=NAVY, fontsize=12)
    fig.tight_layout()
    show_or_save(fig, "02_ref_rref_check")


# ============================================================================
# 3. Gaussian elimination + back-substitution
# ============================================================================
def section3():
    title("3. GAUSSIAN ELIMINATION + BACK-SUBSTITUTION   (Anton 1.2 No.5)")
    A = [[1, 1, 2], [-1, -2, 3], [3, -7, 4]]
    b = [8, 1, 10]
    names = ["x1", "x2", "x3"]
    M = frac_matrix([r + [c] for r, c in zip(A, b)])
    print("Augmented matrix:\n" + mat_str(M, 3))
    sub("Forward phase: leading 1's and zeros BELOW them")
    E, ops = forward_phase(M, 3, bar=3)
    print("\n  This is a ROW ECHELON FORM. Operations used:", ", ".join(op_name(o) for o in ops))
    kind, _, _ = analyse(E, 3, names)
    sub("Back-substitution (bottom row first)")
    x = back_substitution(E, 3, names)
    check(A, b, x, names)
    assert x == [3, 1, 2]


# ============================================================================
# 4. Gauss-Jordan elimination
# ============================================================================
def section4():
    title("4. GAUSS-JORDAN ELIMINATION   (Anton No.9 = No.5 by Gauss-Jordan)")
    M = frac_matrix([[1, 1, 2, 8], [-1, -2, 3, 1], [3, -7, 4, 10]])
    sub("Forward phase (same as section 3)")
    E, _ = forward_phase(M, 3, bar=3)
    sub("Backward phase: from the LAST leading 1 upward, zeros ABOVE each leading 1")
    R, _ = backward_phase(E, bar=3)
    print("\n  REDUCED row echelon form: read the answer directly  x1 = %s, x2 = %s, x3 = %s"
          % tuple(fstr(R[i][3]) for i in range(3)))
    sub("Williams' way: column by column (Williams 1.2 Example 1)")
    W = frac_matrix([[0, 0, 2, -2, 2], [3, 3, -3, 9, 12], [4, 4, -2, 11, 12]])
    print(mat_str(W))
    RW = gauss_jordan_williams(W)
    assert RW == rref(W)
    print("\n  Same RREF as Anton's method would give:", RW == rref(W))


# ============================================================================
# 5. one / none / infinitely many, free variables
# ============================================================================
def section5():
    title("5. ONE, NONE OR INFINITELY MANY SOLUTIONS; PIVOT COLUMNS; FREE VARIABLES")
    systems = [("Williams 5(b)  (needs a row swap)", [[1, 2, 4], [2, 4, 9], [1, 3, 5]], [15, 33, 20]),
               ("Williams 5(d)", [[1, 4, 1], [1, 2, -1], [2, 6, 0]], [2, 0, 3]),
               ("Williams 6(b)", [[1, 2, 1], [1, 2, 2], [2, 4, 3]], [7, 11, 18])]
    names = ["x1", "x2", "x3"]
    for name, A, b in systems:
        sub(name)
        M = frac_matrix([r + [c] for r, c in zip(A, b)])
        print(mat_str(M, 3))
        R = gauss_jordan_williams(M, bar=3)
        kind, pcols, free = analyse(R, 3, names)
        if kind == "one":
            x = [R[i][3] for i in range(3)]
            print("  Solution:", ", ".join(f"{v} = {fstr(xi)}" for v, xi in zip(names, x)))
            check(A, b, x, names)
        elif kind == "many":
            print("  General solution:")
            sol, fr, pn = general_solution(R, 3, names)
            for tval in (0, 1, -2):        # check a few members of the family
                x = [sol[j]["const"] + sum(v * tval for k, v in sol[j].items() if k != "const") for j in range(3)]
                ok = all(sum(Fraction(a) * xi for a, xi in zip(row, x)) == bi for row, bi in zip(A, b))
                print(f"    {pn[0]} = {tval:>2}: {tuple(fstr(v) for v in x)}  satisfies all equations: {ok}")
            sub("WRONG choice of free variable (a pivot variable)")
            print("  Row 2 of the RREF says x3 = 4 exactly. Choosing 'x3 = t free' and t = 0 gives x3 = 0:")
            x = [Fraction(3), Fraction(0), Fraction(0)]
            print("  (3, 0, 0) in equation 2:", sum(Fraction(a) * v for a, v in zip(A[1], x)), "but it should be", b[1],
                  "-> NOT a solution. Free variables are ONLY the non-pivot columns.")


# ============================================================================
# 6. Homogeneous systems
# ============================================================================
def section6():
    title("6. HOMOGENEOUS SYSTEMS  A x = 0")
    print("x = 0 is ALWAYS a solution (the trivial solution).  Free variables = n - r  (Anton Thm 1.2.1).")
    print("More unknowns than equations  =>  r <= m < n  =>  a free variable  =>  nontrivial solutions (Thm 1.2.2).")
    sub("Anton No.13: 3 equations, 4 unknowns -> nontrivial solutions by inspection. Confirm:")
    A = [[2, -3, 4, -1], [7, 1, -8, 9], [2, 8, 1, -1]]
    names = ["x1", "x2", "x3", "x4"]
    M = frac_matrix([r + [0] for r in A])
    print(mat_str(M, 4))
    R = rref(M)
    print("  RREF:\n" + mat_str(R, 4))
    kind, pcols, free = analyse(R, 4, names)
    sol, fr, pn = general_solution(R, 4, names)
    x = [Fraction(-46), Fraction(15), Fraction(55), Fraction(83)]      # t = 83 removes the fractions
    print("  t = 83 gives the integer nontrivial solution", tuple(int(v) for v in x))
    print("  A x =", [int(sum(Fraction(a) * v for a, v in zip(row, x))) for row in A], " (all zero)")
    sub("Williams 7(d): two free variables")
    A = [[1, 2, 0, 4], [-2, -4, 3, -2]]
    M = frac_matrix([r + [0] for r in A])
    print(mat_str(M, 4))
    R = gauss_jordan_williams(M, bar=4)
    analyse(R, 4, names)
    print("  General solution:")
    general_solution(R, 4, names)


# ============================================================================
# 7. Picture: two lines during Gauss-Jordan elimination
# ============================================================================
def eqstr(row):
    """'x + y = 3', '-3y = -6', 'x = 1' (zero terms dropped, coefficient 1 hidden)"""
    out = ""
    for coef, v in zip(row[:2], "xy"):
        if coef == 0:
            continue
        a = abs(coef)
        term = ("" if a == 1 else fstr(a)) + v
        out += (("-" if coef < 0 else "") + term) if not out else (" - " if coef < 0 else " + ") + term
    return f"{out} = {fstr(row[2])}"


def line_xy(a, b, c, lim=(-1.5, 4.5)):
    if b != 0:
        x = np.array(lim)
        return x, (c - a * x) / b
    return np.array([c / a] * 2), np.array(lim)


def section7():
    title("7. TWO LINES: row operations move the lines, NOT the meeting point")
    M = frac_matrix([[1, 1, 3], [2, -1, 0]])
    steps = [("original:  x + y = 3,  2x - y = 0", None)]
    ops = [("add", 1, 0, -2), ("scale", 1, Fraction(-1, 3)), ("add", 0, 1, -1)]
    mats = [M]
    for op in ops:
        mats.append(apply_op(mats[-1], op))
        print(f"  {op_name(op):18s} ->", mats[-1])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    labels = ["original"] + ["after " + op_name(o) for o in ops]
    for k, (ax, Mk, lab) in enumerate(zip(axes, mats, labels)):
        ax.set_xlim(-1.5, 4.5); ax.set_ylim(-1.5, 4.5); ax.set_aspect("equal")
        ax.grid(color="#dfe4ee"); ax.axhline(0, color=GRAY); ax.axvline(0, color=GRAY)
        if k > 0:
            for row, col in zip(mats[k - 1], (BLUE, ORANGE)):
                x, y = line_xy(*[float(v) for v in row])
                ax.plot(x, y, color=col, lw=1.2, ls="--", alpha=0.4)
        for row, col in zip(Mk, (BLUE, ORANGE)):
            x, y = line_xy(*[float(v) for v in row])
            eq = eqstr(row)
            ax.plot(x, y, color=col, lw=2.5, label=eq)
        ax.plot(1, 2, "o", color=NAVY, ms=8, zorder=5)
        ax.annotate("(1, 2)", (1, 2), xytext=(8, 8), textcoords="offset points", color=NAVY, fontweight="bold")
        ax.set_title(lab, fontsize=11)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Gauss-Jordan elimination turns the lines into x = 1 and y = 2; the intersection never moves",
                 color=NAVY)
    fig.tight_layout()
    show_or_save(fig, "07_lines_row_operations")


# ============================================================================
# 8. Picture: three planes, three cases
# ============================================================================
def plane_patch(ax, a, b, c, d, center, h, color):
    coef = np.array([a, b, c], float)
    k = int(np.argmax(np.abs(coef)))
    o = [i for i in range(3) if i != k]
    pts = []
    for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = np.zeros(3)
        p[o[0]] = center[o[0]] + s1 * h
        p[o[1]] = center[o[1]] + s2 * h
        p[k] = (d - coef[o[0]] * p[o[0]] - coef[o[1]] * p[o[1]]) / coef[k]
        pts.append(p)
    ax.add_collection3d(Poly3DCollection([pts], alpha=0.35, facecolor=color, edgecolor=color))
    return np.array(pts)


def section8():
    title("8. THREE PLANES: one point / no common point / a whole line")
    cases = [("one solution: Anton No.5", [[1, 1, 2, 8], [-1, -2, 3, 1], [3, -7, 4, 10]], (3, 1, 2), 2),
             ("no solution: Williams 5(d)", [[1, 4, 1, 2], [1, 2, -1, 0], [2, 6, 0, 3]], (0, 0, 0), 1.5),
             ("infinitely many: Williams 5(c)", [[1, 1, 1, 7], [2, 3, 1, 18], [-1, 1, -3, 1]], (3, 4, 0), 2)]
    fig = plt.figure(figsize=(18, 6))
    for k, (name, P, c, h) in enumerate(cases, 1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        allp = []
        for row, col in zip(P, (BLUE, ORANGE, GREEN)):
            allp.append(plane_patch(ax, *row, c, h, col))
        allp = np.vstack(allp)
        M = frac_matrix(P)
        R = rref(M)
        kind, _, _ = analyse(R, 3, ["x", "y", "z"])
        print("   RREF:\n" + mat_str(R, 3))
        if kind == "one":
            p = [float(R[i][3]) for i in range(3)]
            ax.scatter(*p, color=NAVY, s=50, depthshade=False)
            ax.text(*p, "  (3,1,2)", color=NAVY, fontweight="bold")
        if kind == "many":
            t = np.linspace(-1.5, 2.5, 2)                       # x = 3 - 2t, y = 4 + t, z = t
            ax.plot(3 - 2 * t, 4 + t, t, color=NAVY, lw=3)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(name, fontsize=11)
        ax.view_init(elev=22, azim=-55)
    fig.tight_layout()
    show_or_save(fig, "08_planes_three_cases")


# ============================================================================
# 9. Staircase and pivot positions
# ============================================================================
def section9():
    title("9. STAIRCASE AND PIVOT POSITIONS   (Anton 1.2 Example 9)")
    A = frac_matrix([[0, 0, -2, 0, 7, 12], [2, 4, -10, 6, 12, 28], [2, 4, -5, 6, -5, -1]])
    print("A =\n" + mat_str(A))
    sub("Gaussian elimination (Anton's steps 1-5)")
    E, _ = forward_phase(A, 6)
    sub("Backward phase (step 6)")
    R, _ = backward_phase(E)
    lead = leading(E)
    print("\n  Pivot positions (row, col):", [(i + 1, c + 1) for i, c in lead])
    print("  Pivot columns:", [c + 1 for _, c in lead])
    # Anton's diagram marks the pivots 2, -10, -5 of A. The first one is in row 2 of A
    # because Step 2 swapped rows 1 and 2 before the first leading 1 was made.
    marked = [(1, 0), (1, 2), (2, 4)]
    print("  Pivots marked in A (Anton's diagram):", [fstr(A[i][c]) for i, c in marked])
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.8))
    draw_matrix(axes[0], A, stair=False, pivot_box=set(marked), label="A  (blue boxes = the pivots 2, -10, -5)")
    draw_matrix(axes[1], E, label="a row echelon form")
    draw_matrix(axes[2], R, label="the reduced row echelon form")
    for ax in axes:
        ax.set_xlim(-0.6, 6.6); ax.set_ylim(-4.2, 0.4); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Pivot columns 1, 3, 5 - the leading 1's sit in the same places in every echelon form", color=NAVY)
    fig.tight_layout()
    show_or_save(fig, "09_pivot_positions")


# ============================================================================
# 10. REF is not unique, RREF is unique
# ============================================================================
def section10():
    title("10. ROW ECHELON FORMS ARE NOT UNIQUE, THE REDUCED FORM IS   (Anton No.31)")
    M = frac_matrix([[1, 3], [2, 7]])
    sub("First REF")
    E1 = do(M, ("add", 1, 0, -2))
    sub("Second REF (start with a swap)")
    E2 = do(M, ("swap", 0, 1))
    E2 = do(E2, ("scale", 0, Fraction(1, 2)))
    E2 = do(E2, ("add", 1, 0, -1))
    E2 = do(E2, ("scale", 1, -2))
    print("\n  both REF:", echelon_report(E1)[0], echelon_report(E2)[0], "   different:", E1 != E2)
    print("  RREF of both:", rref(E1) == rref(E2) == [[1, 0], [0, 1]], "-> the identity matrix")


SECTIONS = {i + 1: f for i, f in enumerate([section1, section2, section3, section4, section5,
                                            section6, section7, section8, section9, section10])}

if __name__ == "__main__":
    print(__doc__.split("Requirements:")[0])
    args = [a for a in sys.argv[1:] if a.isdigit() and int(a) in SECTIONS]
    for k in (map(int, args) if args else SECTIONS):
        SECTIONS[k]()
    print("\nDone. All pictures are in", OUT)
