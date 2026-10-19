"""
Linear Algebra - Lecture 1: Systems of Linear Equations
Muhammad Zubair

Companion Python file for the lecture video and slides.
It works through every example from the lecture:

  0. Opening story: the "always 4" number trick and the Nine Chapters grain problem (China, ~200 BC)
  1. The notebook-and-pencil puzzle
  2. Is an equation linear?
  3. Three ways to solve a 2x2 system (substitution, elimination, comparison)
  4. The three possible outcomes: 1, infinitely many, or no solution
  5. Every equation = a straight line (2D plots)
  6. With 3 unknowns, every equation is a plane (3D plots)
  7. A x = b and the augmented matrix [A | b]
  8. The 3 elementary row operations, step by step (exact fractions)
  9. Read the last row
 10. Practice: Anton & Rorres, Elementary Linear Algebra (11th ed.), Exercise Set 1.1
 11. Quick check quiz

Requirements:  pip install numpy matplotlib
Run:           python linear_algebra_lecture1.py
Plots open in windows and are also saved as PNG files next to this script.
"""

from fractions import Fraction
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, BLUE, RED, GOLD = "#3fbf7f", "#4a9fe0", "#e05f5f", "#e8b84a"


def title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers: printing matrices with exact fractions
# ---------------------------------------------------------------------------
def frac(x):
    return Fraction(x).limit_denominator()


def fmt(x):
    x = frac(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def print_augmented(M, n_vars=None):
    """Print an augmented matrix [A | b] neatly."""
    n_vars = n_vars if n_vars is not None else len(M[0]) - 1
    cells = [[fmt(v) for v in row] for row in M]
    width = max(len(c) for row in cells for c in row)
    for row in cells:
        left = "  ".join(c.rjust(width) for c in row[:n_vars])
        right = "  ".join(c.rjust(width) for c in row[n_vars:])
        print(f"   [ {left} | {right} ]")


# ---------------------------------------------------------------------------
# 1. The shopping puzzle
# ---------------------------------------------------------------------------
def shopping_puzzle():
    title("1. A shopping puzzle")
    print("2 notebooks + 1 pencil = Rs 250")
    print("1 notebook  + 1 pencil = Rs 150")
    print("Subtract the second from the first  ->  1 notebook = Rs 100")
    notebook = 250 - 150
    pencil = 150 - notebook
    print(f"Notebook n = Rs {notebook},  pencil p = Rs {pencil}")
    # same thing with numpy
    A = np.array([[2, 1], [1, 1]], dtype=float)
    b = np.array([250, 150], dtype=float)
    print("numpy.linalg.solve gives:", np.linalg.solve(A, b))


# ---------------------------------------------------------------------------
# 2. Is it linear?
# ---------------------------------------------------------------------------
def linear_check():
    title("2. Linear equation")
    print("In my own words: every variable appears only to the power 1,")
    print("and no two variables are multiplied together.\n")
    examples = [
        ("x + 3y = 7", True, "each variable to the power 1"),
        ("x^2 + y = 4", False, "x is squared"),
        ("xy = 5", False, "x and y are multiplied together"),
    ]
    for eq, ok, why in examples:
        print(f"   {eq:<14} {'linear' if ok else 'NOT linear':<11} ({why})")


# ---------------------------------------------------------------------------
# 3. Three methods for one 2x2 system
# ---------------------------------------------------------------------------
def three_methods():
    title("3. One system, three methods:  x + y = 5,  2x - y = 4")
    # Substitution: y = 5 - x  ->  2x - (5 - x) = 4  ->  3x = 9
    x = Fraction(9, 3); y = 5 - x
    print(f"Substitution: y = 5 - x, 2x - (5 - x) = 4, 3x = 9  ->  x = {x}, y = {y}")
    # Elimination: add the equations  ->  3x = 9
    x = Fraction(5 + 4, 3); y = 5 - x
    print(f"Elimination : add the equations, 3x = 9            ->  x = {x}, y = {y}")
    # Comparison: y = 5 - x and y = 2x - 4  ->  5 - x = 2x - 4
    x = Fraction(5 + 4, 3); y = 2 * x - 4
    print(f"Comparison  : 5 - x = 2x - 4                      ->  x = {x}, y = {y}")
    print("KEY IDEA: different roads, the SAME answer (3, 2).")


# ---------------------------------------------------------------------------
# 4. How many solutions?  (uses row reduction from section 8)
# ---------------------------------------------------------------------------
def classify(M):
    """Return 'unique', 'infinite' or 'none' for an augmented matrix M (list of rows)."""
    R, _ = rref(M, verbose=False)
    n = len(M[0]) - 1
    pivots = 0
    for row in R:
        if all(v == 0 for v in row[:n]):
            if row[n] != 0:
                return "none"          # a row  [0 0 ... 0 | c], c != 0
        else:
            pivots += 1
    return "unique" if pivots == n else "infinite"


def three_outcomes():
    title("4. Only 3 possible answers")
    systems = {
        "x + y = 5,   2x - y = 4":   [[1, 1, 5], [2, -1, 4]],
        "x + y = 4,   2x + 2y = 6":  [[1, 1, 4], [2, 2, 6]],
        "x + 2y = 4,  2x + 4y = 8":  [[1, 2, 4], [2, 4, 8]],
    }
    label = {"unique": "exactly ONE solution (consistent)",
             "infinite": "INFINITELY many solutions (consistent)",
             "none": "NO solution (inconsistent)"}
    for name, M in systems.items():
        print(f"   {name:<26} ->  {label[classify(M)]}")
    print("KEY IDEA: every linear system has exactly 1, infinitely many, or no solutions.")


# ---------------------------------------------------------------------------
# 5. Every equation = a straight line
# ---------------------------------------------------------------------------
def plot_lines():
    title("5. Every equation = a straight line  (saving lines_2d.png)")
    x = np.linspace(-2, 6, 200)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cases = [
        ("Lines cross -> 1 solution", [(lambda x: 5 - x, "x + y = 5", GREEN),
                                       (lambda x: 2 * x - 4, "2x - y = 4", GOLD)], (3, 2)),
        ("Parallel -> no solution", [(lambda x: 4 - x, "x + y = 4", GREEN),
                                     (lambda x: 3 - x, "2x + 2y = 6", RED)], None),
        ("Same line -> infinitely many", [(lambda x: (4 - x) / 2, "x + 2y = 4", BLUE),
                                          (lambda x: (8 - 2 * x) / 4, "2x + 4y = 8", GOLD)], None),
    ]
    for ax, (name, lines, point) in zip(axes, cases):
        for i, (f, lab, col) in enumerate(lines):
            ax.plot(x, f(x), color=col, lw=4 if (i == 0 and "Same" in name) else 2,
                    ls="--" if (i == 1 and "Same" in name) else "-", label=lab)
        if point:
            ax.plot(*point, "o", color=GREEN, ms=9)
            ax.annotate(f"{point}", point, textcoords="offset points", xytext=(8, 8))
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-2, 6); ax.set_ylim(-3, 5); ax.set_title(name); ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "lines_2d.png"), dpi=130)

    # the moving point on the 'same line': x = 4 - 2t, y = t
    print("Infinitely many: every point (4 - 2t, t) works, e.g.")
    for t in [-0.5, 0, 1, 2.5]:
        xv, yv = 4 - 2 * t, t
        print(f"   t = {t:>4}:  ({xv:.1f}, {yv:.1f})   check x + 2y = {xv + 2 * yv:.1f}")


# ---------------------------------------------------------------------------
# 6. With 3 unknowns, every equation is a plane
# ---------------------------------------------------------------------------
def plot_planes():
    title("6. With 3 unknowns, every equation is a plane  (saving planes_3d.png)")
    xs, ys = np.meshgrid(np.linspace(-1, 3, 20), np.linspace(0, 4, 20))
    fig = plt.figure(figsize=(14, 4.8))
    systems = [
        ("One common point (1, 2, 3)", [(1, 1, 1, 6), (1, -1, 1, 2), (2, 1, -1, 1)], (1, 2, 3)),
        ("Share a whole line -> infinitely many", [(1, 1, 1, 6), (1, -1, 1, 2), (2, 0, 2, 8)], None),
        ("A 'tent' -> no solution", [(1, 1, 1, 6), (1, -1, 1, 2), (2, 0, 2, 5)], None),
    ]
    colors = [GREEN, GOLD, RED]
    for k, (name, planes, point) in enumerate(systems, 1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        for (a, b, c, d), col in zip(planes, colors):
            if c != 0:
                ax.plot_surface(xs, ys, (d - a * xs - b * ys) / c, color=col, alpha=0.35)
            else:   # plane parallel to the z-axis: a x + b y = d
                zz, yy = np.meshgrid(np.linspace(-2, 8, 20), np.linspace(0, 4, 20))
                ax.plot_surface((d - b * yy) / a, yy, zz, color=col, alpha=0.35)
        if point:
            ax.scatter(*point, color="black", s=60)
        ax.set_title(name, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_zlim(-2, 8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "planes_3d.png"), dpi=130)
    print("KEY IDEA: the solution is where ALL the planes meet.")


# ---------------------------------------------------------------------------
# 7. A x = b
# ---------------------------------------------------------------------------
def matrix_form():
    title("7. The computer's language: A x = b")
    A = np.array([[1, 1, 1], [1, -1, 1], [2, 1, -1]], dtype=float)
    b = np.array([6, 2, 1], dtype=float)
    print("A (coefficients) =\n", A)
    print("b =", b)
    print("Augmented matrix [A | b]:")
    print_augmented(np.column_stack([A, b]).tolist())
    print("numpy.linalg.solve(A, b) =", np.linalg.solve(A, b), "  -> (x, y, z) = (1, 2, 3)")


# ---------------------------------------------------------------------------
# 8. Row operations, step by step
# ---------------------------------------------------------------------------
def swap(M, i, j):
    M[i], M[j] = M[j], M[i]


def scale(M, i, c):
    M[i] = [c * v for v in M[i]]


def replace(M, i, j, c):
    """R_i -> R_i + c * R_j"""
    M[i] = [vi + c * vj for vi, vj in zip(M[i], M[j])]


def rref(M, verbose=True):
    """Reduce an augmented matrix to reduced row echelon form, printing each row operation."""
    M = [[frac(v) for v in row] for row in M]
    rows, cols = len(M), len(M[0]) - 1
    steps = []
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if pivot is None:
            continue
        if pivot != r:
            swap(M, r, pivot); steps.append(f"R{r+1} <-> R{pivot+1}")
            if verbose: print(f"\n {steps[-1]}"); print_augmented(M)
        if M[r][c] != 1:
            k = 1 / M[r][c]
            scale(M, r, k); steps.append(f"R{r+1} -> ({fmt(k)}) R{r+1}")
            if verbose: print(f"\n {steps[-1]}"); print_augmented(M)
        for i in range(rows):
            if i != r and M[i][c] != 0:
                k = -M[i][c]
                replace(M, i, r, k)
                sign = "+" if k > 0 else "-"
                steps.append(f"R{i+1} -> R{i+1} {sign} ({fmt(abs(k))}) R{r+1}")
                if verbose: print(f"\n {steps[-1]}"); print_augmented(M)
        r += 1
        if r == rows:
            break
    return M, steps


def row_operations_demo():
    title("8. The 3 elementary row operations")
    base = [[1, 2, 4], [3, 1, 5]]
    print("Start:"); print_augmented(base)
    M = [r[:] for r in base]; swap(M, 0, 1)
    print("\n1. Swap     R1 <-> R2"); print_augmented(M)
    M = [r[:] for r in base]; scale(M, 0, 2)
    print("\n2. Scale    R1 -> 2 R1"); print_augmented(M)
    M = [r[:] for r in base]; replace(M, 1, 0, -3)
    print("\n3. Replace  R2 -> R2 - 3 R1"); print_augmented(M)
    print("\nKEY IDEA: row operations change how the matrix looks, never the solution.")

    title("8b. Solving our 3-plane system with row operations")
    M = [[1, 1, 1, 6], [1, -1, 1, 2], [2, 1, -1, 1]]
    print("Start:"); print_augmented(M)
    R, steps = rref(M)
    print("\nRead the answer:  x = {}, y = {}, z = {}".format(*(fmt(row[-1]) for row in R)))


# ---------------------------------------------------------------------------
# 9. Read the last row
# ---------------------------------------------------------------------------
def read_last_row():
    title("9. Read the last row")
    cases = [([[1, 2, 3], [0, 0, 5]], "0 = 5  -> no solution"),
             ([[1, 2, 3], [0, 0, 0]], "0 = 0  -> infinitely many"),
             ([[1, 0, 1], [0, 1, 2]], "staircase -> unique (x = 1, y = 2)")]
    for M, meaning in cases:
        print_augmented(M); print(f"   => {meaning}  [classify(): {classify(M)}]\n")


# ---------------------------------------------------------------------------
# 10. Practice: Anton & Rorres, Exercise Set 1.1
# ---------------------------------------------------------------------------
def practice():
    title("10. Practice (Anton & Rorres, Elementary Linear Algebra, 11th ed., Ex. Set 1.1)")

    print("Exercise 9(a): is (3, 1, 1) a solution?")
    A = np.array([[2, -4, -1], [1, -3, 1], [3, -5, -3]]); b = np.array([1, 1, 1])
    lhs = A @ np.array([3, 1, 1])
    for row, val, rhs in zip(A, lhs, b):
        print(f"   {row} . (3,1,1) = {val}  (should be {rhs})  {'OK' if val == rhs else 'X'}")
    print("   ->", "YES, it is a solution" if np.all(lhs == b) else "NO")

    systems = {
        "11(a)  3x - 2y = 4,  6x - 4y = 9": [[3, -2, 4], [6, -4, 9]],
        "11(b)  2x - 4y = 1,  4x - 8y = 2": [[2, -4, 1], [4, -8, 2]],
        "11(c)   x - 2y = 0,   x - 4y = 8": [[1, -2, 0], [1, -4, 8]],
    }
    for name, M in systems.items():
        print(f"\nExercise {name}")
        R, steps = rref(M, verbose=False)
        print_augmented(R)
        kind = classify(M)
        if kind == "unique":
            print(f"   -> unique solution: x = {fmt(R[0][-1])}, y = {fmt(R[1][-1])}")
        elif kind == "infinite":
            print("   -> infinitely many:  y = t,  x = 1/2 + 2t")
        else:
            print("   -> no solution (last row says 0 = nonzero)")

    print("\nExercise 19(a): for which k is [1 k | -4 ; 4 8 | 2] consistent?")
    print("   R2 -> R2 - 4 R1 gives  [0, 8 - 4k | 18]")
    for k in [0, 1, 2, 3]:
        print(f"   k = {k}:  {classify([[1, k, -4], [4, 8, 2]])}")
    print("   -> inconsistent only when k = 2  (0 = 18), so consistent for all k != 2")


# ---------------------------------------------------------------------------
# 11. Quick check quiz
# ---------------------------------------------------------------------------
def quiz(interactive=True):
    title("11. Quick check")
    questions = [
        ("How many solutions?  x + y = 3,  2x + 2y = 6",
         ["Exactly one", "No solution", "Infinitely many"], 2,
         "The second equation is 2 x the first: same line."),
        ("The last row of the augmented matrix is [0 0 | 4]. The system has...",
         ["One solution", "No solution", "Infinitely many"], 1,
         "It says 0 = 4, which is impossible."),
        ("Is  xy + z = 2  a linear equation?",
         ["Yes", "No"], 1,
         "x and y are multiplied together, so it is not linear."),
    ]
    score = 0
    for q, opts, ans, why in questions:
        print("\n" + q)
        for i, o in enumerate(opts):
            print(f"   {'ABC'[i]}) {o}")
        if interactive:
            try:
                choice = input("Your answer (A/B/C): ").strip().upper()[:1]
            except EOFError:
                choice = ""
            if choice == "ABC"[ans]:
                score += 1; print("   Correct!", why)
            else:
                print(f"   Answer: {'ABC'[ans]}) {opts[ans]}.", why)
        else:
            print(f"   Answer: {'ABC'[ans]}) {opts[ans]}.", why)
    if interactive:
        print(f"\nScore: {score}/{len(questions)}")


# ---------------------------------------------------------------------------
# 0. Opening story
# ---------------------------------------------------------------------------
def number_trick():
    title("0a. The number game: why the answer is always 4")
    for n in [1, 7, 20, -100, 3.14, 10**12 + 7]:
        r = (2 * n + 8) / 2 - n
        print(f"   start {n!s:>16}:  double -> {2*n}, +8 -> {2*n+8}, /2 -> {(2*n+8)/2}, minus start -> {r:g}")
    print("   Algebra: y -> 2y -> 2y+8 -> (2y+8)/2 = y+4 -> (y+4) - y = 4")


def grain_problem():
    title("0b. Nine Chapters (China, ~200 BC): three classes of grain")
    print("   3x + 2y +  z = 39")
    print("   2x + 3y +  z = 34")
    print("    x + 2y + 3z = 26")
    M = [[3, 2, 1, 39], [2, 3, 1, 34], [1, 2, 3, 26]]
    print("\nAugmented matrix:"); print_augmented(M)
    R, steps = rref(M)
    x, y, z = (row[-1] for row in R)
    print(f"\nAnswer: x = {fmt(x)} = {float(x)},  y = {fmt(y)} = {float(y)},  z = {fmt(z)} = {float(z)} measures")
    print("   (9 1/4, 4 1/4 and 2 3/4 measures of grain per bundle)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    interactive = sys.stdin.isatty()
    number_trick()
    shopping_puzzle()
    linear_check()
    three_methods()
    three_outcomes()
    plot_lines()
    plot_planes()
    matrix_form()
    row_operations_demo()
    read_last_row()
    practice()
    grain_problem()
    quiz(interactive)
    print("\nNext lecture: Gaussian Elimination")
    if interactive:
        plt.show()
