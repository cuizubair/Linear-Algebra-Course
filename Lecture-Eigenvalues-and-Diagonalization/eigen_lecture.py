"""
Linear Algebra - Eigenvalues, Eigenvectors and Diagonalization
ZMathLab | Muhammad Zubair

What this file does
-------------------
1. Walks through every idea of the lecture with sympy (exact maths) and numpy,
   printing each step: the characteristic polynomial det(lambda*I - A), its
   factored form, the eigenvalues, a basis for every eigenspace, and the check
   A x = lambda x.  Run it and read the console like a second set of notes.
2. Checks every number used in the LaTeX notes (the examples are the same as
   Anton & Rorres Section 5.1 Examples 1-8 and Section 5.2 Examples 1-7).
3. Draws all the figures of the lecture into ./figs (PNG, 200 dpi).

Run:   python eigen_lecture.py
Needs: numpy, sympy, matplotlib

Colour language used in every figure (same as the video series):
    BLUE   = the original object (a vector or a shape before the matrix acts)
    ORANGE = the result (after the matrix acts)
    GOLD   = the eigenvector / eigen-direction (the arrow that never turns)
    TEAL   = a second eigen-direction
    RED    = something that goes wrong (a mistake, a missing direction)

Sections
--------
 1. The story: most arrows turn, eigenvectors do not          (fig 01, 02)
 2. Definition: A x = lambda x  (Anton 5.1 Ex. 1)
 3. Characteristic equation      (Anton 5.1 Ex. 2, Ex. 3)       (fig 03)
 4. Triangular matrices          (Anton 5.1 Ex. 4, Ex. 5)       (fig 04)
 5. Eigenspaces and their bases  (Anton 5.1 Ex. 6, Ex. 7)       (fig 05)
 6. Eigenvalues and invertibility (Anton 5.1 Ex. 8)             (fig 06)
 7. Diagonalization step by step (Anton 5.2 Ex. 1)              (fig 07)
 8. A matrix that is not diagonalizable (Anton 5.2 Ex. 2)       (fig 09)
 9. Recognizing diagonalizability (Anton 5.2 Ex. 3, Ex. 4)
10. Powers of a matrix           (Anton 5.2 Ex. 5, Ex. 6)       (fig 08)
11. Geometric vs algebraic multiplicity                         (fig 10)
12. The converse of Theorem 5.2.2(b) is false (Anton 5.2 Ex. 7) (fig 11)
13. Common student mistakes, checked by computer
"""
import os
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

NAVY = "#0F1B3D"
BLUE = "#2F6FDE"
ORANGE = "#F07C1E"
GOLD = "#F2B705"
TEAL = "#12A39A"
RED = "#D64045"
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
    "savefig.dpi": 200,
})

lam = sp.symbols("lambda")


# ======================================================================
# helpers
# ======================================================================
def say(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


def check(name, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"Check failed: {name}")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print(f"  figure saved -> figs/{name}")


def show(name, M):
    print(f"  {name} =")
    sp.pprint(M, use_unicode=True)


def char_poly_steps(A, name="A"):
    """Print det(lambda I - A) step by step and return (poly, eigen-dict)."""
    n = A.shape[0]
    M = lam * sp.eye(n) - A
    print(f"  Step 1. Form lambda*I - {name}:")
    sp.pprint(M, use_unicode=True)
    p = sp.expand(M.det())
    print(f"  Step 2. det(lambda*I - {name}) = {p}")
    print(f"  Step 3. Factor:  {sp.factor(p)} = 0")
    ev = sp.roots(sp.Poly(p, lam))
    print("  Step 4. Eigenvalues (with algebraic multiplicity):",
          ", ".join(f"{k} (x{v})" for k, v in ev.items()))
    return p, ev


def eigenspace_steps(A, ev, name="A"):
    """For each eigenvalue solve (lambda I - A) x = 0, print a basis and check A x = lambda x."""
    n = A.shape[0]
    bases = {}
    for l0, alg in ev.items():
        M = l0 * sp.eye(n) - A
        R, piv = M.rref()
        basis = M.nullspace()
        bases[l0] = basis
        print(f"\n  lambda = {l0}:  solve ({l0}*I - {name}) x = 0.  Reduced row echelon form:")
        sp.pprint(R, use_unicode=True)
        print(f"    free variables: {n - len(piv)}  ->  geometric multiplicity = {len(basis)}, "
              f"algebraic multiplicity = {alg}")
        for k, v in enumerate(basis, 1):
            v = sp.simplify(v * sp.ilcm(*[sp.fraction(x)[1] for x in v]))  # clear fractions
            basis[k - 1] = v
            print(f"    basis vector p{k} = {list(v)}")
            check(f"{name} p{k} = {l0} p{k}", sp.simplify(A * v - l0 * v) == sp.zeros(n, 1))
    return bases


def arrow(ax, p, q, c, lw=2.6, ms=14, alpha=1.0, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c,
                                 lw=lw, alpha=alpha, zorder=z, shrinkA=0, shrinkB=0))


def plane_axes(ax, lim=3.2, grid=True):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    if grid:
        for t in np.arange(-int(lim), int(lim) + 1):
            ax.axhline(t, color=GREY, lw=0.4, alpha=0.35, zorder=0)
            ax.axvline(t, color=GREY, lw=0.4, alpha=0.35, zorder=0)
    ax.axhline(0, color=NAVY, lw=1.0, zorder=1); ax.axvline(0, color=NAVY, lw=1.0, zorder=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def warped_grid(ax, M, lim=4, color=BLUE, alpha=0.35):
    M = np.array(M, float)
    t = np.linspace(-lim, lim, 60)
    for c in range(-lim, lim + 1):
        P = M @ np.vstack([np.full_like(t, c), t]); ax.plot(P[0], P[1], color=color, lw=0.7, alpha=alpha)
        P = M @ np.vstack([t, np.full_like(t, c)]); ax.plot(P[0], P[1], color=color, lw=0.7, alpha=alpha)


# ======================================================================
# 1. THE STORY - most arrows turn, eigenvectors do not
# ======================================================================
say("1. THE STORY: a matrix turns almost every arrow - except a few special ones")
A = np.array([[3, 0], [8, -1]], float)        # Anton 5.1 Example 1
print("  A = [[3, 0], [8, -1]]  (Anton 5.1 Example 1)")
print("  We push 24 arrows of length 1 through A and measure how much each one turns.")
angles = np.linspace(0, np.pi, 24, endpoint=False)
eig_dirs = []
w, V = np.linalg.eig(A)
order = np.argsort(-w)            # put lambda = 3 first (gold), lambda = -1 second (teal)
w, V = w[order], V[:, order]
for k in range(2):
    eig_dirs.append(V[:, k] / np.linalg.norm(V[:, k]))
for th in angles[:6]:
    x = np.array([np.cos(th), np.sin(th)]); y = A @ x
    turn = np.degrees(np.arctan2(x[0] * y[1] - x[1] * y[0], x @ y))
    print(f"    angle {np.degrees(th):6.1f} deg  ->  turned by {turn:7.2f} deg, length x{np.linalg.norm(y):.2f}")
print("    ...")
for k, v in enumerate(eig_dirs):
    y = A @ v
    print(f"  eigen-direction {k+1}: x = {np.round(v, 3)}  ->  Ax = {np.round(y, 3)} = {w[k]:g} * x  (no turning!)")

fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
for ax, after in zip(axs, (False, True)):
    plane_axes(ax, lim=4.2)
    for th in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        x = np.array([np.cos(th), np.sin(th)])
        if after:
            y = A @ x
            arrow(ax, (0, 0), x * 1.0, BLUE, lw=1.0, ms=8, alpha=0.25, z=2)
            arrow(ax, (0, 0), y / max(1, np.linalg.norm(y) / 3.8), ORANGE, lw=1.6, ms=10, alpha=0.75)
        else:
            arrow(ax, (0, 0), x * 2.2, BLUE, lw=1.6, ms=10, alpha=0.75)
    for k, v in enumerate(eig_dirs):
        c = GOLD if k == 0 else TEAL
        s = 1 if v[1] >= 0 else -1
        ax.plot([-6 * v[0], 6 * v[0]], [-6 * v[1], 6 * v[1]], ls="--", color=c, lw=1.4, zorder=2)
        if after:
            arrow(ax, (0, 0), s * v * min(3.8, abs(w[k]) * 1.0 + 0.2), c, lw=4, ms=20, z=5)
        else:
            arrow(ax, (0, 0), s * v * 2.2, c, lw=4, ms=20, z=5)
    ax.set_title("After A: blue arrows turned (orange),\ngold/teal arrows stayed on their line" if after
                 else "Before: arrows in every direction\n(gold and teal = eigen-directions)")
axs[1].text(-4.1, -4.0, r"gold: $\lambda=3$ (stretched $\times3$)" + "\n" + r"teal: $\lambda=-1$ (flipped)",
            fontsize=10, color=NAVY, va="bottom")
fig.suptitle(r"A = [3 0; 8 -1]:  most arrows turn, eigenvectors only stretch or flip",
             color=NAVY, fontsize=13, fontweight="bold")
save(fig, "01_story_arrows_turn.png")

# fig 02 unit circle -> ellipse (symmetric matrix so the eigen-directions are the ellipse axes)
S = np.array([[2, 1], [1, 2]], float)
ws, Vs = np.linalg.eigh(S)
print(f"\n  Circle -> ellipse with S = [[2,1],[1,2]]: eigenvalues {ws}, eigenvectors (1,-1) and (1,1).")
fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
t = np.linspace(0, 2 * np.pi, 300)
C = np.vstack([np.cos(t), np.sin(t)])
for ax, M, title in ((axs[0], np.eye(2), "Unit circle (before)"), (axs[1], S, "After S: an ellipse")):
    plane_axes(ax, lim=3.5)
    warped_grid(ax, M, color=BLUE if M is not S else ORANGE, alpha=0.25)
    P = M @ C
    ax.fill(P[0], P[1], color=BLUE if M is not S else ORANGE, alpha=0.15)
    ax.plot(P[0], P[1], color=BLUE if M is not S else ORANGE, lw=2.5)
    for k, c in ((1, GOLD), (0, TEAL)):
        v = Vs[:, k] * np.sign(Vs[1, k] if Vs[1, k] != 0 else 1)
        L = ws[k] if M is S else 1
        arrow(ax, (0, 0), v * L, c, lw=4, ms=20, z=5)
        ax.text(*(v * L * 1.12 + np.array([0.05, 0.05])), rf"$\lambda={ws[k]:g}$" if M is S else "", color=c,
                fontsize=12, fontweight="bold")
    # an ordinary vector to show turning
    x = np.array([1, 0.0]); y = M @ x
    arrow(ax, (0, 0), y, NAVY, lw=2, ms=14, z=4)
    ax.text(*(y + np.array([0.08, -0.32])), "ordinary arrow" + (" (turned)" if M is S else ""), fontsize=9, color=NAVY)
    ax.set_title(title)
fig.suptitle(r"S = [2 1; 1 2]: the eigenvectors are the axes of the ellipse",
             color=NAVY, fontsize=13, fontweight="bold")
save(fig, "02_circle_to_ellipse.png")

# ======================================================================
# 2. DEFINITION - A x = lambda x  (Anton 5.1 Example 1)
# ======================================================================
say("2. DEFINITION: x != 0 is an eigenvector of A if A x = lambda x  (Anton 5.1 Ex. 1)")
A1 = sp.Matrix([[3, 0], [8, -1]]); x1 = sp.Matrix([1, 2])
show("A", A1); show("x", x1)
print(f"  A x = {list(A1 * x1)} = 3 * {list(x1)}")
check("A x = 3 x  (eigenvalue 3)", A1 * x1 == 3 * x1)
y = sp.Matrix([1, 0])
print(f"  Compare an ordinary vector y = (1, 0):  A y = {list(A1 * y)}, not a multiple of y -> y is NOT an eigenvector.")

# ======================================================================
# 3. CHARACTERISTIC EQUATION  (Anton 5.1 Ex. 2 and Ex. 3)
# ======================================================================
say("3. CHARACTERISTIC EQUATION det(lambda I - A) = 0  (Anton 5.1 Ex. 2, Ex. 3)")
print(" Why? A x = lambda x  <=>  (lambda I - A) x = 0 has a NONZERO solution  <=>  det(lambda I - A) = 0.\n")
print(" Example 2:  A = [[3,0],[8,-1]]")
p2, ev2 = char_poly_steps(A1)
check("eigenvalues 3 and -1", set(ev2) == {3, -1})
print("\n Example 3:  A = [[0,1,0],[0,0,1],[4,-17,8]]")
A3 = sp.Matrix([[0, 1, 0], [0, 0, 1], [4, -17, 8]])
p3, ev3 = char_poly_steps(A3)
check("p(lambda) = lambda^3 - 8 lambda^2 + 17 lambda - 4", sp.expand(p3 - (lam**3 - 8 * lam**2 + 17 * lam - 4)) == 0)
print("  Integer roots must divide the constant term -4: try +-1, +-2, +-4.  p(4) =", p3.subs(lam, 4))
print("  Divide by (lambda - 4):", sp.factor(sp.div(p3, lam - 4)[0]), "-> quadratic formula gives 2 +- sqrt(3)")
check("eigenvalues 4, 2+sqrt3, 2-sqrt3", set(ev3) == {4, 2 + sp.sqrt(3), 2 - sp.sqrt(3)})
tr3, det3 = A3.trace(), A3.det()
print(f"  Bonus check: sum of eigenvalues = {sp.simplify(sum(ev3))} = trace {tr3};"
      f" product = {sp.simplify(sp.prod(list(ev3)))} = det {det3}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, p, ev, title, rng in ((axs[0], p2, ev2, r"Ex. 2: $p(\lambda)=(\lambda-3)(\lambda+1)$", (-2.5, 4.5)),
                              (axs[1], p3, ev3, r"Ex. 3: $p(\lambda)=\lambda^3-8\lambda^2+17\lambda-4$", (-0.5, 5))):
    f = sp.lambdify(lam, p, "numpy")
    L = np.linspace(*rng, 400)
    ax.axhline(0, color=NAVY, lw=1)
    ax.plot(L, f(L), color=BLUE, lw=2.5)
    for i_r, r in enumerate(sorted(ev, key=lambda z: float(z))):
        rv = float(r)
        ax.plot(rv, 0, "o", ms=11, color=GOLD, mec=NAVY, zorder=5)
        hgt = (f(L).max() - f(L).min()) * (0.18 if i_r % 2 == 0 else -0.30)
        ax.annotate(rf"$\lambda={sp.latex(r)}$", (rv, 0), xytext=(rv + (0.35 if i_r == 2 else 0), hgt),
                    ha="center", color=NAVY, fontsize=11, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_title(title); ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$p(\lambda)=\det(\lambda I-A)$")
    ax.grid(alpha=0.25)
    ax.set_ylim(min(f(L).min(), -1) * 1.1, max(f(L).max(), 1) * 1.1)
fig.suptitle("Eigenvalues = where the characteristic polynomial crosses zero (gold dots)", color=NAVY,
             fontsize=13, fontweight="bold")
save(fig, "03_characteristic_polynomial.png")

# ======================================================================
# 4. TRIANGULAR MATRICES (Anton 5.1 Ex. 4, Ex. 5)
# ======================================================================
say("4. TRIANGULAR MATRICES: eigenvalues = main-diagonal entries  (Anton 5.1 Ex. 4, Ex. 5)")
a = sp.symbols("a11 a12 a13 a14 a22 a23 a24 a33 a34 a44")
U = sp.Matrix([[a[0], a[1], a[2], a[3]], [0, a[4], a[5], a[6]], [0, 0, a[7], a[8]], [0, 0, 0, a[9]]])
pU = sp.factor((lam * sp.eye(4) - U).det())
print("  Upper triangular 4x4 (Ex. 4): det(lambda I - A) =", pU)
L5 = sp.Matrix([[sp.Rational(1, 2), 0, 0], [-1, sp.Rational(2, 3), 0], [5, -8, sp.Rational(-1, 4)]])
p5, ev5 = char_poly_steps(L5, "A (Ex. 5)")
check("lower triangular eigenvalues = 1/2, 2/3, -1/4", set(ev5) == {sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(-1, 4)})

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, M, title in ((axs[0], [["a11", "a12", "a13", "a14"], ["0", "a22", "a23", "a24"], ["0", "0", "a33", "a34"], ["0", "0", "0", "a44"]],
                      "Upper triangular (Anton Ex. 4)"),
                     (axs[1], [["1/2", "0", "0"], ["-1", "2/3", "0"], ["5", "-8", "-1/4"]], "Lower triangular (Anton Ex. 5)")):
    n = len(M)
    for i in range(n):
        for j in range(n):
            diag = i == j
            zero = (j < i) if n == 4 else (j > i)
            fc = GOLD if diag else (LIGHT if not zero else "white")
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=NAVY, lw=1.2, alpha=0.95 if diag else 1))
            ax.text(j + 0.5, n - 0.5 - i, M[i][j], ha="center", va="center", fontsize=13,
                    color=GREY if zero else NAVY, fontweight="bold" if diag else "normal")
    ax.set_xlim(-0.2, n + 3.4); ax.set_ylim(-0.6, n + 0.3); ax.set_aspect("equal"); ax.axis("off")
    eigs = [M[i][i] for i in range(n)]
    ax.text(n + 0.3, n / 2, "eigenvalues =\n" + "\n".join(eigs), va="center", fontsize=12, color=NAVY,
            bbox=dict(boxstyle="round", fc="#FFF6D6", ec=GOLD))
    ax.set_title(title)
fig.suptitle("Triangular matrix: just read the eigenvalues off the gold diagonal", color=NAVY, fontsize=13,
             fontweight="bold")
save(fig, "04_triangular_diagonal.png")

# ======================================================================
# 5. EIGENSPACES (Anton 5.1 Ex. 6, Ex. 7)
# ======================================================================
say("5. EIGENSPACES and their bases  (Anton 5.1 Ex. 6, Ex. 7)")
print(" Example 6:  A = [[-1,3],[2,0]]")
A6 = sp.Matrix([[-1, 3], [2, 0]])
p6, ev6 = char_poly_steps(A6)
b6 = eigenspace_steps(A6, ev6)
check("Ex. 6 eigenvalues 2 and -3", set(ev6) == {2, -3})
print("\n Example 7:  A = [[0,0,-2],[1,2,1],[1,0,3]]")
A7 = sp.Matrix([[0, 0, -2], [1, 2, 1], [1, 0, 3]])
p7, ev7 = char_poly_steps(A7)
b7 = eigenspace_steps(A7, ev7)
check("Ex. 7: (lambda-1)(lambda-2)^2", sp.factor(p7) == (lam - 1) * (lam - 2)**2)
check("Ex. 7: eigenspace of 2 is a plane (2 basis vectors)", len(b7[2]) == 2)

fig = plt.figure(figsize=(13, 5.6))
ax = fig.add_subplot(1, 2, 1)
plane_axes(ax, lim=4)
for l0, c in ((2, GOLD), (-3, TEAL)):
    v = np.array([float(t) for t in b6[l0][0]]); v = v / np.linalg.norm(v)
    ax.plot([-6 * v[0], 6 * v[0]], [-6 * v[1], 6 * v[1]], color=c, lw=3, alpha=0.6)
    arrow(ax, (0, 0), v * 1.6, c, lw=3.5, ms=18, z=5)
    ax.text(*(v * 3.0 + np.array([0.15, 0.1])), rf"$E_{{{l0}}}$: all multiples of $({b6[l0][0][0]},{b6[l0][0][1]})$",
            color=NAVY, fontsize=10)
ax.set_title("Ex. 6 (2x2): each eigenspace is a LINE through 0")
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
p1v, p2v = [np.array([float(t) for t in v]) for v in b7[2]]
q = np.array([float(t) for t in b7[1][0]])
s, tt = np.meshgrid(np.linspace(-1.5, 1.5, 12), np.linspace(-1.5, 1.5, 12))
X = s * p1v[0] + tt * p2v[0]; Y = s * p1v[1] + tt * p2v[1]; Z = s * p1v[2] + tt * p2v[2]
ax3.plot_surface(X, Y, Z, color=GOLD, alpha=0.30, edgecolor=GOLD, lw=0.3)
for v, c in ((p1v, ORANGE), (p2v, ORANGE)):
    ax3.quiver(0, 0, 0, *v, color=c, lw=3, arrow_length_ratio=0.15)
ax3.plot(*[np.array([-1.6, 1.6]) * qi / np.linalg.norm(q) * 2 for qi in q], color=TEAL, lw=3)
ax3.quiver(0, 0, 0, *(q / np.linalg.norm(q) * 1.5), color=TEAL, lw=3, arrow_length_ratio=0.15)
ax3.text(*(p1v * 1.1), r"$p_1$", color=NAVY); ax3.text(*(p2v * 1.1), r"$p_2$", color=NAVY)
ax3.text(*(-q / np.linalg.norm(q) * 2.2), r"$E_1$ (line)", color=TEAL)
ax3.set_title(r"Ex. 7 (3x3): $E_2$ is a gold PLANE, $E_1$ a teal LINE", color=NAVY, fontweight="bold")
ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z"); ax3.view_init(22, -55)
fig.suptitle("Eigenspace = all eigenvectors for one lambda, together with 0", color=NAVY, fontsize=13,
             fontweight="bold")
save(fig, "05_eigenspaces_line_plane.png")

# ======================================================================
# 6. EIGENVALUES AND INVERTIBILITY (Anton 5.1 Ex. 8)
# ======================================================================
say("6. A is invertible  <=>  lambda = 0 is NOT an eigenvalue  (Anton 5.1 Ex. 8, Theorem 5.1.4)")
print(f"  Ex. 7 matrix: eigenvalues {list(ev7)} (no zero)  ->  det = {A7.det()} != 0  ->  invertible.")
check("det(A7) = product of eigenvalues = 4", A7.det() == 1 * 2 * 2)
Z = sp.Matrix([[1, 2], [2, 4]])
pz, evz = char_poly_steps(Z, "B")
check("B = [[1,2],[2,4]] has eigenvalue 0 and det 0", 0 in evz and Z.det() == 0)
print("  lambda = 0 means B x = 0 x = 0 for a NONZERO x -> B squashes a whole line to the origin -> no inverse.")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2))
Zn = np.array(Z, float)
pts = np.random.default_rng(1).uniform(-1.6, 1.6, (2, 260))
for ax, M, title, col in ((axs[0], np.eye(2), "Before: the whole plane", BLUE),
                          (axs[1], Zn, r"After $B$: everything lands on ONE line", ORANGE)):
    plane_axes(ax, lim=6 if M is Zn else 2)
    P = M @ pts
    ax.scatter(P[0], P[1], s=10, color=col, alpha=0.7)
    ax.set_title(title)
k = np.array([2, -1]) / np.sqrt(5)
axs[0].plot([-2 * k[0] * 1.5, 2 * k[0] * 1.5], [-2 * k[1] * 1.5, 2 * k[1] * 1.5], color=RED, lw=3)
axs[0].text(0.3, -1.2, r"red line: $Bx=0$" + "\n" + r"($\lambda=0$ eigenspace)", color=RED, fontsize=10)
axs[1].plot([0], [0], "o", color=RED, ms=10)
axs[1].text(0.3, -1.5, "the red line is crushed\nto this one point", color=RED, fontsize=10)
fig.suptitle(r"B = [1 2; 2 4]: $\lambda=0$ is an eigenvalue $\Rightarrow$ $\det B=0$ $\Rightarrow$ not invertible",
             color=NAVY, fontsize=13, fontweight="bold")
save(fig, "06_lambda_zero_singular.png")

# ======================================================================
# 7. DIAGONALIZATION (Anton 5.2 Ex. 1)
# ======================================================================
say("7. DIAGONALIZATION: P^-1 A P = D  (Anton 5.2 Ex. 1)")
print("  Procedure: (1) find n linearly independent eigenvectors p1..pn,"
      " (2) P = [p1 ... pn], (3) D = diag(lambda1..lambdan) in the SAME order.")
P7 = sp.Matrix.hstack(b7[2][0], b7[2][1], b7[1][0])
show("P", P7)
print("  det P =", P7.det(), "(nonzero -> columns independent -> P invertible)")
D7 = P7.inv() * A7 * P7
show("P^-1 A P", D7)
check("P^-1 A P = diag(2, 2, 1)", D7 == sp.diag(2, 2, 1))
check("A P = P D (no inverse needed for checking!)", A7 * P7 == P7 * sp.diag(2, 2, 1))

# 2D picture of change of basis, with A2 = [[-1,3],[2,0]] from Ex. 6
A2 = np.array([[-1, 3], [2, 0]], float)
P2 = np.array([[1, -3], [1, 2]], float)    # columns: eigvecs for 2 and -3
D2 = np.diag([2.0, -3.0])
print("\n  2-D picture with A = [[-1,3],[2,0]], P = [[1,-3],[1,2]], D = diag(2,-3):")
check("P^-1 A P = D (2x2)", np.allclose(np.linalg.inv(P2) @ A2 @ P2, D2))
x = P2 @ np.array([0.8, 0.4])
s1 = np.linalg.inv(P2) @ x; s2 = D2 @ s1; s3 = P2 @ s2
check("P D P^-1 x = A x", np.allclose(s3, A2 @ x))
print(f"    x = {x},  P^-1 x = {np.round(s1, 3)},  D(P^-1 x) = {np.round(s2, 3)},  P(...) = {np.round(s3, 3)} = A x")
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
steps = [(np.eye(2), x, "1. Start: x in normal\ncoordinates", BLUE),
         (np.linalg.inv(P2), s1, r"2. $P^{-1}x$: coordinates" + "\nalong the eigenvectors", TEAL),
         (D2 @ np.linalg.inv(P2), s2, "3. $D$: just stretch each\ncoordinate (x2 and x(-3))", GOLD),
         (A2, s3, "4. $P$: back to normal\ncoordinates = $Ax$", ORANGE)]
for ax, (M, v, title, col) in zip(axs, steps):
    lim = max(2.2, 1.35 * np.abs(v).max())
    plane_axes(ax, lim=lim)
    if title.startswith("1") or title.startswith("4"):
        for k, c in ((0, GOLD), (1, TEAL)):
            e = P2[:, k] / np.linalg.norm(P2[:, k])
            ax.plot([-8 * e[0], 8 * e[0]], [-8 * e[1], 8 * e[1]], ls="--", color=c, lw=1.4)
    else:
        ax.axhline(0, color=GOLD, ls="--", lw=1.4); ax.axvline(0, color=TEAL, ls="--", lw=1.4)
    arrow(ax, (0, 0), v, col if col != GOLD else ORANGE, lw=3.5, ms=18)
    ax.text(v[0] + 0.15, v[1] + 0.15, f"({v[0]:.2f}, {v[1]:.2f})", fontsize=9, color=NAVY)
    ax.set_title(title, fontsize=11)
fig.suptitle(r"$Ax = P\,D\,P^{-1}x$: change to eigen-coordinates, stretch, change back", color=NAVY,
             fontsize=13, fontweight="bold")
save(fig, "07_diagonalization_change_of_basis.png")

# ======================================================================
# 8. NOT DIAGONALIZABLE (Anton 5.2 Ex. 2) + shear picture
# ======================================================================
say("8. A MATRIX THAT IS NOT DIAGONALIZABLE  (Anton 5.2 Ex. 2)")
A8 = sp.Matrix([[1, 0, 0], [1, 2, 0], [-3, 5, 2]])
p8, ev8 = char_poly_steps(A8)
b8 = eigenspace_steps(A8, ev8)
total = sum(len(v) for v in b8.values())
print(f"  Total independent eigenvectors = {total} < 3  ->  NOT diagonalizable.")
check("Ex. 2 is not diagonalizable", total == 2 and not A8.is_diagonalizable())
Sh = sp.Matrix([[1, 1], [0, 1]])
print("\n  The simplest example: the shear H = [[1,1],[0,1]]")
ph, evh = char_poly_steps(Sh, "H")
bh = eigenspace_steps(Sh, evh, "H")
check("shear has only one eigen-direction", len(bh[1]) == 1)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2))
H = np.array(Sh, float)
for ax, after in zip(axs, (False, True)):
    plane_axes(ax, lim=3.2, grid=False)
    warped_grid(ax, H if after else np.eye(2), lim=4, color=ORANGE if after else BLUE, alpha=0.35)
    for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        v = np.array([np.cos(th), np.sin(th)]) * 1.3
        if abs(np.sin(th)) < 1e-9:
            continue
        arrow(ax, (0, 0), H @ v if after else v, ORANGE if after else BLUE, lw=1.5, ms=10, alpha=0.8)
    for sgn in (1, -1):
        arrow(ax, (0, 0), (sgn * 1.6, 0), GOLD, lw=4, ms=20, z=5)
    ax.axhline(0, color=GOLD, lw=2.5, alpha=0.5)
    ax.set_title(r"After shear $H$: only the x-axis stays put" if after else "Before")
axs[1].text(-3.0, -2.9, "Only ONE eigen-direction in the whole plane\n-> not enough eigenvectors to build P\n-> NOT diagonalizable",
            color=RED, fontsize=10, bbox=dict(boxstyle="round", fc="white", ec=RED))
fig.suptitle(r"H = [1 1; 0 1]: $\lambda=1$ twice, but only one eigenvector",
             color=NAVY, fontsize=13, fontweight="bold")
save(fig, "09_not_diagonalizable_shear.png")

# ======================================================================
# 9. RECOGNIZING DIAGONALIZABILITY (Anton 5.2 Ex. 3, Ex. 4)
# ======================================================================
say("9. RECOGNIZING DIAGONALIZABILITY  (Anton 5.2 Ex. 3, Ex. 4)")
print("  Theorem: n DISTINCT eigenvalues  ->  diagonalizable (enough, but not necessary).")
print(f"  Ex. 3: A = [[0,1,0],[0,0,1],[4,-17,8]] has eigenvalues {list(ev3)}: 3 distinct -> diagonalizable.")
check("Ex. 3 diagonalizable", A3.is_diagonalizable())
T4 = sp.Matrix([[-1, 2, 4, 0], [0, 3, 1, 7], [0, 0, 5, 8], [0, 0, 0, -2]])
print(f"  Ex. 4: triangular with diagonal {[T4[i, i] for i in range(4)]} (all distinct) -> diagonalizable.")
check("Ex. 4 diagonalizable", T4.is_diagonalizable())

# ======================================================================
# 10. POWERS (Anton 5.2 Ex. 5, Ex. 6)
# ======================================================================
say("10. POWERS:  A^k x = lambda^k x  and  A^k = P D^k P^-1  (Anton 5.2 Ex. 5, Ex. 6)")
A87 = A8**7
p87, ev87 = char_poly_steps(A87, "A^7")
check("Ex. 5: eigenvalues of A^7 are 1 and 128", set(ev87) == {1, 128})
A13 = P7 * sp.diag(2**13, 2**13, 1) * P7.inv()
show("A^13 = P D^13 P^-1", A13)
check("Ex. 6: A^13 by diagonalization equals direct power", A13 == A7**13)
check("Ex. 6: A^13 = [[-8190,0,-16382],[8191,8192,8191],[8191,0,16383]]",
      A13 == sp.Matrix([[-8190, 0, -16382], [8191, 8192, 8191], [8191, 0, 16383]]))

M = np.array([[2, 1], [1, 2]], float) / 3 * 1.0     # eigenvalues 1 and 1/3
wv, Vv = np.linalg.eigh(M)
print(f"\n  Picture: M = [[2,1],[1,2]]/3 has eigenvalues {np.round(wv, 4)}."
      " Repeated multiplication kills the 1/3 part, every vector lines up with the gold eigenvector.")
fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
rng = np.random.default_rng(4)
starts = [np.array([np.cos(t), np.sin(t)]) * 1.6 for t in np.linspace(0.2, 2 * np.pi, 10, endpoint=False)]
for ax, k in zip(axs, (0, 1, 3, 10)):
    plane_axes(ax, lim=2.2)
    Mk = np.linalg.matrix_power(M, k)
    for v in starts:
        arrow(ax, (0, 0), Mk @ v, BLUE if k == 0 else ORANGE, lw=1.8, ms=11, alpha=0.85)
    e = Vv[:, 1]
    ax.plot([-3 * e[0], 3 * e[0]], [-3 * e[1], 3 * e[1]], color=GOLD, lw=2.5, ls="--")
    ax.set_title(f"$M^{{{k}}}x$" + ("  (start)" if k == 0 else ""))
fig.suptitle(r"$M^k=P\,D^k P^{-1}$ with $D^k=\mathrm{diag}(1^k,(1/3)^k)$: all arrows line up with the gold eigenvector",
             color=NAVY, fontsize=13, fontweight="bold")
save(fig, "08_powers_align.png")

# ======================================================================
# 11. GEOMETRIC vs ALGEBRAIC MULTIPLICITY
# ======================================================================
say("11. GEOMETRIC vs ALGEBRAIC MULTIPLICITY")
rows = []
for name, Mx in (("Ex. 7 (5.1/5.2)", A7), ("Ex. 2 (5.2)", A8), ("I (3x3)", sp.eye(3)),
                 ("J (5.2 Ex. 7)", sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]]))):
    ev = Mx.eigenvects()
    for l0, alg, vecs in ev:
        rows.append((name, l0, alg, len(vecs)))
        print(f"  {name:16s} lambda = {str(l0):3s}  algebraic = {alg}  geometric = {len(vecs)}"
              f"   {'OK (equal)' if alg == len(vecs) else '<-- geometric smaller: NOT diagonalizable'}")
    print(f"  {'':16s} diagonalizable? {Mx.is_diagonalizable()}")
check("geometric <= algebraic always", all(g <= a_ for _, _, a_, g in rows))
fig, ax = plt.subplots(figsize=(11, 4.4))
labels = [f"{n}\n$\\lambda={sp.latex(l)}$" for n, l, _, _ in rows]
xs = np.arange(len(rows))
ax.bar(xs - 0.18, [a_ for _, _, a_, _ in rows], 0.36, color=BLUE, label="algebraic multiplicity (repeats in p(λ))")
ax.bar(xs + 0.18, [g for _, _, _, g in rows], 0.36, color=GOLD, label="geometric multiplicity (dim of eigenspace)")
for i, (_, _, a_, g) in enumerate(rows):
    if g < a_:
        ax.annotate("gap!", (i + 0.18, g), xytext=(i + 0.18, g + 0.9), ha="center", color=RED,
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9); ax.set_yticks([0, 1, 2, 3])
ax.set_ylim(0, 3.9); ax.legend(loc="upper left", fontsize=9, frameon=False); ax.grid(axis="y", alpha=0.25)
ax.set_title("Diagonalizable  <=>  gold bar = blue bar for EVERY eigenvalue")
save(fig, "10_multiplicity_bars.png")

# ======================================================================
# 12. CONVERSE IS FALSE (Anton 5.2 Ex. 7)
# ======================================================================
say("12. THE CONVERSE OF THEOREM 5.2.2(b) IS FALSE  (Anton 5.2 Ex. 7)")
print("  Theorem 5.2.2(b): n distinct eigenvalues -> diagonalizable.  Converse: diagonalizable -> distinct? FALSE.")
I3 = sp.eye(3); J = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
for name, Mx in (("I", I3), ("J", J)):
    p, ev = char_poly_steps(Mx, name)
    nul = (sp.eye(3) - Mx).nullspace()
    print(f"  dim of eigenspace for lambda = 1: {len(nul)};  diagonalizable: {Mx.is_diagonalizable()}")
check("I diagonalizable with only one eigenvalue", I3.is_diagonalizable())
check("J not diagonalizable", not J.is_diagonalizable())
fig = plt.figure(figsize=(12, 5.2))
for k, (Mx, title, nul) in enumerate(((I3, r"$I$: $\lambda=1$ (x3), eigenspace = ALL of $R^3$", 3),
                                      (J, r"$J$: $\lambda=1$ (x3), eigenspace = one LINE", 1))):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    if nul == 3:
        for v, c in ((np.eye(3)[0], GOLD), (np.eye(3)[1], GOLD), (np.eye(3)[2], GOLD)):
            ax.quiver(0, 0, 0, *v, color=c, lw=3, arrow_length_ratio=0.15)
        g = np.linspace(-1, 1, 5)
        for a_ in g:
            for b_ in g:
                ax.plot([a_, a_], [b_, b_], [-1, 1], color=GOLD, alpha=0.18, lw=0.8)
        ax.text(0.6, 0.6, 1.2, "every vector is an eigenvector\n-> diagonalizable (already diagonal)", color=NAVY, fontsize=9)
    else:
        ax.plot([-1.5, 1.5], [0, 0], [0, 0], color=GOLD, lw=4)
        ax.quiver(0, 0, 0, 1, 0, 0, color=GOLD, lw=3, arrow_length_ratio=0.15)
        for v in (np.array([0, 1, 0.0]), np.array([0, 0, 1.0])):
            ax.quiver(0, 0, 0, *v, color=RED, lw=2, arrow_length_ratio=0.15, alpha=0.7)
            w_ = np.array(J, float) @ v
            ax.quiver(0, 0, 0, *w_, color=ORANGE, lw=2, arrow_length_ratio=0.15)
        ax.text(-1.4, 0.2, 1.2, "only the x-axis survives\n-> NOT diagonalizable", color=RED, fontsize=9)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
    ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold"); ax.view_init(20, -60)
fig.suptitle("Same characteristic polynomial (λ-1)³, different answers: count eigenvectors, not eigenvalues",
             color=NAVY, fontsize=13, fontweight="bold")
save(fig, "11_converse_false_I_vs_J.png")

# ======================================================================
# 13. COMMON STUDENT MISTAKES (checked)
# ======================================================================
say("13. COMMON STUDENT MISTAKES  (each one checked by the computer)")
print("  Mistake 1: 'eigenvalues of A are the diagonal entries' -> only for TRIANGULAR matrices.")
Bm = sp.Matrix([[-1, 3], [2, 0]])
print(f"    A = [[-1,3],[2,0]]: diagonal (-1, 0) but eigenvalues {list(Bm.eigenvals())}.")
check("diagonal != eigenvalues for a non-triangular matrix", set(Bm.eigenvals()) != {-1, 0})
print("  Mistake 2: 'row reduce A first, then read eigenvalues' -> row operations CHANGE eigenvalues.")
R = Bm.echelon_form()
print(f"    echelon form of A = {R.tolist()}, its eigenvalues {list(R.eigenvals())} != {list(Bm.eigenvals())}")
check("row reduction changes eigenvalues", set(R.eigenvals()) != set(Bm.eigenvals()))
print("  Mistake 3: 'x = 0 is an eigenvector' -> NO, eigenvectors must be nonzero (A0 = lambda 0 for every lambda).")
print("  Mistake 4: 'eigenvalues of A+B are sums of eigenvalues' -> false in general.")
Ae = sp.Matrix([[1, 1], [0, 1]]); Be = sp.Matrix([[1, 0], [1, 1]])
print(f"    A = [[1,1],[0,1]], B = [[1,0],[1,1]]: eig(A) = {{1}}, eig(B) = {{1}}, eig(A+B) = {list((Ae + Be).eigenvals())}")
check("eig(A+B) is not {2}", set((Ae + Be).eigenvals()) != {2})
print("  Mistake 5: 'repeated eigenvalue -> not diagonalizable' -> false (look at I, or Anton 5.2 Ex. 1).")
check("Ex. 1 has repeated lambda = 2 yet is diagonalizable", A7.is_diagonalizable())
print("  Mistake 6: putting eigenvalues in D in a different order from the columns of P.")
Pbad = P2[:, ::-1]
print(f"    with P columns swapped, P^-1 A P = {np.round(np.linalg.inv(Pbad) @ A2 @ Pbad, 3).tolist()} (order of D swaps too)")
print("  Mistake 7: forgetting to write det(lambda I - A) (or det(A - lambda I)) consistently: both give the same roots.")
check("det(A - lambda I) = (-1)^n det(lambda I - A)",
      sp.expand((A7 - lam * sp.eye(3)).det() + (lam * sp.eye(3) - A7).det()) == 0)

say("DONE: every check passed, figures are in ./figs")
