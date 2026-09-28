"""Méthodes itératives (Jacobi, Gauss-Seidel, SOR) pour résoudre A x = b
avec A tridiagonale creuse : comparaison du temps, du nombre d'itérations et du temps CPU.

Tous les solveurs ont la même signature (A, b, x0, x_exact, ...) et renvoient
(x, nb_iterations, erreurs, temps), avec erreur_k = ||x_k - x_exact||_2.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import inv, spsolve_triangular

TOL = 1e-6
MAX_ITER = 10_000
OMEGA = 1.2


# ----------------------------------------------------------------------------
# Génération du système linéaire
# ----------------------------------------------------------------------------
def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=10, off_diagonal_value=4):
    """Matrice tridiagonale (creuse CSR et dense) et second membre aléatoire."""
    A = sparse.diags([off_diagonal_value, diagonal_value, off_diagonal_value],
                     [-1, 0, 1], shape=(n, n), format="csr", dtype=float)
    return A, A.toarray(), np.random.rand(n)


def generate_sparse_tridiagonal_matrix(n):
    """Laplacien 1D (2, -1) divisé par h², avec h = 1/(n+1)."""
    A, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n, 2, -1)
    h2 = (1 / (n + 1)) ** 2
    return A / h2, A_dense / h2, b


# ----------------------------------------------------------------------------
# Solveurs
# ----------------------------------------------------------------------------
def jacobi_method(A, b, x0, x_exact, tol=TOL, max_iter=MAX_ITER):
    """Jacobi sur matrice dense (boucle explicite sur les lignes)."""
    n = A.shape[0]
    x = x0.copy()
    errors = []
    start = time.perf_counter()
    for j in range(max_iter):
        x_new = np.empty_like(x)
        for i in range(n):
            x_new[i] = (b[i] - A[i] @ x + A[i, i] * x[i]) / A[i, i]
        x = x_new
        errors.append(np.linalg.norm(x - x_exact))
        if errors[-1] < tol:
            break
    return x, j + 1, errors, time.perf_counter() - start


def jacobi_sparse_with_error(A, b, x0, x_exact, tol=TOL, max_iter=MAX_ITER):
    """Jacobi sur matrice creuse : x_{k+1} = D^-1 (b - B x_k), avec B = A - D."""
    D = A.diagonal()
    B = (A - sparse.diags(D)).tocsr()
    B.eliminate_zeros()
    x = x0.copy()
    errors = []
    start = time.perf_counter()
    for j in range(max_iter):
        x = (b - B @ x) / D
        errors.append(np.linalg.norm(x - x_exact))
        if errors[-1] < tol:
            break
    return x, j + 1, errors, time.perf_counter() - start


def SOR_sparseReel(A, b, x0, x_exact, w=OMEGA, tol=TOL, max_iter=MAX_ITER):
    """SOR composante par composante sur une matrice CSR (x mis à jour sur place)."""
    d = A.diagonal()
    indptr, indices, data = A.indptr, A.indices, A.data
    x = x0.copy()
    errors = []
    start = time.perf_counter()
    for k in range(max_iter):
        for j in range(A.shape[0]):
            lo, hi = indptr[j], indptr[j + 1]
            ligne_fois_x = data[lo:hi] @ x[indices[lo:hi]]  # inclut a_jj * x_j
            x[j] += w * (b[j] - ligne_fois_x) / d[j]
        errors.append(np.linalg.norm(x - x_exact))
        if errors[-1] < tol:
            break
    return x, k + 1, errors, time.perf_counter() - start


def _decomposition(A):
    """A = D - L - U, avec L et U l'opposé des parties strictement inférieure et supérieure."""
    D = sparse.diags(A.diagonal(), format="csc")
    L = -sparse.tril(A, k=-1, format="csc")
    U = -sparse.triu(A, k=1, format="csc")
    return D, L, U


def SOR_sparse(A, b, x0, x_exact, w=OMEGA, tol=TOL, max_iter=MAX_ITER):
    """Variante Benchiha-Muratore : pas de Gauss-Seidel vectoriel, (D-L)^-1 précalculé,
    puis relaxation x_new = w * x_gs + (1 - w) * x. Pour w = 1, c'est Gauss-Seidel."""
    D, L, U = _decomposition(A)
    C1 = inv((D - L).tocsc())
    C1_b = C1 @ b
    C1_U = C1 @ U
    x = x0.copy()
    errors = []
    start = time.perf_counter()  # le précalcul de (D-L)^-1 n'est pas compté
    for j in range(max_iter):
        x = w * (C1_b + C1_U @ x) + (1 - w) * x
        errors.append(np.linalg.norm(x - x_exact))
        if errors[-1] < tol:
            break
    return x, j + 1, errors, time.perf_counter() - start


def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=TOL, max_iter=MAX_ITER):
    return SOR_sparse(A, b, x0, x_exact, w=1.0, tol=tol, max_iter=max_iter)


def SOR_matrice(A, b, x0, x_exact, w=OMEGA, tol=TOL, max_iter=MAX_ITER):
    """SOR sous forme matricielle : (D + wL) x_{k+1} = w b + ((1-w)D - wU) x_k,
    résolu par substitution avant (L, U : parties strictement inférieure / supérieure de A)."""
    d = A.diagonal()
    M = (sparse.diags(d) + w * sparse.tril(A, k=-1)).tocsr()
    N = (sparse.diags((1 - w) * d) - w * sparse.triu(A, k=1)).tocsr()
    x = x0.copy()
    errors = []
    start = time.perf_counter()
    for k in range(max_iter):
        x = spsolve_triangular(M, w * b + N @ x, lower=True)
        errors.append(np.linalg.norm(x - x_exact))
        if errors[-1] < tol:
            break
    return x, k + 1, errors, time.perf_counter() - start


# Méthodes comparées : nom -> fonction (A, A_dense, b, x0, x_exact, w)
METHODES = {
    "SOR": lambda A, Ad, b, x0, xe, w: SOR_sparseReel(A, b, x0, xe, w),
    "Jacobi dense": lambda A, Ad, b, x0, xe, w: jacobi_method(Ad, b, x0, xe),
    "Jacobi sparse": lambda A, Ad, b, x0, xe, w: jacobi_sparse_with_error(A, b, x0, xe),
    "Gauss-Seidel": lambda A, Ad, b, x0, xe, w: gauss_seidel_sparse_with_error(A, b, x0, xe),
    "Benchiha-Muratore": lambda A, Ad, b, x0, xe, w: SOR_sparse(A, b, x0, xe, w),
}


# ----------------------------------------------------------------------------
# Graphiques
# ----------------------------------------------------------------------------
def _finir(fichier):
    if fichier:
        plt.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.show()


def plot_error(errors, name, fichier=None):
    """Erreur en fonction de l'itération (échelle log)."""
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("Itérations")
    plt.ylabel("Erreur")
    plt.title("Erreur vs itérations : " + name)
    plt.grid(True)
    _finir(fichier)


def simple_plot(x, y, name, xlabel="omega", ylabel="Temps d'exécution (s)", fichier=None):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.grid(True)
    _finir(fichier)


def plot_compare(x, series, xlabel, ylabel, title, fichier=None):
    """Trace plusieurs courbes {nom: valeurs} en fonction de x."""
    plt.figure(figsize=(10, 6))
    for (nom, y), marqueur in zip(series.items(), "o*s+x"):
        plt.plot(x, y, marker=marqueur, label=nom)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    _finir(fichier)


# ----------------------------------------------------------------------------
# Expériences
# ----------------------------------------------------------------------------
def trace_sor(A, b, x0, x_exact, repeats=3, omegas=np.arange(0.1, 2.0, 0.01), fichier=None):
    """Temps d'exécution du SOR en fonction de omega (moyenne sur `repeats` exécutions).
    SOR converge pour 0 < omega < 2."""
    temps = np.array([[SOR_sparseReel(A, b, x0, x_exact, w)[3] for w in omegas]
                      for _ in range(repeats)])
    moyenne = temps.mean(axis=0)
    simple_plot(omegas, moyenne, f"Temps d'exécution du SOR en fonction de omega "
                                 f"(moyenne sur {repeats} exécutions)", fichier=fichier)
    return omegas, moyenne


def trace_compare(borne, debut=13, pas=3, w=1.5, dossier=None):
    """Compare les méthodes pour n = debut, debut + pas, ... < borne.
    Temps = boucle d'itération seule ; CPU = fonction entière (préparation incluse)."""
    tailles = list(range(debut, borne, pas))
    temps, iterations, cpu = ({nom: [] for nom in METHODES} for _ in range(3))
    for n in tailles:
        A, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n)
        x0 = np.random.rand(n)
        x_exact = np.linalg.solve(A_dense, b)
        for nom, methode in METHODES.items():
            debut_cpu = time.process_time()
            _, nb_iter, _, tps = methode(A, A_dense, b, x0, x_exact, w)
            cpu[nom].append(time.process_time() - debut_cpu)
            temps[nom].append(tps)
            iterations[nom].append(nb_iter)

    def chemin(nom):
        return os.path.join(dossier, nom) if dossier else None

    xlabel = "Taille de la matrice (n x n)"
    plot_compare(tailles, temps, xlabel, "Temps de résolution (s)",
                 "Temps de résolution en fonction de la taille de la matrice", chemin("temps_vs_n.png"))
    plot_compare(tailles, iterations, xlabel, "Nombre d'itérations",
                 "Nombre d'itérations en fonction de la taille de la matrice", chemin("iterations_vs_n.png"))
    plot_compare(tailles, cpu, xlabel, "Temps CPU (s)",
                 "Temps CPU en fonction de la taille de la matrice", chemin("cpu_vs_n.png"))


if __name__ == "__main__":
    np.random.seed(0)
    os.makedirs("figures", exist_ok=True)

    # Un système de taille n = 10 : résultats de chaque méthode
    n = 10
    A, A_dense, b = generate_simple_sparse_tridiagonal_matrix(n)
    x0 = np.random.rand(n)
    x_exact = np.linalg.solve(A_dense, b)
    for nom, methode in METHODES.items():
        _, nb_iter, erreurs, tps = methode(A, A_dense, b, x0, x_exact, OMEGA)
        print(f"{nom:18s} {nb_iter:4d} itérations   {tps:.4f} s   erreur finale {erreurs[-1]:.2e}")

    # Influence de omega, puis comparaison en fonction de n
    trace_sor(A, b, x0, x_exact, fichier="figures/temps_vs_omega.png")
    trace_compare(100, dossier="figures")
