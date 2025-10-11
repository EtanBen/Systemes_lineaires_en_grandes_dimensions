import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
import time
import tracemalloc
import psutil
import os 



#INITIALISATION DES PARAMÈTRES

n=10
w=1.2
tol = 1e-6
itera = 10000

###########

def generate_sparse_tridiagonal_matrix(n):
    A_sparse, A_dense,b=generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=2, off_diagonal_value=-1)
    h=1/(n+1)
    A_sparse=A_sparse/(h*h)
    A_dense=A_dense/h**2
    return A_sparse,  A_dense, b

    
def generate_sparse_tridiagonal_matrix(n):
    
    A_sparse, A_dense,b=generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=2, off_diagonal_value=-1)
    h=1/(n+1)
    A_sparse=A_sparse/(h*h)
    A_dense=A_dense/h**2
    return A_sparse,  A_dense, b



def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=10, off_diagonal_value=4):   
    main_diag = np.full(n, diagonal_value)
    sub_diag = np.full(2*n-2, off_diagonal_value) #il y a 2n - 2 fois off_diagonal_value
    data = np.concatenate((main_diag,sub_diag), axis=None) #le vecteur valeurs
    rows = np.concatenate((np.arange(n),np.arange(n-1),np.arange(1,n)), axis=None) # le vecteur ligne
    cols = np.concatenate((np.arange(n),np.arange(1,n),np.arange(n-1)), axis=None) # le vecteur colonne 
    A_dense = np.diag(np.full(n,diagonal_value))
    A_dense += np.diag(np.full(n-1,off_diagonal_value),1)
    A_dense += np.diag(np.full(n-1,off_diagonal_value),-1)
    As = csr_matrix((data, (rows, cols)), shape=(n, n)) # Generates CSR Matrix from COO representatio
    b = np.random.rand(n)
    return As, A_dense, b





################################################################

def jacobi_method(A, b, x0, tol=tol, max_iter=itera):


    n = A.shape[0]
    x = x0.copy()
    errors = []


    start_time = time.time()

    for j in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(0,n):
            x_new[i]=(1/A[i,i])*(b[i]- np.dot(A[i],x) + A[i,i]*x[i]) 
    
 
        error = np.linalg.norm(x-x_new)
        errors.append(error)

        x = x_new

        if error < tol:
            break

    end_time = time.time()
    time_taken = end_time - start_time
    
    return x, j + 1, errors, time_taken


def jacobi_sparse_with_error(A, b, x0, x_exact, tol=tol, max_iter=itera):
    errors = []
    n = A.shape[0]
    x = x0.copy()
    

    D = A.diagonal()

    B = A 

    for i in range(n):
        B[i,i]=0
    B.eliminate_zeros()


    start_time = time.time()
    for j in range(max_iter):

        x_new = (b - B.dot(x))/D

        error = np.linalg.norm((x_new-x_exact))
        errors.append(error)
        x = x_new

        if(error) < tol:
            break
    
    end_time = time.time()
    time_taken = end_time - start_time
    return x_new, j+1, errors,time_taken

def SOR_sparseReel(A, b, x0, x_exact, w, tol=1e-6, max_iter=100000):
    n = A.shape[0]
    x = x0.copy()
    start_time = time.time()
    errors = []

    # Itération de Gauss-Seidel
    for i in range(max_iter):
        x_new = x.copy()
        s = 0

        for j in range(n):
            som = A[j, :j].dot(x_new[:j]) + A[j, j+1:].dot(x[j+1:])
            s = (b[j] - som) / A[j, j]
            x_new[j] = w * s + (1 - w) * x[j]

        error = np.linalg.norm(x_new - x_exact)
        errors.append(error)

        if error < tol:
            break

        x = x_new

    end_time = time.time()
    time_taken = end_time - start_time

    return x_new, i + 1, errors, time_taken,w


def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=tol, max_iter=itera):
    errors = []
    n = A.shape[0]
    x = x0.copy()
    

    D=A.copy()
    U=A.copy()
    L=A.copy()
    
    for i in range(n):
        for j in range(0,i+1):
            U[i,j]=0
    U.eliminate_zeros()
    

    for i in range(n):
        for j in range(0,n):
            if i!=j:
                D[i,j]=0
    D.eliminate_zeros()


    L=-L+U+D

    U=-U

    C1=sparse.linalg.spsolve(D-L,sparse.eye(n)) 
    

    C1foisb=C1.dot(b)
    C1foisU=C1.dot(U)
    
    start_time = time.time()
    for j in range(max_iter):

        #x_new = C1*b + C1.dot(U)*x           différentes manieres d'écrire le calcul ( +/- optimisées )
        #x_new = C1*(b + U*x)
        x_new = C1foisb + C1foisU.dot(x) 

        error = np.linalg.norm((x_new-x_exact))
        errors.append(error)
        x = x_new

        if(error) < tol:
            break

    end_time = time.time()
    time_taken = end_time - start_time
    return x_new, j+1, errors, time_taken

def SOR_matrice(A, b, w, x0, x_exact, tol=tol, max_iter=itera):
    n = A.shape[0]
    x=x0.copy()
    D=sparse.diags(A.diagonal(),format='csr')
    DUomega=D-w*sparse.triu(A,format='csr')
    DLomega=D+w*sparse.tril(A,k=-1,format='csr')
    errors=[] 
    bomega=b*w
    start_time = time.time()

    for k in range(max_iter) :
        x_new=sparse.linalg.spsolve_triangular(DLomega,bomega+(DUomega @ x))
        error= np.linalg.norm(x_new-x_exact,ord=np.inf)
        errors.append(error) 
    end_time = time.time()
    time_taken = end_time - start_time
    return x_new, k+1, errors, time_taken
    



def SOR_sparse(A, b, w, x0, x_exact, tol=tol, max_iter=itera) : 

    errors = []
    n = A.shape[0]
    s = x0.copy()
    

    D=A.copy()
    U=A.copy()
    L=A.copy()
    
    for i in range(n):
        for j in range(0,i+1):
            U[i,j]=0
    U.eliminate_zeros()
    

    for i in range(n):
        for j in range(0,n):
            if i!=j:
                D[i,j]=0
    D.eliminate_zeros()

    L=-L+U+D
    U=-U
    C1=inv(D-L) 
    

    C1foisb=C1.dot(b)
    C1foisU=C1.dot(U)
    
    start_time = time.time()
    for j in range(max_iter):

        s_new = C1foisb + C1foisU.dot(s) 
        
        x_new = w*s_new +(1-w)*s

        
        error = np.linalg.norm((x_new-x_exact))
        errors.append(error)
        
        s=x_new 

        if error < tol:
            break

        
    end_time = time.time()
    time_taken = end_time - start_time

    return x_new, j+1, errors, time_taken,w



def plot_error(errors, iterations,name):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations " + name ) 
    plt.grid(True)
    plt.show()

def simple_plot(x,y,name) :
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.ylabel("temps d'éxécution ")
    plt.xlabel("omega")
    plt.title(name ) 
    plt.grid(True)
    plt.show()



#import matplotlib.pyplot as plt

def plot_all_errIT(err1, it1, err2, it2, err3, it3, err4, it4,xname="Itérations",yname="Erreur (log et linéaire)"):
    plt.figure(figsize=(10, 6))  # Taille de la figure

    # Tracer les courbes avec des étiquettes pour les légendes
    #plt.plot(range(it1), err1, marker='o', linestyle='-', label='Courbe 1 (SOR)')
    plt.semilogy(range(it1), err1, marker='o', linestyle='-', label='Courbe SOR (log)')
    
    #plt.plot(range(it2), err2, marker='o', linestyle='-', label='Courbe 2 (Jacobi Dense)')
    plt.semilogy(range(it2), err2, marker='*', linestyle='-', label='Courbe Jacobi Dense (log)')
    
    #plt.plot(range(it3), err3, marker='o', linestyle='-', label='Courbe 3 (linéaire)')
    plt.semilogy(range(it3), err3, marker='s', linestyle='-', label='Courbe Jacobi Sparse (log)')
    
    #plt.plot(range(it4), err4, marker='o', linestyle='-', label='Courbe 4 (linéaire)')
    plt.semilogy(range(it4), err4, marker='+', linestyle='-', label='Courbe Gauss-Seidel (log)')

    # Ajouter des légendes
    plt.legend()

    # Ajouter des titres et des labels aux axes
    plt.title("Évolution des erreurs sur différentes itérations")
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Afficher le graphique
    plt.show()

def plot_all(err1, it1, err2, it2, err3, it3, err4, it4,err5,it5,xname,yname,titre) : 
    plt.figure(figsize=(10, 6))  # Taille de la figure

    # Tracer les courbes avec des étiquettes pour les légendes
    #plt.plot(range(it1), err1, marker='o', linestyle='-', label='Courbe 1 (SOR)')
    plt.plot(it1, err1, marker='o', linestyle='-', label='Courbe SOR ')
    
    #plt.plot(range(it2), err2, marker='o', linestyle='-', label='Courbe 2 (Jacobi Dense)')
    plt.plot(it2, err2, marker='*', linestyle='-', label='Courbe Jacobi Dense ')
    
    #plt.plot(range(it3), err3, marker='o', linestyle='-', label='Courbe 3 (linéaire)')
    plt.plot(it3, err3, marker='s', linestyle='-', label='Courbe Jacobi Sparse ')
    
    #plt.plot(range(it4), err4, marker='o', linestyle='-', label='Courbe 4 (linéaire)')
    plt.plot(it4, err4, marker='+', linestyle='-', label='Courbe Gauss-Seidel')

    plt.plot(it5, err5, marker='+', linestyle='-', label='Courbe de Benchiha-Muratore')

    # Ajouter des légendes
    plt.plot()
    plt.legend() 

    # Ajouter des titres et des labels aux axes
    plt.title(titre)
    plt.xlabel(xname)
    plt.ylabel(yname)


    # Afficher le graphique
    plt.show()







#GÉNÉRATION DE NOTRE SYTEME LINÉAIRE (A et b): A en version dense et sparse

A_sparse,A_dense,b=generate_simple_sparse_tridiagonal_matrix(n)
A_sor = A_sparse.copy() 
A_jsparse = A_sparse.copy()
A_gs = A_sparse.copy()
x0 = np.random.rand(n)

#SOLUTION EXACTE 

x_exact = np.linalg.solve(A_dense, b)

#Appels des fonctions

x_sor, j, err, time_sor,w=SOR_sparseReel(A_sor,b,x0,x_exact,w) 
#SOR

x_jacobi, j_jacobi, err_jacobi, time_jacobi = jacobi_method(A_dense, b, x0) 
#jacobi dance

x_jacospar, j_jacospar, err_jacospar, time_jacospar = jacobi_sparse_with_error(A_jsparse, b, x0, x_exact) 
#jacobi sparse

x_gauss_seidel, j_gauss_seidel, err_gauss_seidel, time_gauss_seidel = gauss_seidel_sparse_with_error(A_gs, b, x0, x_exact) 
#gauss_seidel

#GRAPHIQUES

#plot_all_errIT(err,j,err_jacobi,j_jacobi,err_jacospar,j_jacospar,err_gauss_seidel,j_gauss_seidel)
#print(time_gauss_seidel) 
#plot_all(n, time_sor, n, time_jacobi, n,time_jacospar, n, n,xname="Temps d'éxécution",yname="Erreur (log et linéaire)",titre="titre")


    
#AFFICHAGE DES FONCTIONS  
"""
print("***********************************************************************")
print(" ")
print("Les résultats")
print(f"Iterations (SOR Sparse): {j}, Time (sparse): {time_sor:.4f} seconds, erreur: {err[-1]}, solution (z) : {x_sor[0]}, omega (w) : {w}")
print(" ")
print(f"Iterations (Jacobi dense): {j_jacobi}, Time (dense): {time_jacobi:.4f} seconds, erreur: {err_jacobi[-1]}, solution (z) : {x_jacobi[0]}")
print(" ")
print(f"Iterations (Jacobi sparse): {j_jacospar}, Time (dense): {time_jacospar:.4f} seconds, erreur: {err_jacospar[-1]}, solution (z) : {x_jacospar[0]}")
print(" ")
print(f"Iterations (Gauss-Seidel): {j_gauss_seidel}, Time (dense): {time_gauss_seidel:.4f} seconds, erreur: {err_gauss_seidel[-1]}, solution (z) : {x_gauss_seidel[0]}")
print(" ")
print("la solution exacte est : ", x_exact[0]) 
print("***********************************************************************")
"""


def trace_sor(n):
    # Initialisation
    tps = []
    omega = []
    E = []

    for i in range(1, n): 
        for w in np.arange(0.1, 2.1, 0.01): 
            # Appel de la méthode SOR
            x_sor, j, err, time_sor,w = SOR_sparseReel(A_sor,b,x0,x_exact,w)
            omega.append(w)
            tps.append(time_sor)
            E.append(err[-1])  # Dernière erreur pour chaque w

    # Moyennes par valeur de w
    unique_w = np.arange(0.1, 2.1, 0.01)
    avg_tps = [np.mean([tps[i] for i in range(len(omega)) if omega[i] == w]) for w in unique_w]
    
    return unique_w, avg_tps,n
OM,TPS,nbr=  trace_sor(2)           
simple_plot(OM,TPS,"Temps d'éxécutions en fonction de Omega moyen sur " + str(nbr) + " itérations ")

def trace_compare(borne) : 
    N=[]
    tsor=[]
    tsparse=[]
    tjacobi=[]
    tbm=[]
    tgs=[] 
    isor=[]
    isparse=[]
    ijacobi=[]
    ibm=[]
    igs=[]
    msor=[]
    msparse=[]
    mjacobi=[]
    mbm=[]
    mgs=[]  
    w=1.5 
    for n in range(10,12)  : 
        A_sparse,A_dense,b=generate_simple_sparse_tridiagonal_matrix(n)
        A_sor = A_sparse.copy() 
        A_jsparse = A_sparse.copy()
        A_gs = A_sparse.copy()
        x0 = np.random.rand(n)
        x_exact = np.linalg.solve(A_dense, b)
    

        x_bm, jbm, errbm, time_bm,wbm=SOR_sparse(A_sor, b, w, x0, x_exact, tol=tol, max_iter=itera)

        x_sor, j, err, time_sor,w=SOR_sparseReel(A_sor,b,x0,x_exact,w)
        #msor.append(memoire_utilisee_par_fonction(SOR_sparse,A_sor,b,w,x0,x_exact))
        #SOR

        x_jacobi, j_jacobi, err_jacobi, time_jacobi = jacobi_method(A_dense, b, x0)
        #mjacobi.append(memoire_utilisee_par_fonction(jacobi_method,A_dense, b, x0))
        #jacobi dance
        x_gauss_seidel, j_gauss_seidel, err_gauss_seidel, time_gauss_seidel = gauss_seidel_sparse_with_error(A_gs, b, x0, x_exact) 
        #mgs.append(memoire_utilisee_par_fonction(gauss_seidel_sparse_with_error,A_gs, b, x0, x_exact))
        x_jacospar, j_jacospar, err_jacospar, time_jacospar = jacobi_sparse_with_error(A_jsparse, b, x0, x_exact) 
       # msparse.append(memoire_utilisee_par_fonction(jacobi_sparse_with_error,A_jsparse, b, x0, x_exact))
        test=memoire_utilisee_par_fonction(jacobi_sparse_with_error,A_jsparse, b, x0, x_exact)
        print("resulatst", test)
        #jacobi sparse

    for n in range(13,borne,3)  : 
        A_sparse,A_dense,b=generate_simple_sparse_tridiagonal_matrix(n)
        A_sor = A_sparse.copy() 
        A_jsparse = A_sparse.copy()
        A_gs = A_sparse.copy()
        A_bm = A_sparse.copy()
        x0 = np.random.rand(n)
        x_exact = np.linalg.solve(A_dense, b)

        x_bm, jbm, errbm, time_bm,wbm=SOR_sparse(A_sor, b, w, x0, x_exact, tol=tol, max_iter=itera)
       
        msor.append(cpu_utilise_par_fonction(SOR_sparseReel,A_sor,b,x0,x_exact,w))
        #SOR

        x_sor, j, err, time_sor,w=SOR_sparseReel(A_bm, b, x0, x_exact,w, tol=tol, max_iter=itera)
        mbm.append(cpu_utilise_par_fonction(SOR_sparse,A_sor,b,x0,x_exact,w))

        x_jacobi, j_jacobi, err_jacobi, time_jacobi = jacobi_method(A_dense, b, x0)
        mjacobi.append(cpu_utilise_par_fonction(jacobi_method,A_dense, b, x0))
        #jacobi dance
        x_gauss_seidel, j_gauss_seidel, err_gauss_seidel, time_gauss_seidel = gauss_seidel_sparse_with_error(A_gs, b, x0, x_exact) 
        mgs.append(cpu_utilise_par_fonction(gauss_seidel_sparse_with_error,A_gs, b, x0, x_exact))
        x_jacospar, j_jacospar, err_jacospar, time_jacospar = jacobi_sparse_with_error(A_jsparse, b, x0, x_exact) 
        msparse.append(cpu_utilise_par_fonction(jacobi_sparse_with_error,A_jsparse, b, x0, x_exact))
        #jacobi sparse
        N.append(n) 
        tsor.append(time_sor)
        tsparse.append(time_jacospar)
        tjacobi.append(time_jacobi)
        tbm.append(time_bm)
        tgs.append(time_gauss_seidel)
        isor.append(j)
        ibm.append(jbm)
        isparse.append(j_jacospar)
        ijacobi.append(j_jacobi)
        igs.append(j_gauss_seidel)

    #print(msparse) 
    plot_all( tsor, N, tjacobi, N,tsparse, N, tgs,N,tbm,N,xname="Taille de la matrice (n x n)",yname="Temps de résolution (en secondes)",titre="Temps de résolution en fonction de la taille de la matrice")
    plot_all( isor, N, ijacobi, N,isparse, N, igs,N,ibm,N,xname="Taille de la matrice (n x n)",yname="Nombre d'itérations",titre="Nombre d'itérations nécéssaire en fonction de la taille de la matrice")
    plot_all( msor, N, mjacobi, N,msparse, N, mgs,N,mbm,N,xname="Taille de la matrice (n x n)",yname="Temps CPU (en secondes)",titre="Temps CPU en fonction de la taille de la matrice")

    
def memoire_utilisee_par_fonction(fonction, *args, **kwargs):
    tracemalloc.start()
    # Prendre un instantané avant l'exécution
    start_snapshot = tracemalloc.take_snapshot()
    
    # Exécuter la fonction
    fonction(*args, **kwargs)
    
    # Prendre un instantané après l'exécution
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    # Calcul de la mémoire utilisée (en octets)
    diff = sum([stat.size_diff for stat in end_snapshot.compare_to(start_snapshot, 'lineno')])
    
    # Retourner en Mo
    return diff   # Convertir en mégaoctets (Mo)


def cpu_utilise_par_fonction(fonction, *args, **kwargs):
    """
    Mesure le temps CPU utilisé par une fonction en secondes.
    """
    debut = time.process_time()  # Temps CPU avant l'exécution
    fonction(*args, **kwargs)    # Exécution de la fonction
    fin = time.process_time()    # Temps CPU après l'exécution
    
    return fin - debut  # Temps CPU consommé

def utilisation_cpu(fonction, *args, **kwargs):
    """
    Mesure l'utilisation du CPU en pourcentage pendant l'exécution d'une fonction.
    """
    pid = os.getpid()  # Récupérer l'identifiant du processus actuel
    process = psutil.Process(pid)
    
    # Capture de l'utilisation CPU avant
    cpu_avant = process.cpu_percent(interval=None)
    
    # Exécuter la fonction
    fonction(*args, **kwargs)
    
    # Capture de l'utilisation CPU après
    cpu_apres = process.cpu_percent(interval=None)
    
    # Calculer la différence
    return cpu_apres - cpu_avant




#trace_compare(100) 

            


