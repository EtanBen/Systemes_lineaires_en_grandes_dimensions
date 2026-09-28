# Systèmes linéaires en grandes dimensions

## Présentation

Ce projet porte sur la résolution numérique de systèmes linéaires de grande dimension de la forme

\[
Ax = b
\]

L'objectif est d'étudier et de comparer plusieurs méthodes itératives de résolution, en particulier lorsque la matrice \(A\) est creuse.

Les méthodes sont implémentées en Python et comparées selon plusieurs critères :

- le temps de résolution ;
- le nombre d'itérations nécessaires à la convergence ;
- l'erreur obtenue ;
- le temps CPU ;
- l'influence du paramètre de relaxation pour la méthode SOR.

Le projet permet ainsi d'observer expérimentalement les différences entre les différentes méthodes et leur comportement lorsque la taille du système augmente.

---

## Méthodes étudiées

Plusieurs méthodes de résolution sont implémentées dans le fichier `tous_les_algos.py`.

### Méthode de Jacobi

La méthode de Jacobi calcule successivement une approximation de la solution en utilisant uniquement les valeurs obtenues à l'itération précédente.

Deux versions sont étudiées :

- **Jacobi dense** : utilisation explicite d'une matrice dense ;
- **Jacobi creux** : utilisation d'une matrice creuse afin de réduire les calculs inutiles.

### Méthode de Gauss-Seidel

La méthode de Gauss-Seidel est une amélioration de Jacobi qui utilise immédiatement les nouvelles valeurs calculées au cours d'une même itération.

Dans le programme, elle est obtenue comme un cas particulier de la méthode SOR avec :

\[
\omega = 1
\]

### Méthode SOR

La méthode SOR (*Successive Over-Relaxation*) introduit un paramètre de relaxation \(\omega\) afin d'accélérer la convergence.

Le paramètre vérifie généralement :

\[
0 < \omega < 2
\]

Dans le programme, la valeur utilisée par défaut est :

\[
\omega = 1.2
\]

Une expérience permet également d'étudier l'influence de \(\omega\) sur le temps d'exécution.

### Méthode de Benchiha-Muratore

Une variante matricielle de SOR est également implémentée. Elle repose sur une décomposition de la matrice :

\[
A = D - L - U
\]

et sur le calcul d'une étape de type Gauss-Seidel suivie d'une relaxation.

---

## Structure du projet

Le dépôt contient actuellement les fichiers suivants :

```text
Systemes_lineaires_en_grandes_dimensions/
│
├── tous_les_algos.py
└── README.md
