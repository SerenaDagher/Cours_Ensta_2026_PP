# TP2 – Parallélisation de l’ensemble de Mandelbrot  
## Partie 1 — Partition par blocs de lignes (MPI)

## Note: je travail sur un Apple Mac équipé d’un processeur Apple M4 (architecture ARM)
### Parallélisation

L'image est de taille `H × W` et on a `p` processus MPI :

- Chaque processus `rank` calcule un intervalle de lignes `[start, end[` tel que :
  - tous les pixels de ces lignes sont calculés localement,
  - chaque ligne est calculée par un seul processus,
  - la charge est répartie aussi équitablement que possible.

Chaque processus produit un tableau partiel contenant ses lignes, puis :

- Les tableaux locaux sont envoyés au processus 0 via `MPI_Gatherv`,
- Le processus 0 reconstruit l’image complète,
- L’image est sauvegardée et affichée.

---

### Résultats expérimentaux

Les temps mesurés pour le calcul de Mandelbrot sont :

| Nombre de processus (p) | Temps T(p) (s) | Speedup S(p) = T(1) / T(p) |
|--------------------------|----------------|-----------------------------|
| 1                        | 0.8075         | 1.00                        |
| 2                        | 0.4535         | 1.78                        |
| 4                        | 0.2425         | 3.33                        |
| 8                        | 0.2174         | 3.71                        |

---

### Interprétation

On observe que le speedup augmente avec le nombre de processus, ce qui montre que le calcul de l’ensemble de Mandelbrot est bien parallélisable. Le gain est quasi linéaire entre 1, 2 et 4 processus.

En revanche, au-delà de 4 processus, le speedup progresse beaucoup plus lentement (3.33 → 3.71 entre 4 et 8 processus). Cette saturation s’explique par plusieurs facteurs :

- le coût des communications MPI lors du rassemblement des résultats,
- la partie séquentielle du programme (constitution et sauvegarde de l’image),
- le déséquilibre de charge : certaines lignes de l’image nécessitent plus d’itérations que d’autres,
- l’overhead propre au parallélisme (gestion MPI, synchronisations).

Ces phénomènes limitent le speedup maximal atteignable, conformément à la **loi d’Amdahl**.  
Ainsi, même si le calcul est majoritairement parallèle, la présence de parties séquentielles et de communications empêche d’atteindre un speedup parfaitement linéaire.

### Résultats – Répartition statique cyclique (Q2)

| Nombre de processus (p) | Temps T(p) (s) | Speedup S(p) = T(1)/T(p) |
|--------------------------|----------------|---------------------------|
| 1                        | 0.8292         | 1.00                      |
| 2                        | 0.4247         | 1.95                      |
| 4                        | 0.2257         | 3.67                      |
| 8                        | 0.1596         | 5.19                      |

### Interprétation

La répartition cyclique des lignes améliore l’équilibrage de charge entre les processus.  
Contrairement à la découpe par blocs (Q1), chaque processus reçoit un mélange de lignes “faciles” et “difficiles”, ce qui réduit le temps d’attente du processus le plus lent.

On observe un meilleur speedup, notamment pour 8 processus :
- Q1 (blocs) : S(8) ≈ 3.71  
- Q2 (cyclique) : S(8) ≈ 5.19  

Cette amélioration montre que le déséquilibre de charge était un facteur limitant important dans la version par blocs.

Cependant, cette stratégie présente aussi des inconvénients :
- la reconstruction de l’image est plus complexe (lignes non contiguës),
- un léger surcoût est introduit lors du rassemblement et du réordonnancement,
- la stratégie est adaptée à cette zone de l’image ; pour un autre zoom ou une autre région, elle pourrait ne plus être optimale.

Ainsi, la répartition cyclique constitue une meilleure stratégie statique, mais elle reste dépendante de la structure du problème.


## Partie 3 — Stratégie maître-esclave (Q3)

### Résultats

| Nombre de processus (p) | Temps T(p) (s) | Speedup S(p) = T(1)/T(p) |
|--------------------------|----------------|---------------------------|
| 1                        | 0.7557         | 1.00                      |
| 2                        | 0.7677         | 0.98                      |
| 4                        | 0.2839         | 2.66                      |
| 8                        | 0.1686         | 4.48                      |

### Comparaison avec Q1 et Q2

- Q1 (blocs) :  
  - S(8) ≈ 3.71  
- Q2 (cyclique) :  
  - S(8) ≈ 5.19  
- Q3 (maître-esclave) :  
  - S(8) ≈ 4.48  

On observe que :

- Pour 2 processus, la version maître-esclave est légèrement plus lente que le séquentiel.  
  Cela s’explique par le coût des communications fréquentes (envoi d’une ligne à la fois).
- À partir de 4 et 8 processus, le speedup devient significatif.
- La stratégie maître-esclave équilibre parfaitement la charge, mais introduit un surcoût de communication important.

Par rapport aux autres approches :
- Elle est nettement meilleure que la version par blocs (Q1) en termes d’équilibrage.
- Elle reste cependant moins performante que la répartition cyclique (Q2) dans ce cas précis, car l’envoi ligne par ligne génère beaucoup de messages et le processus maître peut devenir un goulot d’étranglement.

### Conclusion

La stratégie maître-esclave garantit un excellent équilibrage dynamique, indépendamment de la complexité des lignes à calculer.  
Cependant, le coût des communications et la centralisation


## 2.b – Produit matrice–vecteur par lignes

On découpe la matrice par blocs de lignes entre `nbp` tâches.  
Comme \(N\) est divisible par `nbp`, chaque tâche traite :

\[
N_{loc}=\frac{N}{nbp}
\]

Avec \(N=120\) :

| nbp | \(N_{loc}\) |
|-----|-------------|
| 1   | 120         |
| 2   | 60          |
| 4   | 30          |
| 8   | 15          |

Chaque processus calcule uniquement sa partie du vecteur résultat :
\[
v[i_0:i_1]
\]
puis on reconstruit le vecteur complet sur **toutes** les tâches via `MPI_Allgather`.

---

### Résultats expérimentaux

Temps mesurés :

| nbp | Temps T(p) (s) |
|-----|----------------|
| 1   | 0.006377       |
| 2   | 0.002557       |
| 4   | 0.001160       |
| 8   | 0.000912       |

Speedup (référence MPI avec 1 tâche) :

\[
S(p)=\frac{T(1)}{T(p)}
\]

| nbp | Speedup S(p) |
|-----|--------------|
| 1   | 1.00         |
| 2   | 2.49         |
| 4   | 5.50         |
| 8   | 6.99         |

---

### Interprétation

Le découpage par lignes donne un très bon speedup car chaque tâche calcule directement une portion distincte du résultat \(v\). La communication finale (`MPI_Allgather`) consiste uniquement à rassembler les segments de \(v\), ce qui reste léger.

On observe des speedups supérieurs au nombre de tâches pour 2 et 4 processus (super-linéarité), ce qui peut s’expliquer par :
- de meilleurs effets de cache,
- des temps très faibles (ordre de la milliseconde), donc une mesure sensible au bruit et à l’overhead.

À 8 tâches, le speedup reste élevé (≈ 6.99) mais n’est plus linéaire, car l’overhead MPI et les communications commencent à représenter une part significative du temps total.

Globalement, la stratégie par lignes est efficace et souvent plus naturelle que la stratégie par colonnes, car chaque processus produit directement une partie finale du vecteur résultat.
