import numpy as np
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 120
assert N % size == 0

Nloc = N // size
j0 = rank * Nloc
j1 = j0 + Nloc

# Vecteur u (tout le monde doit l'avoir)
u = np.array([i + 1. for i in range(N)], dtype=np.float64)

# Chaque processus calcule seulement les colonnes j0..j1-1
# On ne construit pas toute A, seulement ce qu'il faut.
t0 = time()

v_part = np.zeros(N, dtype=np.float64)

# A[i,j] = ((i + j) % N) + 1
for i in range(N):
    s = 0.0
    for j in range(j0, j1):
        aij = ((i + j) % N) + 1.0
        s += aij * u[j]
    v_part[i] = s

# Somme des contributions -> tout le monde obtient v
v = np.zeros(N, dtype=np.float64)
comm.Allreduce(v_part, v, op=MPI.SUM)

t1 = time()
local_time = t1 - t0
total_time = comm.reduce(local_time, op=MPI.MAX, root=0)

if rank == 0:
    print(f"N={N}, nbp={size}, Nloc={Nloc}")
    print(f"Temps calcul (MPI colonnes) : {total_time:.6f} s")
