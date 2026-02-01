import numpy as np
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 120
assert N % size == 0

Nloc = N // size
i0 = rank * Nloc
i1 = i0 + Nloc

# Vecteur u (tout le monde l'a)
u = np.array([i + 1. for i in range(N)], dtype=np.float64)

t0 = time()

# Chaque processus calcule seulement v[i0:i1]
v_loc = np.zeros(Nloc, dtype=np.float64)

# A[i,j] = ((i + j) % N) + 1
for local_i, i in enumerate(range(i0, i1)):
    s = 0.0
    for j in range(N):
        aij = ((i + j) % N) + 1.0
        s += aij * u[j]
    v_loc[local_i] = s

# Tout le monde récupère v complet
v = np.empty(N, dtype=np.float64)
comm.Allgather(v_loc, v)

t1 = time()
local_time = t1 - t0
total_time = comm.reduce(local_time, op=MPI.MAX, root=0)

if rank == 0:
    print(f"N={N}, nbp={size}, Nloc={Nloc}")
    print(f"Temps calcul (MPI lignes) : {total_time:.6f} s")
