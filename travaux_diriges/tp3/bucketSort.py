#!/usr/bin/env python3

"""
Les etapes que j'ais fait pour ce triage :
1- Le processus rank 0 doit generer N nombres aleatoire dans une liste
2- Le processus distribut des chunk de la liste au processus
3- Chaque processus fait sont propre triage en parallele
4- On choisi de chaque chunk des nombres pivots d'une facon bien repartie sur la liste.
5- On rejoint tous les pivots qu'on a choisi dans une meme liste et on divise en des intervales (le processus 0 doit faire le calcul).
6- On donne a chaque processus un intervale
7- Les processus doivent maintenat echanger leur nombres pour les mettre dans l'interval correspondant.
8- On rejoint tous les intervales sur le processus 0
""" 

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 50
chunkSize = N // size
reste = N % size

local_n = chunkSize + 1 if rank < reste else chunkSize
local = np.empty(local_n, dtype=np.int64)

if rank == 0:
    nombres = np.random.randint(0, 1000, size=N, dtype=np.int64)

    counts = np.array([chunkSize + 1 if i < reste else chunkSize for i in range(size)], dtype=np.int64)
    startIndex = np.zeros(size, dtype=np.int64)
    startIndex[1:] = np.cumsum(counts[:-1])
else:
    nombres = None
    counts = None
    startIndex = None

comm.Scatterv([nombres, counts, startIndex, MPI.INT64_T], local, root=0)

local.sort()
m = len(local)

pivots = []
for j in range(size + 1):
    idx = (j * (m - 1)) // size
    pivots.append(local[idx])

pivots = np.array(pivots, dtype=np.int64)

if rank == 0:
    allpivots = np.empty(size * (size + 1), dtype=np.int64)
else:
    allpivots = None

comm.Gather(pivots,allpivots,root = 0)

if rank == 0:
    intervales = []
    allpivots.sort()
    l = len(allpivots)
    for i in range(size + 1) :
        idx = ( i * (l - 1)) // size 
        intervales.append(allpivots[idx])

    intervales = np.array(intervales, dtype = np.int64)
else:
    intervales = np.empty(size + 1, dtype=np.int64)

comm.Bcast(intervales, root=0)

buckets = [[] for i in range(size)]

for x in local:
    for i in range(size):
        if intervales[i] <= x < intervales[i+1]:
            buckets[i].append(x)
            break

sendcounts = np.array([len(b) for b in buckets], dtype=np.int64)

sendbuf = []
for b in buckets:
    for x in b:
        sendbuf.append(x)

sendbuf = np.array(sendbuf, dtype=np.int64)

recvcounts = np.empty(size, dtype=np.int64)
comm.Alltoall(sendcounts, recvcounts)

sdispls = np.zeros(size, dtype=np.int64)
rdispls = np.zeros(size, dtype=np.int64)

for i in range(1, size):
    sdispls[i] = sdispls[i-1] + sendcounts[i-1]
    rdispls[i] = rdispls[i-1] + recvcounts[i-1]

recvbuf = np.empty(int(np.sum(recvcounts)), dtype=np.int64)

comm.Alltoallv(
    [sendbuf, sendcounts, sdispls, MPI.INT64_T],
    [recvbuf, recvcounts, rdispls, MPI.INT64_T]
)

recvbuf.sort()

final_n = np.array(recvbuf.size, dtype=np.int64)

if rank == 0:
    final_counts = np.empty(size, dtype=np.int64)
else:
    final_counts = None

comm.Gather(final_n, final_counts, root=0)

if rank == 0:
    final_displs = np.zeros(size, dtype=np.int64)
    for i in range(1, size):
        final_displs[i] = final_displs[i-1] + final_counts[i-1]
    result = np.empty(int(np.sum(final_counts)), dtype=np.int64)
else:
    final_displs = None
    result = None

comm.Gatherv(recvbuf, [result, final_counts, final_displs, MPI.INT64_T], root=0)

if rank == 0:
    print("Result: ", result)
