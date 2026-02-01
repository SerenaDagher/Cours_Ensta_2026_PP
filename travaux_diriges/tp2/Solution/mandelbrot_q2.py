# mandelbrot_q2.py — Répartition statique cyclique (Q2)

import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (-0.75 < c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1 - ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations

        z = 0
        for it in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return it + 1 - log(log(abs(z))) / log(2)
                return it
        return self.max_iterations


# Paramètres
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3.0 / width
scaleY = 2.25 / height

# Répartition cyclique des lignes
ys = list(range(rank, height, size))
local_conv = np.empty((len(ys), width), dtype=np.double)

t0 = time()
for idx, y in enumerate(ys):
    cy = -1.125 + scaleY * y
    for x in range(width):
        c = complex(-2.0 + scaleX * x, cy)
        local_conv[idx, x] = mandelbrot_set.convergence(c, smooth=True)
t1 = time()

# Gather
sendbuf = local_conv.ravel()
counts = comm.gather(sendbuf.size, root=0)

recvbuf = None
displs = None
if rank == 0:
    displs = np.zeros(size, dtype=int)
    displs[1:] = np.cumsum(counts[:-1])
    recvbuf = np.empty(sum(counts), dtype=np.double)

comm.Gatherv(sendbuf, (recvbuf, counts, displs, MPI.DOUBLE), root=0)

# Temps global
local_time = t1 - t0
total_time = comm.reduce(local_time, op=MPI.MAX, root=0)

# Reconstruction + Image (rank 0 seulement)
if rank == 0:
    print(f"Temps calcul Mandelbrot (Q2, MPI {size}) : {total_time:.6f} s")

    timg0 = time()
    full = np.empty((height, width), dtype=np.double)

    offset = 0
    for r in range(size):
        ys_r = list(range(r, height, size))
        nrows_r = len(ys_r)
        block = recvbuf[offset : offset + nrows_r * width].reshape((nrows_r, width))
        for idx, y in enumerate(ys_r):
            full[y, :] = block[idx, :]
        offset += nrows_r * width

    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full) * 255))
    image.save("mandelbrot_q2.png")

    timg1 = time()
    print(f"Temps constitution image : {timg1 - timg0:.6f} s")
    image.show()
