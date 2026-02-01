# Calcul de l'ensemble de Mandelbrot en python
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
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

def split_rows(height, rank, size):
    base = height // size
    rem = height % size
    start = rank * base + min(rank, rem)
    nrows = base + (1 if rank < rem else 0)
    end = start + nrows
    return start, end, nrows

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3.0 / width
scaleY = 2.25 / height

y0, y1, nrows = split_rows(height, rank, size)
local_conv = np.empty((nrows, width), dtype=np.double)

t0 = time()
for local_y, y in enumerate(range(y0, y1)):
    cy = -1.125 + scaleY * y
    for x in range(width):
        c = complex(-2.0 + scaleX * x, cy)
        local_conv[local_y, x] = mandelbrot_set.convergence(c, smooth=True)

t1 = time()

sendbuf = local_conv.ravel()
counts = comm.gather(sendbuf.size, root=0)

recvbuf = None
displs = None
if rank == 0:
    displs = np.zeros(size, dtype=int)
    displs[1:] = np.cumsum(counts[:-1])
    recvbuf = np.empty(sum(counts), dtype=np.double)

comm.Gatherv(sendbuf, (recvbuf, counts, displs, MPI.DOUBLE), root=0)

# Temps global (max)
local_time = t1 - t0
total_time = comm.reduce(local_time, op=MPI.MAX, root=0)

if rank == 0:
    print(f"Temps calcul Mandelbrot (MPI {size}) : {total_time:.6f} s")

    timg0 = time()
    full = recvbuf.reshape((height, width))
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full) * 255))
    image.save("mandelbrot_q1.png")
    timg1 = time()

    print(f"Temps constitution image : {timg1 - timg0:.6f} s")
    image.show()
