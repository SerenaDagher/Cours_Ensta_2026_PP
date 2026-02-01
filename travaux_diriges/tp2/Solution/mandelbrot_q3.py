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

# Tags MPI
TAG_REQ  = 1  # worker -> master : demande travail
TAG_JOB  = 2  # master -> worker : envoi d'une ligne y
TAG_RES  = 3  # worker -> master : retour (y + data)
TAG_STOP = 4  # master -> worker : stop

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

def compute_line(y, width, scaleX, scaleY, mandelbrot_set):
    cy = -1.125 + scaleY * y
    line = np.empty(width, dtype=np.double)
    for x in range(width):
        c = complex(-2.0 + scaleX * x, cy)
        line[x] = mandelbrot_set.convergence(c, smooth=True)
    return line

def main():
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024
    scaleX = 3.0 / width
    scaleY = 2.25 / height

    t0 = time()

    if size == 1:
        # fallback séquentiel
        full = np.empty((height, width), dtype=np.double)
        for y in range(height):
            full[y, :] = compute_line(y, width, scaleX, scaleY, mandelbrot_set)
        t1 = time()
        print(f"Temps calcul Mandelbrot (Q3, MPI 1) : {t1 - t0:.6f} s")
        img = Image.fromarray(np.uint8(matplotlib.cm.plasma(full) * 255))
        img.save("mandelbrot_q3.png")
        img.show()
        return

    if rank == 0:
        # MASTER
        full = np.empty((height, width), dtype=np.double)

        next_y = 0
        received = 0

        # Donner 1 job initial à chaque worker
        for worker in range(1, size):
            if next_y < height:
                comm.send(next_y, dest=worker, tag=TAG_JOB)
                next_y += 1
            else:
                comm.send(None, dest=worker, tag=TAG_STOP)

        status = MPI.Status()
        while received < height:
            # recevoir résultat d’un worker
            y, line = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RES, status=status)
            full[y, :] = line
            received += 1
            worker = status.Get_source()

            # envoyer un nouveau job ou STOP
            if next_y < height:
                comm.send(next_y, dest=worker, tag=TAG_JOB)
                next_y += 1
            else:
                comm.send(None, dest=worker, tag=TAG_STOP)

        t1 = time()
        calc_time = t1 - t0
        print(f"Temps calcul Mandelbrot (Q3, MPI {size}) : {calc_time:.6f} s")

        # Image
        timg0 = time()
        img = Image.fromarray(np.uint8(matplotlib.cm.plasma(full) * 255))
        img.save("mandelbrot_q3.png")
        timg1 = time()
        print(f"Temps constitution image : {timg1 - timg0:.6f} s")
        img.show()

    else:
        # WORKER
        while True:
            y = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())

            if y is None:
                break
            line = compute_line(y, width, scaleX, scaleY, mandelbrot_set)
            comm.send((y, line), dest=0, tag=TAG_RES)

if __name__ == "__main__":
    main()
