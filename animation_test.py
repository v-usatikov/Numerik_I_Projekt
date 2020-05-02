import numpy as np
import numpy.linalg as lg
import sympy as sp
from scipy.sparse import spdiags
from scipy import sparse
from scipy.sparse import linalg
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Test:

    def __init__(self: object, x, y, nx, ny, c_mp):
        self.x = x
        self.y = y
        self.nx = nx
        self.ny = ny
        self.gitter_x, self.gitter_y = np.meshgrid(np.linspace(0, self.x, self.nx), np.linspace(0, self.y, self.ny))
        self.c = np.zeros((self.nx, self.ny))  # Colormapping Initialwerte für Scatterplot
        self.c_mp = c_mp

        """
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.scat = self.ax1.scatter(x=np.hstack(self.X_mesh),  # Vektor
                                     y=np.hstack(self.Y_mesh),  # Vektor
                                     c=np.hstack(self.c),  # Vektor
                                     s=5,  # Skalar oder Vektor
                                     cmap='gist_rainbow',
                                     vmin=0,  # Helligkeitsnormalisierung min
                                     vmax=11)  # Helligkeitsnormalisierung max
        self.fig.colorbar(self.scat)
        self.quiver_sol = self.ax2.quiver(self.X_mesh, self.Y_mesh, self.X_mesh, self.Y_mesh)"""


    def plot_scatter(self, x, y, c, titel: str):
        fig, ax = plt.subplots()
        scat = ax.scatter(x, y, c=c)
        fig.colorbar(scat)
        fig.suptitle(titel)

    def plot_quiver(self, x, y, u, v, titel: str):
        fig, ax = plt.subplots()
        quiver = ax.quiver(x, y, u, v)
        fig.colorbar(quiver)
        fig.suptitle(titel)

    def plot_2dcont(self, x, y, c, titel: str):
        fig, ax = plt.subplots()
        contour = ax.contourf(x, y, c)
        fig.colorbar(contour)
        fig.suptitle(titel)

    def fig_init(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)

        self.scat = self.ax1.scatter(x=np.hstack(self.gitter_x),  # Vektor
                                     y=np.hstack(self.gitter_y),  # Vektor
                                     c=np.hstack(self.c),  # Vektor
                                     s=5,  # Skalar oder Vektor
                                     cmap='gist_rainbow',
                                     vmin=0,  # Helligkeitsnormalisierung min
                                     vmax=11)  # Helligkeitsnormalisierung max
        self.fig.colorbar(self.scat)
        self.quiver_sol = self.ax2.quiver(self.gitter_x, self.gitter_y, self.gitter_x, self.gitter_y)
        self.fig.colorbar(self.quiver_sol)
        return self.fig, self.scat, self.quiver_sol

    def animation_init(self):

        self.ax1.set_xlabel(0)
        self.ax2.set_xlabel(0 + 1)
        # Scatter-Plot
        b = self.c * self.c_mp * 0
        u = self.gitter_x + (1* 0)
        v = self.gitter_y + (1* 0)
        self.quiver_sol.set_UVC(u, v)
        #self.scat.set_offsets(np.hstack(self.c))
        print("init")
        self.scat = self.ax1.scatter(x=np.hstack(self.gitter_x),  # Vektor
                                     y=np.hstack(self.gitter_y),  # Vektor
                                     c=np.hstack(self.c),  # Vektor
                                     s=5,  # Skalar oder Vektor
                                     cmap='gist_rainbow',
                                     vmin=0,  # Helligkeitsnormalisierung min
                                     vmax=11)  # Helligkeitsnormalisierung max
        return self.scat, self.quiver_sol,

    def animation_update(self, frame):
        self.ax1.set_xlabel(frame)
        self.ax2.set_xlabel(frame + 1)
        # Scatter-Plot
        b = self.c * self.c_mp * frame
        u = self.gitter_x + (1 * frame)
        v = self.gitter_y + (1 * frame)
        self.quiver_sol.set_UVC(u, v)
        self.scat.set_offsets(np.hstack(self.c))
        print(frame)
        return self.scat, self.quiver_sol,

    def return_anim(self):
        ani = FuncAnimation(self.fig, self.animation_update, frames=9, init_func=self.animation_init, repeat=True,
                            interval=1, blit=True)
        return ani
"""
    def animation_sudo(self):
        global ani
        ani = self.return_anim()
"""

test1 = Test(10, 10, 10, 10, 1)
test1.fig_init()
test2 = Test(10, 10, 10, 10, 2)
test2.fig_init()

#test3 = Test(10, 10, 10, 10, 3)

anim = FuncAnimation(test1.fig, test1.animation_update, frames=9, init_func=test1.animation_init, repeat=True, interval=500, blit=True)
anim2 = FuncAnimation(test2.fig, test2.animation_update, frames=9, init_func=test2.animation_init, repeat=True, interval=500, blit=True)
plt.show()