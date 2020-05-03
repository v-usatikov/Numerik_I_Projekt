from matplotlib.cm import ScalarMappable

from interfaces import Solver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator


class Evaluator:
    def __init__(self, solver: Solver):
        self.solver = solver
        self._iterator = self.solver.__iter__()

    def restart(self):
        self._iterator = self.solver.__iter__()

    def v_feld_animation(self):
        self.restart()
        t, psi, vx, vy = next(self._iterator)

        x = np.arange(self.solver.h, self.solver.L, self.solver.h)
        y = np.arange(self.solver.h, self.solver.d, self.solver.h)

        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(1, 1)
        # Q = axes.quiver(X, Y, vx, vy, pivot='mid', color='black', units='inches')
        Q = axes.quiver(X, Y, vx, vy, color='black', units='inches')
        time_text = axes.text(0.01, 0.01, '', transform=axes.transAxes)

        # X = np.arange(0, self.solver.L + self.solver.h, self.solver.h)
        # Y = np.arange(0, self.solver.d + self.solver.h, self.solver.h)
        # levels = MaxNLocator(nbins=30).tick_values(psi.min(), psi.max())
        # Q2 = axes[0].contourf(X, Y, psi, levels = levels)

        # ax.set_xlim(-1, 7)
        # ax.set_ylim(-1, 7)

        def update_quiver(num, Q):
            """updates the horizontal and vertical vector components by a
            fixed increment on each frame
            """
            t, psi, vx, vy = next(self._iterator)

            Q.set_UVC(vx, vy)
            time_text.set_text('Zeit = %.1f s' % t)

            return Q, time_text

        # you need to set blit=False, or the first set of arrows never gets
        # cleared on subsequent frames
        anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, ),
                                       interval=50, blit=False)

        plt.show()

    def psi_feld_animation(self):
        self.restart()
        t, psi, vx, vy = next(self._iterator)

        X = np.arange(0, self.solver.L + self.solver.h, self.solver.h)
        Y = np.arange(0, self.solver.d + self.solver.h, self.solver.h)

        fig = plt.figure()
        ax = plt.axes(xlabel='x', ylabel='y')

        levels = MaxNLocator(nbins=50).tick_values(psi.min(), psi.max())
        cont = plt.contourf(X, Y, psi, levels = levels)
        cbar = plt.colorbar()

        time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)

        # animation function
        def animate(i, cont):
            t, psi, vx, vy = next(self._iterator)
            ax.collections = []
            # levels = MaxNLocator(nbins=50).tick_values(psi.min(), psi.max())
            # cont.mappable.set_clim(psi.min(), psi.max())
            # cbar.draw_all()
            cont = plt.contourf(X, Y, psi, levels = levels)
            time_text.set_text('Zeit = %.1f s' % t)
            return cont, time_text

        anim = animation.FuncAnimation(fig, animate, fargs=(cont, ), repeat=False)
        # anim.save('animation.mp4', writer=animation.FFMpegWriter())

        plt.show()

    def omega_feld_animation(self):
        self.restart()
        t, psi, vx, vy = next(self._iterator)

        X = np.arange(0, self.solver.L + self.solver.h, self.solver.h)
        Y = np.arange(0, self.solver.d + self.solver.h, self.solver.h)

        fig = plt.figure()
        ax = plt.axes(xlabel='x', ylabel='y')

        levels = MaxNLocator(nbins=50).tick_values(self.solver.omega.min(), self.solver.omega.max())
        cont = plt.contourf(X, Y, self.solver.omega, levels = levels)
        cbar = plt.colorbar()

        min_max_text = ax.text(0.51, 0.01, '', transform=ax.transAxes)
        time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)

        # animation function
        def animate(i, cont):
            t, psi, vx, vy = next(self._iterator)
            ax.collections = []
            levels = MaxNLocator(nbins=50).tick_values(self.solver.omega.min(), self.solver.omega.max())

            # cbar.ScalarMappable.set_clim((self.solver.omega.min(), self.solver.omega.max()))
            cont = plt.contourf(X, Y, self.solver.omega, levels = levels)
            time_text.set_text('Zeit = %.1f s' % t)
            min_max_text.set_text('Min = %.1f, Max = %.1f' % (self.solver.omega.min(), self.solver.omega.max()))
            return cont, time_text, min_max_text

        anim = animation.FuncAnimation(fig, animate, fargs=(cont, ), repeat=False)
        # anim.save('animation.mp4', writer=animation.FFMpegWriter())

        plt.show()


if __name__ == "__main__":
    from iter_solver import IterSolver, RukuTimeSolver

    solver = IterSolver(RukuTimeSolver(), ny=21, L=1, d=1, V_in=0)
    solver.set_omega0_VB()
    evaluator = Evaluator(solver)
    evaluator.psi_feld_animation()

    solver = IterSolver(RukuTimeSolver(), ny=21, L=1, d=1, V_in=0)
    solver.set_omega0_VB()
    evaluator = Evaluator(solver)
    evaluator.v_feld_animation()
    # evaluator.psi_feld_animation()
