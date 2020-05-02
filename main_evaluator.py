from interfaces import Solver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from iter_solver import IterSolver, RukuTimeSolver


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

        fig, ax = plt.subplots(1, 1)
        Q = ax.quiver(X, Y, vx, vy, pivot='mid', color='black', units='inches')
        time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)

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
        anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q,),
                                       interval=50, blit=False)

        plt.show()


if __name__ == "__main__":

    solver = IterSolver(RukuTimeSolver(), ny=21, L=1, d=1)
    evaluator = Evaluator(solver)
    evaluator.v_feld_animation()
