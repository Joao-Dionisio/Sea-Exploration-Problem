import tkinter as tk
from dataclasses import dataclass
from time import sleep
from tkinter import ttk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from seaexp import Probings


@dataclass
class Plotter:
    """

    Usage:
        >>> from seaexp import Seabed
        >>> points = Probings.fromgrid(20)  # zeroed grid
        >>>
        >>> # Standalone plot: f.
        >>> f = Seabed.fromgaussian(0.2, 0.2, -0.1, s=0.2, a=0.9)
        >>> Plotter() << f(points)
        >>>
        >>> # Context of plots: f+g, f+g+h.
        >>> with Plotter() as plt:
        ...     g = Seabed.fromgaussian(0.8, 0.6, 0, s=0.12, a=0.7)
        ...     plt << (f + g)(points)
        ...     h = Seabed.fromgaussian(0.5, 0.5, 0, s=0.07, a=0.8)
        ...     plt << (f + g + h)(points)

    Parameters
    ----------
    other
    name
    xlim
    ylim
    zlim
    inplace
    block
        When used as a context manager ('with ... as:') default value is False
        When used as a standalone plotter ('Plotter()') default value is True
    color

    Returns
    -------

    """
    name: str = None
    xlim: tuple = 0, 1
    ylim: tuple = 0, 1
    zlim: tuple = 0, 1
    inplace: bool = False
    block: bool = None
    color: str = "jet"
    wframe = None
    root = None
    plots = 0

    def __post_init__(self):
        plt.ion()
        if self.inplace:
            self.fig, self.ax = self._setup()

    def _setup(self):
        fig = plt.figure(self.name)
        ax = fig.gca(projection='3d')
        return fig, ax

    def __lshift__(self, other, name=None, xlim=None, ylim=None, zlim=None, inplace=None, block=None, color=None):
        """        Create a new plot from a Probings object. See __init__ doc for details.        """
        print(block)
        #  Reconfigure, if the Plotter object is used as callable.
        name = other.name or name or self.name or ""
        xmin, xmax = self.xlim if xlim is None else xlim
        ymin, ymax = self.ylim if ylim is None else ylim
        zmin, zmax = self.zlim if zlim is None else zlim
        inplace = inplace or self.inplace
        block = block if block is not None else (self.block is None or self.block)
        color = color or self.color

        # Setup/update window.
        fig, ax = (self.fig, self.ax) if inplace else self._setup()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        fig.canvas.set_window_title(name)
        self.plots += 1
        fig.canvas.mpl_connect('key_press_event', self._control_quit)  # ax.set_title(other.name)
        if inplace and self.wframe:
            ax.collections.remove(self.wframe)  # For animation.

        # Plot.
        self.wframe = ax.plot_trisurf(*other.x_y_z, cmap=color)
        if block:
            plt.show(block=True)
        plt.pause(0.001)

    @staticmethod
    def wait():
        root = tk.Tk()
        root.geometry("500x100+0+0")
        Application(root)
        root.mainloop()

    @staticmethod
    def delay(secs=1):
        sleep(secs)

    def __enter__(self):
        return self(block=self.block if self.block is not None else False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root = tk.Tk()
        self.root.geometry("500x100+0+0")
        Application(self.root)
        self.root.mainloop()

    def _control_quit(self, ev):
        if ev.key == "q" and self.root is not None:
            self.plots -= 1
            if self.plots == 0:
                self.root.quit()

    def __hash__(self):
        return id(self)

    def __call__(self, name=None, xlim=None, ylim=None, zlim=None, inplace=None, block=None, color=None):
        from seaexp.reconfiguredplotter import ReconfiguredPlotter
        return ReconfiguredPlotter(self, name, xlim, ylim, zlim, inplace, block, color)


class Application(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        master.geometry("300x60")
        master.wm_title("Main thread ended")
        b = ttk.Button(master, text="Quit", command=lambda: master.quit())
        b.pack(fill=tk.BOTH, expand=1)
        b.focus_set()
