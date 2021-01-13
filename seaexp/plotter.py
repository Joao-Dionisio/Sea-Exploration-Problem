import tkinter as tk
from dataclasses import dataclass
from time import sleep
from tkinter import ttk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from seaexp import Probings


@dataclass
class Plotter:
    xlim: tuple = 0, 1
    ylim: tuple = 0, 1
    zlim: tuple = 0, 1
    name: str = None
    inplace: bool = False
    color: str = "jet"
    block: bool = False
    wframe = None
    root = None
    plots = 0

    def __post_init__(self):
        """

        Usage:
            >>> from seaexp import Seabed
            >>> points = Probings.fromgrid(20)  # zeroed grid
            >>> f = Seabed.fromgaussian(0.2, 0.2, -0.1, s=0.2, a=0.9)
            >>> plt = Plotter()
            >>> plt << f(points)
            >>> plt.delay()
            >>> g = Seabed.fromgaussian(0.8, 0.6, 0, s=0.12, a=0.7)
            >>> plt << (f + g) (points)
            >>> plt.delay()
            >>> h = Seabed.fromgaussian(0.5, 0.5, 0, s=0.07, a=0.8)
            >>> plt << (f + g + h) (points)

        Returns
        -------

        """
        plt.ion()
        if self.inplace:
            self.fig, self.ax = self.setup()

    def setup(self):
        fig = plt.figure(self.name)
        ax = fig.gca(projection='3d')
        return fig, ax

    def __lshift__(self, other, name=None, xlim=None, ylim=None, zlim=None, color="jet", block=None):
        if block is None:
            block = self.block

        fig, ax = (self.fig, self.ax) if self.inplace else self.setup()
        xmin, xmax = self.xlim if xlim is None else xlim
        ymin, ymax = self.ylim if ylim is None else ylim
        zmin, zmax = self.zlim if zlim is None else zlim
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        name = (self.name if other.name is None else other.name) or ""
        fig.canvas.set_window_title(name)
        self.plots += 1

        fig.canvas.mpl_connect('key_press_event', self.control_quit)
        # ax.set_title(other.name)

        if self.inplace and self.wframe:
            ax.collections.remove(self.wframe)
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root = tk.Tk()
        self.root.geometry("500x100+0+0")
        Application(self.root)
        self.root.mainloop()

    def control_quit(self, ev):
        if ev.key == "q" and self.root is not None:
            self.plots -= 1
            if self.plots == 0:
                self.root.quit()

    def __hash__(self):
        return id(self)

    def __call__(self, name=None, xlim=None, ylim=None, zlim=None, color="jet", block=None):
        from seaexp.customlshift import CustomLShift
        return CustomLShift(self, name, xlim, ylim, zlim, color, block)


class Application(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        master.geometry("300x60")
        master.wm_title("Main thread ended")
        b = ttk.Button(master, text="Quit", command=lambda: master.quit())
        b.pack(fill=tk.BOTH, expand=1)
        b.focus_set()
