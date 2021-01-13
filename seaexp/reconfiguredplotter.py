from dataclasses import dataclass

from seaexp.plotter import Plotter


@dataclass
class ReconfiguredPlotter:
    """Helper class to allow a reconfigured plot from an already configured Plotter object"""
    plotter: Plotter
    name: str
    xlim: tuple
    ylim: tuple
    zlim: tuple
    inplace: bool
    block: bool
    color: str

    def __post_init__(self):
        self.wait = self.plotter.wait
        self.delay = self.plotter.delay

    def __lshift__(self, other):
        return self.plotter.__lshift__(
            other, self.name, self.xlim, self.ylim, self.zlim, self.inplace, self.block, self.color
        )

    def __call__(self, name=None, xlim=None, ylim=None, zlim=None, inplace=None, block=None, color=None):
        name = self.name if name is None else name
        xlim = self.xlim if xlim is None else xlim
        ylim = self.ylim if ylim is None else ylim
        zlim = self.zlim if zlim is None else zlim
        inplace = self.inplace if inplace is None else inplace
        block = self.block if block is None else block
        color = self.color if color is None else color
        return ReconfiguredPlotter(self.plotter, name, xlim, ylim, zlim, inplace, block, color)
