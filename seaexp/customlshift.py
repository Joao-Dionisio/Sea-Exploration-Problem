from dataclasses import dataclass

from seaexp.plotter import Plotter


@dataclass
class CustomLShift:
    plotter: Plotter
    name: str = None
    xlim: tuple = 0, 1
    ylim: tuple = 0, 1
    zlim: tuple = 0, 1
    color: str = "jet"
    block: bool = False

    def __lshift__(self, other):
        return self.plotter.__lshift__(other, self.name, self.xlim, self.ylim, self.zlim, self.color, self.block)
