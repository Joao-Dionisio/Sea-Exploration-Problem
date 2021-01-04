from dataclasses import dataclass


@dataclass
class Trip:
    """Sequence of points visited or to visit within a budget

    Every trip start/end at the provided depot."""
    depot: tuple
    offshore_points: list

    def __post_init__(self):
        self.points = [self.depot] + self.offshore_points + [self.depot]

    def __lshift__(self, point):
        """Add a 2D point (x, y) to the trip.

        Usage:
            >>> trip = Trip((0,0), [(1.2,2.0), (3.2,2.3), (3.5,1.9)])
            >>> trip
            Trip(depot=(0, 0), offshore_points=[(1.2, 2.0), (3.2, 2.3), (3.5, 1.9)])
            >>> list(trip)
            [(0, 0), (1.2, 2.0), (3.2, 2.3), (3.5, 1.9), (0, 0)]

        Return an extended new trip including the new point, but still having the depot as ending point."""
        return Trip(self.depot, self.offshore_points + [point])

    def __iter__(self):
        return iter(self.points)
