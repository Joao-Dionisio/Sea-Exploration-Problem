import numpy


class withPairCheck:
    @staticmethod
    def _check_pair(x, y, center):
        if center is None:
            return numpy.array([x, y])
        if x not in [0, center[0]] or y not in [0, center[1]]:
            raise Exception(f"Cannot provide both x={x} (or y{y}) and a center={center}.")
        return numpy.array(center)
