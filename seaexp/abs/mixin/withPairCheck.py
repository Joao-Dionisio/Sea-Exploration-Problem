class withPairCheck:
    @staticmethod
    def _check_pair(x_tup, y):
        if isinstance(x_tup, tuple):
            if y:
                raise Exception(f"Cannot provide both x={x_tup} and y={y}.")
            x, y = x_tup
            return x, y
        return x_tup, y
