import numpy as np

def logical_and(size=1000):
    xs = np.random.random_integers(0, 1, size=(size, 2))
    ys = np.zeros(size, dtype=np.int)
    ys[ys == 0] = -1
    ys[(xs[:, 0] == 1) & (xs[:, 1] == 1)] = 1
    return xs, ys
