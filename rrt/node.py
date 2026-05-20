import numpy as np

class Node:
    def __init__(self, coords):
        self.coords = np.array(coords, dtype=float)
        self.parent = None
        self.cost = 0.0