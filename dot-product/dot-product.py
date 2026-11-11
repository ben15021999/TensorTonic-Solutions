import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    _x = np.asarray(x, dtype=float)
    _y = np.asarray(y, dtype=float)
    return float(np.dot(_x, _y))