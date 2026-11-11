from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    _x = np.asarray(x, dtype=float)
    counts = Counter(x)
    highest_frequency = max(counts.values())
    mode = min(value for value, count in counts.items() if count == highest_frequency)
    return {
        'mean': float(np.mean(_x)),
        'median': float(np.median(_x)),
        'mode': float(mode)
    }
    