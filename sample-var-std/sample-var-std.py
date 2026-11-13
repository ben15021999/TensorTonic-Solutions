import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x)
    variance = float(np.sum(centered ** 2 / (len(x) - 1)))
    return {
        'variance': variance,
        'standard_deviation': float(np.sqrt(variance))
    }