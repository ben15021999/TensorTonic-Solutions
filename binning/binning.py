import math

def binning(values: list, num_bins: int) -> list:
    """
    Returns the equal-width bin index of every value.
    """
    max_val = max(values)
    min_val = min(values)
    bin_width = (max_val - min_val) / num_bins
    if min_val == max_val:
        return [0] * len(values)
    result = []
    for value in values:
        bin_idx = int((value - min_val) / bin_width)
        bin_idx = min(bin_idx, num_bins - 1)
        result.append(bin_idx)
    return result