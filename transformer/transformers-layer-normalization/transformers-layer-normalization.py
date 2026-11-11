import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    # 1. Use axis=-1 and keepdims=True for both statistics
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # 2. Normalize with np.sqrt(variance + eps)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # 3. Broadcast gamma and beta over all leading dimensions
    output = x_normalized * gamma + beta
    
    return output