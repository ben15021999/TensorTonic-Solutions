import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    hidden = np.maximum(0, np.matmul(x, W1) + b1)
    
    # 2. Project hidden with W2 and add b2
    output = np.matmul(hidden, W2) + b2
    
    return output