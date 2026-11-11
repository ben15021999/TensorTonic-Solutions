import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    positions = np.arange(seq_length)[:, np.newaxis]
    # 2. Compute one inverse frequency for each even column
    # (Running over 0 to d_model with a step of 2)
    inv_freq = 1.0 / (10000 ** (np.arange(0, d_model, 2).astype(np.float32) / d_model))
    
    # Initialize the positional encoding matrix
    pe = np.zeros((seq_length, d_model))
    
    # 3. Assign sine and cosine results with 0::2 and 1::2 slices
    pe[:, 0::2] = np.sin(positions * inv_freq)
    pe[:, 1::2] = np.cos(positions * inv_freq)
    
    return pe
    