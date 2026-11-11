import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    # 1. Use axis=-1 and keepdims=True for both statistics
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # 2. Normalize with np.sqrt(variance + eps)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # 3. Broadcast gamma and beta over all leading dimensions
    output = x_normalized * gamma + beta
    
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    batch_size, seq_length, d_model = Q.shape
    head_width = d_model // num_heads
    
    # Linear Projections (using matrix multiplication)
    # Shapes: (batch_size, seq_length, d_model)
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)
    
    # 1. Reshape to batch, sequence, heads, head_width, then transpose the head axis forward
    # Initial reshape: (batch_size, seq_length, num_heads, head_width)
    # After transpose(0, 2, 1, 3): (batch_size, num_heads, seq_length, head_width)
    Q_heads = Q_proj.reshape(batch_size, seq_length, num_heads, head_width).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_length, num_heads, head_width).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_length, num_heads, head_width).transpose(0, 2, 1, 3)
    
    # 2. Transpose only the final two dimensions of projected K
    # Shape becomes: (batch_size, num_heads, head_width, seq_length)
    K_T = K_heads.transpose(0, 1, 3, 2)
    
    # Raw scores and scaling
    scores = np.matmul(Q_heads, K_T) / (head_width ** 0.5)
    
    # Stable Softmax over the final score dimension
    # Subtracting max over the last axis ensures numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Weighted sum of V
    # Shape: (batch_size, num_heads, seq_length, head_width)
    context_heads = np.matmul(attention_weights, V_heads)
    
    # 3. Transpose heads back before reshaping and applying W_o
    # After transpose: (batch_size, seq_length, num_heads, head_width)
    context_back = context_heads.transpose(0, 2, 1, 3)
    
    # Reshape to combine heads: (batch_size, seq_length, d_model)
    # np.ascontiguousarray handles sequential layout adjustments like PyTorch's .contiguous()
    context_flat = np.ascontiguousarray(context_back).reshape(batch_size, seq_length, d_model)
    
    # Final linear projection
    output = np.matmul(context_flat, W_o)
    
    return output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    # 1. Compute hidden = np.maximum(0, x @ W1 + b1)
    # NumPy broadcasts each bias over leading dimensions
    hidden = np.maximum(0, np.matmul(x, W1) + b1)
    
    # 2. Project hidden with W2 and add b2
    output = np.matmul(hidden, W2) + b2
    
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    # 1. Multi-Head Attention Sub-Layer
    # Computes attention context matrix from inputs
    attn_out = multi_head_attention(
        Q=x, K=x, V=x, 
        W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, 
        num_heads=num_heads
    )
    
    # Residual Connection + Post-Layer Normalization
    x = layer_norm(x + attn_out, gamma=gamma1, beta=beta1)
    
    # 2. Position-Wise Feed-Forward Sub-Layer
    # Computes non-linear feature projections
    ffn_out = feed_forward(
        x=x, W1=W1, b1=b1, W2=W2, b2=b2
    )
    
    # Residual Connection + Post-Layer Normalization
    out = layer_norm(x + ffn_out, gamma=gamma2, beta=beta2)
    
    return out
