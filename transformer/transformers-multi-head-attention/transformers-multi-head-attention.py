import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
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