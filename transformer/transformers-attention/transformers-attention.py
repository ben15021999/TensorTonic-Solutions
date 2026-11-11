import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    # 1. Transpose only the final two dimensions of K
    # (Using .transpose(-2, -1) handles any leading batch/head dimensions)
    K_T = K.transpose(-2, -1)
    
    # Raw scores: Q multiplied by transposed K
    scores = torch.matmul(Q, K_T)
    
    # 2. Scale scores by the square root of K's final dimension (K.shape[-1])
    d_k = K.shape[-1]
    scale = d_k ** 0.5
    scaled_scores = scores / scale
    
    # 3. Apply softmax over the final score dimension
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # 4. Return the weighted sum of V without rounding
    output = torch.matmul(attention_weights, V)
    
    return output