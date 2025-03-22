try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False

class RotaryPositionalEncoding():
    """Implements the RoPE (Rotary Position Embeddings) encoding.
       https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d_model, max_len=5000, data_type=np.float32):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.data_type = data_type
        self.build_rope_matrices()

    def build_rope_matrices(self):
        # Generate position indices
        positions = np.arange(self.max_len)[:, np.newaxis]  # [max_len, 1]
        
        # Generate dimension indices
        dims = np.arange(0, self.d_model, 2)  # [d_model/2]
        
        # Compute theta
        freqs = 1.0 / (10000 ** (dims / self.d_model))  # [d_model/2]
        
        # Compute angles
        angles = positions * freqs  # [max_len, d_model/2]
        
        # Convert to complex numbers for rotation
        self.cos = np.cos(angles).astype(self.data_type)
        self.sin = np.sin(angles).astype(self.data_type)

    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = np.stack([-x2, x1], axis=-1).reshape(x.shape)
        return rotated

    def apply_rotary_pos_emb(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        
        cos = self.cos[:seq_len]  # [seq_len, d_model/2]
        sin = self.sin[:seq_len]  # [seq_len, d_model/2]
        
        # Reshape for proper broadcasting
        x_reshaped = x.reshape(batch_size, seq_len, -1, 2)  # [batch_size, seq_len, d_model/2, 2]
        
        # Expand dimensions for broadcasting
        cos = cos[:, :, np.newaxis]  # [seq_len, d_model/2, 1]
        sin = sin[:, :, np.newaxis]  # [seq_len, d_model/2, 1]
        
        # Apply rotary embeddings
        x_rot = np.empty_like(x_reshaped)
        x_rot[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin  # Real part
        x_rot[..., 1] = x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos  # Imaginary part
        
        return x_rot.reshape(x.shape)

    def forward(self, x):
        """Apply rotary position encoding to input tensor.
           x: input tensor of shape [batch_size, seq_len, d_model]
        """
        return self.apply_rotary_pos_emb(x)

    def backward(self, error):
        """Backward pass is simply the error gradient since RoPE is a linear operation."""
        return error
