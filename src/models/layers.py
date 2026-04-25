# src/models/layers.py
import torch
import torch.nn as nn
import numpy as np
import math

class SineLayer(nn.Module):
    """
    SIREN Layer: Implicit Neural Representations with Periodic Activation Functions
    y = sin(w0 * (Wx + b))
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # 첫 레이어는 -1/n ~ 1/n
                bound = 1 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # 이후 레이어는 sqrt(6/n) / omega_0
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SirenDecoder(nn.Module):
    """
    Standard MLP Decoder using Sine Layers.
    Used to decode the Latent Vector into Scalar Potential.
    """
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, omega_0=30.0):
        super().__init__()
        layers = []
        # First layer
        layers.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0))
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
        
        self.net = nn.Sequential(*layers)
        # Final linear layer (To allow arbitrary range output)
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        
        # Final layer initialization (Near Zero)
        with torch.no_grad():
            self.final_layer.weight.uniform_(-np.sqrt(6/hidden_dim)/omega_0, np.sqrt(6/hidden_dim)/omega_0)

    def forward(self, x):
        x = self.net(x)
        return self.final_layer(x)

class GeometryAwareAttention(nn.Module):
    """
    Vectorized Cross-Attention Mechanism.
    
    Query: Spatial Coordinates (B, N_query, 3+Enc)
    Key/Value: Cuboid Features (B, N_cuboid, Feat_Dim)
    
    Output: Context Vector for each query point.
    """
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Projections
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        
        # Multihead Attention (batch_first=True is crucial for efficiency)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                          dropout=dropout, batch_first=True)
        
        # Output processing
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_pos, cuboid_feats, key_padding_mask=None):
        """
        Args:
            query_pos: (B, Nq, Q_dim) - 좌표 정보
            cuboid_feats: (B, Nc, K_dim) - 큐보이드 특징
            key_padding_mask: (B, Nc) - Padding된 큐보이드 무시용 (True=Ignore)
        
        Returns:
            context: (B, Nq, Hidden_Dim)
        """
        # 1. Linear Projection
        Q = self.q_proj(query_pos) # (B, Nq, H)
        K = self.k_proj(cuboid_feats) # (B, Nc, H)
        V = self.v_proj(cuboid_feats) # (B, Nc, H)
        # 2. Attention
        # attn_output: (B, Nq, H)
        # weights: (B, Nq, Nc)
        attn_output, _ = self.attn(query=Q, key=K, value=V, key_padding_mask=key_padding_mask)
        # 3. Residual & Norm
        output = self.norm(Q + self.dropout(attn_output))
        
        return output

class PositionalEncoding3D(nn.Module):
    """
    Fourier Feature Mapping for 3D Coordinates.
    Essential for learning high-frequency functions (sharp boundaries).
    """
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # 2^0, 2^1, ..., 2^(N-1)
        self.register_buffer('freq_bands', 2.0 ** torch.arange(num_freqs))

    def forward(self, x):
        # x: (..., 3)
        # return: (..., 3 + 3*2*num_freqs) if include_input
        embed = [x] if self.include_input else []
        for freq in self.freq_bands:
            embed.append(torch.sin(x * freq * math.pi))
            embed.append(torch.cos(x * freq * math.pi))
        return torch.cat(embed, dim=-1)

class ConditionedSirenDecoder(nn.Module):
    """
    [NEW] Analytical Jacobian이 내장된 Conditioned SIREN Decoder.
    x (좌표)와 z (Context)를 입력받아 phi와 grad_phi(전기장)를 동시에 O(1)으로 산출합니다.
    """
    def __init__(self, coord_dim=3, context_dim=128, hidden_dim=128, out_dim=1, num_layers=4, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.num_layers = num_layers
        
        # Layer 0 (Input Layer: x and z combined)
        self.linear_x0 = nn.Linear(coord_dim, hidden_dim, bias=False)
        self.linear_z0 = nn.Linear(context_dim, hidden_dim, bias=True)
        
        # Hidden Layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output Layer
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        self._init_weights(coord_dim, hidden_dim)

    def _init_weights(self, coord_dim, hidden_dim):
        with torch.no_grad():
            # First layer
            bound = 1 / coord_dim
            self.linear_x0.weight.uniform_(-bound, bound)
            self.linear_z0.weight.uniform_(-bound, bound)
            # Hidden layers
            for layer in self.hidden_layers:
                layer.weight.uniform_(-np.sqrt(6/hidden_dim)/self.omega_0, np.sqrt(6/hidden_dim)/self.omega_0)
            # Final layer
            self.final_layer.weight.uniform_(-np.sqrt(6/hidden_dim)/self.omega_0, np.sqrt(6/hidden_dim)/self.omega_0)

    def forward(self, x, z, compute_grad=True):
        """
        Args:
            x: (B, K, 3) Coordinates
            z: (B, K, D) Context from Attention
            compute_grad: Boolean (True면 Gradient 동시 계산)
        Returns:
            phi: (B, K, 1)
            grad_x: (B, K, 3) -> d(phi)/dx (Analytical)
        """
        B, K, _ = x.shape
        x_flat = x.reshape(-1, 3) # (N, 3)
        z_flat = z.reshape(-1, z.shape[-1]) # (N, D)
        
        # --- Layer 0 ---
        # h0 = sin(w0 * (Wx * x + Wz * z + b))
        out_x = self.linear_x0(x_flat)
        out_z = self.linear_z0(z_flat)
        
        u = (self.omega_0 * out_x) + out_z 
        h = torch.sin(u)
        
        if compute_grad:
            c = torch.cos(u) # (N, H)
            W_x = self.linear_x0.weight.unsqueeze(0) # (1, H, 3)
            # 체인 룰: d(sin(w0*Wx*x + Wz*z))/dx = w0 * cos(u) * Wx
            J = (self.omega_0 * c).unsqueeze(-1) * W_x # (N, H, 3)
        
        # --- Hidden Layers ---
        for layer in self.hidden_layers:
            out = layer(h)
            u = self.omega_0 * out
            h_next = torch.sin(u)
            
            if compute_grad:
                c = self.omega_0 * torch.cos(u) # (N, H)
                # Einsum으로 N*H*H 텐서 방지 (VRAM 최적화)
                W_J = torch.einsum('ij, njc -> nic', layer.weight, J)
                J = c.unsqueeze(-1) * W_J 
                
            h = h_next
            
        # --- Final Layer ---
        phi = self.final_layer(h) # (N, 1)
        
        if compute_grad:
            grad_x = torch.einsum('ij, njc -> nic', self.final_layer.weight, J).squeeze(1) 
            return phi.reshape(B, K, 1), grad_x.reshape(B, K, 3)
            
        return phi.reshape(B, K, 1), None
