import torch
import torch.nn as nn
import numpy as np

class AFDChannelAttention(nn.Module):
    """
    AFD-based Attention Module:
    1) Treat Blaschke bases as N adaptive filters of length L
    2) Project input feature maps to get coefficients A_n
    3) Use |A_n| (or normalized) as channel attention weights
    4) Reconstruct and add as residual
    """
    def __init__(self, in_channels, N=32):
        super().__init__()
        self.C = in_channels
        self.N = N
        # Spatial flattening length L will be inferred at forward
        # Precompute N Blaschke bases of length L once we know L
        self.register_buffer('a_thetas', torch.linspace(0, 2*np.pi, N, endpoint=False))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        L = H * W
        # generate Blaschke bases B_n(t) over t=0..L-1
        thetas = self.a_thetas.to(x.device)  # [N]
        a = 0.9 * torch.exp(1j * thetas)    # [N]
        t = torch.linspace(0, 2*np.pi, L, device=x.device)
        # B: [N, L] complex
        B = (torch.sqrt(1 - torch.abs(a)**2).unsqueeze(1) /
             (1 - torch.conj(a).unsqueeze(1) * torch.exp(1j * t)))
        B = B.real  # use real part as filters

        # Flatten spatial dims
        x_flat = x.view(B, C, L)  # [B, C, L]

        # 1) Projection: compute A_n = <x, B_n> along L
        # proj: [B, C, N]
        proj = torch.einsum('bcl,nl->bcn', x_flat, B)

        # 2) Compute attention weights across N "frequency channels"
        # Use magnitude and softmax along N
        weights = torch.softmax(proj.abs(), dim=-1)  # [B, C, N]

        # 3) Apply weights to each B_n and sum: attention feature [B, C, L]
        weighted_B = B.unsqueeze(0).unsqueeze(0) * weights.unsqueeze(-1)  # [B,C,N,L]
        attn_feature = weighted_B.sum(dim=2)  # [B, C, L]

        # reshape back to [B, C, H, W] and add residual
        attn_feature = attn_feature.view(B, C, H, W)
        return x + attn_feature

# Example usage
if __name__ == '__main__':
    x = torch.randn(4, 8, 16, 16)  # batch of feature maps
    afd_attn = AFDChannelAttention(in_channels=8, N=16)
    out = afd_attn(x)
    print("Output shape:", out.shape)
