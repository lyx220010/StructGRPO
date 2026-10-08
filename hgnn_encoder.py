"""Two-layer backbone encoder with KNN, contact, and attention branches.

The public architectural defaults follow the manuscript: 16 KNN neighbors,
an 8-angstrom contact threshold, 16 geometric input features, and attention
enabled. Dropout probabilities and the attention head count must be supplied
by the caller. This file contains no experiment-specific values for those
settings, checkpoint weights, or private data paths.
"""

from typing import Optional

import torch
import torch.nn as nn


def build_knn_hyperedge(ca_coords: torch.Tensor, k: int, mask: torch.Tensor) -> torch.Tensor:
    """Return [B, N, N] incidence matrices, indexed by node and center.

    Each valid center connects to at most min(k, L - 1) other valid residues,
    where L is that target's valid residue count. Self and padding positions
    are excluded even when a target has fewer than k available neighbors.
    """
    B, N, _ = ca_coords.shape
    if mask.shape != ca_coords.shape[:2]:
        raise ValueError("mask must match the [B, N] coordinate dimensions.")
    if k < 0:
        raise ValueError("The KNN neighbor count must be nonnegative.")

    H = torch.zeros(B, N, N, device=ca_coords.device, dtype=ca_coords.dtype)
    effective_k = min(k, max(N - 1, 0))
    if effective_k == 0:
        return H

    diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)   # [B, N, N, 3]
    dist = torch.norm(diff, dim=-1)                           # [B, N, N]

    valid = mask.bool()
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
    diagonal = torch.eye(N, device=ca_coords.device, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid_pair & ~diagonal
    dist = dist.masked_fill(~valid_pair, float("inf"))

    _, knn_idx = torch.topk(dist, k=effective_k, dim=-1, largest=False)
    H.scatter_(1, knn_idx.transpose(1, 2), 1.0)              # [B, N, N]

    # Invalid top-k filler entries never become members of a hyperedge.
    H = H.masked_fill(~valid_pair, 0.0)
    return H


def build_contact_hyperedge(ca_coords: torch.Tensor, threshold: float, mask: torch.Tensor) -> torch.Tensor:
    """Return a masked pairwise CA contact matrix, excluding self contacts.

    The name is retained for caller compatibility. This branch supplies
    complementary contact-neighborhood information; its input matrix is a
    pairwise contact relation rather than a higher-order KNN incidence matrix.
    """
    _, N, _ = ca_coords.shape
    diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)                           # [B, N, N]

    diag_mask = torch.eye(N, device=ca_coords.device, dtype=ca_coords.dtype).unsqueeze(0)  # [1, N, N]
    dist = dist + diag_mask * 1e9

    H = (dist < threshold).float()

    H = H * mask.unsqueeze(2) * mask.unsqueeze(1)
    return H


def hyperedge_to_laplacian(H: torch.Tensor) -> torch.Tensor:
    """Return P = Dv^(-1/2) H De^(-1) H.T Dv^(-1/2).

    Uniform relation weights are used. P is the normalized propagation matrix
    in the manuscript; the historical function name is retained. H has shape
    [B, N, E], and the returned propagation matrix has shape [B, N, N].
    """
    eps = 1e-6

    Dv = H.sum(dim=-1, keepdim=True).clamp(min=eps)          # [B, N, 1]
    Dv_inv_sqrt = Dv.pow(-0.5)                                # [B, N, 1]

    De = H.sum(dim=1, keepdim=True).clamp(min=eps)            # [B, 1, N]
    De_inv = De.pow(-1.0)                                     # [B, 1, N]

    H_scaled = H * Dv_inv_sqrt                                # [B, N, N]

    HT_scaled = H.transpose(1, 2) * Dv_inv_sqrt.transpose(1, 2)  # [B, N, N]
    HT_scaled = HT_scaled * De_inv.transpose(1, 2)           # [B, N, N]

    G = torch.bmm(H_scaled, HT_scaled)                       # [B, N, N]
    return G


class HGNNConv(nn.Module):
    """Aggregate node features using P, then apply a learnable projection."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Map [B, N, in_dim] features to [B, N, out_dim]."""
        x = torch.bmm(G, x)      # [B, N, in_dim]
        x = self.linear(x)
        return x


class HGNNProteinEncoder(nn.Module):
    """Fuse two stages of KNN and contact propagation with optional MHA.

    Full features are CA(3), virtual CB(3), N-to-CA direction(3), CA-to-C
    direction(3), sin(phi), cos(phi), sin(psi), and cos(psi), in that order.
    Their linear input projection retains the original GELU activation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 k_neighbors: int = 16, contact_threshold: float = 8.0,
                 dropout: Optional[float] = None, use_rich_features: bool = True,
                 use_attention: bool = True, num_attention_heads: Optional[int] = None,
                 attention_dropout: Optional[float] = None):
        """Build the encoder using caller-supplied regularization settings.

        in_dim is used only for the coordinate-only mode; full geometric
        inputs always have 16 features. hidden_dim and out_dim are supplied
        explicitly. Attention settings are required only when MHA is enabled.
        """
        super().__init__()
        if dropout is None:
            raise ValueError("Supply dropout explicitly from your configuration.")
        if use_attention:
            if num_attention_heads is None or attention_dropout is None:
                raise ValueError(
                    "Supply num_attention_heads and attention_dropout explicitly "
                    "when use_attention=True."
                )
            if num_attention_heads < 1:
                raise ValueError("num_attention_heads must be positive.")
            if hidden_dim % num_attention_heads or out_dim % num_attention_heads:
                raise ValueError(
                    "hidden_dim and out_dim must be divisible by num_attention_heads."
                )
        self.k = k_neighbors
        self.contact_threshold = contact_threshold
        self.use_rich_features = use_rich_features
        self.use_attention = use_attention

        if use_rich_features:
            actual_in_dim = 16
        else:
            actual_in_dim = in_dim

        self.input_proj = nn.Linear(actual_in_dim, hidden_dim)

        self.hgnn_knn1 = HGNNConv(hidden_dim, hidden_dim)
        self.hgnn_contact1 = HGNNConv(hidden_dim, hidden_dim)
        self.fuse1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.hgnn_knn2 = HGNNConv(hidden_dim, out_dim)
        self.hgnn_contact2 = HGNNConv(hidden_dim, out_dim)
        self.fuse2 = nn.Linear(out_dim * 2, out_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        if self.use_attention:

            self.attention1 = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            self.attn_norm1 = nn.LayerNorm(hidden_dim)

            self.attention2 = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=num_attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            self.attn_norm2 = nn.LayerNorm(out_dim)

    def forward(self, ca_coords: torch.Tensor, mask: torch.Tensor,
                cb_coords: Optional[torch.Tensor] = None,
                n_to_ca: Optional[torch.Tensor] = None,
                ca_to_c: Optional[torch.Tensor] = None,
                phi: Optional[torch.Tensor] = None,
                psi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a backbone into [B, N, out_dim] residue representations.

        Coordinates and directions have shape [B, N, 3]; angles and the binary
        residue mask have shape [B, N]. All five extra geometric inputs must
        be supplied in the full-feature mode. Nonfinite attention inputs,
        outputs, or residual updates raise RuntimeError instead of silently
        skipping a layer. Empty targets raise ValueError when MHA is enabled.
        """

        H_knn = build_knn_hyperedge(ca_coords, self.k, mask)
        H_contact = build_contact_hyperedge(ca_coords, self.contact_threshold, mask)
        G_knn = hyperedge_to_laplacian(H_knn)
        G_contact = hyperedge_to_laplacian(H_contact)

        if self.use_rich_features and cb_coords is not None \
                and n_to_ca is not None and ca_to_c is not None \
                and phi is not None and psi is not None:

            # CA, CB, directions, then sin/cos of phi and psi: 16 features.
            sin_phi = torch.sin(phi).unsqueeze(-1)  # [B, N, 1]
            cos_phi = torch.cos(phi).unsqueeze(-1)  # [B, N, 1]
            sin_psi = torch.sin(psi).unsqueeze(-1)  # [B, N, 1]
            cos_psi = torch.cos(psi).unsqueeze(-1)  # [B, N, 1]

            x = torch.cat([
                ca_coords,    # [B, N, 3]
                cb_coords,    # [B, N, 3]
                n_to_ca,      # [B, N, 3]
                ca_to_c,      # [B, N, 3]
                sin_phi,      # [B, N, 1]
                cos_phi,      # [B, N, 1]
                sin_psi,      # [B, N, 1]
                cos_psi,      # [B, N, 1]
            ], dim=-1)        # [B, N, 16]
        else:

            if self.use_rich_features:
                raise ValueError(
                    "Full geometric features require cb_coords, n_to_ca, "
                    "ca_to_c, phi, and psi. Check preprocessing or explicitly "
                    "select use_rich_features=False for coordinate-only inputs."
                )
            else:
                x = ca_coords

        x = self.act(self.input_proj(x))                      # [B, N, hidden_dim]
        x = x * mask.unsqueeze(-1)

        h_knn = self.act(self.hgnn_knn1(x, G_knn))
        h_contact = self.act(self.hgnn_contact1(x, G_contact))
        h = self.fuse1(torch.cat([h_knn, h_contact], dim=-1))
        h = self.norm1(self.dropout(h) + x)
        h = h * mask.unsqueeze(-1)

        if self.use_attention:
            attn_mask = ~mask.bool()  # [B, N], True marks padding.
            if (mask.bool().sum(dim=-1) == 0).any():
                raise ValueError("Attention requires at least one valid residue per target.")

            if torch.isnan(h).any() or torch.isinf(h).any():
                raise RuntimeError("Nonfinite HGNN states before the first attention layer.")

            h_attn, _ = self.attention1(
                query=h,
                key=h,
                value=h,
                key_padding_mask=attn_mask,
                need_weights=False
            )
            if torch.isnan(h_attn).any() or torch.isinf(h_attn).any():
                raise RuntimeError("The first attention layer produced nonfinite outputs.")

            h = self.attn_norm1(h + self.dropout(h_attn))
            if torch.isnan(h).any() or torch.isinf(h).any():
                raise RuntimeError("The first attention residual update produced nonfinite states.")
            h = h * mask.unsqueeze(-1)

        h_knn2 = self.act(self.hgnn_knn2(h, G_knn))
        h_contact2 = self.act(self.hgnn_contact2(h, G_contact))
        h_out = self.fuse2(torch.cat([h_knn2, h_contact2], dim=-1))
        h_out = self.norm2(h_out)
        h_out = h_out * mask.unsqueeze(-1)

        if self.use_attention:
            attn_mask = ~mask.bool()

            if torch.isnan(h_out).any() or torch.isinf(h_out).any():
                raise RuntimeError("Nonfinite HGNN states before the second attention layer.")

            h_attn2, _ = self.attention2(
                query=h_out,
                key=h_out,
                value=h_out,
                key_padding_mask=attn_mask,
                need_weights=False
            )
            if torch.isnan(h_attn2).any() or torch.isinf(h_attn2).any():
                raise RuntimeError("The second attention layer produced nonfinite outputs.")

            h_out = self.attn_norm2(h_out + self.dropout(h_attn2))
            if torch.isnan(h_out).any() or torch.isinf(h_out).any():
                raise RuntimeError("The second attention residual update produced nonfinite states.")
            h_out = h_out * mask.unsqueeze(-1)

        return h_out                                           # [B, N, out_dim]
