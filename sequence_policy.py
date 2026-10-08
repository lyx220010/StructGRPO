"""ProteinMPNN-style parallel sequence policy for HGNN backbone embeddings.

Only manuscript architecture defaults are included. Learned weights and local
experiment configuration are not distributed with this source file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:

    B, N, K = neighbor_idx.shape

    flat = neighbor_idx.reshape(B, -1)
    flat = flat.unsqueeze(-1).expand(-1, -1, nodes.size(-1))  # [B, N*K, C]
    out = torch.gather(nodes, 1, flat)                        # [B, N*K, C]
    return out.reshape(B, N, K, nodes.size(-1))               # [B, N, K, C]


def build_neighbor_mask(E_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mark valid central-neighbor pairs, excluding padding and self-loops."""
    valid = mask > 0.5
    neighbor_valid = gather_nodes(valid.unsqueeze(-1), E_idx).squeeze(-1)
    center_idx = torch.arange(mask.size(1), device=E_idx.device).view(1, -1, 1)
    return valid.unsqueeze(-1) & neighbor_valid & (E_idx != center_idx)


def masked_neighbor_mean(messages: torch.Tensor,
                         neighbor_mask: torch.Tensor) -> torch.Tensor:
    """Average messages over valid neighbors; return zero for empty sets."""
    messages = messages.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)
    count = neighbor_mask.sum(dim=-1, keepdim=True).to(messages.dtype)
    return messages.sum(dim=2) / count.clamp(min=1.0)

def build_knn_graph(ca_coords: torch.Tensor, k: int,
                    mask: torch.Tensor,
                    n_to_ca: Optional[torch.Tensor] = None,
                    ca_to_c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    B, N, _ = ca_coords.shape
    if N < 1:
        raise ValueError("The backbone must contain at least one residue position.")
    if k < 1:
        raise ValueError("k must be positive.")
    k = min(k, N - 1)

    valid = mask > 0.5
    coords = ca_coords.masked_fill(~valid.unsqueeze(-1), 0.0)
    graph_coords = coords.float() if coords.dtype in (torch.float16, torch.bfloat16) else coords
    diff = graph_coords.unsqueeze(2) - graph_coords.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)                           # [B, N, N]

    self_mask = torch.eye(N, device=ca_coords.device, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1) & ~self_mask
    dist_masked = dist.masked_fill(~valid_pair, float("inf"))

    _, E_idx = torch.topk(dist_masked, k=k, dim=-1, largest=False)  # [B, N, K]
    neighbor_mask = build_neighbor_mask(E_idx, mask)

    neighbor_coords = gather_nodes(graph_coords, E_idx)
    center_coords = graph_coords.unsqueeze(2).expand_as(neighbor_coords)
    rel_coords = neighbor_coords - center_coords
    dist_knn = torch.norm(rel_coords, dim=-1, keepdim=True)

    centers = torch.linspace(2.0, 20.0, 5, device=ca_coords.device)
    rbf = torch.exp(-((dist_knn - centers) ** 2) / 4.0)      # [B, N, K, 5]

    h_E = torch.cat([rel_coords, dist_knn, rbf], dim=-1)     # [B, N, K, 9]

    if n_to_ca is not None and ca_to_c is not None:
        x_ax = n_to_ca.masked_fill(~valid.unsqueeze(-1), 0.0)
        c_ax = ca_to_c.masked_fill(~valid.unsqueeze(-1), 0.0)
        z_ax = torch.cross(x_ax, c_ax, dim=-1)
        z_ax = z_ax / (z_ax.norm(dim=-1, keepdim=True) + 1e-8)   # [B, N, 3]
        y_ax = torch.cross(z_ax, x_ax, dim=-1)                   # [B, N, 3]

        rel_unit = rel_coords / (dist_knn + 1e-8)                 # [B, N, K, 3]

        x_exp = x_ax.unsqueeze(2).expand_as(rel_unit)
        y_exp = y_ax.unsqueeze(2).expand_as(rel_unit)
        z_exp = z_ax.unsqueeze(2).expand_as(rel_unit)
        local_orient = torch.stack([
            (rel_unit * x_exp).sum(-1),
            (rel_unit * y_exp).sum(-1),
            (rel_unit * z_exp).sum(-1),
        ], dim=-1)                                                # [B, N, K, 3]
        h_E = torch.cat([h_E, local_orient], dim=-1)             # [B, N, K, 12]

    h_E = h_E.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)
    return E_idx, h_E

class MPNNEncLayer(nn.Module):

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()

        in_dim = hidden_dim * 2 + edge_dim
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)

        self.W_edge = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor, h_E: torch.Tensor,
                E_idx: torch.Tensor,
                mask: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N, K, _ = h_E.shape
        if neighbor_mask is None:
            neighbor_mask = build_neighbor_mask(E_idx, mask)

        h_V_nbr = gather_nodes(h_V, E_idx)                   # [B, N, K, D]
        h_V_exp = h_V.unsqueeze(2).expand(-1, -1, K, -1)     # [B, N, K, D]

        h_EV = torch.cat([h_V_exp, h_V_nbr, h_E], dim=-1)   # [B, N, K, 2D+D_e]
        h_EV = h_EV.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)

        msg = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))  # [B, N, K, D]

        dh = masked_neighbor_mean(msg, neighbor_mask)

        h_V = self.norm1(h_V + self.drop(dh))

        h_V = self.norm2(h_V + self.drop(self.ff(h_V)))

        h_V = h_V * mask.unsqueeze(-1)

        h_E = self.norm_e(h_E + self.drop(self.W_edge(h_EV)))
        h_E = h_E.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)
        return h_V, h_E

class MPNNDecLayer(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        in_dim = hidden_dim * 2
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor, h_E: torch.Tensor,
                mask: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:

        h_V_exp = h_V.unsqueeze(2).expand(-1, -1, h_E.size(2), -1)
        h_EV = torch.cat([h_V_exp, h_E], dim=-1)             # [B, N, K, 2D]
        h_EV = h_EV.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)
        msg = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = masked_neighbor_mean(msg, neighbor_mask)
        h_V = self.norm1(h_V + self.drop(dh))
        h_V = self.norm2(h_V + self.drop(self.ff(h_V)))
        h_V = h_V * mask.unsqueeze(-1)
        return h_V

class SequenceDesignPolicy(nn.Module):

    def __init__(self, hgnn_out_dim: int, hidden_dim: int = 128,
                 num_enc_layers: int = 6, num_dec_layers: int = 3,
                 k_neighbors: int = 48, vocab_size: int = 21,
                 dropout: Optional[float] = None,
                 use_online_rel_orient: bool = True):

        super().__init__()
        if dropout is None:
            raise ValueError("Supply dropout explicitly from your local configuration.")
        self.k = k_neighbors
        self.hidden_dim = hidden_dim
        self.use_online_rel_orient = use_online_rel_orient

        edge_dim = 12 if use_online_rel_orient else 9

        self.hgnn_proj = nn.Linear(hgnn_out_dim, hidden_dim)

        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.enc_layers = nn.ModuleList([
            MPNNEncLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_enc_layers)
        ])

        self.dec_layers = nn.ModuleList([
            MPNNDecLayer(hidden_dim, dropout)
            for _ in range(num_dec_layers)
        ])

        self.output_head = nn.Linear(hidden_dim, vocab_size)

        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.bias)

    def encode(self, ca_coords: torch.Tensor, hgnn_h: torch.Tensor,
               mask: torch.Tensor,
               n_to_ca: Optional[torch.Tensor] = None,
               ca_to_c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _n_to_ca = n_to_ca if self.use_online_rel_orient else None
        if self.use_online_rel_orient and (n_to_ca is None or ca_to_c is None):
            raise ValueError("Directional edge features require n_to_ca and ca_to_c.")
        _ca_to_c = ca_to_c if self.use_online_rel_orient else None
        E_idx, h_E_raw = build_knn_graph(ca_coords, self.k, mask, _n_to_ca, _ca_to_c)
        neighbor_mask = build_neighbor_mask(E_idx, mask)

        h_V = self.hgnn_proj(hgnn_h)                         # [B, N, D]
        h_E = self.edge_proj(h_E_raw)                        # [B, N, K, D]
        h_V = h_V.masked_fill(~(mask > 0.5).unsqueeze(-1), 0.0)
        h_E = h_E.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)

        for layer in self.enc_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, neighbor_mask)

        return h_V, h_E, E_idx

    def decode_logits(self, h_V: torch.Tensor, h_E: torch.Tensor,
                      mask: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
        """Decode using the same valid-neighbor mask as the encoder."""

        neighbor_mask = build_neighbor_mask(E_idx, mask)
        h = h_V
        for layer in self.dec_layers:
            h = layer(h, h_E, mask, neighbor_mask)
        logits = self.output_head(h)                          # [B, N, 21]

        if logits.size(-1) > 20:
            logits = logits.clone()
            x_mask_value = -1e4 if logits.dtype in (torch.float16, torch.bfloat16) else -1e9
            logits[..., 20] = x_mask_value

        return logits

    def forward(self, ca_coords: torch.Tensor, hgnn_h: torch.Tensor,
                mask: torch.Tensor,
                n_to_ca: Optional[torch.Tensor] = None,
                ca_to_c: Optional[torch.Tensor] = None) -> torch.Tensor:

        h_V, h_E, E_idx = self.encode(ca_coords, hgnn_h, mask, n_to_ca, ca_to_c)
        logits = self.decode_logits(h_V, h_E, mask, E_idx)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    @torch.no_grad()
    def sample(self, ca_coords: torch.Tensor, hgnn_h: torch.Tensor,
               mask: torch.Tensor, temperature: float,
               n_samples: int,
               n_to_ca: Optional[torch.Tensor] = None,
               ca_to_c: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:

        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if n_samples < 1:
            raise ValueError("n_samples must be at least one.")

        B, N, _ = ca_coords.shape

        ca_rep    = ca_coords.repeat_interleave(n_samples, dim=0)
        hgnn_rep  = hgnn_h.repeat_interleave(n_samples, dim=0)
        mask_rep  = mask.repeat_interleave(n_samples, dim=0)
        n_to_ca_rep = n_to_ca.repeat_interleave(n_samples, dim=0) if n_to_ca is not None else None
        ca_to_c_rep = ca_to_c.repeat_interleave(n_samples, dim=0) if ca_to_c is not None else None

        h_V, h_E, E_idx = self.encode(ca_rep, hgnn_rep, mask_rep, n_to_ca_rep, ca_to_c_rep)
        logits = self.decode_logits(h_V, h_E, mask_rep, E_idx)   # [B*G, N, 21]

        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()
        total_elements = logits.numel()

        if nan_count > 0 or inf_count > 0:

            nan_ratio = nan_count / total_elements * 100
            inf_ratio = inf_count / total_elements * 100

            nan_mask = torch.isnan(logits).any(dim=-1)  # [B*G, N]
            nan_per_seq = nan_mask.sum(dim=-1)  # [B*G]
            nan_per_pos = nan_mask.sum(dim=0)  # [N]

            logger.warning(
                f"Non-finite sampling logits detected:\n"
                f"   NaN: {nan_count}/{total_elements} ({nan_ratio:.2f}%)\n"
                f"   Inf: {inf_count}/{total_elements} ({inf_ratio:.2f}%)\n"
                f"   NaNs per sequence: min={nan_per_seq.min()}, max={nan_per_seq.max()}, "
                f"mean={nan_per_seq.float().mean():.1f}\n"
                f"   Non-finite values will be replaced; sampling quality may be affected."
            )

            if nan_ratio > 10.0:
                logger.error(
                    f"NaN proportion is too high ({nan_ratio:.1f}%); returning None."
                )

                return None

            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        if logits.size(-1) > 20:
            x_mask_value = -1e4 if logits.dtype in (torch.float16, torch.bfloat16) else -1e9
            logits[..., 20] = x_mask_value

        scaled_logits = logits.float() / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        if probs.size(-1) > 20:
            probs[..., 20] = 0.0

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logger.error(
                f"Non-finite probabilities after softmax; temperature={temperature}. "
                f"Returning None."
            )
            return None

        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        BG, N_out, V = probs.shape

        sequences = torch.multinomial(
            probs.reshape(-1, V), num_samples=1
        ).reshape(BG, N_out)                                      # [B*G, N]

        return sequences
