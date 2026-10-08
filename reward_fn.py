"""Five-component structure-feedback reward and sequence evaluation metrics.

Reward coefficients must be supplied by the caller. This module contains no
experiment-specific coefficient defaults or trained model weights. Distances
are in angstroms, residue masks are binary, and pLDDT inputs must be normalized
to [0, 1] before calling the reward function.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
HYDROPHOBIC_AA = 'ACFILMVW'
HYDROPHOBIC_IDX = torch.tensor([ALPHABET.index(aa) for aa in HYDROPHOBIC_AA])


def compute_plddt_reward(plddt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return masked mean confidence for [B, N] pLDDT inputs in [0, 1]."""
    plddt_mean = (plddt * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return plddt_mean


def compute_clash_score(ca_coords: torch.Tensor, mask: torch.Tensor,
                        clash_threshold: float = 3.8) -> torch.Tensor:
    """Return a nonadjacent CA clash diagnostic, not a composite reward term."""
    _, N, _ = ca_coords.shape

    diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)   # [B, N, N, 3]
    dist = torch.norm(diff, dim=-1)                           # [B, N, N]

    idx = torch.arange(N, device=ca_coords.device)
    seq_sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()     # [N, N]
    exclude = (seq_sep <= 1).unsqueeze(0)                     # [1, N, N]

    valid_pair = mask.unsqueeze(2) * mask.unsqueeze(1)        # [B, N, N]
    valid_pair = valid_pair * (~exclude).float()

    valid_bool = valid_pair > 0.5                             # float -> bool
    clash = ((dist < clash_threshold) & valid_bool).float()   # [B, N, N]
    clash_count = clash.sum(dim=(1, 2))                       # [B]
    total_pairs = valid_pair.sum(dim=(1, 2)).clamp(min=1.0)   # [B]

    clash_ratio = clash_count / total_pairs
    return 1.0 - clash_ratio                                  # [B]


def compute_buried_hydrophobic_ratio(sequences: torch.Tensor,
                                     ca_coords: torch.Tensor,
                                     mask: torch.Tensor,
                                     burial_threshold: float = 8.0,
                                     min_contacts: int = 6) -> torch.Tensor:
    """Return the fraction of hydrophobic residues classified as buried.

    The manuscript defines burial using at least six valid CA neighbors within
    8 angstroms. Distances at or below 0.1 angstrom are excluded, preserving the
    existing treatment of self contacts and near-coincident coordinates. This
    is a coarse geometric proxy, not a thermodynamic stability measurement.
    """
    device = sequences.device

    diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)   # [B, N, N, 3]
    dist = torch.norm(diff, dim=-1)                           # [B, N, N]

    contacts = ((dist < burial_threshold) & (dist > 0.1)).float()
    contacts = contacts * mask.unsqueeze(1) * mask.unsqueeze(2)
    contact_count = contacts.sum(dim=-1)                      # [B, N]

    is_buried = (contact_count >= min_contacts).float()       # [B, N]

    hydro_idx = HYDROPHOBIC_IDX.to(device)
    is_hydrophobic = torch.isin(sequences, hydro_idx).float() # [B, N]

    hydro_count = (is_hydrophobic * mask).sum(dim=-1)
    buried_hydro = (is_hydrophobic * is_buried * mask).sum(dim=-1)

    ratio = buried_hydro / (hydro_count + 1e-8)
    return ratio


def kabsch_align(coords_mobile: torch.Tensor,
                 coords_ref: torch.Tensor) -> torch.Tensor:
    """Rigidly align mobile [L, 3] coordinates to reference coordinates.

    Invalid inputs raise ValueError. Numerical alignment failures raise
    RuntimeError instead of returning unaligned coordinates.
    """
    if (coords_mobile.ndim != 2 or coords_mobile.shape[-1] != 3
            or coords_ref.shape != coords_mobile.shape):
        raise ValueError("Kabsch inputs must have the same [L, 3] shape.")
    if coords_mobile.shape[0] == 0:
        raise ValueError("Kabsch alignment requires at least one coordinate pair.")

    c_mob = coords_mobile - coords_mobile.mean(dim=0, keepdim=True)
    c_ref = coords_ref - coords_ref.mean(dim=0, keepdim=True)

    if torch.isnan(c_mob).any() or torch.isinf(c_mob).any():
        raise ValueError("Kabsch mobile coordinates are nonfinite after centering.")
    if torch.isnan(c_ref).any() or torch.isinf(c_ref).any():
        raise ValueError("Kabsch reference coordinates are nonfinite after centering.")

    H = c_mob.T @ c_ref                                       # [3, 3]

    if torch.isnan(H).any() or torch.isinf(H).any():
        raise RuntimeError("Kabsch alignment produced a nonfinite covariance matrix.")

    try:
        U, _, Vt = torch.linalg.svd(H)
    except RuntimeError as exc:
        raise RuntimeError("Kabsch alignment failed during SVD.") from exc

    if (torch.isnan(U).any() or torch.isinf(U).any()
            or torch.isnan(Vt).any() or torch.isinf(Vt).any()):
        raise RuntimeError("Kabsch SVD produced nonfinite singular vectors.")

    d = torch.linalg.det(Vt.T @ U.T)
    if torch.isnan(d) or torch.isinf(d) or d == 0:
        raise RuntimeError("Kabsch alignment produced an invalid rotation determinant.")
    sign_mat = torch.diag(torch.stack([
        torch.ones(1, device=coords_mobile.device).squeeze(),
        torch.ones(1, device=coords_mobile.device).squeeze(),
        torch.sign(d)
    ]))
    R = Vt.T @ sign_mat @ U.T                                 # [3, 3]

    if torch.isnan(R).any() or torch.isinf(R).any():
        raise RuntimeError("Kabsch alignment produced a nonfinite rotation matrix.")

    aligned = c_mob @ R.T + coords_ref.mean(dim=0, keepdim=True)

    if torch.isnan(aligned).any() or torch.isinf(aligned).any():
        raise RuntimeError("Kabsch alignment produced nonfinite aligned coordinates.")

    return aligned


def tm_score_batch(coords1: torch.Tensor, coords2: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Return Kabsch-aligned scTM for [B, N, 3] coordinate batches.

    Preserve the short-length safeguards: d0 is bounded below by 0.1 angstrom,
    and targets with fewer than five valid residues receive a score of zero.
    Nonfinite input coordinates or scores also receive zero. Kabsch alignment
    failures raise RuntimeError with the affected batch index.
    """
    B = coords1.shape[0]
    tm_scores = []

    for b in range(B):
        valid_idx = mask[b].bool()
        L = int(valid_idx.sum().item())
        if L < 5:
            tm_scores.append(torch.tensor(0.0, device=coords1.device))
            continue

        c1 = coords1[b][valid_idx]                            # [L, 3]
        c2 = coords2[b][valid_idx]                            # [L, 3]

        if torch.isnan(c1).any() or torch.isinf(c1).any():
            tm_scores.append(torch.tensor(0.0, device=coords1.device))
            continue
        if torch.isnan(c2).any() or torch.isinf(c2).any():
            tm_scores.append(torch.tensor(0.0, device=coords1.device))
            continue

        c1_centered = c1 - c1.mean(dim=0, keepdim=True)
        c2_centered = c2 - c2.mean(dim=0, keepdim=True)

        try:
            c2_aligned = kabsch_align(c2_centered, c1_centered)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"scTM Kabsch alignment failed for batch item {b}: {exc}"
            ) from exc

        if torch.isnan(c2_aligned).any() or torch.isinf(c2_aligned).any():
            tm_scores.append(torch.tensor(0.0, device=coords1.device))
            continue

        d0 = max(0.1, 1.24 * (max(L - 15, 1) ** (1.0 / 3.0)) - 1.8)
        d = torch.norm(c1_centered - c2_aligned, dim=-1)      # [L]

        if torch.isnan(d).any() or torch.isinf(d).any():
            tm_scores.append(torch.tensor(0.0, device=coords1.device))
            continue

        tm = (1.0 / (1.0 + (d / d0) ** 2)).mean()

        if torch.isnan(tm) or torch.isinf(tm):
            tm_scores.append(torch.tensor(0.0, device=coords1.device))
        else:
            tm_scores.append(tm)

    return torch.stack(tm_scores)                             # [B]


def rmsd_batch(coords1: torch.Tensor, coords2: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Return Kabsch-aligned CA RMSD in angstroms for each target.

    Empty masks and nonfinite calculations retain the existing 20-angstrom
    fallback; this value is a failure penalty rather than a measured RMSD.
    Kabsch alignment failures instead raise RuntimeError with the batch index.
    """
    B = coords1.shape[0]
    rmsds = []
    for b in range(B):
        valid_idx = mask[b].bool()
        L = int(valid_idx.sum().item())
        if L < 1:
            rmsds.append(torch.tensor(20.0, device=coords1.device))
            continue
        c1 = coords1[b][valid_idx]
        c2 = coords2[b][valid_idx]

        if torch.isnan(c1).any() or torch.isinf(c1).any():
            rmsds.append(torch.tensor(20.0, device=coords1.device))
            continue
        if torch.isnan(c2).any() or torch.isinf(c2).any():
            rmsds.append(torch.tensor(20.0, device=coords1.device))
            continue

        c1_centered = c1 - c1.mean(dim=0, keepdim=True)
        c2_centered = c2 - c2.mean(dim=0, keepdim=True)

        try:
            c2_aligned = kabsch_align(c2_centered, c1_centered)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"scRMSD Kabsch alignment failed for batch item {b}: {exc}"
            ) from exc

        if torch.isnan(c2_aligned).any() or torch.isinf(c2_aligned).any():
            rmsds.append(torch.tensor(20.0, device=coords1.device))
            continue

        d2 = ((c1_centered - c2_aligned) ** 2).sum(dim=-1).mean()

        if torch.isnan(d2) or torch.isinf(d2):
            rmsds.append(torch.tensor(20.0, device=coords1.device))
        else:
            rmsd_val = d2.sqrt()
            if torch.isnan(rmsd_val) or torch.isinf(rmsd_val):
                rmsds.append(torch.tensor(20.0, device=coords1.device))
            else:
                rmsds.append(rmsd_val)

    return torch.stack(rmsds)


class RewardFunction(nn.Module):
    """Combine scTM, RMSD reward, pLDDT, hydrophobic fraction, and AAR.

    Supply all five coefficients explicitly from a separate configuration.
    Individual terms can be disabled by passing a zero coefficient.
    """

    def __init__(self, w_sctm: float, w_rmsd: float,
                 w_hydrophobic: float, w_plddt: float, w_aar: float):
        super().__init__()
        self.w_sctm = w_sctm
        self.w_rmsd = w_rmsd
        self.w_hydrophobic = w_hydrophobic
        self.w_plddt = w_plddt
        self.w_aar = w_aar

    def forward(self,
                sequences: torch.Tensor,
                original_coords: torch.Tensor,
                refolded_coords: torch.Tensor,
                lengths: torch.Tensor,
                mask: torch.Tensor,
                plddt: Optional[torch.Tensor] = None,
                native_sequences: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, dict]:
        """Return per-sequence rewards and mean components for logging.

        Sequences, native sequences, masks, and normalized pLDDT have shape
        [B, N]; backbone coordinates have shape [B, N, 3]. Valid positions are
        determined by mask. The lengths argument is retained for callers but
        does not determine the metric denominator. pLDDT and native sequences
        are required whenever their corresponding coefficients are nonzero.
        """
        if self.w_plddt != 0 and plddt is None:
            raise ValueError("Normalized pLDDT is required when w_plddt is nonzero.")
        if self.w_aar != 0 and native_sequences is None:
            raise ValueError("Native sequences are required when w_aar is nonzero.")

        sctm = tm_score_batch(original_coords, refolded_coords, mask)

        rmsd = rmsd_batch(original_coords, refolded_coords, mask)
        rmsd_reward = torch.exp(-rmsd / 5.0)

        hydro = compute_buried_hydrophobic_ratio(sequences, refolded_coords, mask)

        plddt_reward = compute_plddt_reward(plddt, mask) \
            if plddt is not None else torch.zeros_like(sctm)

        aar = torch.zeros_like(sctm)
        if native_sequences is not None:
            aar = compute_sequence_accuracy(sequences, native_sequences, mask)

        reward = (self.w_sctm * sctm
                  + self.w_rmsd * rmsd_reward
                  + self.w_hydrophobic * hydro
                  + self.w_plddt * plddt_reward
                  + self.w_aar * aar)

        info = {
            "sctm":        sctm.mean().item(),
            "rmsd":        rmsd.mean().item(),
            "rmsd_reward": rmsd_reward.mean().item(),
            "hydrophobic": hydro.mean().item(),
            "plddt":       plddt_reward.mean().item(),
            "aar":         aar.mean().item(),
            "total_reward": reward.mean().item(),
        }
        return reward, info


def compute_sequence_accuracy(designed_seq: torch.Tensor,
                               native_seq: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
    """Return AAR using only the manuscript's valid-residue mask.

    Every position with mask=1 contributes to the denominator, including
    native X tokens. Padding positions must be excluded by the supplied mask.
    """
    valid = mask > 0.5
    valid_f = valid.float()                                   # [B, N]

    match = (designed_seq == native_seq).float()              # [B, N]
    valid_count = valid_f.sum(dim=-1).clamp(min=1.0)          # [B]
    accuracy = (match * valid_f).sum(dim=-1) / valid_count    # [B]
    return accuracy


def compute_perplexity(log_probs: torch.Tensor,
                       native_seq: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """Return exp(mean native NLL) over the supplied valid-residue mask.

    Inputs are log probabilities with shape [B, N, V]. Native tokens at valid
    positions must index this vocabulary; X is included when mask=1. Neither
    the mean negative log likelihood nor the perplexity is capped.
    """
    valid = mask > 0.5

    # Replace indices only at masked-out positions, before gathering.
    native_seq_safe = native_seq.masked_fill(~valid, 0)       # [B, N]
    if ((native_seq_safe < 0) | (native_seq_safe >= log_probs.size(-1))).any():
        raise ValueError("Native tokens at valid positions must index the log-probability vocabulary.")

    native_logp = log_probs.gather(
        -1, native_seq_safe.unsqueeze(-1)
    ).squeeze(-1)                                             # [B, N]

    # Mask before reduction so padding with NaN or -inf cannot affect NLL.
    native_logp = native_logp.masked_fill(~valid, 0.0)
    valid_f = valid.float()
    valid_count = valid_f.sum(dim=-1).clamp(min=1.0)          # [B]
    nll = -native_logp.sum(dim=-1) / valid_count             # [B]

    perplexity = torch.exp(nll)                              # [B]
    return perplexity


def compute_diversity(sequences: torch.Tensor,
                      mask: torch.Tensor,
                      group_size: int) -> torch.Tensor:
    """Return mean normalized pairwise Hamming distance for each backbone.

    Sequences and masks have shape [B*G, N], with each target's G candidates
    stored consecutively. All candidates for a target should share the same
    valid-residue mask, as assumed in the manuscript's diversity definition.
    """
    BG, N = sequences.shape
    if mask.shape != sequences.shape:
        raise ValueError("sequences and mask must have the same [B*G, N] shape.")
    if group_size < 1 or BG == 0 or BG % group_size != 0:
        raise ValueError("A nonempty candidate batch must contain complete groups.")
    B = BG // group_size
    G = group_size

    seqs = sequences.reshape(B, G, N)
    masks = mask.reshape(B, G, N)

    diversity_list = []
    for b in range(B):
        pair_dists = []
        for i in range(G):
            for j in range(i + 1, G):
                valid = (masks[b, i] * masks[b, j])          # [N]
                valid_count = valid.sum().clamp(min=1.0)

                diff = (seqs[b, i] != seqs[b, j]).float() * valid
                hamming = diff.sum() / valid_count
                pair_dists.append(hamming)

        if len(pair_dists) > 0:
            diversity_list.append(torch.stack(pair_dists).mean())
        else:
            diversity_list.append(torch.tensor(0.0, device=sequences.device))

    return torch.stack(diversity_list)                        # [B]
