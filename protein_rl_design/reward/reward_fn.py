from typing import Optional, Tuple
import torch
import torch.nn as nn

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
HYDROPHOBIC_AA = "ACFILMVW"
HYDROPHOBIC_IDX = torch.tensor([ALPHABET.index(aa) for aa in HYDROPHOBIC_AA])


def compute_plddt_reward(plddt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    plddt_mean = (plddt * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return plddt_mean


def compute_buried_hydrophobic_ratio(
    sequences: torch.Tensor,
    ca_coords: torch.Tensor,
    mask: torch.Tensor,
    burial_threshold: float = 8.0,
    min_contacts: int = 6,
) -> torch.Tensor:
    device = sequences.device
    diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    contacts = ((dist < burial_threshold) & (dist > 0.1)).float()
    contacts = contacts * mask.unsqueeze(1) * mask.unsqueeze(2)
    contact_count = contacts.sum(dim=-1)
    is_buried = (contact_count >= min_contacts).float()
    hydro_idx = HYDROPHOBIC_IDX.to(device)
    is_hydrophobic = torch.isin(sequences, hydro_idx).float()
    hydro_count = (is_hydrophobic * mask).sum(dim=-1)
    buried_hydro = (is_hydrophobic * is_buried * mask).sum(dim=-1)
    ratio = buried_hydro / (hydro_count + 1e-08)
    return ratio


def kabsch_align(coords_mobile: torch.Tensor, coords_ref: torch.Tensor) -> torch.Tensor:
    if (
        coords_mobile.ndim != 2
        or coords_mobile.shape[-1] != 3
        or coords_ref.shape != coords_mobile.shape
    ):
        raise ValueError("Kabsch inputs must have the same [L, 3] shape.")
    if coords_mobile.shape[0] == 0:
        raise ValueError("Kabsch alignment requires at least one coordinate pair.")
    c_mob = coords_mobile - coords_mobile.mean(dim=0, keepdim=True)
    c_ref = coords_ref - coords_ref.mean(dim=0, keepdim=True)
    if torch.isnan(c_mob).any() or torch.isinf(c_mob).any():
        raise ValueError("Kabsch mobile coordinates are nonfinite after centering.")
    if torch.isnan(c_ref).any() or torch.isinf(c_ref).any():
        raise ValueError("Kabsch reference coordinates are nonfinite after centering.")
    H = c_mob.T @ c_ref
    if torch.isnan(H).any() or torch.isinf(H).any():
        raise RuntimeError("Kabsch alignment produced a nonfinite covariance matrix.")
    try:
        U, _, Vt = torch.linalg.svd(H)
    except RuntimeError as exc:
        raise RuntimeError("Kabsch alignment failed during SVD.") from exc
    if (
        torch.isnan(U).any()
        or torch.isinf(U).any()
        or torch.isnan(Vt).any()
        or torch.isinf(Vt).any()
    ):
        raise RuntimeError("Kabsch SVD produced nonfinite singular vectors.")
    d = torch.linalg.det(Vt.T @ U.T)
    if torch.isnan(d) or torch.isinf(d) or d == 0:
        raise RuntimeError("Kabsch alignment produced an invalid rotation determinant.")
    sign_mat = torch.diag(
        torch.stack(
            [
                torch.ones(1, device=coords_mobile.device).squeeze(),
                torch.ones(1, device=coords_mobile.device).squeeze(),
                torch.sign(d),
            ]
        )
    )
    R = Vt.T @ sign_mat @ U.T
    if torch.isnan(R).any() or torch.isinf(R).any():
        raise RuntimeError("Kabsch alignment produced a nonfinite rotation matrix.")
    aligned = c_mob @ R.T + coords_ref.mean(dim=0, keepdim=True)
    if torch.isnan(aligned).any() or torch.isinf(aligned).any():
        raise RuntimeError("Kabsch alignment produced nonfinite aligned coordinates.")
    return aligned


def structural_scores_batch(
    coords1: torch.Tensor, coords2: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    tm_scores = []
    rmsds = []
    for b in range(coords1.shape[0]):
        valid_idx = mask[b].bool()
        length = int(valid_idx.sum().item())
        zero = torch.tensor(0.0, device=coords1.device)
        failure_rmsd = torch.tensor(20.0, device=coords1.device)
        if length < 1:
            tm_scores.append(zero)
            rmsds.append(failure_rmsd)
            continue
        reference = coords1[b][valid_idx]
        mobile = coords2[b][valid_idx]
        if not torch.isfinite(reference).all() or not torch.isfinite(mobile).all():
            tm_scores.append(zero)
            rmsds.append(failure_rmsd)
            continue
        reference = reference - reference.mean(dim=0, keepdim=True)
        mobile = mobile - mobile.mean(dim=0, keepdim=True)
        try:
            aligned = kabsch_align(mobile, reference)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Structural scoring Kabsch alignment failed for batch item {b}: {exc}"
            ) from exc
        squared_distances = (reference - aligned).square().sum(dim=-1)
        if not torch.isfinite(squared_distances).all():
            tm_scores.append(zero)
            rmsds.append(failure_rmsd)
            continue
        rmsd = squared_distances.mean().sqrt()
        if not torch.isfinite(rmsd):
            rmsd = failure_rmsd
        rmsds.append(rmsd)
        if length < 5:
            tm_scores.append(zero)
            continue
        d0 = max(0.5, 1.24 * max(length - 15, 1) ** (1.0 / 3.0) - 1.8)
        tm = (1.0 / (1.0 + squared_distances / (d0 * d0))).mean()
        tm_scores.append(tm if torch.isfinite(tm) else zero)
    return (torch.stack(tm_scores), torch.stack(rmsds))


def tm_score_batch(
    coords1: torch.Tensor, coords2: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    scores, _ = structural_scores_batch(coords1, coords2, mask)
    return scores


def rmsd_batch(coords1: torch.Tensor, coords2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _, rmsds = structural_scores_batch(coords1, coords2, mask)
    return rmsds


class RewardFunction(nn.Module):

    def __init__(
        self, w_sctm: float, w_rmsd: float, w_hydrophobic: float, w_plddt: float, w_aar: float
    ):
        super().__init__()
        self.w_sctm = w_sctm
        self.w_rmsd = w_rmsd
        self.w_hydrophobic = w_hydrophobic
        self.w_plddt = w_plddt
        self.w_aar = w_aar

    def forward(
        self,
        sequences: torch.Tensor,
        original_coords: torch.Tensor,
        refolded_coords: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        plddt: Optional[torch.Tensor] = None,
        native_sequences: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.w_plddt != 0 and plddt is None:
            raise ValueError("Normalized pLDDT is required when w_plddt is nonzero.")
        if self.w_aar != 0 and native_sequences is None:
            raise ValueError("Native sequences are required when w_aar is nonzero.")
        sctm, rmsd = structural_scores_batch(original_coords, refolded_coords, mask)
        rmsd_reward = torch.exp(-rmsd / 5.0)
        hydro = compute_buried_hydrophobic_ratio(sequences, refolded_coords, mask)
        plddt_reward = (
            compute_plddt_reward(plddt, mask) if plddt is not None else torch.zeros_like(sctm)
        )
        aar = torch.zeros_like(sctm)
        if native_sequences is not None:
            aar = compute_sequence_accuracy(sequences, native_sequences, mask)
        reward = (
            self.w_sctm * sctm
            + self.w_rmsd * rmsd_reward
            + self.w_hydrophobic * hydro
            + self.w_plddt * plddt_reward
            + self.w_aar * aar
        )
        info = {
            "sctm": sctm.mean().item(),
            "rmsd": rmsd.mean().item(),
            "rmsd_reward": rmsd_reward.mean().item(),
            "hydrophobic": hydro.mean().item(),
            "plddt": plddt_reward.mean().item(),
            "aar": aar.mean().item(),
            "total_reward": reward.mean().item(),
        }
        return (reward, info)


def compute_sequence_accuracy(
    designed_seq: torch.Tensor, native_seq: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = mask > 0.5
    valid_f = valid.float()
    match = (designed_seq == native_seq).float()
    valid_count = valid_f.sum(dim=-1).clamp(min=1.0)
    accuracy = (match * valid_f).sum(dim=-1) / valid_count
    return accuracy


def compute_perplexity(
    log_probs: torch.Tensor, native_seq: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = mask > 0.5
    native_seq_safe = native_seq.masked_fill(~valid, 0)
    if ((native_seq_safe < 0) | (native_seq_safe >= log_probs.size(-1))).any():
        raise ValueError(
            "Native tokens at valid positions must index the log-probability vocabulary."
        )
    native_logp = log_probs.gather(-1, native_seq_safe.unsqueeze(-1)).squeeze(-1)
    native_logp = native_logp.masked_fill(~valid, 0.0)
    valid_f = valid.float()
    valid_count = valid_f.sum(dim=-1).clamp(min=1.0)
    nll = -native_logp.sum(dim=-1) / valid_count
    perplexity = torch.exp(nll)
    return perplexity


def compute_diversity(sequences: torch.Tensor, mask: torch.Tensor, group_size: int) -> torch.Tensor:
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
                valid = masks[b, i] * masks[b, j]
                valid_count = valid.sum().clamp(min=1.0)
                diff = (seqs[b, i] != seqs[b, j]).float() * valid
                hamming = diff.sum() / valid_count
                pair_dists.append(hamming)
        if len(pair_dists) > 0:
            diversity_list.append(torch.stack(pair_dists).mean())
        else:
            diversity_list.append(torch.tensor(0.0, device=sequences.device))
    return torch.stack(diversity_list)
