import logging
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from protein_rl_design.reward.reward_fn import (
    compute_sequence_accuracy,
    compute_perplexity,
    compute_diversity,
)

logger = logging.getLogger(__name__)


class GRPOTrainer:

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        hgnn_encoder: nn.Module,
        ref_hgnn_encoder: nn.Module,
        reward_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        group_size: int = 8,
        clip_range: float = 0.2,
        beta_kl: float = 2.0,
        max_grad_norm: float = 1.0,
        kl_ref_logp_gap_clamp: float = 0.5,
        temperature: float = 1.0,
        mixed_precision: bool = True,
        device: str = "cuda",
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.hgnn = hgnn_encoder
        self.ref_hgnn = ref_hgnn_encoder
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.group_size = group_size
        self.clip_range = clip_range
        self.beta_kl = beta_kl
        self.max_grad_norm = max_grad_norm
        self.kl_ref_logp_gap_clamp = float(kl_ref_logp_gap_clamp)
        self.temperature = temperature
        self.device = device
        self.mixed_precision = mixed_precision
        self.amp_dtype = torch.bfloat16 if mixed_precision else torch.float32
        if isinstance(device, str):
            self.device_type = device.split(":")[0]
        else:
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        pytorch_version = tuple((int(x) for x in torch.__version__.split(".")[:2]))
        self.use_device_type = pytorch_version >= (2, 0)
        if not self.use_device_type and mixed_precision:
            logger.warning(
                f"Detected PyTorch {torch.__version__} < 2.0. Using the legacy CUDA autocast path with its default float16 dtype."
            )
        if self.device_type == "cpu":
            self.amp_dtype = torch.float32
            if mixed_precision:
                logger.warning("Mixed precision is disabled on CPU in this trainer.")
                self.mixed_precision = False
        self.scaler = GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            dtype_name = "bfloat16" if self.amp_dtype == torch.bfloat16 else "float16"
            logger.info(
                f"Mixed-precision training enabled: {dtype_name} (PyTorch {torch.__version__})"
            )
        else:
            logger.info("Mixed precision disabled; using float32.")
        self._call_counter = 0
        self._success_counter = 0
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        for p in self.ref_hgnn.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()
        self.ref_hgnn.eval()

    def _autocast(self):
        if self.use_device_type:
            from torch.amp import autocast

            return autocast(
                device_type=self.device_type, dtype=self.amp_dtype, enabled=self.mixed_precision
            )
        else:
            from torch.cuda.amp import autocast
            from contextlib import nullcontext

            if self.device_type == "cuda":
                if self.amp_dtype == torch.bfloat16 and self.mixed_precision:
                    if not hasattr(self, "_bfloat16_warning_shown"):
                        logger.warning(
                            f"Legacy PyTorch {torch.__version__} autocast path uses float16. Use the PyTorch 2.x path for the configured bfloat16 mode."
                        )
                        self._bfloat16_warning_shown = True
                return autocast(enabled=self.mixed_precision)
            else:
                return nullcontext()

    def _encode_backbone(
        self,
        ca_coords: torch.Tensor,
        mask: torch.Tensor,
        cb_coords: Optional[torch.Tensor] = None,
        n_to_ca: Optional[torch.Tensor] = None,
        ca_to_c: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        psi: Optional[torch.Tensor] = None,
        use_ref: bool = False,
    ) -> torch.Tensor:
        hgnn = self.ref_hgnn if use_ref else self.hgnn
        return hgnn(ca_coords, mask, cb_coords, n_to_ca, ca_to_c, phi, psi)

    def _compute_sequence_logp(
        self,
        policy: nn.Module,
        ca_coords: torch.Tensor,
        hgnn_h: torch.Tensor,
        mask: torch.Tensor,
        sequences: torch.Tensor,
        n_to_ca: Optional[torch.Tensor] = None,
        ca_to_c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_probs = policy(ca_coords, hgnn_h, mask, n_to_ca, ca_to_c)
        token_logp = log_probs.gather(-1, sequences.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * mask
        seq_len = mask.sum(dim=-1).clamp(min=1.0)
        seq_logp = token_logp.sum(dim=-1) / seq_len
        return seq_logp

    def _aggregate_token_logp(self, token_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logger.warning(
            "_aggregate_token_logp() is deprecated; use _compute_sequence_logp() for consistent sequence scoring."
        )
        token_logp = token_logp * mask
        seq_len = mask.sum(dim=-1).clamp(min=1.0)
        seq_logp = token_logp.sum(dim=-1) / seq_len
        return seq_logp

    def _grpo_loss(
        self,
        seq_logp_new: torch.Tensor,
        seq_logp_old: torch.Tensor,
        seq_logp_ref: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float, float, float, float, float, float]:
        seq_logp_new_fp32 = seq_logp_new.float()
        seq_logp_old_fp32 = seq_logp_old.detach().float()
        seq_logp_ref_fp32 = seq_logp_ref.detach().float()
        advantages_fp32 = advantages.float()
        ratio = torch.exp(seq_logp_new_fp32 - seq_logp_old_fp32)
        surr1 = ratio * advantages_fp32
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_fp32
        policy_loss = -torch.min(surr1, surr2).mean()
        logp_diff = seq_logp_old_fp32 - seq_logp_new_fp32
        same_param_diff = logp_diff.abs().mean()
        same_param_diff_raw = logp_diff.mean()
        logp_gap = seq_logp_new_fp32 - seq_logp_ref_fp32
        mc_ref_gap = logp_gap.mean()
        gap_abs = logp_gap.abs()
        with torch.no_grad():
            mean_abs_ref_gap = gap_abs.mean()
            p90_abs_ref_gap = torch.quantile(gap_abs.flatten(), 0.9)
        c = self.kl_ref_logp_gap_clamp
        if c > 0.0:
            logp_gap_kl = torch.clamp(logp_gap, min=-c, max=c)
        else:
            logp_gap_kl = logp_gap
        mean_sq_ref_gap = (logp_gap_kl**2).mean()
        if same_param_diff > 0.001:
            logger.warning(
                f"Large same-parameter score difference: {same_param_diff:.5f} (signed mean: {same_param_diff_raw.item():.5f})\n   This compares two forward evaluations before optimization. Check precision and stochastic forward operations."
            )
        elif same_param_diff > 0.0005:
            logger.info(
                f"  [Numerical diagnostic] Elevated same-parameter score difference: {same_param_diff:.5f} (signed mean: {same_param_diff_raw.item():.5f})\n   This may reflect precision differences between forward evaluations."
            )
        if abs(mc_ref_gap.item()) > 0.15:
            logger.info(
                f"  [Reference score gap] mean_gap={mc_ref_gap.item():.5f}\n   Current scores differ from the frozen reference. Current reference penalty coefficient: {self.beta_kl})"
            )
        kl_loss = self.beta_kl * mean_sq_ref_gap
        kl_loss_val = float(kl_loss.detach().item())
        total_loss = policy_loss + kl_loss
        return (
            total_loss,
            policy_loss.item(),
            same_param_diff.item(),
            mc_ref_gap.item(),
            mean_abs_ref_gap.item(),
            mean_sq_ref_gap.item(),
            p90_abs_ref_gap.item(),
            kl_loss_val,
        )

    @torch.no_grad()
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        G = self.group_size
        B = rewards.shape[0] // G
        rewards_grouped = rewards.reshape(B, G)
        if torch.isnan(rewards_grouped).any() or torch.isinf(rewards_grouped).any():
            logger.error("Rewards contain NaN/Inf; returning zero advantages.")
            return torch.zeros_like(rewards)
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = (rewards_grouped - mean) / (std + 1e-08)
        return advantages.reshape(rewards.shape[0])

    @torch.no_grad()
    def _validate_sequences(self, sequences: torch.Tensor, mask: torch.Tensor) -> Tuple[bool, str]:
        HYDROPHOBIC_IDX = torch.tensor([0, 1, 4, 7, 9, 10, 17, 18], device=sequences.device)
        for i, (seq, m) in enumerate(zip(sequences, mask)):
            valid_seq = seq[m.bool()]
            seq_len = len(valid_seq)
            if seq_len < 5:
                continue
            unique_aa = torch.unique(valid_seq)
            unique_ratio = len(unique_aa) / 21.0
            if unique_ratio < 0.15:
                return (
                    False,
                    f"Sequence {i} has too few distinct amino acids: {len(unique_aa)}/21 ({unique_ratio:.1%})",
                )
            for aa_idx in range(21):
                aa_count = (valid_seq == aa_idx).sum().item()
                aa_ratio = aa_count / seq_len
                if aa_ratio > 0.6:
                    return (
                        False,
                        f"Sequence {i} has a high fraction of amino acid {aa_idx}: {aa_ratio:.1%}",
                    )
            is_hydrophobic = torch.isin(valid_seq, HYDROPHOBIC_IDX)
            hydrophobic_ratio = is_hydrophobic.float().mean().item()
            if hydrophobic_ratio < 0.15:
                return (
                    False,
                    f"Sequence {i} has a low hydrophobic-residue fraction: {hydrophobic_ratio:.1%}",
                )
            if hydrophobic_ratio > 0.85:
                return (
                    False,
                    f"Sequence {i} has a high hydrophobic-residue fraction: {hydrophobic_ratio:.1%}",
                )
            max_repeat = 1
            current_repeat = 1
            for j in range(1, seq_len):
                if valid_seq[j] == valid_seq[j - 1]:
                    current_repeat += 1
                    max_repeat = max(max_repeat, current_repeat)
                else:
                    current_repeat = 1
            if max_repeat > 5:
                return (False, f"Sequence {i} has {max_repeat} identical consecutive amino acids")
        return (True, "Sequence checks passed")

    def train_step(
        self,
        ca_coords: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        cb_coords: Optional[torch.Tensor] = None,
        n_to_ca: Optional[torch.Tensor] = None,
        ca_to_c: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        psi: Optional[torch.Tensor] = None,
        native_sequences: Optional[torch.Tensor] = None,
        esm_fold_fn=None,
    ) -> Dict[str, float]:
        if native_sequences is None:
            raise ValueError(
                "native_sequences is required for native-sequence recovery and perplexity during training."
            )
        B, N, _ = ca_coords.shape
        G = self.group_size
        logger.debug("Step 1: encode backbones in evaluation mode for sampling.")
        self.policy.eval()
        self.hgnn.eval()
        with torch.no_grad():
            with self._autocast():
                hgnn_h_for_sampling = self._encode_backbone(
                    ca_coords, mask, cb_coords, n_to_ca, ca_to_c, phi, psi
                )
        logger.debug("Step 2: sample candidate sequences.")
        sequences = self.policy.sample(
            ca_coords,
            hgnn_h_for_sampling,
            mask,
            temperature=self.temperature,
            n_samples=G,
            n_to_ca=n_to_ca,
            ca_to_c=ca_to_c,
        )
        if sequences is None:
            logger.error(
                "Sampling failed because logits contain too many NaNs; skipping this batch."
            )
            safe_metrics = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "same_param_diff": 0.0,
                "approx_kl": -1.0,
                "mc_ref_gap": 0.0,
                "mean_abs_ref_gap": 0.0,
                "mean_sq_ref_gap": 0.0,
                "p90_abs_ref_gap": 0.0,
                "accuracy": 0.0,
                "perplexity": float("inf"),
                "diversity": 0.0,
                "x_sample_rate": 0.0,
                "grad_norm": 0.0,
                "clip_grad_norm_return": 0.0,
                "kl_loss": 0.0,
                "beta_kl": float(self.beta_kl),
                "skipped": True,
                "skip_reason": "Sampling failed: too many NaN logits.",
                "sctm": 0.0,
                "rmsd": 0.0,
                "rmsd_reward": 0.0,
                "hydrophobic": 0.0,
                "plddt": 0.0,
                "aar": 0.0,
                "total_reward": 0.0,
            }
            return safe_metrics
        ca_rep = ca_coords.repeat_interleave(G, dim=0)
        mask_rep = mask.repeat_interleave(G, dim=0)
        lengths_rep = lengths.repeat_interleave(G, dim=0)
        phi_rep = phi.repeat_interleave(G, dim=0) if phi is not None else None
        psi_rep = psi.repeat_interleave(G, dim=0) if psi is not None else None
        n_to_ca_rep = n_to_ca.repeat_interleave(G, dim=0) if n_to_ca is not None else None
        ca_to_c_rep = ca_to_c.repeat_interleave(G, dim=0) if ca_to_c is not None else None
        logger.debug("Step 2.5: recompute pre-update sequence scores.")
        with torch.no_grad():
            with self._autocast():
                hgnn_h_rep_for_old = hgnn_h_for_sampling.repeat_interleave(G, dim=0)
                seq_logp_old = self._compute_sequence_logp(
                    self.policy,
                    ca_rep,
                    hgnn_h_rep_for_old,
                    mask_rep,
                    sequences,
                    n_to_ca_rep,
                    ca_to_c_rep,
                )
        logger.debug("Step 3: refold candidates with ESMFold.")
        is_valid, invalid_reason = self._validate_sequences(sequences, mask_rep)
        if not is_valid:
            logger.warning(f"Sequence diagnostic: {invalid_reason} (continuing training)")
        if esm_fold_fn is not None:
            fold_result = esm_fold_fn(sequences, lengths_rep)
            refolded_coords = fold_result["ca_coords"].float()
            plddt = fold_result.get("plddt", None)
        else:
            refolded_coords = ca_rep.float().clone()
            plddt = None
            logger.warning(
                "esm_fold_fn was not provided; target coordinates are used as a debugging surrogate."
            )
        has_nan = False
        skip_reason = ""
        if torch.isnan(refolded_coords).any() or torch.isinf(refolded_coords).any():
            logger.warning(f"Refolded coordinates contain NaN/Inf.")
            has_nan = True
            if skip_reason:
                skip_reason += " + refolded coordinates contain NaN/Inf"
            else:
                skip_reason = "ESMFold coordinates contain NaN/Inf"
        if has_nan:
            safe_metrics = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "same_param_diff": 0.0,
                "approx_kl": -1.0,
                "mc_ref_gap": 0.0,
                "mean_abs_ref_gap": 0.0,
                "mean_sq_ref_gap": 0.0,
                "p90_abs_ref_gap": 0.0,
                "accuracy": 0.0,
                "perplexity": float("inf"),
                "diversity": 0.0,
                "x_sample_rate": 0.0,
                "grad_norm": 0.0,
                "clip_grad_norm_return": 0.0,
                "kl_loss": 0.0,
                "beta_kl": float(self.beta_kl),
                "skipped": True,
                "skip_reason": skip_reason,
                "sctm": 0.0,
                "rmsd": 0.0,
                "rmsd_reward": 0.0,
                "hydrophobic": 0.0,
                "plddt": 0.0,
                "aar": 0.0,
                "total_reward": 0.0,
            }
            return safe_metrics
        logger.debug("Step 4: compute rewards.")
        native_sequences_rep = native_sequences.repeat_interleave(G, dim=0)
        with torch.no_grad():
            rewards, reward_info = self.reward_fn(
                sequences=sequences,
                original_coords=ca_rep,
                refolded_coords=refolded_coords,
                lengths=lengths_rep,
                mask=mask_rep,
                plddt=plddt,
                native_sequences=native_sequences_rep,
            )
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            logger.warning(f"Rewards contain NaN/Inf.")
            safe_metrics = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "same_param_diff": 0.0,
                "approx_kl": -1.0,
                "mc_ref_gap": 0.0,
                "mean_abs_ref_gap": 0.0,
                "mean_sq_ref_gap": 0.0,
                "p90_abs_ref_gap": 0.0,
                "accuracy": 0.0,
                "perplexity": float("inf"),
                "diversity": 0.0,
                "x_sample_rate": 0.0,
                "grad_norm": 0.0,
                "clip_grad_norm_return": 0.0,
                "kl_loss": 0.0,
                "beta_kl": float(self.beta_kl),
                "skipped": True,
                "skip_reason": "The reward function returned NaN/Inf",
                **{k: 0.0 for k in reward_info.keys()},
            }
            return safe_metrics
        advantages = self._compute_advantages(rewards)
        logger.debug("Step 5: update the sequence policy and HGNN.")
        self.policy.train()
        self.hgnn.train()
        self.optimizer.zero_grad()
        with self._autocast():
            self.policy.eval()
            self.hgnn.eval()
            hgnn_rep_new = self._encode_backbone(
                ca_rep,
                mask_rep,
                cb_coords.repeat_interleave(G, dim=0) if cb_coords is not None else None,
                n_to_ca.repeat_interleave(G, dim=0) if n_to_ca is not None else None,
                ca_to_c.repeat_interleave(G, dim=0) if ca_to_c is not None else None,
                phi_rep,
                psi_rep,
                use_ref=False,
            )
            seq_logp_new = self._compute_sequence_logp(
                self.policy, ca_rep, hgnn_rep_new, mask_rep, sequences, n_to_ca_rep, ca_to_c_rep
            )
            with torch.no_grad():
                ref_hgnn_h = self._encode_backbone(
                    ca_rep,
                    mask_rep,
                    cb_coords.repeat_interleave(G, dim=0) if cb_coords is not None else None,
                    n_to_ca.repeat_interleave(G, dim=0) if n_to_ca is not None else None,
                    ca_to_c.repeat_interleave(G, dim=0) if ca_to_c is not None else None,
                    phi_rep,
                    psi_rep,
                    use_ref=True,
                )
                seq_logp_ref = self._compute_sequence_logp(
                    self.ref_policy,
                    ca_rep,
                    ref_hgnn_h,
                    mask_rep,
                    sequences,
                    n_to_ca_rep,
                    ca_to_c_rep,
                )
            (
                loss,
                policy_loss_val,
                same_param_diff_val,
                mc_ref_gap_val,
                mean_abs_ref_gap_val,
                mean_sq_ref_gap_val,
                p90_abs_ref_gap_val,
                kl_loss_val,
            ) = self._grpo_loss(seq_logp_new, seq_logp_old, seq_logp_ref, advantages)
            self.policy.train()
            self.hgnn.train()
        self.policy.eval()
        self.hgnn.eval()
        with torch.no_grad():
            with self._autocast():
                hgnn_rep_eval = self._encode_backbone(
                    ca_rep,
                    mask_rep,
                    cb_coords.repeat_interleave(G, dim=0) if cb_coords is not None else None,
                    n_to_ca.repeat_interleave(G, dim=0) if n_to_ca is not None else None,
                    ca_to_c.repeat_interleave(G, dim=0) if ca_to_c is not None else None,
                    phi_rep,
                    psi_rep,
                    use_ref=False,
                )
                log_probs_full = self.policy(
                    ca_rep, hgnn_rep_eval, mask_rep, n_to_ca_rep, ca_to_c_rep
                )
            designed_argmax = log_probs_full.argmax(dim=-1)
            accuracy = compute_sequence_accuracy(designed_argmax, sequences, mask_rep).mean().item()
            perplexity = (
                compute_perplexity(log_probs_full, native_sequences_rep, mask_rep).mean().item()
            )
            diversity = compute_diversity(sequences, mask_rep, self.group_size).mean().item()
            x_count = ((sequences == 20).float() * mask_rep).sum()
            valid_count = mask_rep.sum().clamp(min=1.0)
            x_sample_rate = (x_count / valid_count).item()
        self.policy.train()
        self.hgnn.train()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self._call_counter += 1
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float("inf")
        num_params_with_grad = 0
        num_params_total = 0
        policy_norm = 0.0
        hgnn_norm = 0.0
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        adv_max = advantages.max().item()
        adv_min = advantages.min().item()
        ratio = torch.exp(seq_logp_new.float() - seq_logp_old.detach().float())
        ratio_mean = ratio.mean().item()
        ratio_std = ratio.std().item()
        ratio_clipped = (
            ((ratio < 1.0 - self.clip_range) | (ratio > 1.0 + self.clip_range))
            .float()
            .mean()
            .item()
        )
        for name, p in self.policy.named_parameters():
            num_params_total += 1
            if p.grad is not None:
                num_params_with_grad += 1
                param_norm = p.grad.data.norm(2).item()
                policy_norm += param_norm**2
                total_norm += param_norm**2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                min_grad = min(min_grad, p.grad.data.abs().min().item())
        for name, p in self.hgnn.named_parameters():
            num_params_total += 1
            if p.grad is not None:
                num_params_with_grad += 1
                param_norm = p.grad.data.norm(2).item()
                hgnn_norm += param_norm**2
                total_norm += param_norm**2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                min_grad = min(min_grad, p.grad.data.abs().min().item())
        total_norm = total_norm**0.5
        policy_norm = policy_norm**0.5
        hgnn_norm = hgnn_norm**0.5
        has_nan_grad = False
        if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
            logger.error(f"Gradients contain NaN/Inf; skipping the optimizer update.")
            has_nan_grad = True
        elif total_norm < 1e-08:
            logger.warning(
                f"Very small gradient norm: {total_norm:.2e} (Policy: {policy_norm:.2e}, HGNN: {hgnn_norm:.2e})\n   Parameters with gradients: {num_params_with_grad}/{num_params_total}\n   Advantage statistics: mean={adv_mean:.4f}, std={adv_std:.4f}, range=[{adv_min:.4f}, {adv_max:.4f}]\n   Geometric score ratio: mean={ratio_mean:.4f}, std={ratio_std:.4f}, clipped={ratio_clipped * 100:.1f}%\n   Continuing the optimizer update."
            )
        elif total_norm > self.max_grad_norm * 3:
            logger.warning(
                f"Large gradient norm: {total_norm:.4f} (clipping threshold: {self.max_grad_norm}, maximum absolute gradient: {max_grad:.2e})\n   Policy gradient norm: {policy_norm:.4f}, HGNN gradient norm: {hgnn_norm:.4f}\n   Advantage statistics: mean={adv_mean:.4f}, std={adv_std:.4f}\n   Gradient clipping will be applied."
            )
        elif self._call_counter % 50 == 0:
            logger.info(
                f"  [Gradient diagnostic] Total norm: {total_norm:.4f} (Policy: {policy_norm:.4f}, HGNN: {hgnn_norm:.4f}), maximum: {max_grad:.2e}, minimum: {min_grad:.2e}\n  [Advantage diagnostic] mean={adv_mean:.4f}, std={adv_std:.4f}, range=[{adv_min:.4f}, {adv_max:.4f}]\n  [Geometric score ratio] mean={ratio_mean:.4f}, std={ratio_std:.4f}, clipped={ratio_clipped * 100:.1f}%"
            )
        if has_nan_grad:
            self.scaler.update()
            skip_reason = "Gradients contain NaN/Inf"
            logger.warning(
                f"Call #{self._call_counter}: skipping this batch ({skip_reason})\n   Completed optimizer steps: {self._success_counter}"
            )
            safe_metrics = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "same_param_diff": 0.0,
                "approx_kl": -1.0,
                "mc_ref_gap": 0.0,
                "mean_abs_ref_gap": 0.0,
                "mean_sq_ref_gap": 0.0,
                "p90_abs_ref_gap": 0.0,
                "accuracy": 0.0,
                "perplexity": float("inf"),
                "diversity": 0.0,
                "x_sample_rate": 0.0,
                "grad_norm": total_norm,
                "clip_grad_norm_return": 0.0,
                "kl_loss": kl_loss_val,
                "beta_kl": float(self.beta_kl),
                "skipped": True,
                "skip_reason": skip_reason,
                "sctm": 0.0,
                "rmsd": 0.0,
                "rmsd_reward": 0.0,
                "hydrophobic": 0.0,
                "plddt": 0.0,
                "aar": 0.0,
                "total_reward": 0.0,
            }
            return safe_metrics
        clip_grad_norm_return = nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.hgnn.parameters()), self.max_grad_norm
        )
        clip_grad_norm_return_val = float(
            clip_grad_norm_return.item()
            if hasattr(clip_grad_norm_return, "item")
            else clip_grad_norm_return
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._success_counter += 1
        approx_kl_post_step = None
        if self._success_counter % 50 == 0:
            with torch.no_grad():
                with self._autocast():
                    self.policy.eval()
                    self.hgnn.eval()
                    hgnn_rep_post = self._encode_backbone(
                        ca_rep,
                        mask_rep,
                        cb_coords.repeat_interleave(G, dim=0) if cb_coords is not None else None,
                        n_to_ca.repeat_interleave(G, dim=0) if n_to_ca is not None else None,
                        ca_to_c.repeat_interleave(G, dim=0) if ca_to_c is not None else None,
                        phi_rep,
                        psi_rep,
                        use_ref=False,
                    )
                    seq_logp_post = self._compute_sequence_logp(
                        self.policy,
                        ca_rep,
                        hgnn_rep_post,
                        mask_rep,
                        sequences,
                        n_to_ca_rep,
                        ca_to_c_rep,
                    )
                    logp_diff_post = seq_logp_old.float() - seq_logp_post.float()
                    approx_kl_post_step = logp_diff_post.abs().mean().item()
                    self.policy.train()
                    self.hgnn.train()
            logger.info(
                f"  [Update diagnostic] Mean absolute score change (post-step)={approx_kl_post_step:.5f} (pre-update to post-update)\n  [Numerical diagnostic] Same-parameter score difference={same_param_diff_val:.5f}\n  Interpret this diagnostic together with precision and gradient statistics."
            )
        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss_val,
            "same_param_diff": same_param_diff_val,
            "approx_kl": approx_kl_post_step if approx_kl_post_step is not None else -1.0,
            "mc_ref_gap": mc_ref_gap_val,
            "mean_abs_ref_gap": mean_abs_ref_gap_val,
            "mean_sq_ref_gap": mean_sq_ref_gap_val,
            "p90_abs_ref_gap": p90_abs_ref_gap_val,
            "kl_loss": kl_loss_val,
            "beta_kl": float(self.beta_kl),
            "accuracy": accuracy,
            "perplexity": perplexity,
            "diversity": diversity,
            "x_sample_rate": x_sample_rate,
            "grad_norm": total_norm,
            "clip_grad_norm_return": clip_grad_norm_return_val,
            **reward_info,
        }
        return metrics
