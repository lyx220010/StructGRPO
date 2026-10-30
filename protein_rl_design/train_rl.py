import os
import csv
import math
import atexit

RESUME_FROM = None
PRETRAIN_FROM = None


def setup_gpu():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if "gpu_id" in line and ":" in line and ("str" in line) and ("=" in line):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        value = parts[1].strip()
                        value = value.split("#")[0].strip()
                        value = value.strip('"').strip("'")
                        if value:
                            os.environ["CUDA_VISIBLE_DEVICES"] = value
                            print(f"[GPU setup] config.py physical GPU {value} -> logical cuda:0")
                            return
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print(f"[GPU setup] No gpu_id found in config.py; using physical GPU 1 -> logical cuda:0")
    except Exception as e:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print(
            f"[GPU setup] Configuration parsing failed ({e}); using physical GPU 1 -> logical cuda:0"
        )


setup_gpu()
import copy
import logging
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_original_torch_tensor = torch.tensor


def _patched_torch_tensor(*args, **kwargs):
    if args and hasattr(args[0], "dtype") and hasattr(args[0].dtype, "type"):
        import numpy as np

        if isinstance(args[0], (np.ndarray, np.generic)):
            if "dtype" not in kwargs:
                np_dtype = args[0].dtype
                if np_dtype == np.int64:
                    kwargs["dtype"] = torch.int64
                elif np_dtype == np.int32:
                    kwargs["dtype"] = torch.int32
                elif np_dtype == np.float64:
                    kwargs["dtype"] = torch.float64
                elif np_dtype == np.float32:
                    kwargs["dtype"] = torch.float32
    return _original_torch_tensor(*args, **kwargs)


torch.tensor = _patched_torch_tensor
from protein_rl_design.config import TrainConfig
from protein_rl_design.models.hgnn_encoder import HGNNProteinEncoder
from protein_rl_design.models.sequence_policy import SequenceDesignPolicy
from protein_rl_design.reward.reward_fn import RewardFunction
from protein_rl_design.rl.grpo_trainer import GRPOTrainer
from protein_rl_design.utils.esm_fold_wrapper import ESMFoldWrapper
from protein_rl_design.utils.checkpoint import load_model_weights, save_pretrain_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info("Installed the NumPy dtype compatibility wrapper for torch.tensor.")


def _format_approx_kl(approx_kl: float) -> str:
    if approx_kl < 0:
        return "N/A (computed every 50 optimizer steps)"
    status = ""
    if approx_kl > 0.05:
        status = "large score change"
    elif approx_kl < 0.0001:
        status = "small score change"
    else:
        status = "normal"
    return f"{approx_kl:.5f}  {status}"


def validate_batch_data(
    ca_coords: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
) -> Tuple[bool, str]:
    B, N, _ = ca_coords.shape
    if torch.isnan(ca_coords).any():
        nan_count = torch.isnan(ca_coords).sum().item()
        return (False, f"Coordinates contain {nan_count} NaN values")
    if torch.isinf(ca_coords).any():
        inf_count = torch.isinf(ca_coords).sum().item()
        return (False, f"Coordinates contain {inf_count} Inf values")
    max_coord = ca_coords.abs().max().item()
    if max_coord > 1000.0:
        return (False, f"Coordinate magnitude is too large: {max_coord:.1f} Angstrom")
    for i in range(B):
        valid_mask = mask[i].bool()
        valid_coords = ca_coords[i][valid_mask]
        if len(valid_coords) == 0:
            return (False, f"Sample {i} has no valid coordinates")
        if len(valid_coords) < 3:
            return (
                False,
                f"Sample {i} has too few valid residues: {len(valid_coords)} (at least 3 are required by this validation routine)",
            )
        if len(valid_coords) != lengths[i].item():
            return (
                False,
                f"Sample {i} has inconsistent lengths: mask={len(valid_coords)}, lengths={lengths[i].item()}",
            )
        if len(valid_coords) > 1:
            dists = torch.norm(valid_coords[1:] - valid_coords[:-1], dim=-1)
            min_dist = dists.min().item()
            max_dist = dists.max().item()
            if min_dist < 0.5:
                return (
                    False,
                    f"Sample {i} has an adjacent CA distance that is too small: {min_dist:.2f} Angstrom (< 0.5; overlapping coordinates)",
                )
    return (True, "Batch checks passed")


class BackboneDataset(Dataset):

    def __init__(self, data_dir: str, max_len: int = 300):
        input_path = Path(data_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"Training data directory does not exist: {input_path}")
        self.files = sorted((path for path in input_path.glob("*.pt") if path.is_file()))
        self.max_len = max_len
        if len(self.files) == 0:
            raise ValueError(f"No .pt training files were found in {input_path}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        actual_file_idx = idx % len(self.files)
        pt_path = self.files[actual_file_idx]
        data = torch.load(pt_path, map_location="cpu")
        N = int(data["length"])
        if N > self.max_len:
            raise ValueError(f"{pt_path.name}: length {N} exceeds limit {self.max_len}; truncation is disabled.")
        result = {
            "ca_coords": data["ca_coords"][:N],
            "mask": data["mask"][:N],
            "length": N,
            "sample_idx": idx,
            "file_idx": actual_file_idx,
            "target_id": str(
                data.get("target_id", data.get("domain_id", data.get("cath_domain", pt_path.stem)))
            ),
            "source_file": pt_path.name,
        }
        if "sequence" in data:
            result["sequence"] = data["sequence"][:N]
        for key in ["n_coords", "c_coords", "cb_coords", "n_to_ca", "ca_to_c", "phi", "psi"]:
            if key in data:
                result[key] = data[key][:N]
        return result


def collate_fn(batch):
    max_len = max((item["length"] for item in batch))
    B = len(batch)
    ca_coords = torch.zeros(B, max_len, 3)
    mask = torch.zeros(B, max_len)
    lengths = torch.zeros(B, dtype=torch.long)
    sample_indices = torch.zeros(B, dtype=torch.long)
    file_indices = torch.zeros(B, dtype=torch.long)
    has_seq = all(("sequence" in item for item in batch))
    sequences = torch.full((B, max_len), -1, dtype=torch.long) if has_seq else None
    required_geom_keys = ["n_coords", "c_coords", "cb_coords", "n_to_ca", "ca_to_c", "phi", "psi"]
    has_geom = all((all((key in item for key in required_geom_keys)) for item in batch))
    if has_geom:
        n_coords = torch.zeros(B, max_len, 3)
        c_coords = torch.zeros(B, max_len, 3)
        cb_coords = torch.zeros(B, max_len, 3)
        n_to_ca = torch.zeros(B, max_len, 3)
        ca_to_c = torch.zeros(B, max_len, 3)
        phi = torch.zeros(B, max_len)
        psi = torch.zeros(B, max_len)
    target_ids: List[str] = []
    source_files: List[str] = []
    for i, item in enumerate(batch):
        N = item["length"]
        ca_coords[i, :N] = item["ca_coords"]
        mask[i, :N] = item["mask"]
        lengths[i] = N
        sample_indices[i] = item.get("sample_idx", -1)
        file_indices[i] = item.get("file_idx", -1)
        target_ids.append(str(item.get("target_id", f"sample_{item.get('sample_idx', i)}")))
        source_files.append(str(item.get("source_file", "")))
        if has_seq:
            sequences[i, :N] = item["sequence"]
        if has_geom:
            n_coords[i, :N] = item["n_coords"]
            c_coords[i, :N] = item["c_coords"]
            cb_coords[i, :N] = item["cb_coords"]
            n_to_ca[i, :N] = item["n_to_ca"]
            ca_to_c[i, :N] = item["ca_to_c"]
            phi[i, :N] = item["phi"]
            psi[i, :N] = item["psi"]
    result = {
        "ca_coords": ca_coords,
        "mask": mask,
        "lengths": lengths,
        "sample_indices": sample_indices,
        "file_indices": file_indices,
        "target_id": target_ids,
        "source_file": source_files,
    }
    if has_seq:
        result["sequence"] = sequences
    if has_geom:
        result.update(
            {
                "n_coords": n_coords,
                "c_coords": c_coords,
                "cb_coords": cb_coords,
                "n_to_ca": n_to_ca,
                "ca_to_c": ca_to_c,
                "phi": phi,
                "psi": psi,
            }
        )
    return result


def build_models(cfg: TrainConfig):
    hgnn = HGNNProteinEncoder(
        in_dim=3,
        hidden_dim=cfg.hgnn.hidden_dim,
        out_dim=cfg.hgnn.out_dim,
        k_neighbors=cfg.hgnn.k_neighbors,
        contact_threshold=cfg.hgnn.contact_threshold,
        dropout=cfg.hgnn.dropout,
        use_rich_features=True,
        use_attention=cfg.hgnn.use_attention,
        num_attention_heads=cfg.hgnn.num_attention_heads,
        attention_dropout=cfg.hgnn.attention_dropout,
    )
    policy = SequenceDesignPolicy(
        hgnn_out_dim=cfg.hgnn.out_dim,
        hidden_dim=cfg.mpnn.hidden_dim,
        num_enc_layers=cfg.mpnn.num_encoder_layers,
        num_dec_layers=cfg.mpnn.num_decoder_layers,
        k_neighbors=cfg.mpnn.k_neighbors,
        vocab_size=cfg.mpnn.vocab_size,
        dropout=cfg.mpnn.dropout,
        use_online_rel_orient=cfg.mpnn.use_online_rel_orient,
    )
    if cfg.gradient_checkpointing:
        if hasattr(hgnn, "gradient_checkpointing_enable"):
            hgnn.gradient_checkpointing_enable()
        if hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()
    reward_fn = RewardFunction(
        w_sctm=cfg.rl.w_sctm,
        w_rmsd=cfg.rl.w_rmsd,
        w_hydrophobic=cfg.rl.w_hydrophobic,
        w_plddt=cfg.rl.w_plddt,
        w_aar=cfg.rl.w_aar,
    )
    return (hgnn, policy, reward_fn)


def _sync_ref_models(policy, hgnn, ref_policy, ref_hgnn) -> None:
    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()
    ref_hgnn.load_state_dict(hgnn.state_dict())
    for p in ref_hgnn.parameters():
        p.requires_grad_(False)
    ref_hgnn.eval()


def _restore_frozen_references(checkpoint, ref_policy, ref_hgnn, pretrain_checkpoint, device):
    if "ref_policy" in checkpoint and "ref_hgnn" in checkpoint:
        ref_policy.load_state_dict(checkpoint["ref_policy"])
        ref_hgnn.load_state_dict(checkpoint["ref_hgnn"])
        logger.info("Restored the frozen reference models from the training checkpoint.")
    else:
        if not pretrain_checkpoint:
            raise ValueError(
                "The checkpoint does not contain both frozen reference models. Provide the original supervised pretraining checkpoint through PRETRAIN_FROM or config.pretrain_ckpt_path to resume training."
            )
        load_model_weights(pretrain_checkpoint, ref_hgnn, ref_policy, device=device)
        logger.info(
            "Restored the frozen reference models from the supplied pretraining checkpoint."
        )
    for model in (ref_policy, ref_hgnn):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()


def train(cfg: TrainConfig, resume_from: str = None, pretrain_from: str = None):
    cfg.rl.beta_kl = 2.0
    cfg.rl.kl_ref_logp_gap_clamp = 0.5
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            f"Using cuda (logical GPU 0, physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')})"
        )
    elif cfg.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info(f"Using cpu")
    else:
        logger.warning(
            f"Detected cfg.device={cfg.device}. Using cuda on logical GPU 0. Set device=cuda and select physical devices with gpu_id in config.py."
        )
        device = torch.device("cuda")
    hgnn, policy, reward_fn = build_models(cfg)
    hgnn = hgnn.to(device)
    policy = policy.to(device)
    reward_fn = reward_fn.to(device)
    ref_policy = copy.deepcopy(policy).to(device)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()
    ref_hgnn = copy.deepcopy(hgnn).to(device)
    for p in ref_hgnn.parameters():
        p.requires_grad_(False)
    ref_hgnn.eval()
    policy_params = list(policy.parameters()) + list(hgnn.parameters())
    optimizer = torch.optim.AdamW(policy_params, lr=cfg.rl.lr, weight_decay=0.0001)
    total_steps = int(cfg.total_steps)
    if cfg.rl.use_warmup_cosine_lr:
        warmup_steps = min(300, max(50, int(total_steps * 0.03)))
        init_lr = float(cfg.rl.lr)
        peak_lr = float(cfg.rl.peak_lr)
        final_lr = float(cfg.rl.final_lr)
        peak_ratio = peak_lr / init_lr
        final_ratio = final_lr / init_lr

        def lr_lambda(step):
            if step < warmup_steps:
                alpha = step / max(1, warmup_steps)
                return 1.0 + (peak_ratio - 1.0) * alpha
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (peak_ratio - final_ratio) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        logger.info(
            f"RL learning-rate schedule: warmup and cosine (initial={init_lr:.2e}, peak={peak_lr:.2e}, final={final_lr:.2e}, warmup_steps={warmup_steps})"
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        logger.info(f"RL learning-rate schedule: constant (lr={cfg.rl.lr:.2e})")
    start_step = 0
    start_batch_count = 0
    start_skip_count = 0
    if resume_from:
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        start_step = int(checkpoint["step"])
        if start_step >= total_steps:
            logger.info(
                f"Checkpoint already contains {start_step} successful updates; the requested total is {total_steps}. No further training is needed."
            )
            return
        policy.load_state_dict(checkpoint["policy"])
        hgnn.load_state_dict(checkpoint["hgnn"])
        _restore_frozen_references(
            checkpoint, ref_policy, ref_hgnn, pretrain_from or cfg.pretrain_ckpt_path, device
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_batch_count = checkpoint.get("batch_count", start_step)
        start_skip_count = checkpoint.get("skip_count", 0)
        loaded_problematic_batches = checkpoint.get("problematic_batches", {})
        num_problematic = len(loaded_problematic_batches)
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            for _ in range(start_step):
                scheduler.step()
        logger.info(f"Checkpoint loaded; continuing from step {start_step}")
        logger.info(
            f"  Batch statistics: {start_batch_count} batches ({start_step} optimizer steps, {start_skip_count} skipped)"
        )
        logger.info(f"  Current learning rate: {scheduler.get_last_lr()[0]:.2e}")
        if num_problematic > 0:
            top_problematic = sorted(
                loaded_problematic_batches.items(), key=lambda x: x[1], reverse=True
            )[:5]
            problematic_info = ", ".join(
                (f"sample_{idx}({count} occurrences)" for idx, count in top_problematic)
            )
            logger.info(f"  Recorded problematic samples: {num_problematic} [{problematic_info}]")
    else:
        loaded_problematic_batches = {}
    dataset = BackboneDataset(cfg.data_dir, max_len=getattr(cfg, "max_seq_len", 300))
    esm = ESMFoldWrapper(
        model_name=cfg.esm_fold_model,
        local_model_path=cfg.esm_fold_local_path,
        device="cuda" if device.type == "cuda" else "cpu",
        chunk_size=64,
    )

    def esm_fold_fn(sequences, lengths):
        return esm.fold(sequences, lengths, max_batch=cfg.esm_fold_batch_size)

    trainer = GRPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        hgnn_encoder=hgnn,
        ref_hgnn_encoder=ref_hgnn,
        reward_fn=reward_fn,
        optimizer=optimizer,
        group_size=cfg.rl.group_size,
        clip_range=cfg.rl.clip_range,
        beta_kl=cfg.rl.beta_kl,
        max_grad_norm=cfg.rl.max_grad_norm,
        kl_ref_logp_gap_clamp=cfg.rl.kl_ref_logp_gap_clamp,
        temperature=cfg.rl.temperature,
        mixed_precision=cfg.mixed_precision,
        device="cuda" if device.type == "cuda" else "cpu",
    )
    logger.info(f"Current reference penalty coefficient: {trainer.beta_kl:.4f}")
    trainer._call_counter = start_batch_count
    trainer._success_counter = start_step
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    pretrain_dir = cfg.pretrain_output_dir or cfg.output_dir
    os.makedirs(pretrain_dir, exist_ok=True)
    train_metrics_csv_path = os.path.join(cfg.output_dir, "train_metrics.csv")
    skip_events_csv_path = os.path.join(cfg.output_dir, "skip_events.csv")
    train_metrics_headers = [
        "batch_count",
        "epoch",
        "batch_idx",
        "global_step",
        "total_steps",
        "skipped",
        "skip_reason",
        "skip_count",
        "skip_rate",
        "lr",
        "loss",
        "policy_loss",
        "approx_kl",
        "mc_ref_gap",
        "mean_abs_ref_gap",
        "mean_sq_ref_gap",
        "p90_abs_ref_gap",
        "grad_norm",
        "clip_grad_norm_return",
        "beta_kl",
        "kl_loss",
        "accuracy",
        "perplexity",
        "diversity",
        "x_sample_rate",
        "sctm",
        "rmsd",
        "rmsd_reward",
        "hydrophobic",
        "plddt",
        "aar",
        "total_reward",
    ]
    skip_events_headers = [
        "batch_count",
        "epoch",
        "batch_idx",
        "global_step",
        "sample_idx",
        "file_idx",
        "problem_count",
        "skip_reason",
    ]

    def _truncate_csv_to_step(csv_path, headers, step_col, max_step):
        kept_rows = []
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if int(row[step_col]) <= max_step:
                            kept_rows.append(row)
                    except (ValueError, KeyError):
                        pass
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows([{h: row.get(h, "") for h in headers} for row in kept_rows])
        return len(kept_rows)

    if resume_from:
        kept = _truncate_csv_to_step(
            train_metrics_csv_path, train_metrics_headers, "global_step", start_step
        )
        _truncate_csv_to_step(skip_events_csv_path, skip_events_headers, "global_step", start_step)
        logger.info(f"Training CSV truncated to step {start_step}; retained {kept} records")
    csv_mode = "a" if resume_from else "w"
    train_metrics_file = open(train_metrics_csv_path, csv_mode, newline="", encoding="utf-8")
    skip_events_file = open(skip_events_csv_path, csv_mode, newline="", encoding="utf-8")
    train_metrics_writer = csv.DictWriter(train_metrics_file, fieldnames=train_metrics_headers)
    skip_events_writer = csv.DictWriter(skip_events_file, fieldnames=skip_events_headers)

    def _close_csv_files():
        for f in (train_metrics_file, skip_events_file):
            if f is not None and (not f.closed):
                f.close()

    atexit.register(_close_csv_files)
    if not resume_from:
        train_metrics_writer.writeheader()
        train_metrics_file.flush()
        skip_events_writer.writeheader()
        skip_events_file.flush()
    pretrain_ckpt = pretrain_from or cfg.pretrain_ckpt_path
    if pretrain_ckpt and (not resume_from):
        logger.info(f"Loading supervised weights and skipping pretraining: {pretrain_ckpt}")
        loaded = load_model_weights(pretrain_ckpt, hgnn, policy, device=device)
        _sync_ref_models(policy, hgnn, ref_policy, ref_hgnn)
        logger.info(
            f"Supervised weights loaded (type={loaded.get('_loaded_type')}, step={loaded.get('step', 'unknown')})"
        )
    elif cfg.pretrain_steps > 0 and (not resume_from):
        logger.info(f"Starting supervised pretraining for {cfg.pretrain_steps} steps.")
        pretrain_optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(hgnn.parameters()),
            lr=5e-06,
            betas=(0.9, 0.98),
            eps=1e-09,
        )
        pretrain_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            pretrain_optimizer,
            max_lr=0.005,
            total_steps=cfg.pretrain_steps,
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=1000,
            final_div_factor=1.0,
        )
        pretrain_loader = DataLoader(
            dataset,
            batch_size=cfg.pretrain_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        pretrain_step = 0
        pretrain_skip_count = 0
        pretrain_iter = iter(pretrain_loader)
        pretrain_pass_start_step = pretrain_step
        while pretrain_step < cfg.pretrain_steps:
            try:
                batch = next(pretrain_iter)
            except StopIteration:
                if pretrain_step == pretrain_pass_start_step:
                    _close_csv_files()
                    raise RuntimeError(
                        "An entire pretraining pass completed without a successful update. Check input geometry and batch validation failures."
                    )
                pretrain_pass_start_step = pretrain_step
                pretrain_iter = iter(pretrain_loader)
                batch = next(pretrain_iter)
            ca_coords = batch["ca_coords"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            native_seq = batch.get("sequence")
            is_valid, invalid_reason = validate_batch_data(ca_coords, mask, lengths)
            if not is_valid:
                pretrain_skip_count += 1
                logger.warning(
                    f"Pretraining step {pretrain_step}: batch validation failed (skip {pretrain_skip_count})\n  Reason: {invalid_reason}\n  Skipping this pretraining batch."
                )
                continue
            cb_coords = batch.get("cb_coords", None)
            n_to_ca = batch.get("n_to_ca", None)
            ca_to_c = batch.get("ca_to_c", None)
            phi = batch.get("phi", None)
            psi = batch.get("psi", None)
            if cb_coords is not None:
                cb_coords = cb_coords.to(device)
            if n_to_ca is not None:
                n_to_ca = n_to_ca.to(device)
            if ca_to_c is not None:
                ca_to_c = ca_to_c.to(device)
            if phi is not None:
                phi = phi.to(device)
            if psi is not None:
                psi = psi.to(device)
            if native_seq is None:
                raise ValueError("Native sequence records are required for supervised pretraining.")
            native_seq = native_seq.to(device)
            pretrain_optimizer.zero_grad()
            hgnn_h = hgnn(ca_coords, mask, cb_coords, n_to_ca, ca_to_c, phi, psi)
            log_probs = policy(ca_coords, hgnn_h, mask, n_to_ca, ca_to_c)
            B, N, V = log_probs.shape
            target = native_seq.clone()
            valid_native = (mask > 0.5) & (target >= 0) & (target != 20)
            target = torch.where(valid_native, target, torch.full_like(target, -1))
            loss = torch.nn.functional.nll_loss(
                log_probs.reshape(-1, V), target.reshape(-1), ignore_index=-1, reduction="none"
            )
            valid = valid_native.reshape(-1).float()
            loss = (loss * valid).sum() / valid.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(hgnn.parameters()), 1.0)
            pretrain_optimizer.step()
            pretrain_scheduler.step()
            pretrain_step += 1
            if cfg.pretrain_save_every > 0 and pretrain_step % cfg.pretrain_save_every == 0:
                ckpt_path = os.path.join(pretrain_dir, f"pretrain_step{pretrain_step}.pt")
                save_pretrain_checkpoint(ckpt_path, policy, hgnn, pretrain_step)
                logger.info(f"  Pretraining checkpoint saved: {ckpt_path}")
            if pretrain_step % 100 == 0:
                with torch.no_grad():
                    pred_argmax = log_probs.argmax(dim=-1)
                    valid_mask = mask > 0.5
                    argmax_x = ((pred_argmax == 20) & valid_mask).sum().item()
                    valid_n = max(int(valid_mask.sum().item()), 1)
                    argmax_x_rate = argmax_x / valid_n
                logger.info(
                    f"  Pretraining step {pretrain_step}/{cfg.pretrain_steps}  loss={loss.item():.4f}  lr={pretrain_optimizer.param_groups[0]['lr']:.2e}  argmax_X_rate={argmax_x_rate:.2e}"
                )
        if pretrain_step == 0:
            raise RuntimeError(
                "Supervised pretraining completed no valid optimizer steps. Check that input files contain sequence records."
            )
        final_pretrain_path = os.path.join(pretrain_dir, cfg.pretrain_save_name)
        save_pretrain_checkpoint(final_pretrain_path, policy, hgnn, pretrain_step)
        logger.info(
            f"Supervised weights saved to {final_pretrain_path}; starting GRPO-inspired fine-tuning."
        )
        _sync_ref_models(policy, hgnn, ref_policy, ref_hgnn)
        logger.info("Reference policy and HGNN synchronized with the pretrained models.")
    global_step = start_step
    batch_count = start_batch_count
    skip_count = start_skip_count
    problematic_samples = loaded_problematic_batches if resume_from else {}

    def _save_checkpoint(checkpoint_path):
        torch.save(
            {
                "step": global_step,
                "batch_count": batch_count,
                "skip_count": skip_count,
                "problematic_batches": problematic_samples,
                "policy": policy.state_dict(),
                "hgnn": hgnn.state_dict(),
                "ref_policy": ref_policy.state_dict(),
                "ref_hgnn": ref_hgnn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "beta_kl": trainer.beta_kl,
                "kl_ref_logp_gap_clamp": trainer.kl_ref_logp_gap_clamp,
            },
            checkpoint_path,
        )

    if start_step > 0:
        logger.info(f"Continuing GRPO-inspired fine-tuning from step {start_step}")
    else:
        logger.info("Starting GRPO-inspired fine-tuning.")
    last_logged_temperature = None
    logger.info("=" * 80)
    epoch = 0
    while global_step < total_steps:
        pass_start_step = global_step
        for batch_idx, batch in enumerate(dataloader):
            ca_coords = batch["ca_coords"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            sample_indices = batch["sample_indices"]
            file_indices = batch["file_indices"]
            is_valid, invalid_reason = validate_batch_data(ca_coords, mask, lengths)
            if not is_valid:
                skip_count += 1
                batch_count += 1
                skip_rate = skip_count / batch_count * 100
                skip_reason = f"Batch validation failed: {invalid_reason}"
                for i, sample_idx in enumerate(sample_indices.tolist()):
                    if sample_idx not in problematic_samples:
                        problematic_samples[sample_idx] = 0
                    problematic_samples[sample_idx] += 1
                    skip_events_writer.writerow(
                        {
                            "batch_count": batch_count,
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                            "sample_idx": sample_idx,
                            "file_idx": (
                                int(file_indices[i].item()) if i < len(file_indices) else -1
                            ),
                            "problem_count": problematic_samples[sample_idx],
                            "skip_reason": skip_reason,
                        }
                    )
                train_metrics_writer.writerow(
                    {
                        "batch_count": batch_count,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "skipped": True,
                        "skip_reason": skip_reason,
                        "skip_count": skip_count,
                        "skip_rate": skip_rate,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": 0.0,
                        "policy_loss": 0.0,
                        "approx_kl": -1.0,
                        "mc_ref_gap": 0.0,
                        "mean_abs_ref_gap": 0.0,
                        "mean_sq_ref_gap": 0.0,
                        "p90_abs_ref_gap": 0.0,
                        "grad_norm": 0.0,
                        "clip_grad_norm_return": 0.0,
                        "beta_kl": float(trainer.beta_kl),
                        "kl_loss": 0.0,
                        "accuracy": 0.0,
                        "perplexity": float("inf"),
                        "diversity": 0.0,
                        "sctm": 0.0,
                        "rmsd": 0.0,
                        "rmsd_reward": 0.0,
                        "hydrophobic": 0.0,
                        "plddt": 0.0,
                        "aar": 0.0,
                        "total_reward": 0.0,
                    }
                )
                skip_events_file.flush()
                train_metrics_file.flush()
                logger.warning(
                    f"\n{'=' * 60}\nWarning: step {global_step} (Batch {batch_count}) failed batch validation (skip {skip_count})\n{'=' * 60}\n  Reason: {invalid_reason}\n  Skipping this batch before model evaluation. The optimizer-step count remains {global_step})\n  Skip rate: {skip_rate:.1f}% ({skip_count}/{batch_count})\n  Affected sample indices: {sample_indices.tolist()}\n  Inspect the affected input files.\n{'=' * 60}\n"
                )
                continue
            cb_coords = batch.get("cb_coords", None)
            n_to_ca = batch.get("n_to_ca", None)
            ca_to_c = batch.get("ca_to_c", None)
            phi = batch.get("phi", None)
            psi = batch.get("psi", None)
            native_sequences = batch.get("sequence", None)
            if cb_coords is not None:
                cb_coords = cb_coords.to(device)
            if n_to_ca is not None:
                n_to_ca = n_to_ca.to(device)
            if ca_to_c is not None:
                ca_to_c = ca_to_c.to(device)
            if phi is not None:
                phi = phi.to(device)
            if psi is not None:
                psi = psi.to(device)
            if native_sequences is not None:
                native_sequences = native_sequences.to(device)
            if cfg.rl.use_temperature_schedule:
                interval = max(1, int(cfg.rl.temperature_decay_interval))
                stage = global_step // interval
                scheduled_temperature = max(
                    float(cfg.rl.min_temperature),
                    float(cfg.rl.temperature) - float(cfg.rl.temperature_decay) * stage,
                )
            else:
                scheduled_temperature = float(cfg.rl.temperature)
            trainer.temperature = scheduled_temperature
            if (
                last_logged_temperature is None
                or abs(last_logged_temperature - scheduled_temperature) > 1e-12
            ):
                logger.info(
                    f"Sampling-temperature schedule: step={global_step} temperature={scheduled_temperature:.3f} (base={cfg.rl.temperature:.3f}, stage={global_step // max(1, int(cfg.rl.temperature_decay_interval))})"
                )
                last_logged_temperature = scheduled_temperature
            metrics = trainer.train_step(
                ca_coords=ca_coords,
                mask=mask,
                lengths=lengths,
                cb_coords=cb_coords,
                n_to_ca=n_to_ca,
                ca_to_c=ca_to_c,
                phi=phi,
                psi=psi,
                native_sequences=native_sequences,
                esm_fold_fn=esm_fold_fn,
            )
            if metrics.get("skipped", False):
                skip_count += 1
                batch_count += 1
                skip_rate = skip_count / batch_count * 100
                skip_reason = metrics.get("skip_reason", "unknown")
                for i, sample_idx in enumerate(sample_indices.tolist()):
                    if sample_idx not in problematic_samples:
                        problematic_samples[sample_idx] = 0
                    problematic_samples[sample_idx] += 1
                    skip_events_writer.writerow(
                        {
                            "batch_count": batch_count,
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                            "sample_idx": sample_idx,
                            "file_idx": (
                                int(file_indices[i].item()) if i < len(file_indices) else -1
                            ),
                            "problem_count": problematic_samples[sample_idx],
                            "skip_reason": skip_reason,
                        }
                    )
                train_metrics_writer.writerow(
                    {
                        "batch_count": batch_count,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "skipped": True,
                        "skip_reason": skip_reason,
                        "skip_count": skip_count,
                        "skip_rate": skip_rate,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": metrics.get("loss", 0.0),
                        "policy_loss": metrics.get("policy_loss", 0.0),
                        "approx_kl": metrics.get("approx_kl", -1.0),
                        "mc_ref_gap": metrics.get("mc_ref_gap", 0.0),
                        "mean_abs_ref_gap": metrics.get("mean_abs_ref_gap", 0.0),
                        "mean_sq_ref_gap": metrics.get("mean_sq_ref_gap", 0.0),
                        "p90_abs_ref_gap": metrics.get("p90_abs_ref_gap", 0.0),
                        "grad_norm": metrics.get("grad_norm", 0.0),
                        "clip_grad_norm_return": metrics.get("clip_grad_norm_return", 0.0),
                        "beta_kl": metrics.get("beta_kl", float(trainer.beta_kl)),
                        "kl_loss": metrics.get("kl_loss", 0.0),
                        "accuracy": metrics.get("accuracy", 0.0),
                        "perplexity": metrics.get("perplexity", float("inf")),
                        "diversity": metrics.get("diversity", 0.0),
                        "x_sample_rate": metrics.get("x_sample_rate", 0.0),
                        "sctm": metrics.get("sctm", 0.0),
                        "rmsd": metrics.get("rmsd", 0.0),
                        "rmsd_reward": metrics.get("rmsd_reward", 0.0),
                        "hydrophobic": metrics.get("hydrophobic", 0.0),
                        "plddt": metrics.get("plddt", 0.0),
                        "aar": metrics.get("aar", 0.0),
                        "total_reward": metrics.get("total_reward", 0.0),
                    }
                )
                skip_events_file.flush()
                train_metrics_file.flush()
                repeat_counts = [problematic_samples[idx] for idx in sample_indices.tolist()]
                max_repeat = max(repeat_counts)
                is_repeated = max_repeat > 1
                logger.warning(
                    f"\n{'=' * 60}\nWarning: step {global_step} (Batch {batch_count}) returned a skipped update (skip {skip_count})\n{'=' * 60}\n  Reason: {skip_reason}\n  This batch did not update model parameters. The optimizer-step count remains {global_step})\n  Skip rate: {skip_rate:.1f}% ({skip_count}/{batch_count})\n  Affected sample indices: {sample_indices.tolist()}{(f' (observed up to {max_repeat} times)' if is_repeated else '')}\n  Diagnostic: {('High skip rate; inspect the data and numerical stability.' if skip_rate > 10 else 'Continuing training.')}\n"
                )
                if max_repeat >= 3:
                    repeated_samples = [
                        (idx, problematic_samples[idx])
                        for idx in sample_indices.tolist()
                        if problematic_samples[idx] >= 3
                    ]
                    logger.error(
                        f"{'=' * 60}\nRepeatedly skipped samples detected.\n{'=' * 60}\n  Sample indices and counts: {repeated_samples}\n  Inspect input geometry and numerical diagnostics for these samples.\n{'=' * 60}"
                    )
                logger.warning(f"{'=' * 60}\n")
                if batch_count % 50 == 0:
                    skip_rate_for_save = skip_count / batch_count * 100
                    ckpt_path = os.path.join(
                        cfg.output_dir, f"ckpt_batch{batch_count}_step{global_step}.pt"
                    )
                    _save_checkpoint(ckpt_path)
                    logger.info(
                        f"\n{'=' * 60}\n  Checkpoint saved after a skipped batch: {ckpt_path}\n  Batch progress: {batch_count} batches ({global_step} optimizer steps, {skip_count} skipped)\n  Skip rate: {skip_rate_for_save:.1f}%\n  Number of problematic samples: {len(problematic_samples)}\n{'=' * 60}"
                    )
                continue
            scheduler.step()
            global_step += 1
            batch_count += 1
            skip_rate = skip_count / batch_count * 100 if batch_count > 0 else 0.0
            train_metrics_writer.writerow(
                {
                    "batch_count": batch_count,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "skipped": False,
                    "skip_reason": "",
                    "skip_count": skip_count,
                    "skip_rate": skip_rate,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": metrics.get("loss", 0.0),
                    "policy_loss": metrics.get("policy_loss", 0.0),
                    "approx_kl": metrics.get("approx_kl", -1.0),
                    "mc_ref_gap": metrics.get("mc_ref_gap", 0.0),
                    "mean_abs_ref_gap": metrics.get("mean_abs_ref_gap", 0.0),
                    "mean_sq_ref_gap": metrics.get("mean_sq_ref_gap", 0.0),
                    "p90_abs_ref_gap": metrics.get("p90_abs_ref_gap", 0.0),
                    "grad_norm": metrics.get("grad_norm", 0.0),
                    "clip_grad_norm_return": metrics.get("clip_grad_norm_return", 0.0),
                    "beta_kl": metrics.get("beta_kl", float(trainer.beta_kl)),
                    "kl_loss": metrics.get("kl_loss", 0.0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "perplexity": metrics.get("perplexity", float("inf")),
                    "diversity": metrics.get("diversity", 0.0),
                    "x_sample_rate": metrics.get("x_sample_rate", 0.0),
                    "sctm": metrics.get("sctm", 0.0),
                    "rmsd": metrics.get("rmsd", 0.0),
                    "rmsd_reward": metrics.get("rmsd_reward", 0.0),
                    "hydrophobic": metrics.get("hydrophobic", 0.0),
                    "plddt": metrics.get("plddt", 0.0),
                    "aar": metrics.get("aar", 0.0),
                    "total_reward": metrics.get("total_reward", 0.0),
                }
            )
            train_metrics_file.flush()
            if global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                grad_norm = metrics.get("grad_norm", 0.0)
                max_grad_norm = cfg.rl.max_grad_norm
                logger.info(
                    f"\n{'-' * 60}\n  Step {global_step:5d} / {total_steps}   lr={lr:.2e}\n{'-' * 60}\n  [Loss]\n    Total loss: {metrics['loss']:.4f}\n    Policy loss: {metrics['policy_loss']:.4f}\n    Reference score penalty: {metrics.get('kl_loss', 0.0):.4f}  beta={metrics.get('beta_kl', 0.0):.4f}\n    clip_grad_norm_return   : {metrics.get('clip_grad_norm_return', 0.0):.4f}  (total gradient norm before clipping)\n  [RL metrics]\n    Mean absolute score change: {_format_approx_kl(metrics['approx_kl'])}\n    Mean reference score gap: {metrics['mc_ref_gap']:.5f}  {('(signed mean; distinct from the squared reference penalty)' if abs(metrics['mc_ref_gap']) < 0.15 else 'large score gap')}  mean_abs_gap={metrics.get('mean_abs_ref_gap', 0.0):.5f}  mean_squared_gap={metrics.get('mean_sq_ref_gap', 0.0):.5f}  p90_abs_gap={metrics.get('p90_abs_ref_gap', 0.0):.5f}  (reference penalty = beta * mean_squared_clipped_gap)\n    Total reward: {metrics['total_reward']:.4f}\n    Gradient norm: {grad_norm:.4f}  {('very large' if grad_norm > max_grad_norm * 3 else 'large' if grad_norm > max_grad_norm * 2 else 'clipped' if grad_norm > max_grad_norm else 'normal' if grad_norm > 1e-06 else 'small')}\n  [Reward components]\n    scTM: {metrics['sctm']:.4f}  {('high' if metrics['sctm'] > 0.6 else 'training' if metrics['sctm'] > 0.4 else 'low')}\n    scRMSD: {metrics['rmsd']:.2f} Angstrom  {('high' if metrics['rmsd'] < 2.0 else 'training' if metrics['rmsd'] < 5.0 else 'low')}\n    Hydrophobic core ratio: {metrics['hydrophobic']:.3f}\n  [Sequence metrics]\n    AAR against native sequence: {metrics['aar'] * 100:.1f}%  {('high' if metrics['aar'] > 0.4 else 'training' if metrics['aar'] > 0.3 else 'low')}\n    Sample-argmax agreement: {metrics['accuracy'] * 100:.1f}%  {('high' if metrics['accuracy'] > 0.7 else 'moderate' if metrics['accuracy'] > 0.5 else 'low')}\n    Perplexity against native sequence: {metrics['perplexity']:.2f}  {('high' if metrics['perplexity'] < 5 else 'training' if metrics['perplexity'] < 10 else 'low')}\n    Candidate-sequence diversity: {metrics['diversity']:.4f}  {('high' if metrics['diversity'] > 0.5 else 'low candidate-sequence diversity' if metrics['diversity'] < 0.2 else 'normal')}\n  [Prediction confidence]\n    pLDDT (0-1): {metrics['plddt']:.3f}  {('high' if metrics['plddt'] > 0.8 else 'moderate' if metrics['plddt'] > 0.6 else 'low')}\n{'-' * 60}"
                )
            if global_step % 50 == 0 or global_step >= total_steps:
                skip_rate = skip_count / batch_count * 100 if batch_count > 0 else 0
                ckpt_path = os.path.join(cfg.output_dir, f"ckpt_step{global_step}.pt")
                _save_checkpoint(ckpt_path)
                num_problematic = len(problematic_samples)
                if num_problematic > 0:
                    top_problematic = sorted(
                        problematic_samples.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    problematic_info = ", ".join(
                        (f"sample_{idx}({count} occurrences)" for idx, count in top_problematic)
                    )
                else:
                    problematic_info = "none"
                logger.info(
                    f"\n{'=' * 60}\n  Checkpoint saved: {ckpt_path}\n  Training progress: {global_step}/{total_steps} optimizer steps ({global_step / max(total_steps, 1) * 100:.1f}%)\n  Batch statistics: {batch_count} batches ({global_step} optimizer steps, {skip_count} skipped, skip rate {skip_rate:.1f}%)\n  Problematic samples: {num_problematic} [{problematic_info}]\n{'=' * 60}"
                )
            if global_step >= total_steps:
                logger.info(
                    f"\n{'=' * 60}\n  Training completed.\n  Final scTM: {metrics['sctm']:.4f}\n  Final scRMSD: {metrics['rmsd']:.2f} Angstrom\n  Final pLDDT: {metrics['plddt']:.3f}\n  Checkpoint directory: {cfg.output_dir}\n{'=' * 60}"
                )
                _close_csv_files()
                return
        if global_step == pass_start_step:
            _close_csv_files()
            raise RuntimeError(
                "An entire reinforcement-learning pass completed without a successful update. Check input validation failures and skipped-update diagnostics."
            )
        epoch += 1
    _close_csv_files()


def main():
    cfg = TrainConfig()
    train(cfg, resume_from=RESUME_FROM, pretrain_from=PRETRAIN_FROM)


if __name__ == "__main__":
    main()
