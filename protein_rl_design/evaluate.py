import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from protein_rl_design.train_rl import BackboneDataset, collate_fn
from protein_rl_design.config import TrainConfig
from protein_rl_design.models.hgnn_encoder import HGNNProteinEncoder
from protein_rl_design.models.sequence_policy import SequenceDesignPolicy
from protein_rl_design.reward.reward_fn import (
    RewardFunction,
    structural_scores_batch,
    compute_plddt_reward,
    compute_buried_hydrophobic_ratio,
    compute_sequence_accuracy,
    compute_perplexity,
    compute_diversity,
)
from protein_rl_design.utils.esm_fold_wrapper import ESMFoldWrapper
from protein_rl_design.utils.checkpoint import (
    CHECKPOINT_TYPE_PRETRAIN,
    detect_checkpoint_type,
    load_model_weights,
)

CHECKPOINT_PATH = None
TEST_DATA_DIR = Path("data/cath42_pt_filtered/test")
OUTPUT_DIR = Path("outputs/test_evaluation")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MissingNativeSequenceError(ValueError):
    pass


class EvalMetricsAggregator:

    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.per_target_rows: List[Dict[str, Any]] = []
        self.batch_count = 0
        self.skip_count = 0
        self.nan_count = 0
        self.invalid_seq_count = 0
        self.total_designs = 0
        self.total_time = 0.0
        self.design_time = 0.0
        self.fold_time = 0.0
        self.metric_time = 0.0

    def add_rows(self, rows: List[Dict[str, Any]]):
        self.rows.extend(rows)
        self.total_designs += len(rows)

    def add_per_target_rows(self, rows: List[Dict[str, Any]]):
        self.per_target_rows.extend(rows)

    def add_batch_stats(
        self,
        batch_time: float,
        design_time: float,
        fold_time: float,
        metric_time: float,
        skipped: bool = False,
        has_nan: bool = False,
        invalid_seq_count: int = 0,
    ):
        self.batch_count += 1
        self.total_time += float(batch_time)
        self.design_time += float(design_time)
        self.fold_time += float(fold_time)
        self.metric_time += float(metric_time)
        if skipped:
            self.skip_count += 1
        if has_nan:
            self.nan_count += 1
        self.invalid_seq_count += int(invalid_seq_count)

    def _length_binned(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if df.empty:
            return {}
        groups = {
            "0-150": df[(df["length"] >= 0) & (df["length"] < 150)],
            "150-300": df[(df["length"] >= 150) & (df["length"] <= 300)],
        }
        out = {}
        for key, sub in groups.items():
            if len(sub) == 0:
                continue
            out[key] = {
                "count": int(len(sub)),
                "sctm_mean": float(sub["sctm"].mean()),
                "sctm_std": float(sub["sctm"].std(ddof=0)),
                "plddt_mean": float(sub["plddt_mean"].mean()),
                "plddt_std": float(sub["plddt_mean"].std(ddof=0)),
                "aar_mean": float(sub["aar"].mean()),
                "aar_std": float(sub["aar"].std(ddof=0)),
                "rmsd_mean": float(sub["rmsd"].mean()),
                "rmsd_std": float(sub["rmsd"].std(ddof=0)),
            }
        return out

    def summarize(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.rows)
        if df.empty:
            peak_mem = 0.0
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1000000000.0
            return {
                "basic": {},
                "confidence": {},
                "efficiency": {
                    "total_time": self.total_time,
                    "design_time": self.design_time,
                    "fold_time": self.fold_time,
                    "metric_time": self.metric_time,
                    "peak_gpu_memory_gb": peak_mem,
                },
                "length_binned": {},
            }
        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1000000000.0
        basic = {
            "sctm_mean": float(df["sctm"].mean()),
            "sctm_std": float(df["sctm"].std(ddof=0)),
            "rmsd_mean": float(df["rmsd"].mean()),
            "rmsd_std": float(df["rmsd"].std(ddof=0)),
            "aar_mean": float(df["aar"].mean()),
            "aar_std": float(df["aar"].std(ddof=0)),
            "perplexity_mean": float(df["perplexity"].mean()),
            "perplexity_std": float(df["perplexity"].std(ddof=0)),
            "diversity_mean": float(df["group_diversity"].mean()),
            "diversity_std": float(df["group_diversity"].std(ddof=0)),
            "hydrophobic_mean": float(df["hydrophobic"].mean()),
            "hydrophobic_std": float(df["hydrophobic"].std(ddof=0)),
        }
        confidence = {
            "plddt_mean": float(df["plddt_mean"].mean()),
        }
        efficiency = {
            "total_time": self.total_time,
            "design_time": self.design_time,
            "fold_time": self.fold_time,
            "metric_time": self.metric_time,
            "peak_gpu_memory_gb": peak_mem,
        }
        return {
            "basic": basic,
            "confidence": confidence,
            "efficiency": efficiency,
            "length_binned": self._length_binned(df),
        }


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _repeat_for_samples(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    return x.repeat_interleave(n_samples, dim=0)


def _extract_names(batch: Dict[str, Any], batch_size: int) -> List[str]:
    names = batch.get("name", None)
    if isinstance(names, list) and len(names) == batch_size:
        return [str(x) for x in names]
    tids = batch.get("target_id", None)
    if isinstance(tids, list) and len(tids) == batch_size:
        return [str(x) for x in tids]
    if names is None:
        return [f"sample_{i}" for i in range(batch_size)]
    if isinstance(names, list):
        return [str(x) for x in names]
    return [str(names)] * batch_size


def _extract_source_files(batch: Dict[str, Any], batch_size: int) -> List[str]:
    files = batch.get("source_file", None)
    if isinstance(files, list) and len(files) == batch_size:
        return [str(x) for x in files]
    return [""] * batch_size


def build_models(cfg: TrainConfig, device: torch.device):
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
    ).to(device)
    policy = SequenceDesignPolicy(
        hgnn_out_dim=cfg.hgnn.out_dim,
        hidden_dim=cfg.mpnn.hidden_dim,
        num_enc_layers=cfg.mpnn.num_encoder_layers,
        num_dec_layers=cfg.mpnn.num_decoder_layers,
        k_neighbors=cfg.mpnn.k_neighbors,
        vocab_size=cfg.mpnn.vocab_size,
        dropout=cfg.mpnn.dropout,
        use_online_rel_orient=cfg.mpnn.use_online_rel_orient,
    ).to(device)
    reward_fn = RewardFunction(
        w_sctm=cfg.rl.w_sctm,
        w_rmsd=cfg.rl.w_rmsd,
        w_hydrophobic=cfg.rl.w_hydrophobic,
        w_plddt=cfg.rl.w_plddt,
        w_aar=cfg.rl.w_aar,
    ).to(device)
    return (hgnn, policy, reward_fn)


def load_checkpoint(ckpt_path: str, hgnn, policy, device: torch.device) -> Dict[str, Any]:
    ckpt = load_model_weights(ckpt_path, hgnn, policy, device=device)
    ckpt_type = ckpt.get("_loaded_type") or detect_checkpoint_type(ckpt)
    if ckpt_type == CHECKPOINT_TYPE_PRETRAIN:
        print(f"")
    else:
        print(f"Loaded reinforcement-learning checkpoint: {ckpt_path}")
    return ckpt


def build_esm(cfg: TrainConfig, device: torch.device) -> ESMFoldWrapper:
    return ESMFoldWrapper(
        model_name=cfg.esm_fold_model,
        local_model_path=cfg.esm_fold_local_path,
        device="cuda" if device.type == "cuda" else "cpu",
        chunk_size=64,
    )


def evaluate_batch(
    batch: Dict[str, Any],
    hgnn,
    policy,
    esm: ESMFoldWrapper,
    device: torch.device,
    num_samples: int,
    temperature: float,
    esm_fold_batch_size: int,
    verbose_progress: bool = False,
    batch_desc: str = "",
) -> Dict[str, Any]:
    batch = _to_device(batch, device)
    hgnn.eval()
    policy.eval()
    ca_coords = batch["ca_coords"]
    mask = batch["mask"]
    lengths = batch["lengths"]
    native_seq = batch.get("sequence", None)
    if native_seq is None:
        raise MissingNativeSequenceError(
            "Native sequence records are required to calculate AAR and native-sequence perplexity. At least one input file in this batch is missing its native sequence."
        )
    cb_coords = batch.get("cb_coords", None)
    n_to_ca = batch.get("n_to_ca", None)
    ca_to_c = batch.get("ca_to_c", None)
    phi = batch.get("phi", None)
    psi = batch.get("psi", None)
    B0 = ca_coords.shape[0]
    names = _extract_names(batch, B0)
    source_files = _extract_source_files(batch, B0)
    batch_start = time.perf_counter()
    with torch.no_grad():
        t0 = time.perf_counter()
        hgnn_h = hgnn(ca_coords, mask, cb_coords, n_to_ca, ca_to_c, phi, psi)
        sequences = policy.sample(
            ca_coords,
            hgnn_h,
            mask,
            temperature=temperature,
            n_samples=num_samples,
            n_to_ca=n_to_ca,
            ca_to_c=ca_to_c,
        )
        design_time = time.perf_counter() - t0
        B = ca_coords.shape[0]
        N = ca_coords.shape[1]
        total_designs = B * num_samples
        ca_rep = _repeat_for_samples(ca_coords, num_samples)
        mask_rep = _repeat_for_samples(mask, num_samples)
        lengths_rep = _repeat_for_samples(lengths, num_samples)
        n_to_ca_rep = _repeat_for_samples(n_to_ca, num_samples) if n_to_ca is not None else None
        ca_to_c_rep = _repeat_for_samples(ca_to_c, num_samples) if ca_to_c is not None else None
        native_rep = _repeat_for_samples(native_seq, num_samples)
        t1 = time.perf_counter()
        if verbose_progress:
            print(
                f"{batch_desc} ESMFold start: total_designs={total_designs}, esm_fold_batch_size={esm_fold_batch_size}",
                flush=True,
            )
        esm_out = esm.fold(
            sequences=sequences,
            lengths=lengths_rep,
            original_coords=ca_rep,
            max_batch=esm_fold_batch_size,
        )
        fold_time = time.perf_counter() - t1
        if verbose_progress:
            print(f"{batch_desc} ESMFold done: fold_time={fold_time:.2f}s", flush=True)
        refolded_ca = esm_out["ca_coords"].float()
        plddt = esm_out["plddt"].float()
        t2 = time.perf_counter()
        sctm, rmsd = structural_scores_batch(ca_rep.float(), refolded_ca, mask_rep.float())
        plddt_mean = compute_plddt_reward(plddt, mask_rep.float())
        hydrophobic = compute_buried_hydrophobic_ratio(sequences, refolded_ca, mask_rep.float())
        aar = compute_sequence_accuracy(sequences, native_rep, mask_rep.float())
        log_probs = policy(
            ca_rep, _repeat_for_samples(hgnn_h, num_samples), mask_rep, n_to_ca_rep, ca_to_c_rep
        )
        perplexity = compute_perplexity(log_probs, native_rep, mask_rep.float())
        group_diversity = compute_diversity(sequences, mask_rep.float(), num_samples)
        metric_time = time.perf_counter() - t2
    batch_time = time.perf_counter() - batch_start
    invalid_sequences = 0
    if torch.isnan(sequences.float()).any() or torch.isinf(sequences.float()).any():
        invalid_sequences = int(
            torch.isnan(sequences.float()).any().item()
            or torch.isinf(sequences.float()).any().item()
        )
    rows: List[Dict[str, Any]] = []
    per_target_rows: List[Dict[str, Any]] = []
    repeated_names = []
    repeated_source_files = []
    for name, src in zip(names, source_files):
        repeated_names.extend([name] * num_samples)
        repeated_source_files.extend([src] * num_samples)
    for i in range(total_designs):
        target_idx = i // num_samples
        sample_idx = i % num_samples
        row = {
            "target_name": repeated_names[i],
            "source_file": repeated_source_files[i],
            "target_index": int(target_idx),
            "sample_index": int(sample_idx),
            "length": int(lengths_rep[i].item()),
            "sctm": float(sctm[i].item()),
            "rmsd": float(rmsd[i].item()),
            "plddt_mean": float(plddt_mean[i].item()),
            "aar": float(aar[i].item()),
            "perplexity": float(perplexity[i].item()),
            "group_diversity": float(group_diversity[target_idx].item()),
            "hydrophobic": float(hydrophobic[i].item()),
            "batch_time": float(batch_time),
            "design_time": float(design_time),
            "fold_time": float(fold_time),
            "metric_time": float(metric_time),
            "designed_sequence": " ".join(
                map(str, sequences[i].detach().cpu().tolist()[: int(lengths_rep[i].item())])
            ),
        }
        rows.append(row)
    for b in range(B):
        start = b * num_samples
        end = (b + 1) * num_samples
        sub_sctm = sctm[start:end]
        sub_rmsd = rmsd[start:end]
        sub_plddt = plddt_mean[start:end]
        sub_aar = aar[start:end]
        sub_perp = perplexity[start:end]
        sub_div = group_diversity[b : b + 1]
        sub_hydro = hydrophobic[start:end]
        per_target_rows.append(
            {
                "target_name": names[b],
                "source_file": source_files[b],
                "length": int(lengths[b].item()),
                "num_samples": int(num_samples),
                "mean_sctm": float(sub_sctm.mean().item()),
                "mean_rmsd": float(sub_rmsd.mean().item()),
                "mean_plddt": float(sub_plddt.mean().item()),
                "mean_aar": float(sub_aar.mean().item()),
                "mean_perplexity": float(sub_perp.mean().item()),
                "group_diversity": float(sub_div.mean().item()),
                "mean_hydrophobic": float(sub_hydro.mean().item()),
            }
        )
    return {
        "rows": rows,
        "per_target_rows": per_target_rows,
        "batch_time": batch_time,
        "design_time": design_time,
        "fold_time": fold_time,
        "metric_time": metric_time,
        "invalid_sequences": invalid_sequences,
        "has_nan": bool(
            torch.isnan(sctm).any().item()
            or torch.isnan(rmsd).any().item()
            or torch.isnan(plddt_mean).any().item()
        ),
    }


def save_results(
    output_dir: Path,
    per_design_rows: List[Dict[str, Any]],
    per_target_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_design_path = output_dir / "per_design_results.csv"
    per_target_path = output_dir / "per_target_results.csv"
    summary_path = output_dir / "summary_metrics.json"
    pd.DataFrame(per_design_rows).to_csv(per_design_path, index=False)
    pd.DataFrame(per_target_rows).to_csv(per_target_path, index=False)
    payload = {"meta": meta, "summary": summary}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 80)
    print("Evaluation finished")
    print("=" * 80)
    print(f"Per-design results: {per_design_path}")
    print(f"Per-target results: {per_target_path}")
    print(f"Summary results:    {summary_path}")
    print("=" * 80 + "\n")


def run_evaluation(cfg: TrainConfig) -> Dict[str, Any]:
    seed = int(getattr(cfg, "seed", 42))
    num_samples = int(getattr(cfg, "eval_num_samples", 8))
    temperature = float(getattr(cfg, "eval_temperature", 0.5))
    batch_size = int(getattr(cfg, "eval_batch_size", 2))
    esm_fold_batch_size = int(getattr(cfg, "eval_esm_fold_batch_size", 4))
    max_seq_len = int(getattr(cfg, "max_seq_len", 300))
    log_every = int(getattr(cfg, "eval_log_every", 10))
    data_dir = getattr(cfg, "test_dir", None) or TEST_DATA_DIR
    ckpt_path = (
        CHECKPOINT_PATH
        or getattr(cfg, "eval_ckpt_path", None)
        or getattr(cfg, "pretrain_ckpt_path", None)
    )
    if not ckpt_path:
        raise ValueError(
            "Set CHECKPOINT_PATH, config.eval_ckpt_path or config.pretrain_ckpt_path to the checkpoint to evaluate."
        )
    set_seed(seed)
    device = torch.device(
        getattr(cfg, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    hgnn, policy, reward_fn = build_models(cfg, device)
    ckpt = load_checkpoint(ckpt_path, hgnn, policy, device)
    hgnn.eval()
    policy.eval()
    esm = build_esm(cfg, device)
    dataset = BackboneDataset(data_dir, max_len=max_seq_len)
    print(f"Evaluating all {len(dataset)} test targets from {data_dir}.")
    print(f"Maximum input length: {max_seq_len} residues.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    agg = EvalMetricsAggregator()
    total_batches = len(loader)
    processed_targets = 0
    for batch_idx, batch in enumerate(loader):
        batch_size_now = int(batch["ca_coords"].shape[0])
        batch_desc = f"[Batch {batch_idx + 1}/{total_batches}]"
        print(
            f"{batch_desc} start: batch_size={batch_size_now}, processed_targets={processed_targets}",
            flush=True,
        )
        try:
            out = evaluate_batch(
                batch=batch,
                hgnn=hgnn,
                policy=policy,
                esm=esm,
                device=device,
                num_samples=num_samples,
                temperature=temperature,
                esm_fold_batch_size=esm_fold_batch_size,
                verbose_progress=True,
                batch_desc=batch_desc,
            )
            agg.add_rows(out["rows"])
            agg.add_per_target_rows(out["per_target_rows"])
            agg.add_batch_stats(
                batch_time=out["batch_time"],
                design_time=out["design_time"],
                fold_time=out["fold_time"],
                metric_time=out["metric_time"],
                skipped=False,
                has_nan=out["has_nan"],
                invalid_seq_count=out["invalid_sequences"],
            )
            print(
                f"{batch_desc} done: batch_time={out['batch_time']:.2f}s, total_designs={agg.total_designs}, processed_targets={processed_targets + batch_size_now}",
                flush=True,
            )
        except MissingNativeSequenceError:
            raise
        except Exception as e:
            print(f"Batch {batch_idx + 1} evaluation failed and was skipped: {e}")
            agg.add_batch_stats(
                batch_time=0.0,
                design_time=0.0,
                fold_time=0.0,
                metric_time=0.0,
                skipped=True,
                has_nan=False,
                invalid_seq_count=0,
            )
        processed_targets += batch_size_now
        if (batch_idx + 1) % log_every == 0:
            print(
                f"Processed {processed_targets}/{len(dataset)} test targets; generated {agg.total_designs} evaluated sequences"
            )
    summary = agg.summarize()
    flat_summary = {
        "aar": summary.get("basic", {}).get("aar_mean", 0.0),
        "sctm": summary.get("basic", {}).get("sctm_mean", 0.0),
        "plddt": summary.get("confidence", {}).get("plddt_mean", 0.0),
        "rmsd": summary.get("basic", {}).get("rmsd_mean", 0.0),
        "diversity": summary.get("basic", {}).get("diversity_mean", 0.0),
    }
    meta = {
        "checkpoint": str(ckpt_path),
        "dataset_dir": str(data_dir),
        "seed": seed,
        "num_samples": num_samples,
        "temperature": temperature,
        "batch_size": batch_size,
        "esm_fold_batch_size": esm_fold_batch_size,
        "max_seq_len": max_seq_len,
        "device": str(device),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "num_available_targets": len(dataset.files),
        "num_processed_targets": int(processed_targets),
        "num_successfully_evaluated_targets": len(agg.per_target_rows),
        "num_failed_targets": int(processed_targets - len(agg.per_target_rows)),
        "num_skipped_batches": int(agg.skip_count),
        "num_nan_batches": int(agg.nan_count),
        "num_batches_with_invalid_sequences": int(agg.invalid_seq_count),
        "evaluation_complete": len(agg.per_target_rows) == len(dataset.files),
    }
    save_results(
        output_dir=OUTPUT_DIR,
        per_design_rows=agg.rows,
        per_target_rows=agg.per_target_rows,
        summary=summary,
        meta=meta,
    )
    print(
        f"Test targets processed: {processed_targets}/{len(dataset)}; successfully evaluated: {len(agg.per_target_rows)}; failed: {processed_targets - len(agg.per_target_rows)}."
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"summary": summary, "flat_summary": flat_summary}


def main():
    cfg = TrainConfig()
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
