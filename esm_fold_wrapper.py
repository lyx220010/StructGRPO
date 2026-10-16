"""ESMFold inference utilities for predicted refolding feedback.

Model locations, devices, chunk sizes, and inference batch sizes are supplied
by the caller. The composite reward is calculated outside this wrapper.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def sequences_to_strings(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    alphabet: str = MPNN_ALPHABET,
) -> List[str]:
    """Convert integer-encoded sequences, excluding trailing padding."""
    result = []
    for i in range(sequences.shape[0]):
        L = int(lengths[i].item())
        seq = sequences[i, :L].tolist()
        aa_str = ''.join(
            alphabet[idx] if 0 <= idx < len(alphabet) else 'X'
            for idx in seq
        )
        result.append(aa_str)
    return result


class ESMFoldWrapper:
    """Refold sequences and return coordinates and confidence outputs.

    Loading and initialization failures raise errors without simulated outputs.
    """

    def __init__(
        self,
        model_name: str = "facebook/esmfold_v1",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ):
        if device is None:
            raise ValueError("Pass the ESMFold device explicitly.")
        if chunk_size is None:
            raise ValueError("Pass the ESMFold chunk size explicitly.")
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("ESMFold chunk size must be a positive integer.")
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model(model_name, local_model_path, chunk_size)

    def _load_model(
        self, model_name: str, local_model_path: Optional[str], chunk_size: int,
    ):
        """Load a model identifier or local directory, with local fallback."""
        self.model = None
        self.tokenizer = None
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("ESMFold model_name must be a nonempty model identifier or path.")
        try:
            from transformers import EsmForProteinFolding, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ESMFold requires Transformers and its folding dependencies."
            ) from exc

        is_local_path = model_name.startswith('/') or model_name.startswith('./')
        if is_local_path:
            try:
                logger.info("Loading ESMFold from local path: %s", model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = EsmForProteinFolding.from_pretrained(
                    model_name, low_cpu_mem_usage=True
                )
                logger.info("ESMFold loaded from local path.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load ESMFold from local path {model_name!r}. "
                    "No folding results were generated."
                ) from exc
        else:
            try:
                logger.info("Loading ESMFold model identifier: %s", model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = EsmForProteinFolding.from_pretrained(
                    model_name, low_cpu_mem_usage=True
                )
                logger.info("ESMFold loaded from model identifier.")
            except Exception as exc:
                if local_model_path is None:
                    raise RuntimeError(
                        f"Failed to load ESMFold model {model_name!r}, and no "
                        "local fallback was supplied. No folding results were generated."
                    ) from exc
                logger.warning("ESMFold loading failed (%s); trying local fallback.", exc)
                try:
                    logger.info("Loading ESMFold from local path: %s", local_model_path)
                    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                    model = EsmForProteinFolding.from_pretrained(
                        local_model_path, low_cpu_mem_usage=True
                    )
                    logger.info("ESMFold loaded from local path.")
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Failed to load ESMFold model {model_name!r} and local "
                        f"fallback {local_model_path!r}. No folding results were generated."
                    ) from fallback_exc

        try:
            model = model.to(self.device)
            model.eval()
            if hasattr(model.esm.encoder, 'layer_norm'):
                model.esm.encoder.layer_norm.float()
            if hasattr(model, 'set_chunk_size'):
                model.set_chunk_size(chunk_size)
        except Exception as exc:
            raise RuntimeError(
                f"ESMFold initialization failed on device {self.device!r}. "
                "No folding results were generated."
            ) from exc
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def fold_sequence_to_backbone(
        self, sequence: str,
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """
        Refold one AA string and return (N, CA, C) backbone coordinates [L, 3].
        Uses the same autocast path as fold() for GPU compatibility.
        """
        import numpy as np
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("ESMFold is not loaded; folding cannot proceed.")
        self.model.eval()
        inputs = self.tokenizer(
            sequence, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        use_amp = (
            torch.cuda.is_available()
            and self.device != "cpu"
            and not str(self.device).startswith("cpu")
            and torch.cuda.is_bf16_supported()
        )
        with torch.no_grad():
            if use_amp:
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(**inputs)
                except (AttributeError, TypeError):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        positions = outputs.positions[-1][0].detach().cpu().float().numpy()
        L = min(len(sequence), positions.shape[0])
        n_coords = positions[:L, 0, :]
        ca_coords = positions[:L, 1, :]
        c_coords = positions[:L, 2, :]
        return n_coords, ca_coords, c_coords

    @torch.no_grad()
    def fold(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        original_coords: Optional[torch.Tensor] = None,
        max_batch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return padded C-alpha coordinates and residue confidence.

        ``max_batch`` controls inference mini-batching, not the number of
        candidate sequences sampled for a target. Reward weights are external.
        ``original_coords`` is accepted for call compatibility and is not used
        to construct predictions.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("ESMFold is not loaded; folding cannot proceed.")
        if max_batch is None:
            raise ValueError("Pass the ESMFold inference batch size explicitly.")
        if (
            not isinstance(max_batch, int)
            or isinstance(max_batch, bool)
            or max_batch <= 0
        ):
            raise ValueError("ESMFold inference batch size must be a positive integer.")
        B, N = sequences.shape
        logger.debug("ESMFold inference: batch_size=%s, seq_len=%s", B, N)

        aa_strings = sequences_to_strings(sequences, lengths)
        all_ca, all_plddt = [], []
        for start in range(0, B, max_batch):
            batch_seqs = aa_strings[start:start + max_batch]
            logger.debug(
                "ESMFold mini-batch [%s:%s], sequence lengths=%s",
                start, start + len(batch_seqs), [len(s) for s in batch_seqs],
            )
            inputs = self.tokenizer(
                batch_seqs, return_tensors="pt",
                padding=True, add_special_tokens=False
            ).to(self.device)

            with torch.no_grad():
                use_amp = (torch.cuda.is_available() and
                          torch.cuda.is_bf16_supported() and
                          self.device != 'cpu')
                if use_amp:
                    try:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = self.model(**inputs)
                    except (AttributeError, TypeError):
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            logger.debug(
                "ESMFold output fields: %s",
                list(outputs.keys()) if hasattr(outputs, 'keys') else dir(outputs),
            )

            positions = outputs.positions[-1]                 # [B, L, atoms, 3]
            ca = positions[:, :, 1, :]                        # [B, L, 3]

            plddt = outputs.plddt
            if plddt.dim() == 3:
                plddt = plddt.mean(dim=-1)

            L_pred = ca.shape[1]
            if L_pred < N:
                pad_ca = torch.zeros(ca.shape[0], N - L_pred, 3, device=ca.device)
                ca = torch.cat([ca, pad_ca], dim=1)
                pad_pl = torch.zeros(plddt.shape[0], N - L_pred, device=plddt.device)
                plddt = torch.cat([plddt, pad_pl], dim=1)
            else:
                ca = ca[:, :N, :]
                plddt = plddt[:, :N]

            all_ca.append(ca)
            all_plddt.append(plddt)

        ca_coords = torch.cat(all_ca, dim=0)      # [B, N, 3]
        plddt_out = torch.cat(all_plddt, dim=0)   # [B, N]

        target_device = sequences.device
        if ca_coords.device != target_device:
            ca_coords = ca_coords.to(target_device)
            plddt_out = plddt_out.to(target_device)
        return {
            'ca_coords': ca_coords,
            'plddt': plddt_out,
        }
