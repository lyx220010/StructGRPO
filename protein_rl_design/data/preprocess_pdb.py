"""Preprocess single-chain CATH domain backbones without residue truncation."""

import hashlib
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
INPUT_DIR = Path("data/cath42/dompdb")
OUTPUT_DIR = Path("data/cath42_pt_all")
MIN_LEN = 0
MAX_LEN = 300
CHAIN_ID = None
CATH_DOMAIN_LIST = None
PEPTIDE_BOND_MAX_DISTANCE = 1.8
INTRA_RESIDUE_BOND_MIN_DISTANCE = 0.8
INTRA_RESIDUE_BOND_MAX_DISTANCE = 2.0
MIN_CA_SEPARATION = 0.5
MIN_BACKBONE_SINE = 1e-4


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
STANDARD_AA = set(AA3_TO_AA1)
MODIFIED_POLYMER_NAMES = {"MSE", "SEC", "PYL", "HYP", "ASX", "GLX", "UNK"}
BACKBONE_ATOMS = {"N", "CA", "C"}
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA1_TO_IDX = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}


def compute_virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    N = ca.shape[0]
    cb = np.zeros_like(ca)
    for i in range(N):
        b = ca[i] - n[i]
        c_vec = c[i] - ca[i]
        b = b / (np.linalg.norm(b) + 1e-08)
        c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-08)
        a = np.cross(b, c_vec)
        a = a / (np.linalg.norm(a) + 1e-08)
        cb_bond_length = 1.52
        angle = 1.93
        dihedral = 2.14
        cb[i] = ca[i] + cb_bond_length * (
            -np.cos(angle) * b
            + np.sin(angle) * np.cos(dihedral) * c_vec
            + np.sin(angle) * np.sin(dihedral) * a
        )
    return cb


def compute_backbone_orientations(
    n: np.ndarray, ca: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_to_ca = ca - n
    ca_to_c = c - ca
    n_to_ca = n_to_ca / (np.linalg.norm(n_to_ca, axis=-1, keepdims=True) + 1e-08)
    ca_to_c = ca_to_c / (np.linalg.norm(ca_to_c, axis=-1, keepdims=True) + 1e-08)
    return (n_to_ca, ca_to_c)


def compute_backbone_dihedrals(
    n: np.ndarray, ca: np.ndarray, c: np.ndarray, peptide_bonds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    N = ca.shape[0]
    phi = np.zeros(N, dtype=np.float32)
    psi = np.zeros(N, dtype=np.float32)
    distances = np.linalg.norm(c[:-1] - n[1:], axis=-1)
    connected = (
        np.isfinite(distances) & (distances > 1e-08) & (distances < PEPTIDE_BOND_MAX_DISTANCE)
    )
    if peptide_bonds is not None:
        peptide_bonds = np.asarray(peptide_bonds, dtype=bool)
        if peptide_bonds.shape != (max(N - 1, 0),):
            raise ValueError("Peptide-bond indicators do not match the number of residues.")
        connected &= peptide_bonds
    for i in range(N):
        if i > 0 and connected[i - 1]:
            phi[i] = compute_dihedral(c[i - 1], n[i], ca[i], c[i])
        if i < N - 1 and connected[i]:
            psi[i] = compute_dihedral(n[i], ca[i], c[i], n[i + 1])
    return (phi, psi)


def compute_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    b2_norm = b2 / (np.linalg.norm(b2) + 1e-08)
    n1 = np.cross(b1, b2)
    n1 = n1 / (np.linalg.norm(n1) + 1e-08)
    n2 = np.cross(b2, b3)
    n2 = n2 / (np.linalg.norm(n2) + 1e-08)
    m1 = np.cross(n1, b2_norm)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.arctan2(y, x)


def _new_audit(pdb_path):
    return {
        "domain_id": Path(pdb_path).stem,
        "source_file": str(pdb_path),
        "source_sha256": "",
        "status": "rejected",
        "reasons": [],
        "details": [],
        "input_chains": [],
        "selected_chain": None,
        "selected_model": "first",
        "observed_residue_count": 0,
        "valid_backbone_count": 0,
        "reference_length": None,
        "reference_length_check": "not_provided",
        "missing_backbone_residue_count": 0,
        "invalid_backbone_residue_count": 0,
        "unsupported_residue_count": 0,
        "numbering_gap_count": 0,
        "alternate_conformer_residue_count": 0,
        "internal_missing_residue_records": [],
        "retained_residue_ids": [],
        "final_length": 0,
        "truncated": False,
        "output_file": "",
    }


def _issue(audit, reason, detail=None):
    if reason not in audit["reasons"]:
        audit["reasons"].append(reason)
    if detail is not None:
        audit["details"].append(detail)


def _residue_id(position):
    chain, segment, number, insertion = position
    return "{}:{}{}@{}".format(chain or "_", number, insertion, segment)


def _missing_residue_record(line):
    """Parse residue data rows from PDB REMARK 465."""
    if not line.startswith("REMARK 465"):
        return None
    fields = line[10:].split()
    if len(fields) not in (3, 4):
        return None
    name, chain, number = fields[-3:]
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", number)
    if name not in STANDARD_AA | MODIFIED_POLYMER_NAMES or len(chain) != 1 or not match:
        return None
    model = fields[0] if len(fields) == 4 else None
    if model is not None and not model.isdigit():
        return None
    return {"resname": name, "chain": chain, "number": int(match[1]),
            "insertion_code": match[2], "model": model}


def _select_backbone_records(pdb_path, chain_id=None, audit_info=None):
    """Select complete, consistent backbone conformers from the first model."""
    audit = audit_info if audit_info is not None else _new_audit(pdb_path)
    raw = Path(pdb_path).read_bytes()
    audit["source_sha256"] = hashlib.sha256(raw).hexdigest()
    residues = {}
    remarks = []
    segment = 0
    model_started = False
    atoms_started = False
    selected_model = None
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        missing = _missing_residue_record(line)
        if missing is not None:
            remarks.append(missing)
        record = line[:6].strip()
        if record == "MODEL":
            if model_started or atoms_started:
                break
            model_started = True
            selected_model = line[10:14].strip() or "1"
            audit["selected_model"] = selected_model
            continue
        if record in {"ENDMDL", "END"}:
            break
        if record == "TER":
            segment += 1
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 27:
            _issue(audit, "malformed_atom_record", {"line": line_number})
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()

        if (record == "HETATM" and res_name not in STANDARD_AA | MODIFIED_POLYMER_NAMES
                and atom_name not in BACKBONE_ATOMS):
            continue
        atoms_started = True
        chain = line[21].strip()
        number = line[22:26].strip()
        insertion = line[26].strip()
        position = (chain, segment, number, insertion)
        residue = residues.setdefault(position, {"order": len(residues), "variants": {},
                                                  "polymer_hint": False, "atoms_seen": set()})
        residue["polymer_hint"] |= (record == "ATOM"
                                     or res_name in STANDARD_AA | MODIFIED_POLYMER_NAMES)
        residue["atoms_seen"].add(atom_name)
        variant = residue["variants"].setdefault((res_name, line[16].strip()), {})
        if atom_name not in BACKBONE_ATOMS:
            continue
        error = None
        try:
            if len(line) < 54:
                raise ValueError("Incomplete coordinate columns")
            xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
            if not all(math.isfinite(value) for value in xyz):
                error = "nonfinite_coordinates"
        except ValueError:
            xyz = None
            error = "malformed_coordinates"
        occupancy_text = line[54:60].strip()
        try:
            occupancy = float(occupancy_text) if occupancy_text else 1.0
            if not math.isfinite(occupancy) or occupancy <= 0:
                error = error or "unobserved_or_invalid_occupancy"
        except ValueError:
            occupancy = 0.0
            error = error or "invalid_occupancy"
        variant.setdefault(atom_name, []).append({"line": line, "occupancy": occupancy,
                                                  "error": error})
    polymer_residues = {}
    for position, residue in residues.items():

        if (residue["polymer_hint"] or ("CA" in residue["atoms_seen"]
                                       and bool({"N", "C"} & residue["atoms_seen"]))):
            polymer_residues[position] = residue
    chains = sorted({position[0] for position in polymer_residues})
    audit["input_chains"] = chains
    if chain_id is None:
        if len(chains) != 1:
            _issue(audit, "no_polymer_residues" if not chains else "multiple_polymer_chains",
                   {"chains": chains})
        selected_chain = chains[0] if len(chains) == 1 else None
    else:
        selected_chain = str(chain_id).strip()
        if selected_chain not in chains:
            _issue(audit, "requested_chain_not_found", {"requested_chain": selected_chain})
    audit["selected_chain"] = selected_chain
    selected_residues = [(p, r) for p, r in polymer_residues.items()
                         if selected_chain is None or p[0] == selected_chain]
    audit["observed_residue_count"] = len(selected_residues)
    selected = []
    residue_ids = []
    for residue_order, (position, residue) in enumerate(selected_residues):
        rid = _residue_id(position)
        variants = residue["variants"]
        if any(alt for (_, alt) in variants):
            audit["alternate_conformer_residue_count"] += 1
        candidates = []
        failures = set()
        for (res_name, altloc), alternative in variants.items():
            if res_name not in STANDARD_AA:
                failures.add("unsupported_residue")
                continue
            atoms = dict(variants.get((res_name, ""), {}))
            atoms.update(alternative)
            if not BACKBONE_ATOMS.issubset(atoms):
                failures.add("missing_backbone_atoms")
                continue
            if any(len(atoms[name]) != 1 for name in BACKBONE_ATOMS):
                failures.add("ambiguous_duplicate_backbone_atoms")
                continue
            chosen = {name: atoms[name][0] for name in BACKBONE_ATOMS}
            errors = {atom["error"] for atom in chosen.values() if atom["error"]}
            if errors:
                failures.update(errors)
                continue
            occupancy = sum(atom["occupancy"] for atom in chosen.values()) / 3.0
            priority = (-occupancy, altloc != "", altloc != "A", altloc, res_name)
            candidates.append((priority, chosen))
        if not candidates:
            if "unsupported_residue" in failures:
                audit["unsupported_residue_count"] += 1
            if "missing_backbone_atoms" in failures:
                audit["missing_backbone_residue_count"] += 1
            audit["invalid_backbone_residue_count"] += 1
            _issue(audit, "incomplete_or_invalid_backbone",
                   {"residue": rid, "causes": sorted(failures)})
            continue
        chosen = min(candidates, key=lambda item: item[0])[1]
        residue_ids.append(rid)
        for name in ("N", "CA", "C"):
            selected.append((chosen[name]["line"], position[1], residue_order))
    audit["valid_backbone_count"] = len(residue_ids)

    numeric_numbers = []
    for position, _ in selected_residues:
        try:
            numeric_numbers.append(int(position[2]))
        except ValueError:

            pass
    if numeric_numbers and selected_chain is not None:
        lo, hi = min(numeric_numbers), max(numeric_numbers)
        for missing in remarks:
            if (missing["chain"] == selected_chain and lo <= missing["number"] <= hi
                    and (missing["model"] is None or selected_model is None
                         or missing["model"] == selected_model)):
                audit["internal_missing_residue_records"].append(missing)
    return selected, residue_ids


def parse_pdb(pdb_path: str, chain_id: Optional[str] = None,
              residue_info: Optional[list] = None, audit_info: Optional[dict] = None):
    """Return N, CA, C, mask and sequence, or None values for a rejected domain."""
    audit = audit_info if audit_info is not None else _new_audit(pdb_path)
    selected, residue_ids = _select_backbone_records(pdb_path, chain_id, audit)
    if audit["reasons"] or not selected:
        if not audit["reasons"]:
            _issue(audit, "no_valid_backbone")
        return (None, None, None, None, None)
    n_coords, ca_coords, c_coords, sequence = [], [], [], []
    for index in range(0, len(selected), 3):
        group = selected[index:index + 3]
        for destination, (line, _, _) in zip((n_coords, ca_coords, c_coords), group):
            destination.append([float(line[start:start + 8]) for start in (30, 38, 46)])
        line, segment, order = group[0]
        sequence.append(AA3_TO_AA1[line[17:20].strip()])
        if residue_info is not None:
            residue_info.append((line[21].strip(), segment, order))
    with np.errstate(over="ignore", invalid="ignore"):
        arrays = tuple(np.asarray(value, dtype=np.float32)
                       for value in (n_coords, ca_coords, c_coords))
    if not all(np.isfinite(value).all() for value in arrays):
        _issue(audit, "coordinates_not_representable_as_float32")
        return (None, None, None, None, None)
    audit["retained_residue_ids"] = residue_ids
    audit["numbering_gap_count"] = sum(
        int(right.split(":", 1)[1].split("@", 1)[0])
        > int(left.split(":", 1)[1].split("@", 1)[0]) + 1
        for left, right in zip(residue_ids[:-1], residue_ids[1:])
        if re.fullmatch(r"-?\d+", left.split(":", 1)[1].split("@", 1)[0])
        and re.fullmatch(r"-?\d+", right.split(":", 1)[1].split("@", 1)[0])
    )
    return (*arrays, np.ones(len(sequence), dtype=np.float32), "".join(sequence))


def _check_geometry(n, ca, c, residue_info, audit):
    for name, vectors in (("N_CA", ca - n), ("CA_C", c - ca)):
        distances = np.linalg.norm(vectors.astype(np.float64), axis=-1)
        invalid = np.flatnonzero((distances < INTRA_RESIDUE_BOND_MIN_DISTANCE)
                                 | (distances > INTRA_RESIDUE_BOND_MAX_DISTANCE))
        if len(invalid):
            _issue(audit, "abnormal_intra_residue_backbone_geometry",
                   {"bond": name, "indices": invalid.tolist(),
                    "distances": distances[invalid].tolist()})
    b, d = (ca - n).astype(np.float64), (c - ca).astype(np.float64)
    denominator = np.linalg.norm(b, axis=-1) * np.linalg.norm(d, axis=-1)
    sine = np.linalg.norm(np.cross(b, d), axis=-1) / np.maximum(denominator, 1e-12)
    if np.any(sine < MIN_BACKBONE_SINE):
        _issue(audit, "degenerate_local_backbone_frame",
               {"indices": np.flatnonzero(sine < MIN_BACKBONE_SINE).tolist()})
    connected = np.array([left[0] == right[0] and left[1] == right[1]
                          and right[2] == left[2] + 1
                          for left, right in zip(residue_info[:-1], residue_info[1:])],
                         dtype=bool)
    distances = np.linalg.norm((c[:-1] - n[1:]).astype(np.float64), axis=-1)
    connected &= ((distances > 1e-08)
                  & (distances < PEPTIDE_BOND_MAX_DISTANCE))
    residue_positions = []
    for identifier in audit["retained_residue_ids"]:
        match = re.fullmatch(r"([^:]*):(-?\d+)([^@]*)@\d+", identifier)
        residue_positions.append((int(match[2]), match[3]) if match else None)
    for missing in audit["internal_missing_residue_records"]:
        position = (missing["number"], missing["insertion_code"])
        for index, (left, right) in enumerate(zip(residue_positions[:-1], residue_positions[1:])):
            if left is not None and right is not None and left < position < right:
                connected[index] = False
    ca_distances = np.linalg.norm(ca[:, None, :].astype(np.float64)
                                  - ca[None, :, :].astype(np.float64), axis=-1)
    pairs = np.argwhere(np.triu(ca_distances < MIN_CA_SEPARATION, k=1))
    if len(pairs):
        _issue(audit, "overlapping_ca_positions", {"pairs": pairs.tolist()})
    return connected


def _validate_limits(min_len, max_len):
    if not isinstance(min_len, int) or not isinstance(max_len, int):
        raise ValueError("min_len and max_len must be integers")
    if not 0 <= min_len <= max_len <= MAX_LEN:
        raise ValueError("Require 0 <= min_len <= max_len <= 300 for this protocol")


def inspect_single_pdb(pdb_path: str, min_len: int = MIN_LEN, max_len: int = MAX_LEN,
                       chain_id: Optional[str] = None, reference_length: Optional[int] = None):
    """Inspect a whole domain and return NumPy features with validation details."""
    _validate_limits(min_len, max_len)
    audit = _new_audit(pdb_path)
    audit["reference_length"] = reference_length
    residue_info = []
    try:
        n, ca, c, mask, sequence = parse_pdb(pdb_path, chain_id, residue_info, audit)
    except (OSError, UnicodeError) as error:
        _issue(audit, "input_read_error", str(error))
        return None, audit
    N = audit["valid_backbone_count"]
    if reference_length is not None:
        matched = reference_length == audit["observed_residue_count"] == N
        audit["reference_length_check"] = "matched" if matched else "mismatch"
        if not matched:
            _issue(audit, "reference_length_mismatch",
                   {"reference": reference_length, "observed": audit["observed_residue_count"],
                    "complete_backbone": N})
    if N < min_len:
        _issue(audit, "too_short", {"valid_length": N, "minimum": min_len})
    if N > max_len:
        _issue(audit, "too_long", {"valid_length": N, "maximum": max_len})
    if ca is None or audit["reasons"]:
        audit["retained_residue_ids"] = []
        return None, audit
    peptide_bonds = _check_geometry(n, ca, c, residue_info, audit)
    if audit["reasons"]:
        audit["retained_residue_ids"] = []
        return None, audit
    cb = compute_virtual_cb(n, ca, c)
    n_to_ca, ca_to_c = compute_backbone_orientations(n, ca, c)
    phi, psi = compute_backbone_dihedrals(n, ca, c, peptide_bonds)
    if not all(np.isfinite(value).all() for value in (cb, n_to_ca, ca_to_c, phi, psi)):
        _issue(audit, "nonfinite_geometric_features")
        audit["retained_residue_ids"] = []
        return None, audit
    audit["status"] = "accepted"
    audit["final_length"] = N
    features = {
        "ca_coords": ca, "mask": mask, "length": N,
        "sequence": np.array([AA1_TO_IDX[aa] for aa in sequence], dtype=np.int64),
        "name": Path(pdb_path).stem, "n_coords": n, "c_coords": c, "cb_coords": cb,
        "n_to_ca": n_to_ca, "ca_to_c": ca_to_c, "phi": phi, "psi": psi,
        "residue_ids": list(audit["retained_residue_ids"]),
        "peptide_bonds": peptide_bonds,
        "preprocessing": {
            "protocol": "whole_domain_no_truncation_v1",
            "source_sha256": audit["source_sha256"],
            "observed_residue_count": audit["observed_residue_count"],
            "valid_backbone_count": N, "reference_length": reference_length,
            "reference_length_check": audit["reference_length_check"],
            "selected_chain": audit["selected_chain"], "selected_model": audit["selected_model"],
            "max_length": max_len, "truncated": False,
        },
    }
    return features, audit


def _to_tensors(features):
    import torch
    return {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
            for key, value in features.items()}


def process_single_pdb(pdb_path: str, min_len: int = MIN_LEN, max_len: int = MAX_LEN,
                       chain_id: Optional[str] = None, audit_info: Optional[dict] = None,
                       reference_length: Optional[int] = None):
    """Return torch tensors for an accepted domain, otherwise None."""
    features, audit = inspect_single_pdb(pdb_path, min_len, max_len, chain_id, reference_length)
    if audit_info is not None:
        audit_info.update(audit)
    return _to_tensors(features) if features is not None else None


def load_cath_domain_lengths(path):
    lengths = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 12:
                raise ValueError("Invalid CATH domain-list row {}".format(number))
            length = int(fields[10])
            if length <= 0 or fields[0] in lengths:
                raise ValueError("Invalid/duplicate CATH domain ID at row {}".format(number))
            lengths[fields[0]] = length
    if not lengths:
        raise ValueError("CATH domain list is empty")
    return lengths


def preprocess_directory(input_dir: str, output_dir: str, min_len: int = MIN_LEN,
                         max_len: int = MAX_LEN, chain_id: Optional[str] = None,
                         num_workers: int = 1, cath_domain_list=None):
    _validate_limits(min_len, max_len)
    input_path, output_path = Path(input_dir), Path(output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError("Input structure directory does not exist: {}".format(input_path))
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output directories must differ")
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError("Use an empty output directory; existing results are not overwritten: {}"
                              .format(output_path))
    pdb_files = sorted(path for path in input_path.iterdir()
                       if path.is_file() and (path.suffix.lower() in {".pdb", ".ent"}
                                              or "." not in path.name))
    if not pdb_files:
        raise ValueError("No .pdb, .ent, or extensionless domain files found")
    names = [path.stem.casefold() for path in pdb_files]
    if len(set(names)) != len(names):
        raise ValueError("Multiple inputs map to the same output domain name")
    references = load_cath_domain_lengths(cath_domain_list) if cath_domain_list else None
    import torch
    if references is None:
        logger.warning("No CATH reference lengths supplied: unobserved terminal coverage is not verified.")
    if num_workers != 1:
        logger.warning("This auditable implementation processes files sequentially (num_workers ignored).")
    output_path.mkdir(parents=True, exist_ok=True)
    counts, reason_counts = Counter(), Counter()
    for index, pdb_file in enumerate(pdb_files, 1):
        domain_id = pdb_file.stem
        expected = references.get(domain_id) if references is not None else None
        features, audit = inspect_single_pdb(str(pdb_file), min_len, max_len, chain_id, expected)
        if references is not None and expected is None:
            _issue(audit, "domain_id_not_in_cath_reference")
            audit.update(status="rejected", final_length=0, retained_residue_ids=[])
            features = None
        if features is not None:
            output_file = output_path / (domain_id + ".pt")
            created = False
            try:
                with output_file.open("xb") as destination:
                    created = True
                    torch.save(_to_tensors(features), destination)
                audit["output_file"] = output_file.name
                counts["saved"] += 1
            except Exception as error:
                if created:
                    output_file.unlink(missing_ok=True)
                audit.update(status="save_failed", final_length=0, retained_residue_ids=[])
                _issue(audit, "save_error", str(error))
        counts[audit["status"]] += 1
        reason_counts.update(audit["reasons"])
        if audit["reasons"]:
            logger.debug("%s: %s", domain_id, ", ".join(audit["reasons"]))
        if index % 100 == 0 or index == len(pdb_files):
            logger.info("Checked %d/%d: %d accepted, %d rejected, %d saved", index,
                        len(pdb_files), counts["accepted"], counts["rejected"], counts["saved"])

    stats = {
        "input_files": len(pdb_files), "accepted": counts["accepted"],
        "rejected": counts["rejected"], "save_failed": counts["save_failed"],
        "saved": counts["saved"], "reason_counts": dict(reason_counts),
        "truncated_count": 0,
        "min_length": min_len, "max_length": max_len,
    }
    logger.info("Finished: accepted=%d rejected=%d saved=%d save_failed=%d",
                stats["accepted"], stats["rejected"], stats["saved"], stats["save_failed"])
    if reason_counts:
        logger.info("Exclusion reasons: %s", dict(reason_counts))
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = preprocess_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        min_len=MIN_LEN,
        max_len=MAX_LEN,
        chain_id=CHAIN_ID,
        cath_domain_list=CATH_DOMAIN_LIST,
    )
    return 1 if summary["save_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
