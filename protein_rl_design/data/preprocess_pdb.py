import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
INPUT_DIR = Path('data/cath42/dompdb')
OUTPUT_DIR = Path('data/cath42_pt_all')
MIN_LEN = 0
MAX_LEN = 300
CHAIN_ID = None
PEPTIDE_BOND_MAX_DISTANCE = 1.8
STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
AA3_TO_AA1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
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
        cb[i] = ca[i] + cb_bond_length * (-np.cos(angle) * b + np.sin(angle) * np.cos(dihedral) * c_vec + np.sin(angle) * np.sin(dihedral) * a)
    return cb

def compute_backbone_orientations(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_to_ca = ca - n
    ca_to_c = c - ca
    n_to_ca = n_to_ca / (np.linalg.norm(n_to_ca, axis=-1, keepdims=True) + 1e-08)
    ca_to_c = ca_to_c / (np.linalg.norm(ca_to_c, axis=-1, keepdims=True) + 1e-08)
    return (n_to_ca, ca_to_c)

def compute_backbone_dihedrals(n: np.ndarray, ca: np.ndarray, c: np.ndarray, peptide_bonds: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    N = ca.shape[0]
    phi = np.zeros(N, dtype=np.float32)
    psi = np.zeros(N, dtype=np.float32)
    distances = np.linalg.norm(c[:-1] - n[1:], axis=-1)
    connected = np.isfinite(distances) & (distances > 1e-08) & (distances < PEPTIDE_BOND_MAX_DISTANCE)
    if peptide_bonds is not None:
        peptide_bonds = np.asarray(peptide_bonds, dtype=bool)
        if peptide_bonds.shape != (max(N - 1, 0),):
            raise ValueError('Peptide-bond indicators do not match the number of residues.')
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

def _select_backbone_records(pdb_path: str, chain_id: Optional[str]=None):
    residues = {}
    target_chain = chain_id
    model_started = False
    atoms_started = False
    segment = 0
    with open(pdb_path, 'r') as handle:
        for line in handle:
            record = line[:6].strip()
            if record == 'MODEL':
                if model_started or atoms_started:
                    break
                model_started = True
                continue
            if record in {'ENDMDL', 'END'}:
                break
            if record == 'TER':
                segment += 1
                continue
            if record != 'ATOM' or len(line) < 54:
                continue
            atoms_started = True
            chain = line[21].strip()
            if target_chain is None:
                target_chain = chain
            if chain != target_chain:
                continue
            res_num = line[22:26].strip()
            insertion_code = line[26].strip()
            position = (chain, segment, res_num, insertion_code)
            if position not in residues:
                residues[position] = (len(residues), {})
            order, variants = residues[position]
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            if res_name not in STANDARD_AA or atom_name not in {'N', 'CA', 'C'}:
                continue
            try:
                tuple((float(line[start:start + 8]) for start in (30, 38, 46)))
            except ValueError:
                continue
            try:
                occupancy = float(line[54:60].strip() or '0')
            except ValueError:
                occupancy = 0.0
            if not np.isfinite(occupancy):
                occupancy = 0.0
            altloc = line[16].strip()
            atoms = variants.setdefault((res_name, altloc), {})
            if atom_name not in atoms or occupancy > atoms[atom_name][1]:
                atoms[atom_name] = (line, occupancy)
    selected = []
    for position, (order, variants) in residues.items():
        candidates = []
        for (res_name, altloc), alternative_atoms in variants.items():
            atoms = dict(variants.get((res_name, ''), {}))
            atoms.update(alternative_atoms)
            if not {'N', 'CA', 'C'}.issubset(atoms):
                continue
            occupancy = sum((value[1] for value in alternative_atoms.values())) / len(alternative_atoms)
            priority = (-occupancy, altloc != '', altloc != 'A', altloc, res_name)
            candidates.append((priority, atoms))
        if not candidates:
            continue
        atoms = min(candidates, key=lambda candidate: candidate[0])[1]
        for atom_name in ('N', 'CA', 'C'):
            selected.append((atoms[atom_name][0], position[1], order))
    return selected

def parse_pdb(pdb_path: str, chain_id: Optional[str]=None, residue_info: Optional[list]=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    n_dict = {}
    ca_dict = {}
    c_dict = {}
    res_dict = {}
    residue_metadata = {}
    target_chain = chain_id
    selected_records = _select_backbone_records(pdb_path, chain_id)
    for line, segment, order in selected_records:
        if not line.startswith('ATOM'):
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21].strip()
        res_num = line[22:26].strip()
        if target_chain is None:
            target_chain = chain
        if chain != target_chain:
            continue
        if res_name in STANDARD_AA:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if atom_name == 'N':
                    n_dict[res_num] = (x, y, z)
                    res_dict[res_num] = res_name
                elif atom_name == 'CA':
                    ca_dict[res_num] = (x, y, z)
                    res_dict[res_num] = res_name
                elif atom_name == 'C':
                    c_dict[res_num] = (x, y, z)
                    res_dict[res_num] = res_name
                residue_metadata[res_num] = (chain, segment, order)
            except ValueError:
                continue
    if len(ca_dict) == 0:
        return (None, None, None, None, None, None)
    sorted_keys = sorted(ca_dict.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    valid_keys = [k for k in sorted_keys if k in n_dict and k in ca_dict and (k in c_dict)]
    if len(valid_keys) == 0:
        return (None, None, None, None, None, None)
    n_coords = np.array([n_dict[k] for k in valid_keys], dtype=np.float32)
    ca_coords = np.array([ca_dict[k] for k in valid_keys], dtype=np.float32)
    c_coords = np.array([c_dict[k] for k in valid_keys], dtype=np.float32)
    sequence = ''.join((AA3_TO_AA1.get(res_dict[k], 'X') for k in valid_keys))
    mask = np.ones(len(valid_keys), dtype=np.float32)
    if residue_info is not None:
        residue_info.extend((residue_metadata[k] for k in valid_keys))
    return (n_coords, ca_coords, c_coords, mask, sequence)

def process_single_pdb(pdb_path: str, min_len: int=MIN_LEN, max_len: int=MAX_LEN, chain_id: Optional[str]=None) -> Optional[dict]:
    residue_info = []
    try:
        n_coords, ca_coords, c_coords, mask, sequence = parse_pdb(pdb_path, chain_id, residue_info)
    except Exception as e:
        logger.debug(f'Failed to parse {pdb_path}: {e}')
        return None
    if ca_coords is None:
        return None
    N = len(ca_coords)
    if N < min_len:
        return None
    if N > max_len:
        n_coords = n_coords[:max_len]
        ca_coords = ca_coords[:max_len]
        c_coords = c_coords[:max_len]
        mask = mask[:max_len]
        sequence = sequence[:max_len]
        N = max_len
    residue_info = residue_info[:N]
    if np.any(np.isnan(ca_coords)) or np.any(np.isnan(n_coords)) or np.any(np.isnan(c_coords)):
        return None
    try:
        cb_coords = compute_virtual_cb(n_coords, ca_coords, c_coords)
        n_to_ca, ca_to_c = compute_backbone_orientations(n_coords, ca_coords, c_coords)
        peptide_bonds = np.array([left[0] == right[0] and left[1] == right[1] and (right[2] == left[2] + 1) for left, right in zip(residue_info[:-1], residue_info[1:])], dtype=bool)
        phi, psi = compute_backbone_dihedrals(n_coords, ca_coords, c_coords, peptide_bonds)
    except Exception as e:
        logger.debug(f'Failed to compute geometric features for {pdb_path}: {e}')
        return None
    seq_indices = np.array([AA1_TO_IDX.get(aa, AA1_TO_IDX['X']) for aa in sequence], dtype=np.int64)
    name = Path(pdb_path).stem
    return {'ca_coords': torch.tensor(ca_coords, dtype=torch.float32), 'mask': torch.tensor(mask, dtype=torch.float32), 'length': N, 'sequence': torch.tensor(seq_indices, dtype=torch.long), 'name': name, 'n_coords': torch.tensor(n_coords, dtype=torch.float32), 'c_coords': torch.tensor(c_coords, dtype=torch.float32), 'cb_coords': torch.tensor(cb_coords, dtype=torch.float32), 'n_to_ca': torch.tensor(n_to_ca, dtype=torch.float32), 'ca_to_c': torch.tensor(ca_to_c, dtype=torch.float32), 'phi': torch.tensor(phi, dtype=torch.float32), 'psi': torch.tensor(psi, dtype=torch.float32)}

def preprocess_directory(input_dir: str, output_dir: str, min_len: int=MIN_LEN, max_len: int=MAX_LEN, chain_id: Optional[str]=None, num_workers: int=1):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdb_files = list(input_path.glob('*.pdb')) + list(input_path.glob('*.ent')) + [f for f in input_path.iterdir() if f.is_file() and '.' not in f.name]
    if len(pdb_files) == 0:
        logger.error(f'No .pdb, .ent, or extensionless structure files found in {input_dir}.')
        return
    logger.info(f'Found {len(pdb_files)} structure files; starting preprocessing.')
    success = 0
    failed = 0
    skipped = 0
    for pdb_file in tqdm(pdb_files, desc='Processing structure files'):
        result = process_single_pdb(str(pdb_file), min_len=min_len, max_len=max_len, chain_id=chain_id)
        if result is None:
            skipped += 1
            continue
        out_file = output_path / f"{result['name']}.pt"
        try:
            torch.save(result, out_file)
            success += 1
        except Exception as e:
            logger.debug(f'Failed to save {out_file}: {e}')
            failed += 1
    logger.info(f'Processing completed: {success} saved, {skipped} skipped, {failed} save failures')
    logger.info(f'Output directory: {output_dir}')

def main():
    preprocess_directory(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, min_len=MIN_LEN, max_len=MAX_LEN, chain_id=CHAIN_ID)
if __name__ == '__main__':
    main()
