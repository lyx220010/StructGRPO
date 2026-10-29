import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


CATH_FILE = Path("data/cath-domain-list-v4_2_0.txt")
PT_DIR = Path("data/cath42_pt_all")
OUTPUT_DIR = Path("data/cath42_pt_split")
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
SEED = 42
USE_SYMLINK = False

logger = logging.getLogger(__name__)


def load_cath_classification(cath_file: str) -> dict:
    cath_path = Path(cath_file)
    if not cath_path.is_file():
        raise FileNotFoundError(f"CATH classification file not found: {cath_path}")

    mapping = {}
    skipped_lines = 0
    with cath_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                skipped_lines += 1
                continue
            try:
                domain_id = parts[0]
                classification = tuple(int(value) for value in parts[1:5])
            except ValueError:
                skipped_lines += 1
                continue
            mapping[domain_id] = classification

    logger.info("Loaded classifications for %d domains from %s", len(mapping), cath_path)
    if skipped_lines:
        logger.warning("Skipped %d invalid classification records", skipped_lines)
    return mapping


def _validate_disjoint(file_groups: dict, topology_groups: dict):
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for label, groups in (("Domain files", file_groups), ("Topology groups", topology_groups)):
        for left, right in pairs:
            overlap = set(groups[left]) & set(groups[right])
            if overlap:
                message = (
                    f"{label} overlap between {left} and {right}: "
                    f"{len(overlap)} shared entries. Split validation failed."
                )
                logger.error(message)
                raise RuntimeError(message)
    logger.info("Split validation passed: domain files and topologies are disjoint")


def split_by_topology(
    pt_dir: str,
    cath_mapping: dict,
    output_dir: str,
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
    use_symlink: bool = USE_SYMLINK,
):
    random.seed(seed)
    input_path = Path(pt_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    pt_files = [path for path in input_path.glob("*.pt") if path.is_file()]
    if not pt_files:
        raise ValueError(f"No .pt files were found in the input directory: {input_path}")
    logger.info("Found %d .pt files", len(pt_files))

    topology_groups = defaultdict(list)
    missing_domains = []
    for pt_file in tqdm(pt_files, desc="Grouping domains by topology"):
        domain_id = pt_file.stem
        if domain_id in cath_mapping:
            topology_key = tuple(cath_mapping[domain_id][:3])
            topology_groups[topology_key].append(pt_file)
        else:
            missing_domains.append(domain_id)

    if not topology_groups:
        raise ValueError(
            "None of the input .pt files has a matching CATH classification. "
            "Check the classification file and domain identifiers."
        )
    logger.info("Grouped %d classified domains", len(pt_files) - len(missing_domains))
    if missing_domains:
        logger.warning(
            "Skipped %d domains without classifications; first entries: %s",
            len(missing_domains), missing_domains[:10],
        )

    all_topologies = list(topology_groups)
    topology_sizes = [len(files) for files in topology_groups.values()]
    logger.info(
        "Found %d topology groups; mean/min/max group size: %.1f/%d/%d",
        len(all_topologies), sum(topology_sizes) / len(all_topologies),
        min(topology_sizes), max(topology_sizes),
    )
    random.shuffle(all_topologies)

    n_topologies = len(all_topologies)
    n_train = int(n_topologies * train_ratio)
    n_test = int(n_topologies * test_ratio)
    subset_topologies = {
        "train": all_topologies[:n_train],
        "test": all_topologies[n_train:n_train + n_test],
        "val": all_topologies[n_train + n_test:],
    }
    subset_files = {
        subset: [
            pt_file.name
            for topology in topologies
            for pt_file in topology_groups[topology]
        ]
        for subset, topologies in subset_topologies.items()
    }
    _validate_disjoint(subset_files, subset_topologies)

    output_path = Path(output_dir)
    subset_dirs = {subset: output_path / subset for subset in subset_topologies}
    existing_dirs = [
        str(directory)
        for directory in subset_dirs.values()
        if directory.exists() and any(directory.iterdir())
    ]
    if existing_dirs:
        raise RuntimeError(
            "Output subset directories are not empty: "
            + ", ".join(existing_dirs)
            + ". Choose an empty output directory before running this script."
        )
    for directory in subset_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("Using %s mode", "symbolic link" if use_symlink else "file copy")
    for subset, topologies in subset_topologies.items():
        for topology in tqdm(topologies, desc=f"Writing {subset} subset"):
            for pt_file in topology_groups[topology]:
                destination = subset_dirs[subset] / pt_file.name
                if use_symlink:
                    os.symlink(os.path.abspath(pt_file), destination)
                else:
                    shutil.copy(pt_file, destination)

    for subset, filenames in subset_files.items():
        (output_path / f"{subset}.txt").write_text(
            "\n".join(sorted(filenames)), encoding="utf-8"
        )
        topology_names = [
            f"{topology[0]}.{topology[1]}.{topology[2]}"
            for topology in subset_topologies[subset]
        ]
        (output_path / f"{subset}_topologies.txt").write_text(
            "\n".join(sorted(topology_names)), encoding="utf-8"
        )

    split_info = {
        "split_strategy": "CATH Topology (C.A.T) - Random Split",
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": 1 - train_ratio - test_ratio,
        "test_ratio": test_ratio,
        "use_symlink": use_symlink,
    }
    for subset in ("train", "val", "test"):
        split_info[subset] = {
            "num_files": len(subset_files[subset]),
            "num_topologies": len(subset_topologies[subset]),
            "example_topologies": [
                f"{topology[0]}.{topology[1]}.{topology[2]}"
                for topology in subset_topologies[subset][:10]
            ],
        }
    split_info.update({
        "missing_domains": len(missing_domains),
        "total_processed": sum(len(files) for files in subset_files.values()),
        "total_pt_files": len(pt_files),
        "total_unique_topologies": n_topologies,
    })
    with (output_path / "split_info.json").open("w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2)

    logger.info("Split completed; output directory: %s", output_path)
    for subset in ("train", "test", "val"):
        logger.info(
            "%s: %d domains in %d topology groups",
            subset, len(subset_files[subset]), len(subset_topologies[subset]),
        )
    return split_info


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cath_mapping = load_cath_classification(CATH_FILE)
    split_by_topology(
        pt_dir=PT_DIR,
        cath_mapping=cath_mapping,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        use_symlink=USE_SYMLINK,
    )


if __name__ == "__main__":
    main()
