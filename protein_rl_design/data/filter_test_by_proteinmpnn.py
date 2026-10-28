import csv
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

INPUT_TEST_DIR = Path("data/cath42_pt_split/test")
PROTEINMPNN_TRAIN_FASTA = Path("data/proteinmpnn_train.fasta")
OUTPUT_TEST_DIR = Path("data/cath42_pt_filtered/test")
MMSEQS_EXECUTABLE = "mmseqs"
THREADS = max(1, min(8, os.cpu_count() or 1))
MAX_TARGET_LENGTH = 300
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_training_fasta(path):
    records = []
    seen = set()
    identifier = None
    fragments = []

    def append_record():
        sequence = "".join(fragments).upper()
        if not sequence or set(sequence) - set(ALPHABET):
            raise ValueError(f"Invalid amino acid sequence for training record: {identifier}")
        records.append((identifier, sequence))

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    append_record()
                header = line[1:].split()
                if not header or header[0] in seen:
                    raise ValueError(
                        f"Missing or duplicate training FASTA identifier at line {line_number}."
                    )
                identifier = header[0]
                seen.add(identifier)
                fragments = []
            else:
                if identifier is None:
                    raise ValueError(
                        "Training FASTA contains sequence data before its first header."
                    )
                fragments.append(line)
    if identifier is not None:
        append_record()
    if not records:
        raise ValueError("The ProteinMPNN training FASTA contains no sequences.")
    return records


def read_target(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or any(key not in data for key in ("sequence", "mask", "length")):
        raise ValueError(f"Target must contain native sequence, mask, and length: {path}")
    length = int(data["length"])
    if length != data["length"] or not 1 <= length <= MAX_TARGET_LENGTH:
        raise ValueError(
            f"Target length must be between 1 and {MAX_TARGET_LENGTH} without truncation: {path}"
        )
    mask = torch.as_tensor(data["mask"])
    if mask.ndim != 1 or len(mask) != length or not torch.isfinite(mask).all():
        raise ValueError(f"Invalid valid-position mask: {path}")
    if not ((mask == 0) | (mask == 1)).all() or not mask.bool().any():
        raise ValueError(f"Mask must contain binary values and at least one valid position: {path}")
    native = data["sequence"]
    if isinstance(native, str):
        sequence = native.upper()
        if len(sequence) != length:
            raise ValueError(f"Native sequence length does not match target length: {path}")
        sequence = "".join(sequence[index] for index in range(length) if mask[index])
    else:
        indices = torch.as_tensor(native)
        if indices.ndim != 1 or len(indices) != length:
            raise ValueError(f"Native sequence indices do not match target length: {path}")
        selected = indices[mask.bool()]
        if not torch.isfinite(selected).all() or not ((selected >= 0) & (selected < 20)).all():
            raise ValueError(f"Native sequence contains invalid amino acid indices: {path}")
        if not (selected == selected.long()).all():
            raise ValueError(f"Native sequence indices must be integers: {path}")
        sequence = "".join(ALPHABET[index] for index in selected.long().tolist())
    if set(sequence) - set(ALPHABET[:20]):
        raise ValueError(f"Valid target positions must contain standard amino acids: {path}")
    return {"target_id": path.stem, "path": path, "sequence": sequence, "length": len(sequence)}


def run_command(command, log_path):
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"MMseqs2 command failed with exit code {result.returncode}. See {log_path}."
        )


def search_alignments(executable, targets, references, work_dir, log_path):
    query_fasta = work_dir / "queries.fasta"
    reference_fasta = work_dir / "training.fasta"
    with query_fasta.open("w", encoding="ascii") as handle:
        for index, target in enumerate(targets):
            handle.write(f">q{index:08d}\n{target['sequence']}\n")
    with reference_fasta.open("w", encoding="ascii") as handle:
        for index, (_, sequence) in enumerate(references):
            handle.write(f">r{index:08d}\n{sequence}\n")
    query_db = str(work_dir / "query_db")
    reference_db = str(work_dir / "reference_db")
    result_db = str(work_dir / "result_db")
    hits_path = work_dir / "alignments.tsv"
    common = ["--threads", str(THREADS)]
    commands = [
        [executable, "createdb", str(query_fasta), query_db, "--dbtype", "1", "--shuffle", "0"],
        [
            executable,
            "createdb",
            str(reference_fasta),
            reference_db,
            "--dbtype",
            "1",
            "--shuffle",
            "0",
        ],
        [
            executable,
            "search",
            query_db,
            reference_db,
            result_db,
            str(work_dir / "search_tmp"),
            "--prefilter-mode",
            "2",
            "--alignment-mode",
            "3",
            "-a",
            "1",
            "--search-type",
            "1",
            "--min-seq-id",
            "0.0",
            "-c",
            "0.0",
            "-e",
            "1e30",
            "--max-seqs",
            str(len(references)),
            "--max-accept",
            str(len(references)),
            "--max-rejected",
            str(len(references)),
            "--mask",
            "0",
            "--comp-bias-corr",
            "0",
            "--max-seq-len",
            str(max(65535, max(len(sequence) for _, sequence in references))),
        ]
        + common,
        [
            executable,
            "convertalis",
            query_db,
            reference_db,
            result_db,
            str(hits_path),
            "--format-output",
            "query,target,nident,qlen,alnlen,qstart,qend",
        ]
        + common,
    ]
    for command in commands:
        run_command(command, log_path)
    if not hits_path.is_file():
        raise RuntimeError("MMseqs2 did not produce the alignment table.")
    return hits_path, commands


def best_query_alignments(hits_path, targets, references, archive_path):
    best = {}
    with (
        hits_path.open("r", encoding="utf-8") as source,
        gzip.open(archive_path, "wt", encoding="utf-8") as archive,
    ):
        archive.write("query\ttarget\tnident\tqlen\talnlen\tqstart\tqend\n")
        for line_number, line in enumerate(source, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 7:
                raise ValueError(f"Invalid alignment table at line {line_number}.")
            query_id, reference_id = fields[:2]
            try:
                query_index = int(query_id[1:])
                reference_index = int(reference_id[1:])
                nident, qlen, alnlen, qstart, qend = map(int, fields[2:])
            except ValueError as error:
                raise ValueError(f"Invalid alignment fields at line {line_number}.") from error
            if (
                query_id != f"q{query_index:08d}"
                or reference_id != f"r{reference_index:08d}"
                or not 0 <= query_index < len(targets)
                or not 0 <= reference_index < len(references)
                or qlen != targets[query_index]["length"]
                or not 0 <= nident <= min(qlen, alnlen)
                or not 1 <= qstart <= qend <= qlen
                or qend - qstart + 1 > alnlen
            ):
                raise ValueError(
                    f"Alignment does not match the supplied sequences at line {line_number}."
                )
            archive.write(line if line.endswith("\n") else line + "\n")
            hit = {
                "reference_index": reference_index,
                "nident": nident,
                "alnlen": alnlen,
                "qstart": qstart,
                "qend": qend,
            }
            previous = best.get(query_index)
            if previous is None or (-nident, reference_index) < (
                -previous["nident"],
                previous["reference_index"],
            ):
                best[query_index] = hit
    return best


def build_report_rows(targets, references, best):
    rows = []
    for index, target in enumerate(targets):
        length = target["length"]
        threshold = 40 if length < 150 else 30
        hit = best.get(index)
        nident = hit["nident"] if hit else 0
        exclude = 100 * nident > threshold * length
        rows.append(
            {
                "query_id": f"q{index:08d}",
                "target_id": target["target_id"],
                "valid_length": length,
                "best_training_id": references[hit["reference_index"]][0] if hit else "",
                "nident": nident,
                "alignment_length": hit["alnlen"] if hit else 0,
                "query_normalized_identity": nident / length,
                "alignment_identity": nident / hit["alnlen"] if hit else 0.0,
                "query_coverage": (hit["qend"] - hit["qstart"] + 1) / length if hit else 0.0,
                "threshold_percent": threshold,
                "alignment_status": "reported_alignment" if hit else "no_reported_alignment",
                "decision": "excluded" if exclude else "retained",
                "source_sha256": file_sha256(target["path"]),
            }
        )
    return rows


def filter_test_set(
    input_dir=INPUT_TEST_DIR,
    training_fasta=PROTEINMPNN_TRAIN_FASTA,
    output_dir=OUTPUT_TEST_DIR,
    executable=MMSEQS_EXECUTABLE,
):
    input_dir = Path(input_dir).resolve()
    training_fasta = Path(training_fasta).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input test directory does not exist: {input_dir}")
    if not training_fasta.is_file():
        raise FileNotFoundError(
            f"Provide native sequences from the ProteinMPNN training subset in FASTA format: {training_fasta}"
        )
    if output_dir == input_dir or output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Choose a new output directory; existing data will not be overwritten: {output_dir}"
        )
    executable_path = shutil.which(str(executable))
    if executable_path is None:
        raise FileNotFoundError(
            "MMseqs2 is required. Install release 15 or newer and make mmseqs available on PATH."
        )
    version_result = subprocess.run(
        [executable_path, "version"], check=True, capture_output=True, text=True
    )
    mmseqs_version = version_result.stdout.strip()
    files = sorted(path for path in input_dir.glob("*.pt") if path.is_file())
    if not files:
        raise ValueError(f"No .pt target files found in {input_dir}.")
    targets = [read_target(path) for path in files]
    references = read_training_fasta(training_fasta)
    logger.info(
        "Loaded %d test targets and %d supplied ProteinMPNN training sequences.",
        len(targets),
        len(references),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="proteinmpnn_identity_", dir=output_dir.parent
    ) as temporary:
        work_dir = Path(temporary)
        staged = work_dir / "filtered_test"
        staged.mkdir()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=output_dir.name + "_mmseqs_",
            suffix=".log",
            dir=output_dir.parent,
            delete=False,
        ) as handle:
            log_path = Path(handle.name)
            handle.write("MMseqs2 " + mmseqs_version + "\n")
        hits_path, commands = search_alignments(
            executable_path, targets, references, work_dir, log_path
        )
        best = best_query_alignments(hits_path, targets, references, staged / "alignments.tsv.gz")
        rows = build_report_rows(targets, references, best)
        with (staged / "identity_filter.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counts = {"retained": 0, "excluded": 0}
        for decision in counts:
            with (staged / (decision + "_targets.txt")).open("w", encoding="utf-8") as handle:
                for target, row in zip(targets, rows):
                    if row["decision"] == decision:
                        handle.write(target["target_id"] + "\n")
                        counts[decision] += 1
                        if decision == "retained":
                            shutil.copy2(target["path"], staged / target["path"].name)
        with (staged / "training_reference_ids.tsv").open("w", encoding="utf-8") as handle:
            handle.write("reference_id\ttraining_id\n")
            for index, (identifier, _) in enumerate(references):
                handle.write(f"r{index:08d}\t{identifier}\n")
        metadata = {
            "rule": "maximum nident / valid target length among reported MMseqs2 protein local alignments",
            "prefilter": "exhaustive, --prefilter-mode 2",
            "short_target_rule": "valid length < 150; exclude only when identity > 0.40",
            "medium_long_target_rule": "150 <= valid length <= 300; exclude only when identity > 0.30",
            "reference_requirement": "native sequences from the actual ProteinMPNN training subset",
            "training_fasta_sha256": file_sha256(training_fasta),
            "training_reference_count": len(references),
            "input_target_count": len(targets),
            "counts": counts,
            "mmseqs_version": mmseqs_version,
            "commands": commands,
        }
        with (staged / "identity_filter.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        shutil.move(str(log_path), str(staged / "mmseqs.log"))
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output directory appeared during filtering: {output_dir}")
        staged.rename(output_dir)
    logger.info(
        "Filtering completed: %d retained, %d excluded. Output: %s",
        counts["retained"],
        counts["excluded"],
        output_dir,
    )
    return metadata


def main():
    filter_test_set()


if __name__ == "__main__":
    main()
