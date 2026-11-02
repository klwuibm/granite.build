"""Convert intrinsics_training's tokenized answerability parquet into a staged
dataset dir the hf-peft-sft step can load.

The source (data/train/answerability/parquet_granite4.2_8b/{train,test}.parquet)
is the output of transform_data.py: pre-tokenized `input_ids` / `labels` /
`attention_mask` next to the conversational `prompt` / `completion` message
lists they were tokenized from. hf-peft-sft accepts the `prompt` + `completion`
format but must NOT see `input_ids`: TRL treats a dataset that already has
`input_ids` as tokenized and skips its own chat-template tokenization and
completion-only masking. So keep exactly `prompt` and `completion`.

Output layout (a dir `load_dataset(<dir>, split=...)` resolves):
    <out>/data/train.parquet
    <out>/data/test.parquet
    <out>/README.md   provenance: source sha256s, row counts

Usage:
    python -I convert_answerability_parquet.py <src_dir> <out_dir>
"""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

KEEP = ["prompt", "completion"]
SPLITS = ["train", "test"]
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_rows(table, split: str) -> None:
    """Every row: a non-empty prompt of messages ending in a user turn, and a
    single assistant completion. Fails on the first bad row."""
    prompts = table.column("prompt").to_pylist()
    completions = table.column("completion").to_pylist()
    for i, (p, c) in enumerate(zip(prompts, completions)):
        where = f"{split} row {i}"
        if not p or not all(isinstance(m, dict) and "role" in m for m in p):
            raise ValueError(f"{where}: prompt is not a non-empty message list")
        if p[-1]["role"] != "user":
            raise ValueError(f"{where}: prompt ends with role {p[-1]['role']!r}")
        if len(c) != 1 or c[0]["role"] != "assistant":
            raise ValueError(f"{where}: completion is not one assistant message")


def main(src: Path, out: Path) -> None:
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in SPLITS:
        f = src / f"{split}.parquet"
        with f.open("rb") as fh:
            if fh.read(len(LFS_POINTER)) == LFS_POINTER:
                raise SystemExit(f"{f} is a Git LFS pointer, not the parquet data")
        table = pq.read_table(f)
        missing = [c for c in KEEP if c not in table.column_names]
        if missing:
            raise SystemExit(
                f"{f}: missing columns {missing}; has {table.column_names}"
            )
        out_table = table.select(KEEP)
        check_rows(out_table, split)
        pq.write_table(out_table, data / f"{split}.parquet", compression="zstd")
        labels = out_table.column("completion").to_pylist()
        counts = {}
        for c in labels:
            counts[c[0]["content"]] = counts.get(c[0]["content"], 0) + 1
        report[split] = {
            "source": str(f),
            "source_sha256": sha256(f),
            "source_columns": table.column_names,
            "rows": out_table.num_rows,
            "completion_counts": counts,
        }
        print(json.dumps({split: report[split]}, indent=2))
    (out / "README.md").write_text(
        "# answerability prompt/completion (granite-4.2-8b)\n\n"
        "Converted from intrinsics_training "
        "`data/train/answerability/parquet_granite4.2_8b` by keeping only the "
        "`prompt` and `completion` columns (dropping the pre-tokenized "
        "`input_ids` / `labels` / `attention_mask`).\n\n"
        "```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
