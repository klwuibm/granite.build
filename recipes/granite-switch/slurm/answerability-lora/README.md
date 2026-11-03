# Answerability LoRA adapter (hf-peft-sft, BlueVela SLURM)

Trains a regular LoRA answerability adapter on `ibm-granite/granite-4.2-8b`
with the [`hf-peft-sft`](../../../../steps/train/hf-peft-sft/skypilot/USAGE.md)
step, on IBM BlueVela over SLURM. It registers the **adapter**, not a merged
model, as the `adapter` artifact.

```bash
gb build start -f build.yaml --param-file parameters.yaml --param RUN_ID=<unique>
```

The build has two targets:

```
prepare ──(dataset)──> train
```

- **`prepare`** (CPU, `cpu` partition) pulls the answerability parquet from the
  private HF dataset repo `ibm-research/touma` and writes a prompt/completion
  copy to shared `/proj`. The `hf://` URI ends in a path inside the repo, so
  only that directory is downloaded, not the whole repo.
- **`train`** (4×H200, DDP) stages the base model, binds `prepare.dataset`,
  and trains the adapter.

## The one prerequisite: an `HF_TOKEN` secret

`ibm-research/touma` is private, so the pull needs a read-only token from an
account with access to the `ibm-research` org. The download runs on a cluster
node, which can't see your shell, so register the token once by name:

```bash
gb secret create HF_TOKEN --personal --value "$HF_TOKEN" --skip-version-check
gb secret list --personal --skip-version-check    # -> HF_TOKEN
```

`HF_TOKEN` is the credential name the Hugging Face assetstore looks for
(`token_secretname` in its `store.yaml`), so a secret under another name is not
found. The token never appears in `build.yaml`, `parameters.yaml`, or a command
line.

`ibm-research/touma` is a **dataset** repo, so the URI needs `datasets/`:
`hf:///datasets/ibm-research/touma/...`. Without it, the name resolves to a
model repo and the Hub answers 404 "Repository not found".

## The data

`SOURCE_DATA_URI` points at
`training_data/answerability/parquet_granite4.2_8b` in `ibm-research/touma`:

| file | sha256 |
|---|---|
| `train.parquet` (~114 MB, 12,849 rows) | `b81f26558bd09933bd1125ee81fcf3116d2f1b9ae973efdbb30c94e9865f91a1` |
| `test.parquet` (~12 MB, 1,428 rows) | `7c9d38dc402f936f8b610a7c886be74c1cc45f5f4767df63c1eda396c8a05bca` |

`prepare` records these hashes in the dataset's own `README.md`. To pin the
data, replace `main` in the URI with a commit hash.

### What `prepare` does

The parquet holds pre-tokenized `input_ids` / `labels` / `attention_mask` next
to the conversational `prompt` and `completion` message lists they were
tokenized from. hf-peft-sft accepts TRL's `prompt` + `completion` format, so
`prepare` keeps exactly those two columns.

It must drop `input_ids`. TRL treats a dataset that already has `input_ids` as
tokenized and skips its own chat-template tokenization and completion masking.
The step's collator then has no `completion_mask`, and the run trains on the
whole sequence.

It also drops any row whose prompt doesn't end in a user turn, because the
completion must answer one. Six train rows (507, 695, 997, 5041, 7418, 10176)
end with an `agent` turn, so 12,843 train rows remain. The dropped rows are
logged and listed in the dataset's `README.md`. If more than 1% of a split
would be dropped, `prepare` fails instead, since that points to a data
problem rather than stray rows.

The output is `$DATA_ROOT/$RUN_ID/data/{train,test}.parquet` on `/proj`, under
the container-visible `/proj/data-eng/llmb-read-write` tree. SkyPilot's SLURM
backend bind-mounts only that tree and home, not all of `/proj`.

### Checking the tokenization (optional)

`check_tokenization.py` replays the step's tokenization on the source rows
and compares the result with the stored `input_ids` / `labels`. It needs
`transformers` and `jinja2`, no GPU and no weights:

```bash
hf download ibm-research/touma --repo-type dataset \
  --include 'training_data/answerability/parquet_granite4.2_8b/*' --local-dir .
python -I check_tokenization.py ibm-granite/granite-4.2-8b \
  training_data/answerability/parquet_granite4.2_8b/train.parquet 500
```

All rows should match. See [What is trained](#what-is-trained).

## What is trained

The loss is on the completion only. TRL's `completion_only_loss` defaults on
for `prompt` / `completion` data.

Granite 4.2's template ends a generation prompt with
`<|im_start|>assistant\n<think>\n`, but renders a finished assistant turn as
`<|im_start|>assistant\n<think></think>"answerable"<|im_end|>`. TRL starts the
completion where the two diverge, so the trained span is
`</think>"answerable"<|im_end|>\n` (or `"unanswerable"`). This should equal the
stored `labels`; `check_tokenization.py` verifies that.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `RUN_ID` | `it1-r1` | Names the build, the dataset dir (`$DATA_ROOT/$RUN_ID`), the adapter dir (`$ADAPTER_ROOT/$RUN_ID`) and the ClearML task. Use a new one per run: re-using one replaces that run's outputs. |
| `SOURCE_DATA_URI` | `hf:///datasets/ibm-research/touma/main/training_data/answerability/parquet_granite4.2_8b` | The source parquet dir. The path after the revision limits the pull to it. |
| `DATA_ROOT` | `/proj/data-eng/llmb-read-write/touma/data/answerability-lora` | Where `prepare` writes, on shared `/proj`. |
| `PREP_IMAGE` / `PREP_MEMORY` | the hf-peft-sft image / `32` | `prepare` runs in the step image (its venv has pyarrow), so the node's image cache serves both targets. |
| `BASE_MODEL_URI` | `hf:///ibm-granite/granite-4.2-8b` | Staged by an hfpull job on the cpu partition. If that job fails (it runs in a small cgroup), point this at an existing copy: `env:///proj/...`. |
| `DATASET_SPLIT` | `train` | `load_dataset` split; `train[:256]` for a smoke run. |
| `ADAPTER_ROOT` | `/proj/data-eng/llmb-read-write/touma/adapters/answerability-lora` | Shared, so the adapter survives teardown. |
| `ACCELERATORS` | `H200:4` | DDP over 4 GPUs. Effective batch 4 × 1 × 8 = 32. |
| `MEMORY` | `512` | Host GiB for `train`. On SLURM, gbserver drops `compute_config` memory. |
| `NUM_TRAIN_EPOCHS` / `MAX_STEPS` | `3` / `-1` | A positive `MAX_STEPS` caps the run. |
| `CLEARML` | `false` | `true` turns tracking on and declares space secrets `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY`, which must exist. |

A smoke run, to try the plumbing on one GPU before the full run:

```bash
gb build start -f build.yaml --param-file parameters.yaml \
  --param RUN_ID=smoke1 --param ACCELERATORS=H200:1 \
  --param DATASET_SPLIT='train[:64]' --param MAX_STEPS=4
```

## What this recipe does not do

- **No evaluation, and no checkpoint per epoch.** hf-peft-sft does not
  evaluate and owns `save_strategy` (it passes `--save_strategy no`), so one
  adapter is registered at the end. `prepare` converts `test.parquet`, but
  `train` doesn't use it.
- **TRL re-tokenizes `prompt` / `completion`.** The step accepts only text
  formats, so the stored `input_ids` / `labels` are not used. Pre-tokenized
  input is planned for iteration 2.
- **`max_length: 8192` truncates; it doesn't drop rows.** The stored rows
  already fit in 8192 tokens, so in practice nothing is truncated.
- **`use_reentrant` is set by the step** (`false` for LoRA + DDP), and
  `mixed_precision: bf16` is how this step asks for bf16.
