# Answerability LoRA adapter (hf-peft-sft, BlueVela SLURM)

Trains a regular LoRA answerability adapter on `ibm-granite/granite-4.2-8b`
with the [`hf-peft-sft`](../../../../steps/train/hf-peft-sft/skypilot/USAGE.md)
step, on IBM BlueVela over SLURM. One target, `train`. It registers the
**adapter**, not a merged model, as the `adapter` artifact.

```bash
gb build start -f build.yaml --param-file parameters.yaml --param RUN_ID=<unique>
```

The data is intrinsics_training's
`data/train/answerability/parquet_granite4.2_8b`, converted (see below). The
hyperparameters mirror that repo's
`configs/answerability/granite_4.2_8b_lora_trl.yaml`.

## Before the first run: stage the data

The build reads the training data from BlueVela's shared filesystem
(`DATASET_URI`, an `env://` path). It does not fetch the data, so stage it
first:

1. **Get the parquet files.** They are Git LFS objects (`train.parquet`
   ~114 MB, `test.parquet` ~12 MB). A clone without LFS has only 133-byte
   pointer files, and the converter refuses those:

   ```bash
   git clone https://github.ibm.com/touma/intrinsics_training.git
   cd intrinsics_training
   git lfs pull --include='data/train/answerability/parquet_granite4.2_8b/*'
   ```

   Expected sha256 (the LFS oids):

   | file | sha256 |
   |---|---|
   | `train.parquet` | `b81f26558bd09933bd1125ee81fcf3116d2f1b9ae973efdbb30c94e9865f91a1` |
   | `test.parquet` | `7c9d38dc402f936f8b610a7c886be74c1cc45f5f4767df63c1eda396c8a05bca` |

2. **Convert.** This needs `pyarrow` only:

   ```bash
   python -I convert_answerability_parquet.py \
     <intrinsics_training>/data/train/answerability/parquet_granite4.2_8b \
     ./granite-4.2-8b-prompt-completion
   ```

   It writes `data/train.parquet`, `data/test.parquet` and a `README.md` that
   records the source sha256s, row counts and label counts.

3. **(Optional) Check the tokenization.** This needs `transformers` and
   `jinja2`, no GPU and no weights:

   ```bash
   python -I check_tokenization.py ibm-granite/granite-4.2-8b \
     <intrinsics_training>/data/train/answerability/parquet_granite4.2_8b/train.parquet 500
   ```

   It replays the step's tokenization on the converted rows and compares them
   with the stored `input_ids` / `labels`. All rows should match. See
   [What is trained](#what-is-trained).

4. **Copy it to BlueVela**, under the container-visible
   `/proj/data-eng/llmb-read-write` tree. SkyPilot's SLURM backend bind-mounts
   only that tree and home, not all of `/proj`:

   ```bash
   rsync -a ./granite-4.2-8b-prompt-completion/ \
     bluevela:/proj/data-eng/llmb-read-write/touma/data/answerability/granite-4.2-8b-prompt-completion/
   ```

   Then set `DATASET_URI` to `env://` plus that path, if it differs from the
   default.

## The data conversion

`parquet_granite4.2_8b` is what `transform_data.py` emits: pre-tokenized
`input_ids` / `labels` / `attention_mask`, plus the conversational `prompt` and
`completion` message lists they were tokenized from. hf-peft-sft accepts TRL's
`prompt` + `completion` format, so the converter keeps exactly those two
columns.

It must drop `input_ids`. TRL treats a dataset that already has `input_ids` as
tokenized and skips its own chat-template tokenization and completion masking.
The step's collator then has no `completion_mask`, and the run trains on the
whole sequence.

Nothing is lost by dropping the other columns:

- With `prompt_version: user_doc_message`, the documents are a user message
  inside `prompt`.
- The transform's per-row `chat_template_kwargs.documents` is not in the
  parquet, and Granite 4.2's chat template never reads `documents` anyway.

## What is trained

The loss is on the completion only. TRL's `completion_only_loss` defaults on
for `prompt` / `completion` data.

Granite 4.2's template ends a generation prompt with
`<|im_start|>assistant\n<think>\n`, but renders a finished assistant turn as
`<|im_start|>assistant\n<think></think>"answerable"<|im_end|>`. TRL starts the
completion where the two diverge, so the trained span is
`</think>"answerable"<|im_end|>\n` (or `"unanswerable"`).

The intrinsics transform tokenized through the same TRL path, so this should
equal the stored `labels`. `check_tokenization.py` verifies that on your copy
of the data.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `RUN_ID` | `it1-r1` | Names the build, the adapter dir (`$ADAPTER_ROOT/$RUN_ID`) and the ClearML task. Use a new one per run: re-using one replaces that run's adapter. |
| `BASE_MODEL_URI` | `hf:///ibm-granite/granite-4.2-8b` | Staged by an hfpull job on the cpu partition. If that job fails (it runs in a small cgroup), point this at an existing copy: `env:///proj/...`. |
| `DATASET_URI` | `env:///proj/data-eng/llmb-read-write/touma/data/answerability/granite-4.2-8b-prompt-completion` | The converted dir. |
| `DATASET_SPLIT` | `train` | `load_dataset` split; `train[:256]` for a smoke run. |
| `ADAPTER_ROOT` | `/proj/data-eng/llmb-read-write/touma/adapters/answerability-lora` | Shared, so the adapter survives teardown. |
| `ACCELERATORS` | `H200:4` | DDP over 4 GPUs, as in the source config's `gpus_per_node: 4`. Effective batch 4 × 1 × 8 = 32. |
| `MEMORY` | `512` | Host GiB. On SLURM, gbserver drops `compute_config` memory. |
| `NUM_TRAIN_EPOCHS` / `MAX_STEPS` | `3` / `-1` | A positive `MAX_STEPS` caps the run. |
| `CLEARML` | `false` | `true` turns tracking on and declares space secrets `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY`, which must exist. |

A smoke run, to try the plumbing on one GPU before the full run:

```bash
gb build start -f build.yaml --param-file parameters.yaml \
  --param RUN_ID=smoke1 --param ACCELERATORS=H200:1 \
  --param DATASET_SPLIT='train[:64]' --param MAX_STEPS=4
```

## Differences from `granite_4.2_8b_lora_trl.yaml`

| Source config | Here | Why |
|---|---|---|
| `eval_strategy: epoch`, `load_best_model_at_end`, `metric_for_best_model: eval_loss` | none; the final adapter is registered | The step does not evaluate. It passes `--save_strategy no`, and `test.parquet` is converted but not used. |
| `save_strategy: epoch` (a checkpoint per epoch) | one adapter at the end | The step owns `save_strategy`. |
| custom collator over the stored `input_ids` / `labels` | TRL re-tokenizes `prompt` / `completion` | The step takes text formats only. Pre-tokenized input is iteration 2. |
| over-length rows dropped at transform time | `max_length: 8192` truncates instead | The stored rows already fit 8192, so in practice nothing is truncated. |
| `gradient_checkpointing_kwargs: {use_reentrant: false}` | the step sets `use_reentrant` itself (`false` for LoRA + DDP) | The kwargs are step-owned. |
| `bf16: true` | `mixed_precision: bf16` | Same. |
