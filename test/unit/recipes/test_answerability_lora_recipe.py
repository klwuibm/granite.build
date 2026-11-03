#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the granite-switch answerability-lora recipe.

The recipe is rendered with its parameters and then through hf-peft-sft's own
step-template blocks with the strict renderer, so a config error that would fail
the build at step creation fails here instead. The prepare target's command is
rendered the way the command step renders it and run on a small parquet with
the schema of the HF source (pre-tokenized columns next to prompt/completion).
"""

import pathlib
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters
from gbserver.utils.template import fill_template

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RECIPE = _REPO / "recipes" / "granite-switch" / "slurm" / "answerability-lora"
_STEP = _REPO / "steps" / "train" / "hf-peft-sft" / "skypilot"
_TEMPLATE = yaml.safe_load((_STEP / "step-template.yaml").read_text())
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["hf_peft_sft"][
    "config"
]
_SCRIPTS = ("hf_peft_sft_guard.sh", "hf_peft_sft_setup.sh", "hf_peft_sft_run.sh")


def _build(tmp_path, *params):
    text = apply_parameters(
        (_RECIPE / "build.yaml").read_text(),
        list(params),
        get_params_from_file(str(_RECIPE / "parameters.yaml")),
        str(tmp_path),
    )
    return yaml.safe_load(text)["granite.build"]


def _target(tmp_path, *params):
    return _build(tmp_path, *params)["targets"]["train"]


def _prepare(tmp_path, *params):
    return _build(tmp_path, *params)["targets"]["prepare"]


def _step_config(target):
    (step,) = target["steps"]
    assert step["step_uri"] == "space://steps/train/hf-peft-sft"
    return step["config"]


def _run_argv(target):
    """The trainer argv the step's run block hands hf_peft_sft_run.sh."""
    config = _step_config(target)
    cfg = {**_TEMPLATE["config"]["hf_peft_sft_config"], **config["hf_peft_sft_config"]}
    cfg["image_id"] = "docker:quay.io/x/img:t"
    ctx = {
        "config": {**config, "hf_peft_sft_config": cfg},
        "bindings": {
            "base_model": {"binding": {"path": "/staged/base"}},
            "dataset": {"binding": {"path": "/staged/ds"}},
        },
    }
    for block in ("setup", "run"):
        text = fill_template(_LAUNCHER[block], ctx, strict=True)
    for s in _SCRIPTS:
        stub = (
            'f() { for a in "$@"; do echo "ARG:$a"; done; }; f'
            if s == "hf_peft_sft_run.sh"
            else "true"
        )
        text = text.replace(f"bash /opt/hf-peft-sft/{s}", stub)
    proc = subprocess.run(["bash", "-c", text], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("ARG:")]


def _flags(argv):
    """`--key value` pairs after the `--` that ends the run script's own options."""
    rest = argv[argv.index("--") + 1 :]
    return dict(zip(rest[::2], rest[1::2]))


def test_environment_is_bluevela_slurm(tmp_path):
    target = _target(tmp_path)
    assert (
        target["environment_uri"] == "space://environments/skypilot/slurm/ibm-bluevela"
    )
    resources = _step_config(target)["launcher_config"]["resources"]
    assert resources["accelerators"].startswith("H200:")
    assert isinstance(resources["memory"], str)


def test_regular_lora_with_the_prepared_dataset(tmp_path):
    target = _target(tmp_path)
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert cfg["method"] == "lora"
    assert "dataset_name" not in cfg
    assert target["inputs"]["dataset"] == {"binding": "prepare.dataset"}


def test_prepare_pulls_only_the_parquet_dir_from_hf(tmp_path):
    from gbcommon.uri.hf import HfType, HfURI

    source = _prepare(tmp_path)["inputs"]["source"]
    assert source["type"] == "dataset"
    uri = HfURI.parse(source["uri"])
    assert uri.get_hf_type() == HfType.DATASET
    assert f"{uri.get_owner()}/{uri.get_repo()}" == "ibm-research/touma"
    assert uri.get_path_in_repo() == "training_data/answerability/parquet_granite4.2_8b"


def test_prepare_writes_to_shared_storage_on_the_cpu_partition(tmp_path):
    prepare = _prepare(tmp_path, "RUN_ID=abc")
    out = prepare["outputs"]["dataset"]
    assert out["uri"].startswith("env:///proj/data-eng/llmb-read-write/")
    assert out["uri"].endswith("/abc")
    (step,) = prepare["steps"]
    assert step["step_uri"] == "space://steps/command"
    assert step["config"]["zone"] == "cpu"
    assert step["config"]["launcher_config"]["sbatch_options"] == {"constraint": None}
    assert "accelerators" not in step["config"]["launcher_config"]["resources"]


def test_output_path_matches_the_output_uri(tmp_path):
    target = _target(tmp_path, "RUN_ID=abc")
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert target["outputs"][cfg["output"]]["type"] == "model"
    assert target["outputs"][cfg["output"]]["uri"] == "env://" + cfg["output_path"]
    assert cfg["output_path"].endswith("/abc")


def test_hyperparameters_mirror_the_source_config(tmp_path):
    """The answerability LoRA recipe: r16/a32, lr 5e-5 cosine, batch 32."""
    argv = _run_argv(_target(tmp_path))
    flags = _flags(argv)
    assert argv[argv.index("--method") + 1] == "lora"
    assert {k: flags[k] for k in ("--lora_r", "--lora_alpha", "--lora_dropout")} == {
        "--lora_r": "16",
        "--lora_alpha": "32",
        "--lora_dropout": "0.05",
    }
    assert flags["--lora_target_modules"] == "all-linear"
    expected = {
        "--max_length": "8192",
        "--per_device_train_batch_size": "1",
        "--gradient_accumulation_steps": "8",
        "--learning_rate": "5e-05",
        "--lr_scheduler_type": "cosine",
        "--warmup_steps": "0.05",
        "--weight_decay": "0.01",
        "--num_train_epochs": "3",
        "--seed": "0",
    }
    assert {k: flags[k] for k in expected} == expected
    assert flags["--dataset_path"] == "/staged/ds"


def test_clearml_off_by_default_declares_no_secrets(tmp_path):
    target = _target(tmp_path)
    argv = _run_argv(target)
    assert argv[argv.index("--clearml-mode") + 1] == "off"
    assert "skypilot" not in _step_config(target)


def test_clearml_on_declares_the_three_secrets(tmp_path):
    target = _target(tmp_path, "CLEARML=true")
    argv = _run_argv(target)
    assert argv[argv.index("--clearml-mode") + 1] == "on"
    declared = _step_config(target)["skypilot"]["secrets"][
        "secret_names_to_use_as_env_variable"
    ]
    assert {d["env_name"] for d in declared} == {
        "CLEARML_API_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
    }


def test_smoke_overrides_render(tmp_path):
    target = _target(
        tmp_path, "DATASET_SPLIT=train[:64]", "MAX_STEPS=4", "ACCELERATORS=H200:1"
    )
    flags = _flags(_run_argv(target))
    assert flags["--dataset_split"] == "train[:64]"
    assert flags["--max_steps"] == "4"


# ── prepare ──────────────────────────────────────────────────────────────────


def _prepare_script(tmp_path, src, out_root, *params):
    """prepare's command, rendered as the command step renders it, with the
    hf-peft-sft image's venv python swapped for this interpreter."""
    import sys

    step = _prepare(tmp_path, f"DATA_ROOT={out_root}", *params)["steps"][0]
    command = fill_template(
        step["config"]["command_config"]["command"],
        {"bindings": {"source": {"binding": {"path": str(src)}}}},
        strict=True,
    )
    return command.replace("/opt/hf-peft-sft/.venv/bin/python", sys.executable)


def _source_table(n, bad=()):
    """The HF source's columns: pre-tokenized ids next to prompt/completion.

    Rows in *bad* end their prompt with an `agent` turn, as 6 rows of the real
    train split do.
    """
    prompts = [
        [
            {"role": "user", "content": "You have access to the following documents:"},
            {"role": "agent" if i in bad else "user", "content": f"q{i}?"},
        ]
        for i in range(n)
    ]
    completions = [
        [
            {
                "role": "assistant",
                "content": '"answerable"' if i % 2 else '"unanswerable"',
            }
        ]
        for i in range(n)
    ]
    ids = pa.array([[1, 2, 3]] * n, type=pa.list_(pa.int32()))
    return pa.table(
        {
            "input_ids": ids,
            "labels": ids,
            "attention_mask": ids,
            "prompt": pa.array(prompts),
            "completion": pa.array(completions),
        }
    )


def _write_src(tmp_path, train_rows=6, bad=()):
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(_source_table(train_rows, bad), src / "train.parquet")
    pq.write_table(_source_table(2), src / "test.parquet")
    return src


def _run_prepare(tmp_path, src):
    script = _prepare_script(tmp_path, src, tmp_path / "data", "RUN_ID=r1")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_prepare_keeps_only_prompt_and_completion(tmp_path):
    proc = _run_prepare(tmp_path, _write_src(tmp_path))
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "data" / "r1"
    for split, rows in (("train", 6), ("test", 2)):
        table = pq.read_table(out / "data" / f"{split}.parquet")
        # No input_ids: TRL would take the data as pre-tokenized and skip its
        # chat-template tokenization and completion-only masking.
        assert table.column_names == ["prompt", "completion"]
        assert table.num_rows == rows
    readme = (out / "README.md").read_text()
    assert "source_sha256" in readme
    assert "hf:///datasets/ibm-research/touma/main/" in readme
    assert f"GB_ARTIFACT_ID:dataset GB_ARTIFACT_PATH:{out}" in proc.stdout


def test_prepare_output_matches_the_trl_prompt_completion_format(tmp_path):
    hf_train = pytest.importorskip("hf_peft_sft_train", reason="needs step src on path")
    assert _run_prepare(tmp_path, _write_src(tmp_path)).returncode == 0
    cols = pq.read_table(
        tmp_path / "data" / "r1" / "data" / "train.parquet"
    ).column_names
    assert hf_train.detect_format(cols) == "prompt_completion"


def test_prepare_fails_without_the_parquet(tmp_path):
    (tmp_path / "src").mkdir()
    proc = _run_prepare(tmp_path, tmp_path / "src")
    assert proc.returncode != 0
    assert "train.parquet not found" in proc.stderr
    assert "GB_ARTIFACT_ID" not in proc.stdout


def test_prepare_drops_and_records_a_prompt_not_ending_in_a_user_turn(tmp_path):
    proc = _run_prepare(tmp_path, _write_src(tmp_path, train_rows=200, bad={7}))
    assert proc.returncode == 0, proc.stderr
    assert "train: dropping 1 row(s)" in proc.stdout
    out = tmp_path / "data" / "r1"
    table = pq.read_table(out / "data" / "train.parquet")
    assert table.num_rows == 199
    assert "q7?" not in str(table.column("prompt").to_pylist())
    readme = (out / "README.md").read_text()
    assert '"row": 7' in readme and '"last_prompt_role": "agent"' in readme


def test_prepare_fails_when_too_many_rows_are_bad(tmp_path):
    proc = _run_prepare(tmp_path, _write_src(tmp_path, train_rows=50, bad={1, 2}))
    assert proc.returncode != 0
    assert "2 of 50 rows" in proc.stderr
    assert "GB_ARTIFACT_ID" not in proc.stdout
