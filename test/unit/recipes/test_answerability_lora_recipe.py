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
the build at step creation fails here instead. The converter is run on a small
parquet with the schema intrinsics_training's transform_data.py writes.
"""

import importlib.util
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


def test_regular_lora_with_a_bound_dataset(tmp_path):
    cfg = _step_config(_target(tmp_path))["hf_peft_sft_config"]
    assert cfg["method"] == "lora"
    assert "dataset_name" not in cfg
    assert _target(tmp_path)["inputs"]["dataset"]["uri"].startswith("env:///proj/")


def test_output_path_matches_the_output_uri(tmp_path):
    target = _target(tmp_path, "RUN_ID=abc")
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert target["outputs"][cfg["output"]]["type"] == "model"
    assert target["outputs"][cfg["output"]]["uri"] == "env://" + cfg["output_path"]
    assert cfg["output_path"].endswith("/abc")


def test_hyperparameters_mirror_the_source_config(tmp_path):
    """intrinsics_training configs/answerability/granite_4.2_8b_lora_trl.yaml."""
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


# ── converter ────────────────────────────────────────────────────────────────


def _converter():
    spec = importlib.util.spec_from_file_location(
        "convert_answerability_parquet",
        _RECIPE / "convert_answerability_parquet.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _transform_table(n, prompt_end="user"):
    """The columns transform_data.py's make_record_table writes."""
    prompts = [
        [
            {"role": "user", "content": "You have access to the following documents:"},
            {"role": prompt_end, "content": f"q{i}?"},
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


def _write_src(tmp_path, **kw):
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(_transform_table(6, **kw), src / "train.parquet")
    pq.write_table(_transform_table(2, **kw), src / "test.parquet")
    return src


def test_converter_keeps_only_prompt_and_completion(tmp_path):
    out = tmp_path / "out"
    _converter().main(_write_src(tmp_path), out)
    for split, rows in (("train", 6), ("test", 2)):
        table = pq.read_table(out / "data" / f"{split}.parquet")
        # No input_ids: TRL would take the data as pre-tokenized and skip its
        # chat-template tokenization and completion-only masking.
        assert table.column_names == ["prompt", "completion"]
        assert table.num_rows == rows
    assert "source_sha256" in (out / "README.md").read_text()


def test_converter_output_matches_the_trl_prompt_completion_format(tmp_path):
    hf_train = pytest.importorskip("hf_peft_sft_train", reason="needs step src on path")
    out = tmp_path / "out"
    _converter().main(_write_src(tmp_path), out)
    cols = pq.read_table(out / "data" / "train.parquet").column_names
    assert hf_train.detect_format(cols) == "prompt_completion"


def test_converter_refuses_an_lfs_pointer(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "train.parquet").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 1\n"
    )
    with pytest.raises(SystemExit, match="LFS pointer"):
        _converter().main(src, tmp_path / "out")


def test_converter_refuses_a_prompt_not_ending_in_a_user_turn(tmp_path):
    with pytest.raises(ValueError, match="prompt ends with role 'assistant'"):
        _converter().main(
            _write_src(tmp_path, prompt_end="assistant"), tmp_path / "out"
        )
