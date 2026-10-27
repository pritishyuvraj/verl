"""CPU-only composition and lifecycle contracts; no model/GPU launches."""

import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sft_extensions_cpu", ROOT / "verl/trainer/sft_extensions.py")
extensions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extensions)


def recipe(overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return compose(config_name="gemma_sft", overrides=list(overrides))


def test_native_groups_and_model_data_extensions():
    config = recipe()
    assert config.model._target_ == "verl.workers.config.HFModelConfig"
    assert config.engine._target_ == "verl.workers.config.FSDPEngineConfig"
    assert config.optim._target_ == "verl.workers.config.FSDPOptimizerConfig"
    assert config.data.custom_cls.path == "pkg://verl.extensions.gemma.dataset"
    assert config.data.custom_cls.name == "GemmaMaskedSFTDataset"
    assert config.data.truncation == "error"
    assert config.data.use_dynamic_bsz is False
    assert config.model.use_remove_padding is False
    assert config.data.val_files is None
    assert config.trainer.test_freq == -1


def test_training_and_evaluation_overrides_are_independent():
    config = recipe(
        [
            "model.path=/model",
            "model.tokenizer_path=/tokenizer",
            "data.train_files=/train.parquet",
            "optim.lr=1e-5",
            "model.lora_rank=16",
            "trainer.stop_after_step=20",
            "generation_validation.patience=4",
            "generation_inference.episode_concurrency=128",
        ]
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    assert resolved["optim"]["lr"] == 1e-5
    assert resolved["model"]["lora_rank"] == 16
    assert resolved["generation_inference"]["episode_concurrency"] == 128
    assert resolved["generation_validation"]["patience"] == 4
    assert resolved["trainer"]["total_training_steps"] == 157
    assert resolved["trainer"]["stop_after_step"] == 20
    assert resolved["engine"]["seed"] == resolved["trainer"]["seed"]


def test_missing_real_panel_blocks_before_distributed_initialization():
    with pytest.raises(RuntimeError, match="qualification are pending"):
        extensions.validate_launch(recipe())
    extensions.validate_launch(OmegaConf.create({"trainer": {}}))


def test_sampling_and_generation_selection_contract():
    config = recipe()
    inference = config.generation_inference
    assert (inference.data_parallel_size, inference.tensor_parallel_size) == (8, 1)
    assert (inference.temperature, inference.top_p, inference.top_k) == (1.0, 0.95, 64)
    assert (inference.max_model_len, inference.max_output_tokens, inference.thinking_budget) == (32768, 16384, 4096)
    assert inference.language_model_only and inference.include_thoughts and inference.enable_thinking
    assert inference.dtype == inference.lora_dtype == "bfloat16"
    assert config.trainer.callbacks[-1].dtype == inference.lora_dtype
    assert config.generation_validation.selection_metric == "independently_tested_solved_count"
    assert config.generation_validation.include_untuned_step_zero
    assert config.generation_validation.nll_diagnostic_only
    assert config.trainer.max_ckpt_to_keep is None


@pytest.mark.parametrize("resume,total,stop,expected", [(0, 157, None, 157), (0, 157, 10, 10), (10, 157, 20, 20)])
def test_phase_boundary_does_not_change_schedule(resume, total, stop, expected):
    assert extensions.phase_stop_step(resume, total, stop) == expected


@pytest.mark.parametrize(
    "resume,total,stop", [(10, 157, 10), (0, 157, 158), (0, 157, 0), (0, 157, True), (157, 157, None)]
)
def test_invalid_phase_boundaries_fail(resume, total, stop):
    with pytest.raises(ValueError):
        extensions.phase_stop_step(resume, total, stop)


def test_optional_hook_order_and_failure_propagation():
    events = []

    class Callback:
        def on_checkpoint(self, step):
            events.append(step)

    extensions.notify([object(), Callback(), Callback()], "on_checkpoint", step=10)
    assert events == [10, 10]

    class Broken:
        def on_train_start(self):
            raise RuntimeError("No export")

    with pytest.raises(RuntimeError, match="No export"):
        extensions.notify([Broken()], "on_train_start")


def test_hydra_creates_callbacks_from_target():
    callbacks = extensions.build_callbacks([{"_target_": "collections.Counter", "count": 2}])
    assert callbacks[0]["count"] == 2
