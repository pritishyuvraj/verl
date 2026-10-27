"""CPU lifecycle tests with explicit fake model/evaluator stages; never GPU runs."""

import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("generation_sft_cpu", ROOT / "verl/trainer/generation_sft.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def recipe(overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return compose(
            config_name="gemma_sft",
            overrides=[
                "model.path=/model",
                "model.tokenizer_path=/tokenizer",
                "data.train_files=/data/train.parquet",
                *overrides,
            ],
        )


class Driver:
    def __init__(self, scores=(0, 1, 2), broken_step=None):
        self.events = []
        self.scores = iter(scores)
        self.broken_step = broken_step

    def qualify(self, config, deadline):
        self.events.append(("qualify",))
        return {"task_ids": ["heldout-a", "heldout-b"], "final_evaluation_overlap": False}

    def assert_gpus_idle(self, config, deadline):
        self.events.append(("idle",))

    def train(self, resume_checkpoint, stop_after_step, config, deadline):
        assert config.trainer.total_training_steps == 40
        assert config.trainer.stop_after_step == stop_after_step
        assert config.trainer.test_freq == -1
        assert "generation_validation" not in config
        assert config.trainer.resume_from_path == resume_checkpoint
        self.events.append(("train", stop_after_step, resume_checkpoint))
        return f"/native/global_step_{stop_after_step}"

    def verify_checkpoint(self, path, step, config, deadline):
        self.events.append(("checkpoint", step))

    def evaluate(self, step, checkpoint, task_ids, config, deadline):
        self.events.append(("eval", step, tuple(task_ids)))
        return {
            "state": "completed",
            "step": step,
            "task_ids": list(task_ids),
            "solved_task_ids": list(task_ids)[: next(self.scores)],
            "cleanup_complete": step != self.broken_step,
            "diagnostic_nll": 0.0,
        }


def test_visible_batch_size_and_full_twenty_step_schedule():
    config = recipe()
    plan = workflow.plan(config)
    assert plan["global_batch_size"] == 32
    assert plan["micro_batch_size_per_gpu"] == 1
    assert plan["gradient_accumulation_steps"] == 4
    assert plan["generation_eval_steps"] == [0, 20, 40, 60, 80, 100, 120, 140, 157]
    assert config.trainer.save_freq == 20
    assert plan["full_dataset"] and not plan["executed"] and not plan["inputs_configured"]


def test_native_phase_exit_idle_full_eval_then_resume():
    config = recipe(["trainer.total_training_steps=40"])
    driver = Driver()
    result = workflow.run(config, driver)
    assert result["selected_step"] == 40 and result["selected_solved"] == 2
    assert not result["nll_used_for_selection"]
    assert driver.events == [
        ("qualify",),
        ("idle",),
        ("eval", 0, ("heldout-a", "heldout-b")),
        ("idle",),
        ("idle",),
        ("train", 20, None),
        ("checkpoint", 20),
        ("idle",),
        ("eval", 20, ("heldout-a", "heldout-b")),
        ("idle",),
        ("idle",),
        ("train", 40, "/native/global_step_20"),
        ("checkpoint", 40),
        ("idle",),
        ("eval", 40, ("heldout-a", "heldout-b")),
        ("idle",),
    ]


def test_failed_eval_cleanup_blocks_next_train():
    config = recipe(["trainer.total_training_steps=40"])
    driver = Driver(broken_step=20)
    with pytest.raises(ValueError, match="cleanup blocks"):
        workflow.run(config, driver)
    assert not any(event[:2] == ("train", 40) for event in driver.events)


def test_final_evaluation_overlap_blocks_before_any_gpu_phase():
    driver = Driver()
    driver.qualify = lambda **kwargs: {"task_ids": ["dev-task"], "final_evaluation_overlap": True}
    with pytest.raises(ValueError, match="disjoint"):
        workflow.run(recipe(), driver)
    assert driver.events == []


@pytest.mark.parametrize(
    "override",
    [
        "generation_validation.max_tasks=1",
        "generation_validation.full_dataset=false",
        "generation_inference.data_parallel_size=4",
        "trainer.test_freq=20",
        "data.train_batch_size=31",
        "generation_validation.include_untuned_step_zero=false",
    ],
)
def test_wrong_selection_or_parallelism_is_rejected(override):
    with pytest.raises(ValueError):
        workflow.plan(recipe([override]))


def test_generation_patience_uses_solved_counts_and_earlier_ties():
    config = recipe(["trainer.total_training_steps=40", "generation_validation.patience=1"])
    driver = Driver(scores=(1, 1, 2))
    result = workflow.run(config, driver)
    assert result["selected_step"] == 0
    assert [row["step"] for row in result["observations"]] == [0, 20]


def test_missing_qualified_evaluator_refuses_training():
    with pytest.raises(ValueError, match="no training started"):
        workflow.NativePhases().qualify(recipe(), 1e99)


def test_native_checkpoint_requires_every_rank_state_and_adapter(tmp_path):
    config = recipe()
    path = tmp_path / "global_step_20"
    path.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20")
    for rank in range(8):
        for kind in ("model", "optim", "extra_state"):
            (path / f"{kind}_world_size_8_rank_{rank}.pt").write_bytes(b"state")
        (path / f"data_{rank}.pt").write_bytes(b"loader")
    (path / "adapter").mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (path / "adapter" / name).write_bytes(b"adapter")
    driver = workflow.NativePhases()
    driver.verify_checkpoint(path, 20, config, 1e99)
    (path / "optim_world_size_8_rank_7.pt").unlink()
    with pytest.raises(ValueError, match="Missing native"):
        driver.verify_checkpoint(path, 20, config, 1e99)


def test_evaluation_has_separate_finite_budget_and_blocks_resume():
    config = recipe(["trainer.total_training_steps=40", "generation_validation.max_wall_seconds=5"])
    driver = Driver()
    moment = [0]
    original = driver.evaluate

    def too_slow(**kwargs):
        assert kwargs["deadline"] == 5
        result = original(**kwargs)
        moment[0] = 6
        return result

    driver.evaluate = too_slow
    with pytest.raises(TimeoutError, match="no training resume"):
        workflow.run(config, driver, clock=lambda: moment[0])
    assert not any(event[0] == "train" for event in driver.events)


def test_native_training_command_uses_eight_workers_and_preserves_resume(tmp_path, monkeypatch):
    config = recipe([f"trainer.default_local_dir={tmp_path}"])
    phase = workflow.native_phase_config(config, "/native/global_step_20", 40)
    launched = []

    class Process:
        def wait(self, timeout):
            return 0

    def launch(command, **kwargs):
        launched.append(command)
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(workflow.shutil, "which", lambda name: "/usr/bin/timeout")
    monkeypatch.setattr(workflow.subprocess, "Popen", launch)
    monkeypatch.setattr(workflow.time, "monotonic", lambda: 0)
    result = workflow.NativePhases(training_python="/train/python").train("/native/global_step_20", 40, phase, 1000)
    command = launched[0]
    assert "/train/python" in command
    assert "torch.distributed.run" in command and "--nproc_per_node=8" in command
    assert "verl.trainer.sft_trainer" in command
    saved = OmegaConf.load(tmp_path / "phases/step-000040/native_phase.yaml")
    assert saved.trainer.total_training_steps == 157 and saved.trainer.stop_after_step == 40
    assert saved.trainer.resume_mode == "resume_path"
    assert saved.trainer.resume_from_path == "/native/global_step_20"
    assert result == str(tmp_path / "global_step_40")
