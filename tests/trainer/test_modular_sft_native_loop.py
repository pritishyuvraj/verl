"""Run the actual native fit loop with CPU fakes in an installed verl runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf  # noqa: E402

from verl.trainer import sft_trainer as native  # noqa: E402


@pytest.mark.parametrize("resume,boundary,expected", [(0, 2, [1, 2]), (2, 4, [3, 4])])
def test_native_fit_preserves_horizon_and_checkpoints_before_callback(monkeypatch, resume, boundary, expected):
    events = []
    trainer = native.SFTTrainer.__new__(native.SFTTrainer)
    trainer.config = OmegaConf.create(
        {
            "model": {"use_remove_padding": False},
            "data": {
                "use_dynamic_bsz": False,
                "max_token_len_per_gpu": 32,
                "micro_batch_size_per_gpu": 1,
                "pad_mode": "no_padding",
            },
            "trainer": {
                "total_epochs": 1,
                "default_local_dir": "/not-written",
                "project_name": "cpu-test",
                "experiment_name": "step-timing",
                "logger": ["console"],
            },
        }
    )
    trainer.engine = SimpleNamespace(is_mp_src_rank_with_outputs=lambda: True, get_data_parallel_rank=lambda: 0)
    trainer.model_config = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    trainer.device_name = "cpu"
    trainer.rank = 0
    trainer.resume_global_step = resume
    trainer.total_training_steps = 5
    trainer.stop_after_step = boundary
    trainer.steps_per_epoch = 5
    trainer.global_batch_size = 1
    trainer.test_freq = 0
    trainer.save_freq = 1
    trainer.start_profile_step = trainer.end_profile_step = -1
    trainer.val_dataloader = None
    trainer.train_sampler = Mock()
    trainer.train_dataloader = [{} for _ in range(5 - resume)]
    trainer._get_batch_seqlens = lambda data: [2]
    trainer.training_client = SimpleNamespace(
        train_batch=lambda data: {"metrics": {"loss": 0.5}}, step_profile=lambda: None
    )
    trainer.ckpt_handler = SimpleNamespace(save_checkpoint=lambda step: events.append(("native", step)))

    class Callback:
        def on_train_start(self, trainer):
            events.append(("start", trainer.resume_global_step))

        def on_checkpoint(self, trainer, step, checkpoint_path):
            assert checkpoint_path == f"/not-written/global_step_{step}"
            events.append(("export", step))

    trainer.callbacks = [Callback()]
    monkeypatch.setattr(native, "aggressive_empty_cache", lambda **kwargs: None)
    monkeypatch.setattr(native, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(native.tu, "get_tensordict", lambda **kwargs: {})
    monkeypatch.setattr(native.tu, "assign_non_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(native.tu, "get", lambda output, key: output[key])
    clock = iter([10.0, 10.25, 20.0, 20.75])
    monkeypatch.setattr(native, "synchronized_time", lambda device: next(clock))
    tracking = Mock()
    monkeypatch.setattr(native, "Tracking", lambda **kwargs: tracking)
    trainer.fit()
    assert trainer.total_training_steps == 5
    assert events == [("start", resume)] + [(kind, step) for step in expected for kind in ("native", "export")]
    assert [call.kwargs["data"]["perf/step_time_seconds"] for call in tracking.log.call_args_list] == [0.25, 0.75]


def test_synchronized_timer_waits_before_reading_clock(monkeypatch):
    from verl.trainer import sft_extensions

    events = []
    monkeypatch.setattr(sft_extensions, "perf_counter", lambda: events.append("clock") or 42.0)
    device = SimpleNamespace(synchronize=lambda: events.append("synchronize"))
    assert sft_extensions.synchronized_time(device) == 42.0
    assert events == ["synchronize", "clock"]


def test_pending_validation_blocks_before_process_group_initialization(monkeypatch):
    initialized = Mock()
    monkeypatch.setattr("verl.utils.distributed.initialize_global_process_group", initialized)
    with pytest.raises(RuntimeError, match="qualification are pending"):
        native.run_sft(OmegaConf.create({"generation_validation": {"status": "integration_pending"}}))
    initialized.assert_not_called()


def test_process_group_cleanup_when_constructor_fails(monkeypatch):
    monkeypatch.setattr("verl.utils.distributed.initialize_global_process_group", Mock())
    destroyed = Mock()
    monkeypatch.setattr(native, "destroy_global_process_group", destroyed)
    monkeypatch.setattr(native, "SFTTrainer", Mock(side_effect=RuntimeError("Constructor failed")))
    with pytest.raises(RuntimeError, match="Constructor failed"):
        native.run_sft(OmegaConf.create({}))
    destroyed.assert_called_once()
