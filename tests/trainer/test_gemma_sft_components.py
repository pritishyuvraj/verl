"""CPU adapter export and canonical-mask rejection; no model downloads."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
from omegaconf import OmegaConf  # noqa: E402

from verl.extensions.gemma.callbacks import ExportAdapter  # noqa: E402
from verl.extensions.gemma.dataset import GemmaMaskedSFTDataset  # noqa: E402


def dataset(tmp_path, rows, **overrides):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    config = OmegaConf.create(
        {"pad_mode": "no_padding", "truncation": "error", "train_batch_size": 4, "max_length": 8, **overrides}
    )
    return GemmaMaskedSFTDataset(str(path), SimpleNamespace(pad_token_id=0), config)


def test_exact_tokens_and_neutral_batch_completion(tmp_path):
    data = dataset(tmp_path, [{"input_ids": [2, 3, 4], "loss_mask": [0, 0, 1]}])
    assert len(data) == 4 and data.real_count == 1
    assert data[0]["input_ids"].tolist() == [2, 3, 4]
    assert data[0]["loss_mask"].tolist() == [0, 0, 1]
    assert data[3]["loss_mask"].sum().item() == 0


@pytest.mark.parametrize(
    "ids,mask", [([1, 2, 3], [0, 2, 1]), ([1, 2], [1, 1]), ([1] * 9, [0] + [1] * 8), ([1, 2], [0, 0])]
)
def test_invalid_masks_and_overlength_rejected_whole(tmp_path, ids, mask):
    with pytest.raises(ValueError, match="Invalid canonical"):
        dataset(tmp_path, [{"input_ids": ids, "loss_mask": mask}])


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("finite", [True, False])
def test_portable_adapter_export_and_distributed_error_propagation(tmp_path, monkeypatch, finite, dtype):
    import peft
    from safetensors.torch import load_file

    module = torch.nn.Linear(2, 1, bias=False)
    if not finite:
        with torch.no_grad():
            module.weight.fill_(float("nan"))

    class PeftConfig:
        def save_pretrained(self, path):
            (path / "adapter_config.json").write_text("{}")

    module.peft_config = {"default": PeftConfig()}
    trainer = SimpleNamespace(rank=0, engine=SimpleNamespace(module=module))
    monkeypatch.setattr(peft, "get_peft_model_state_dict", lambda module, state_dict: state_dict)
    broadcast = Mock()
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    original = module.weight.detach().clone()
    if finite:
        ExportAdapter(dtype=dtype).on_checkpoint(trainer, 10, tmp_path)
        saved = load_file(tmp_path / "adapter/adapter_model.safetensors")["weight"]
        assert saved.dtype == getattr(torch, dtype)
        assert torch.equal(saved, original.to(dtype=saved.dtype))
        assert module.weight.dtype == torch.float32 and torch.equal(module.weight, original)
        assert (tmp_path / "adapter/training_step.txt").read_text() == "10\n"
        with pytest.raises(RuntimeError, match="FileExistsError"):
            ExportAdapter().on_checkpoint(trainer, 10, tmp_path)
    else:
        with pytest.raises(RuntimeError, match="nonfinite"):
            ExportAdapter().on_checkpoint(trainer, 10, tmp_path)
        assert not (tmp_path / "adapter").exists()
    assert broadcast.called


def test_invalid_export_dtype_rejected():
    with pytest.raises(ValueError, match="export dtype"):
        ExportAdapter(dtype="int4")


def test_bf16_cast_overflow_blocks_export(tmp_path, monkeypatch):
    import peft

    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(torch.finfo(torch.float32).max)
    trainer = SimpleNamespace(rank=0, engine=SimpleNamespace(module=module))
    monkeypatch.setattr(peft, "get_peft_model_state_dict", lambda module, state_dict: state_dict)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", Mock())
    with pytest.raises(RuntimeError, match="dtype conversion"):
        ExportAdapter(dtype="bfloat16").on_checkpoint(trainer, 20, tmp_path)
    assert not (tmp_path / "adapter").exists()
    assert torch.isfinite(module.weight).all()
