"""Sequential native SFT / full generation-test validation controller.

The parent owns no model. A driver trains in a subprocess to the requested
boundary, waits for all training ranks to exit, proves GPU release, runs full
generation+grading, and proves cleanup before returning. Driver methods must be
synchronous and respect the shared absolute deadline. The concrete SWE driver
and execution-qualified held-out panel remain required inputs, never invented
by this controller. Native SFT alone does not implement generation validation.

Driver API (Hydra _target_):
  qualify(config, deadline) -> {task_ids, final_evaluation_overlap: false}
  assert_gpus_idle(config, deadline)
  evaluate(step, checkpoint, task_ids, config, deadline) ->
      {state: completed, step, task_ids, solved_task_ids, cleanup_complete: true}
  train(resume_checkpoint, stop_after_step, config, deadline) -> checkpoint path
  verify_checkpoint(path, step, config, deadline)

evaluate must use the supplied native vLLM DP/TP and sampler settings, retain
one multi-turn prediction/task, execute independent tests, verify their actual
artifacts, and return only after all GPU processes/owned containers are drained.
No real evaluation driver is bundled until those obligations are qualified.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict


def plan(config):
    total = config.trainer.total_training_steps
    frequency = config.generation_validation.every_steps
    batch = config.data.train_batch_size
    micro = config.data.micro_batch_size_per_gpu
    replicas = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if any(type(value) is not int or value <= 0 for value in (total, frequency, batch, micro, replicas)):
        raise ValueError("Steps, frequency, batch sizes and GPU count must be positive integers")
    if batch % (micro * replicas):
        raise ValueError("Global batch must be divisible by microbatch times training DP size")
    if config.engine.ulysses_sequence_parallel_size != 1:
        raise ValueError("This Gemma recipe uses training DP with sequence parallel size one")
    if config.trainer.nnodes != 1 or replicas != 8:
        raise ValueError("This recipe requires one node with all eight GPUs")
    if (config.generation_inference.data_parallel_size, config.generation_inference.tensor_parallel_size) != (8, 1):
        raise ValueError("Generation validation requires native vLLM DP8/TP1")
    if config.generation_inference.engine != "vllm":
        raise ValueError("Generation validation requires vLLM")
    if not config.generation_validation.full_dataset or config.generation_validation.max_tasks is not None:
        raise ValueError("Evaluate the complete qualified held-out panel without task sampling")
    if config.trainer.test_freq != -1 and (
        not config.generation_validation.nll_diagnostic_only or not config.data.val_files
    ):
        raise ValueError("Native NLL requires validation data and must remain diagnostic only")
    if config.generation_validation.selection_metric != "independently_tested_solved_count":
        raise ValueError("Only executable-test solved count can select checkpoints")
    if not config.generation_validation.include_untuned_step_zero:
        raise ValueError("The untuned baseline must be evaluated before training")
    if type(config.generation_validation.patience) is not int or config.generation_validation.patience < 1:
        raise ValueError("Generation patience must be a positive integer")
    if config.trainer.resume_mode != "disable":
        raise ValueError("Workflow restart needs a verified history; automatic fresh reruns are forbidden")
    if config.trainer.max_ckpt_to_keep is not None:
        raise ValueError("Retain checkpoints until generation-based best selection completes")
    if not 0 < config.workflow.max_wall_seconds <= 86400:
        raise ValueError("A workflow requires a finite wall budget of at most one day")
    steps = list(range(frequency, total + 1, frequency))
    if not steps or steps[-1] != total:
        steps.append(total)
    return {
        "mode": "generation_and_executable_tests",
        "global_batch_size": batch,
        "micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": batch // (micro * replicas),
        "generation_eval_steps": [0, *steps],
        "full_dataset": True,
        "data_parallel_size": 8,
        "tensor_parallel_size": 1,
        "phase_order": [
            "train",
            "native_checkpoint",
            "training_exit",
            "gpu_release",
            "generate_and_grade",
            "eval_cleanup",
            "resume",
        ],
        "inputs_configured": (
            config.generation_validation.driver is not None
            and config.generation_validation.driver.get("evaluator") is not None
            and config.generation_validation.panel is not None
        ),
        "executed": False,
    }


def native_phase_config(config, resume_checkpoint, boundary):
    """Keep the full optimizer horizon; restore native model/optimizer/RNG/loader state."""
    phase = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    with open_dict(phase):
        del phase["generation_validation"]
        del phase["generation_inference"]
        del phase["workflow"]
    phase.trainer.stop_after_step = boundary
    phase.trainer.resume_mode = "disable" if resume_checkpoint is None else "resume_path"
    phase.trainer.resume_from_path = resume_checkpoint
    return phase


class NativePhases:
    """Native torchrun phases; the domain evaluator owns generation and grading.

    An evaluator is mandatory. It must qualify source/holdout/environment pins,
    enforce deadline/DP8/sampling, authenticate actual grading, and prove GPU and
    container cleanup. Those domain obligations cannot be replaced by a config
    boolean or NLL result. Training retains the upstream native fit loop.
    """

    def __init__(self, training_python=None, evaluator=None):
        self.training_python = training_python or sys.executable
        self.evaluator = evaluator

    def qualify(self, config, deadline):
        if self.evaluator is None:
            raise ValueError("Missing qualified SWE generation/grading evaluator; no training started")
        return self.evaluator.qualify(config=config, deadline=deadline)

    def assert_gpus_idle(self, config, deadline):
        self.evaluator.assert_gpus_idle(config=config, deadline=deadline)

    def evaluate(self, **kwargs):
        return self.evaluator.evaluate(**kwargs)

    def train(self, resume_checkpoint, stop_after_step, config, deadline):
        # Native resume consumes model, optimizer, RNG and loader state. The
        # driver cannot reduce total_training_steps to the phase boundary.
        if config.trainer.resume_from_path != resume_checkpoint:
            raise ValueError("Native resume path differs from the previous checkpoint")
        timeout = shutil.which("timeout")
        remaining = deadline - time.monotonic()
        if timeout is None or remaining <= 90:
            raise ValueError("Native phases require GNU timeout and at least 90 seconds cleanup headroom")
        root = Path(config.trainer.default_local_dir).resolve()
        phase_dir = root / "phases" / f"step-{stop_after_step:06d}"
        phase_dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(config, phase_dir / "native_phase.yaml")
        command = [
            timeout,
            "--signal=TERM",
            "--kill-after=30s",
            f"{int(remaining - 60)}s",
            self.training_python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={config.trainer.n_gpus_per_node}",
            "--module",
            "verl.trainer.sft_trainer",
            "--config-path",
            str(phase_dir),
            "--config-name",
            "native_phase",
        ]
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
        with (phase_dir / "train.log").open("x") as log:
            # GNU timeout owns this new process group and forwards termination.
            # Any nonzero exit stops the workflow; no automatic training retry.
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=max(1, min(30, deadline - time.monotonic())))
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)
        return str(root / f"global_step_{stop_after_step}")

    def verify_checkpoint(self, path, step, config, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("Native checkpoint verification deadline exceeded")
        checkpoint = Path(path)
        if checkpoint.name != f"global_step_{step}" or checkpoint.is_symlink():
            raise ValueError("Unexpected native checkpoint path")
        world = config.trainer.n_gpus_per_node
        required = [
            checkpoint / f"{kind}_world_size_{world}_rank_{rank}.pt"
            for rank in range(world)
            for kind in ("model", "optim", "extra_state")
        ]
        required += [checkpoint / f"data_{rank}.pt" for rank in range(world)]
        required += [checkpoint / "adapter" / name for name in ("adapter_model.safetensors", "adapter_config.json")]
        if any(path.is_symlink() or not path.is_file() or path.stat().st_size == 0 for path in required):
            raise ValueError("Missing native rank/optimizer/RNG/dataloader/portable-adapter state")
        tracker = checkpoint.parent / "latest_checkpointed_iteration.txt"
        if tracker.is_symlink() or tracker.read_text().strip() != str(step):
            raise ValueError("Native checkpoint tracker differs from the completed phase")


def run(config, driver, *, clock=time.monotonic, record=lambda value: None):
    schedule = plan(config)
    deadline = clock() + config.workflow.max_wall_seconds

    def remaining():
        if clock() >= deadline:
            raise TimeoutError("Generation SFT workflow deadline exceeded; no next phase authorized")

    remaining()
    qualification = driver.qualify(config=config, deadline=deadline)
    task_ids = qualification["task_ids"]
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task, str) or not task for task in task_ids)
        or len(set(task_ids)) != len(task_ids)
        or (
            qualification.get("final_evaluation_overlap") is not False
            and not (
                qualification.get("final_evaluation_overlap") is True
                and config.generation_validation.get("allow_dev_selection") is True
            )
        )
    ):
        raise ValueError("A qualified panel must be disjoint unless dev-set selection is explicitly enabled")
    best_step, best_solved, bad = 0, -1, 0
    checkpoint = None
    observations = []
    for step in schedule["generation_eval_steps"]:
        remaining()
        driver.assert_gpus_idle(config=config, deadline=deadline)
        if step:
            phase = native_phase_config(config, checkpoint, step)
            checkpoint = driver.train(
                resume_checkpoint=checkpoint, stop_after_step=step, config=phase, deadline=deadline
            )
            driver.verify_checkpoint(path=checkpoint, step=step, config=phase, deadline=deadline)
            remaining()
            driver.assert_gpus_idle(config=config, deadline=deadline)
        evaluation_deadline = min(deadline, clock() + config.generation_validation.max_wall_seconds)
        outcome = driver.evaluate(
            step=step, checkpoint=checkpoint, task_ids=tuple(task_ids), config=config, deadline=evaluation_deadline
        )
        remaining()
        # Never resume/declare success merely because every task has a prediction.
        driver.assert_gpus_idle(config=config, deadline=deadline)
        if clock() >= evaluation_deadline:
            raise TimeoutError("Generation evaluation exceeded its wall budget; no training resume")
        solved = outcome.get("solved_task_ids")
        if (
            outcome.get("state") != "completed"
            or type(outcome.get("step")) is not int
            or outcome.get("step") != step
            or outcome.get("cleanup_complete") is not True
            or outcome.get("task_ids") != task_ids
            or not isinstance(solved, list)
            or any(not isinstance(task, str) for task in solved)
            or len(set(solved)) != len(solved)
            or not set(solved) <= set(task_ids)
        ):
            raise ValueError("Incomplete generation/test grading or cleanup blocks checkpoint selection and resume")
        observation = {
            "step": step,
            "checkpoint": checkpoint,
            "solved": len(solved),
            "total": len(task_ids),
            "dev_set_used_for_selection": qualification.get("final_evaluation_overlap") is True,
        }
        observations.append(observation)
        if len(solved) > best_solved:
            best_step, best_solved, bad = step, len(solved), 0
        elif step:
            bad += 1
        record({"observations": observations, "selected_step": best_step, "selected_solved": best_solved})
        if bad >= config.generation_validation.patience:
            break
    remaining()
    return {
        "state": "completed_generation_selection",
        "selected_step": best_step,
        "selected_solved": best_solved,
        "observations": observations,
        "nll_used_for_selection": False,
    }


@hydra.main(config_path="config", config_name="gemma_sft", version_base=None)
def main(config):
    schedule = plan(config)
    if not config.workflow.execute:
        print(json.dumps(schedule))
        return
    if not schedule["inputs_configured"]:
        raise ValueError(
            "Missing execution-qualified held-out panel and SWE generation/grading driver; no training started"
        )
    output = Path(config.trainer.default_local_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    driver = instantiate(config.generation_validation.driver)

    def record(value):
        # Append exclusive history snapshots; never overwrite prior predictions or scores.
        path = output / f"generation-history-{len(value['observations']):04d}.yaml"
        with path.open("x") as stream:
            OmegaConf.save(OmegaConf.create(value), stream)

    print(json.dumps(run(config, driver, record=record)))


if __name__ == "__main__":
    main()
