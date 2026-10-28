"""Hor-3 GPU phases, existing qualified Hor-1 tool/test environments.

SSH uses existing host authentication. Only adapters, configuration and small
receipts move; credentials and environment-admission claims never move hosts.
The already exercised relaxed-budget solver/proxy/grader sources stay frozen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

from omegaconf import OmegaConf

from verl.extensions.gemma.swe_dev import MODEL, SWEDevEvaluator, sha, write

PROFILE = "outputs/runtime/gemma-relaxed-128k-60m-20261027-attempt004-baseline-continuation"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]


def require_settings(config):
    expected = {"data_parallel_size": 8, "tensor_parallel_size": 1, "episode_concurrency": 128,
                "episodes_per_rank": 16, "max_num_seqs": 256, "max_num_batched_tokens": 65536,
                "gpu_memory_utilization": 0.9, "max_model_len": 131072, "max_prompt_tokens": 114688,
                "max_output_tokens": 16384, "thinking_budget": 4096, "temperature": 1.0,
                "top_p": 0.95, "top_k": 64, "seed": 42, "enable_thinking": True, "include_thoughts": True}
    if any(config.generation_inference.get(k) != v for k, v in expected.items()):
        raise ValueError("Remote validation differs from the exercised matched relaxed profile")
    if dict(config.generation_validation.episode) != {"max_minutes": 60, "max_tools": 60, "max_turns": 60}:
        raise ValueError("Remote episode budgets differ")


class RemoteSWEDevEvaluator(SWEDevEvaluator):
    def __init__(self, project_root, baseline_report, protocol, output_dir, cpu_host, gpu_host,
                 cpu_project_root, cpu_source_dir="/home/ubuntu/workspace/verl-gemma-hor3-20261028",
                 cpu_output_dir="outputs/eval/e4b-packed-lora-lr1e5-hor3-20261028"):
        super().__init__(project_root, baseline_report, protocol, output_dir)
        self.cpu_host, self.gpu_host = cpu_host, gpu_host
        self.cpu_root, self.cpu_source = Path(cpu_project_root), Path(cpu_source_dir)
        self.cpu_output = self.cpu_root / cpu_output_dir
        self.baseline_report, self.protocol = baseline_report, protocol

    def request(self, config, action, step, deadline, adapter=None):
        return {"action": action, "step": step, "cpu_root": str(self.cpu_root),
                "cpu_output": str(self.cpu_output), "gpu_host": self.gpu_host,
                "baseline_report": self.baseline_report, "protocol": self.protocol,
                "adapter": adapter, "deadline_unix": time.time() + deadline - time.monotonic(),
                "config": OmegaConf.to_container(config, resolve=True)}

    def call_worker(self, request, local_dir, deadline):
        seconds = int(deadline - time.monotonic() - 30)
        if seconds < 60:
            raise TimeoutError("No remote-worker cleanup budget")
        local_dir.mkdir(parents=True, exist_ok=False)
        write(local_dir / "request.json", request)
        script = self.cpu_source / "verl/extensions/gemma/remote_swe_dev.py"
        command = ["env", f"PYTHONPATH={self.cpu_source}",
                   f"LD_LIBRARY_PATH={self.cpu_root}/.runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat",
                   "timeout", "--signal=TERM", "--kill-after=15s", f"{seconds}s",
                   self.cpu_root / ".runtime/venvs/sft-verl/bin/python", script, "--worker"]
        with (local_dir / "worker.log").open("x") as log:
            subprocess.run(SSH + [self.cpu_host, shlex.join(map(str, command))],
                           input=json.dumps(request), text=True, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=seconds + 25)
        remote = self.cpu_output / f"worker-{request['action']}-{request['step']:06d}.json"
        raw = subprocess.check_output(SSH + [self.cpu_host, "cat " + shlex.quote(str(remote))], timeout=20)
        result = json.loads(raw)
        (local_dir / "result.json").write_bytes(raw)
        return result

    def qualify(self, config, deadline):
        require_settings(config)
        if subprocess.check_output(["hostname"], text=True).strip() != "zyphra-3":
            raise ValueError("This experiment is authorized only on hor-3")
        self.output.mkdir(parents=True, exist_ok=False)
        result = self.call_worker(self.request(config, "qualify", 0, deadline), self.output / "qualification", deadline)
        self.ids = result["task_ids"]
        return result

    def evaluate(self, step, checkpoint, task_ids, config, deadline):
        require_settings(config)
        if list(task_ids) != self.ids:
            raise ValueError("Selection cohort changed")
        stage = self.output / f"step-{step:06d}"
        stage.mkdir(exist_ok=False)
        if step == 0:
            from verl.trainer.generation_sft import NativePhases, native_phase_config

            initial = native_phase_config(config, None, 1)
            initial.trainer.validation_only = True
            # The native process exits after step-zero NLL/PPL/entropy, without
            # optimizer updates or a checkpoint. Its phase directory is step0.
            NativePhases(evaluator=self, training_python=config.generation_validation.driver.training_python).train(
                None, 0, initial, min(deadline - 9000, time.monotonic() + 3600)
            )
            self.assert_gpus_idle(config, deadline)
        adapter = None if step == 0 else Path(checkpoint) / "adapter"
        adapter_arg = None
        name = "baseline" if adapter is None else f"sft_step_{step}"
        if adapter is not None:
            from safetensors import safe_open
            with safe_open(adapter / "adapter_model.safetensors", framework="pt") as tensors:
                if not tensors.keys() or any(tensors.get_slice(k).get_dtype() != "BF16" for k in tensors.keys()):
                    raise ValueError("Inference requires the actual BF16 export")
            remote = self.cpu_output / "adapters" / name
            subprocess.run(SSH + [self.cpu_host, "mkdir -p " + shlex.quote(str(remote.parent)) + " && test ! -e " + shlex.quote(str(remote))], check=True, timeout=30)
            subprocess.run(["rsync", "-a", "--timeout=120", str(adapter) + "/", f"{self.cpu_host}:{remote}/"], check=True, timeout=300)
            adapter_arg = {"path": str(remote), "sha256": sha(adapter / "adapter_model.safetensors")}
        command = [self.python, "-m", "vllm.entrypoints.cli.main", "serve", self.model,
                   "--host", "127.0.0.1", "--port", "8400", "--data-parallel-size", "8",
                   "--tensor-parallel-size", "1", "--data-parallel-backend", "mp", "--api-server-count", "8",
                   "--language-model-only", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.9",
                   "--max-model-len", "131072", "--max-num-seqs", "256", "--max-num-batched-tokens", "65536",
                   "--enable-auto-tool-choice", "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4",
                   "--reasoning-config", '{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}',
                   "--no-scheduler-reserve-full-isl", "--served-model-name", MODEL,
                   "--override-generation-config", '{"temperature":1.0,"top_p":0.95,"top_k":64}']
        if adapter is not None:
            command += ["--enable-lora", "--lora-dtype", "bfloat16", "--max-loras", "1", "--max-lora-rank", "32", "--lora-modules", f"{name}={adapter}"]
        write(stage / "server-command.json", list(map(str, command)))
        server = self._spawn(command, stage / "server.log", deadline - time.monotonic() - 90)
        result = None
        try:
            until = min(deadline - 7800, time.monotonic() + 1200)
            while time.monotonic() < until:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited; no automatic retry or memory reduction")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8400/v1/models", timeout=3) as response:
                        if (name if adapter is not None else MODEL) in {v["id"] for v in json.load(response)["data"]}:
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("vLLM startup did not leave full evaluation headroom")
            result = self.call_worker(self.request(config, "evaluate", step, deadline, adapter_arg), stage / "cpu", deadline)
        finally:
            self._stop(server)
            self.assert_gpus_idle(config, deadline)
            write(stage / "gpu-cleanup.json", {"gpu_contexts_released": True, "server_returncode": server.returncode})
        if result.get("state") != "completed" or result.get("cleanup_complete") is not True:
            raise RuntimeError("Remote generation/grading/cleanup incomplete")
        write(stage / "results.json", result)
        return result


def worker(request):
    """CPU-only side; invokes the preserved, verified relaxed profile."""
    root, output = Path(request["cpu_root"]), Path(request["cpu_output"])
    if subprocess.check_output(["hostname"], text=True).strip() != "zyphra-1":
        raise ValueError("Existing host qualification applies only to hor-1")
    config = OmegaConf.create(request["config"])
    require_settings(config)
    deadline = time.monotonic() + request["deadline_unix"] - time.time()
    profile = root / PROFILE
    spec = importlib.util.spec_from_file_location("verified_relaxed_profile", profile / "run_relaxed.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    output.mkdir(parents=True, exist_ok=True)
    step = request["step"]
    phase = output / ("host-qualification" if request["action"] == "qualify" else f"step-{step:06d}")
    evaluator = previous.Evaluator(root, request["baseline_report"], request["protocol"], phase / "qualification")
    result = evaluator.qualify(config, deadline)
    if request["action"] == "evaluate":
        # All pinned tool/client/compaction/rank-map source bytes remain unchanged.
        # Only the per-phase output location and the served adapter change.
        control = phase / "control"
        control.mkdir()
        for file in ("diagnostic_solver.py", "diagnostic_agent_prepare.py", "recording_proxy.py", "context-override.json", "rank-map.json"):
            shutil.copy2(profile / file, control / file)
        previous.CONTROL, previous.OUTPUT = control, phase / "episodes-and-grading"
        previous.OUTPUT.mkdir()
        tunnel_args = SSH + ["-o", "ExitOnForwardFailure=yes", "-N", "-L", "127.0.0.1:8400:127.0.0.1:8400", request["gpu_host"]]
        tunnel = evaluator._spawn(tunnel_args, control / "tunnel.log", deadline - time.monotonic() - 60)
        try:
            class Endpoint:
                def poll(self):
                    if tunnel.poll() is not None:
                        return 1
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:8400/v1/models", timeout=3) as response:
                            return None if response.status == 200 else 1
                    except OSError:
                        return 1
            endpoint = Endpoint()
            until = min(deadline - 7800, time.monotonic() + 30)
            while endpoint.poll() is not None and time.monotonic() < until:
                time.sleep(1)
            if endpoint.poll() is not None:
                raise RuntimeError("Hor-3 endpoint tunnel unavailable")
            adapter = request["adapter"]
            name = "baseline" if adapter is None else f"sft_step_{step}"
            if adapter and sha(Path(adapter["path"]) / "adapter_model.safetensors") != adapter["sha256"]:
                raise ValueError("Copied adapter hash differs")
            previous.run_arm(evaluator, name, None if adapter is None else Path(adapter["path"]),
                             None if adapter is None else adapter["sha256"], config, deadline, endpoint)
            raw_path = previous.OUTPUT / name / "results.json"
            raw = json.loads(raw_path.read_text())
            result = {"state": raw["state"], "step": step, "task_ids": raw["task_ids"],
                      "solved_task_ids": [t for t, value in raw["outcomes"].items() if value["resolved"]],
                      "cleanup_complete": raw["cleanup_complete"], "cpu_result_path": str(raw_path),
                      "cpu_result_sha256": sha(raw_path), "generation_seconds": raw["generation_seconds"],
                      "elapsed_seconds": raw["elapsed_seconds"], "gpu_host": request["gpu_host"],
                      "cpu_host": "zyphra-1", "baseline_reused": False,
                      "empty_patches": raw["empty_patches"]}
        finally:
            evaluator._stop(tunnel)
        if evaluator.containers() - evaluator.initial_containers:
            raise RuntimeError("New CPU containers remain; no training resume")
    write(output / f"worker-{request['action']}-{step:06d}.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    worker(json.load(sys.stdin))
