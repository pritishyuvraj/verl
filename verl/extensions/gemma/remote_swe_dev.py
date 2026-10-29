"""Hor-3 GPU phases, existing qualified Hor-1 tool/test environments.

SSH uses existing host authentication. Only adapters, configuration and small
receipts move; credentials and environment-admission claims never move hosts.
The already exercised relaxed-budget solver/proxy/grader sources stay frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

from omegaconf import OmegaConf

from verl.extensions.gemma.swe_dev import MODEL, SWEDevEvaluator, read_report, sha, write

PROFILE = "outputs/runtime/gemma-relaxed-128k-60m-20261027-attempt004-baseline-continuation"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]


def atomic_status(path, value):
    """Concurrent progress writers must never share a temporary filename."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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


def verify_transport(log_path):
    """Transport failures invalidate selection, rather than score as model failures."""
    failures = []
    with Path(log_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("client_cancelled") or row.get("error_type") or row.get("status") != 200:
                failures.append({k: row.get(k) for k in
                                 ("task_id", "status", "error_type", "classification", "start", "end")})
    if failures:
        write(Path(log_path).with_name("invalid-transport.json"),
              {"state": "invalid_infrastructure", "failures": failures})
        raise RuntimeError(f"{len(failures)} incomplete/failed model responses invalidate selection; predictions preserved")


def preserved_transport_failure(raw_path, task_ids, step):
    """Allow bounded continuation only after a fully graded, clean transport failure.

    The invalid arm is retained verbatim and never becomes a selection score.
    Model errors, missing grades, changed evidence and cleanup failures do not
    qualify for this recovery path.
    """
    raw_path = Path(raw_path)
    raw = json.loads(raw_path.read_text())
    errors = raw.get("errors", {})
    outcomes = raw.get("outcomes", {})
    if (step == 0 or raw.get("state") != "incomplete"
            or raw.get("task_ids") != list(task_ids)
            or set(outcomes) != set(task_ids)
            or raw.get("cleanup_complete") is not True or raw.get("new_containers") != []
            or set(errors) != {"transport"}
            or not isinstance(errors["transport"].get("unexpected_failures"), int)
            or errors["transport"]["unexpected_failures"] < 1
            or any(e.get("error") == "ordinary_request_missing_or_wrong_rank_binding"
                   for e in errors["transport"].get("examples", []))):
        raise ValueError("Failure is not a clean, fully graded transport-only arm")
    failed = 0
    for line in (raw_path.parent / "model-requests-responses.jsonl").open():
        row = json.loads(line)
        if "chat/completions" not in row.get("path", ""):
            continue
        if row.get("client_cancelled"):
            raise ValueError("Unfinished cancelled request blocks continuation")
        if "error" in row or row.get("status", 0) >= 500:
            failed += 1
        elif row.get("status") != 200:
            raise ValueError("Unexpected transport response blocks continuation")
    if failed != errors["transport"]["unexpected_failures"]:
        raise ValueError("Transport failures differ from recorded result")
    for task, outcome in outcomes.items():
        slug = task + "-" + hashlib.sha256(task.encode()).hexdigest()[:12]
        prediction = raw_path.parent / "episodes" / slug / "generation/predictions.jsonl"
        if (type(outcome.get("resolved")) is not bool
                or prediction.is_symlink() or sha(prediction) != outcome.get("prediction_sha256")):
            raise ValueError("Saved prediction differs; no regeneration permitted")
        payload = json.loads(prediction.read_text())
        if payload.get("instance_id") != task:
            raise ValueError("Saved prediction has a different task identity")
        if outcome.get("reason") == "empty_patch":
            if outcome["resolved"] or payload.get("model_patch", "").strip():
                raise ValueError("Invalid empty-patch outcome")
        else:
            report = Path(outcome["report"])
            if report.is_symlink() or sha(report) != outcome.get("report_sha256"):
                raise ValueError("Saved grade differs")
    return {"state": "invalid_infrastructure", "reason_code": "model_transport_failure",
            "step": step, "task_ids": list(task_ids), "graded_count": len(outcomes),
            "cleanup_complete": True, "cpu_result_path": str(raw_path),
            "cpu_result_sha256": sha(raw_path), "transport_failures": failed,
            "raw_solved_not_selection": sum(v["resolved"] for v in outcomes.values()),
            "predictions_preserved": True, "generation_rerun": False}


def verify_recovered_baseline(reference, evaluator):
    """Reconcile an explicitly pinned, fully graded saved baseline without rerolls."""
    path = Path(reference["path"])
    if sha(path) != reference["sha256"]:
        raise ValueError("Recovered baseline receipt changed")
    value = json.loads(path.read_text())
    if (value.get("state") != "completed" or value.get("step") != 0
            or value.get("cleanup_complete") is not True or value.get("generation_rerun") is not False
            or value.get("task_ids") != evaluator.ids or set(value.get("artifact_rows", {})) != set(evaluator.ids)):
        raise ValueError("Recovered baseline is incomplete or a different cohort")
    protocol = Path(value["original_protocol"])
    if sha(protocol) != value["protocol_sha256"]:
        raise ValueError("Recovered baseline protocol changed")
    binding = json.loads(protocol.read_text())
    if binding["fixture_v4"] != evaluator.fixture or binding["qualified_ids"] != evaluator.ids:
        raise ValueError("Recovered baseline environment differs")
    for filename, expected in {**binding["source_pins"], **value["input_pins"]}.items():
        if sha(filename) != expected:
            raise ValueError(f"Recovered baseline evidence changed: {filename}")
    solved = []
    for task in evaluator.ids:
        row = value["artifact_rows"][task]
        if sha(row["prediction"]) != row["prediction_sha256"] or sha(row["outcome"]) != row["outcome_sha256"]:
            raise ValueError("Recovered prediction or grade changed")
        prediction = json.loads(Path(row["prediction"]).read_text())
        outcome = json.loads(Path(row["outcome"]).read_text())
        if prediction["instance_id"] != task or outcome["prediction_sha256"] != row["prediction_sha256"]:
            raise ValueError("Recovered grade identity differs")
        if outcome.get("report"):
            if sha(outcome["report"]) != outcome["report_sha256"]:
                raise ValueError("Recovered official report changed")
            resolved = read_report(outcome["report"], task)
        elif not prediction["model_patch"].strip() and outcome.get("reason") == "empty_patch":
            resolved = False
        elif outcome.get("reason") == "official_patch_apply_failed_malformed_model_patch":
            evidence = outcome["evidence"]
            if len(evidence) != 3 or any(sha(e["path"]) != e["sha256"] for e in evidence):
                raise ValueError("Invalid-patch evidence changed")
            manifest, aggregate = [json.loads(Path(e["path"]).read_text()) for e in evidence[:2]]
            log = Path(evidence[2]["path"]).read_text()
            if not (manifest["instance_ids"] == [task] and manifest["predictions_sha256"] == row["prediction_sha256"]
                    and aggregate["submitted_ids"] == [task] and aggregate["error_ids"] == [task]
                    and aggregate["completed_instances"] == aggregate["resolved_instances"] == 0
                    and "Patch Apply Failed" in log and "patch: **** malformed patch" in log):
                raise ValueError("Unproven invalid-patch failure")
            resolved = False
        else:
            raise ValueError("Missing independent grading evidence")
        if type(outcome.get("resolved")) is not bool or outcome["resolved"] != resolved:
            raise ValueError("Recovered outcome disagrees with executable tests")
        if resolved:
            solved.append(task)
    if value["solved_task_ids"] != solved or evaluator.containers() != evaluator.initial_containers:
        raise ValueError("Recovered aggregate or current cleanup differs")
    return {**value, "recovery_receipt_sha256": reference["sha256"]}


class RemoteSWEDevEvaluator(SWEDevEvaluator):
    def __init__(self, project_root, baseline_report, protocol, output_dir, cpu_host, gpu_host,
                 cpu_project_root, cpu_source_dir="/home/ubuntu/workspace/verl-gemma-hor3-20261028",
                 cpu_output_dir="outputs/eval/e4b-packed-lora-lr1e5-hor3-20261028",
                 recovered_step_zero=None):
        super().__init__(project_root, baseline_report, protocol, output_dir)
        self.cpu_host, self.gpu_host = cpu_host, gpu_host
        self.cpu_root, self.cpu_source = Path(cpu_project_root), Path(cpu_source_dir)
        self.cpu_output = self.cpu_root / cpu_output_dir
        self.baseline_report, self.protocol = baseline_report, protocol
        self.recovered_step_zero = (
            OmegaConf.to_container(recovered_step_zero, resolve=True)
            if OmegaConf.is_config(recovered_step_zero) else recovered_step_zero
        )

    def request(self, config, action, step, deadline, adapter=None):
        return {"action": action, "step": step, "cpu_root": str(self.cpu_root),
                "cpu_output": str(self.cpu_output), "gpu_host": self.gpu_host,
                "baseline_report": self.baseline_report, "protocol": self.protocol,
                "adapter": adapter, "deadline_unix": time.time() + deadline - time.monotonic(),
                "recovered_step_zero": self.recovered_step_zero,
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
        parser_pin = config.generation_inference.get("tool_parser_sha256")
        if parser_pin is not None:
            raw = subprocess.check_output([str(self.python), "-c",
                "import sysconfig,pathlib,hashlib,json; "
                "p=pathlib.Path(sysconfig.get_paths()['purelib'])/'vllm/tool_parsers/gemma4_tool_parser.py'; "
                "print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"],
                timeout=15)
            pin = json.loads(raw)
            if pin["sha256"] != parser_pin:
                raise ValueError("vLLM Gemma parser lacks the pinned upstream41991 repair")
            write(stage / "tool-parser-pin.json", pin)
        if step == 0 and self.recovered_step_zero is not None:
            result = self.call_worker(self.request(config, "recover_step_zero", step, deadline), stage / "cpu", deadline)
            self.assert_gpus_idle(config, deadline)
            write(stage / "results.json", result)
            return result
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
        accepted_states = {"completed"}
        if config.workflow.get("max_invalid_evaluations", 0):
            accepted_states.add("invalid_infrastructure")
        if result.get("state") not in accepted_states or result.get("cleanup_complete") is not True:
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
    # Preserve historical source bytes. Its watcher, main collector and eight
    # graders otherwise race on the same status.tmp pathname.
    previous.atomic = atomic_status
    output.mkdir(parents=True, exist_ok=True)
    step = request["step"]
    phase = output / ("host-qualification" if request["action"] == "qualify" else f"step-{step:06d}")
    evaluator = previous.Evaluator(root, request["baseline_report"], request["protocol"], phase / "qualification")
    result = evaluator.qualify(config, deadline)
    if request["action"] == "recover_step_zero":
        if step != 0:
            raise ValueError("Only the pretraining baseline can be recovered without a native checkpoint")
        result = verify_recovered_baseline(request["recovered_step_zero"], evaluator)
    if request["action"] == "evaluate":
        # All pinned tool/client/compaction/rank-map source bytes remain unchanged.
        # Only the per-phase output location and the served adapter change.
        control = phase / "control"
        control.mkdir()
        for file in ("diagnostic_solver.py", "diagnostic_agent_prepare.py", "recording_proxy.py", "context-override.json", "rank-map.json"):
            shutil.copy2(profile / file, control / file)
        transport_timeout = config.generation_validation.get("request_timeout_seconds")
        if transport_timeout is not None:
            if type(transport_timeout) is not int or transport_timeout != 600:
                raise ValueError("Only the reviewed 600s transport guard is admitted")
            proxy = control / "recording_proxy.py"
            source = proxy.read_text()
            old = "ClientTimeout(total=3750)"
            if source.count(old) != 1:
                raise ValueError("Unexpected pinned proxy timeout source")
            original_sha = sha(proxy)
            proxy.write_text(source.replace(old, "ClientTimeout(total=600)"))
            write(control / "transport-guard.json", {"original_sha256": original_sha,
                  "new_sha256": sha(proxy), "request_timeout_seconds": 600,
                  "episode_timeout_seconds_unchanged": 3600,
                  "timeout_is_infrastructure_failure_not_model_score": True})
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
            raw_path = previous.OUTPUT / name / "results.json"
            failed_transport = False
            try:
                previous.run_arm(evaluator, name, None if adapter is None else Path(adapter["path"]),
                                 None if adapter is None else adapter["sha256"], config, deadline, endpoint)
            except AssertionError as error:
                if (str(error) != "Incomplete arm; do not advance or relabel failures"
                        or not config.workflow.get("max_invalid_evaluations", 0)):
                    raise
                result = preserved_transport_failure(raw_path, evaluator.ids, step)
                failed_transport = True
            if not failed_transport and transport_timeout is not None:
                verify_transport(previous.OUTPUT / name / "model-requests-responses.jsonl")
            raw = json.loads(raw_path.read_text())
            if not failed_transport:
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
