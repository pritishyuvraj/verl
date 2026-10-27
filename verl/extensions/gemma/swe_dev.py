"""Generated SWE repairs with the existing qualified organizer solver and grader.

This adapter invokes pinned environment assets; it does not replace the agent
loop or tests. The recorded dev cohort may be used for selection only through an
explicit config opt-in. Its resulting scores are selection scores, not holdout
estimates. A byte-preserving loopback proxy retains every model request/response.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = "gemma-4-e4b-it-qat-w4a16-ct"
DOCKER = "unix:///dev/shm/pritish-gemma4-20261025/eval-docker/docker.sock"
DAEMON = "9469315a-6cc5-4032-a513-6427b656879d"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_report(path, task):
    """Verify actual official artifacts, never an exit code or prediction alone."""
    path = Path(path)
    worker = json.loads(path.read_text())
    matches = []
    for entry in worker.get("official_reports", []):
        actual = json.loads((path.parent / entry["path"]).read_text())
        if actual != entry["report"]:
            raise ValueError("Official report differs from worker evidence")
        if task in actual:
            matches.append(actual[task])
    if len(matches) != 1 or type(matches[0].get("resolved")) is not bool:
        raise ValueError("Missing unambiguous executable-test outcome")
    result = matches[0]
    if result["resolved"]:
        if result.get("patch_successfully_applied") is not True:
            raise ValueError("Resolved patch was not applied")
        if any(result["tests_status"][name]["failure"] for name in ("FAIL_TO_PASS", "PASS_TO_PASS")):
            raise ValueError("Resolved report contains failing required tests")
    return result["resolved"]


class SWEDevEvaluator:
    def __init__(self, project_root, baseline_report, protocol, output_dir):
        self.root = Path(project_root).absolute()
        self.baseline_path = self.root / baseline_report
        self.protocol_path = self.root / protocol
        self.output = Path(output_dir)
        self.python = self.root / ".runtime/venvs/scorer-v31/bin/python"
        self.grader_python = self.root / "downloads/.venv-grader/bin/python"
        self.policy = self.root / "outputs/runtime/context-policy-admission.json"
        self.model = self.root / ".runtime/models/kaggle-e4b-v2"
        self.solver = self.root / "outputs/eval/swebench-dev-e48e2bd1e9fe/solver/all.jsonl"
        self.daemon_receipt = self.root / "outputs/transfer/hor1-image-plan/private-daemon-ready.json"
        self.env = {
            **os.environ,
            "DOCKER_HOST": DOCKER,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHON_DOTENV_DISABLED": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_TELEMETRY": "False",
            "OTEL_SDK_DISABLED": "true",
            "VLLM_NO_USAGE_STATS": "1",
            "LOCAL_API_KEY": "local-only",
            "OPENAI_API_KEY": "local-only",
        }

    def command(self, args, *, seconds=30):
        return subprocess.check_output([str(x) for x in args], cwd=self.root, env=self.env, timeout=seconds, text=True)

    def containers(self):
        info = self.command(["sudo", "-n", "docker", "--host", DOCKER, "info", "--format", "{{.ID}}"])
        if info.strip() != DAEMON:
            raise ValueError("Private Docker identity changed")
        return set(self.command(["sudo", "-n", "docker", "--host", DOCKER, "ps", "-aq"]).split())

    def assert_gpus_idle(self, config, deadline):
        until = min(deadline, time.monotonic() + 90)
        while time.monotonic() < until:
            rows = self.command(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
            )
            processes = self.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
            parsed = [[int(x.strip()) for x in row.split(",")] for row in rows.splitlines()]
            if (
                len(parsed) == 8
                and not processes.strip()
                and all(memory <= 4 and utilization == 0 for _, memory, utilization in parsed)
            ):
                return
            time.sleep(2)
        raise RuntimeError("GPUs did not release; no next phase or foreign cleanup")

    def qualify(self, config, deadline):
        if config.generation_validation.get("allow_dev_selection") is not True:
            raise ValueError("Existing dev tasks require explicit selection opt-in")
        protocol = json.loads(self.protocol_path.read_text())
        self.fixture = protocol["fixture_v4"]
        self.ids = protocol["qualified_ids"]
        self.excluded = protocol["unqualified_ids"]
        if len(self.ids) != 221 or len(set(self.ids)) != 221 or len(self.excluded) != 4:
            raise ValueError("Expected the existing 225-target/221-qualified cohort")
        if set(self.ids) & set(self.excluded):
            raise ValueError("Overlapping included/excluded IDs")
        self.images = json.loads((self.root / "downloads/swebench-full-image-map.json").read_text())
        code = (
            "import json,sys;from pathlib import Path;sys.path.insert(0,'scripts');"
            "from qualified_host_v4 import verify_qualified_host;"
            "v=verify_qualified_host(Path('.').absolute(),live=False);"
            "print(json.dumps({'binding':v['binding'],'binding_sha256':v['binding_sha256']}))"
        )
        if json.loads(self.command([self.python, "-c", code], seconds=120)) != self.fixture:
            raise ValueError("Qualified source/profile graph changed")
        baseline = json.loads(self.baseline_path.read_text())
        if (
            baseline.get("status") != "complete"
            or baseline.get("pending") != 0
            or baseline.get("grade_errors") != 0
            or len(baseline["rows"]) != 221
            or {r["instance_id"] for r in baseline["rows"]} != set(self.ids)
        ):
            raise ValueError("Baseline is incomplete")
        solved = []
        for row in baseline["rows"]:
            pred = self.root / row["prediction"]
            if sha(pred) != row["prediction_sha256"]:
                raise ValueError("Baseline prediction changed")
            values = [json.loads(x) for x in pred.read_text().splitlines() if x.strip()]
            if len(values) != 1 or values[0]["instance_id"] != row["instance_id"]:
                raise ValueError("Invalid baseline prediction identity")
            if values[0]["model_patch"].strip():
                report = self.root / row["worker_report"]
                if sha(report) != row["worker_report_sha256"]:
                    raise ValueError("Baseline grading changed")
                resolved = read_report(report, row["instance_id"])
            else:
                resolved = False
            if resolved != row["resolved"]:
                raise ValueError("Baseline summary disagrees with tests")
            if resolved:
                solved.append(row["instance_id"])
        if len(solved) != baseline["resolved"]:
            raise ValueError("Baseline solved count disagrees")
        self.baseline_solved = solved
        self.initial_containers = self.containers()
        self.output.mkdir(parents=True, exist_ok=False)
        self.pins = {
            str(p): sha(p)
            for p in [
                self.baseline_path,
                self.protocol_path,
                self.policy,
                self.root / "configs/token_policy.json",
                self.root / "configs/official_runtime.json",
            ]
        }
        write(
            self.output / "qualification.json",
            {
                "task_ids": self.ids,
                "excluded_ids": self.excluded,
                "baseline_solved": len(solved),
                "baseline_reused": True,
                "pins": self.pins,
                "dev_set_used_for_selection": True,
                "unbiased_holdout_score": False,
                "preexisting_container_ids_untouched": sorted(self.initial_containers),
            },
        )
        return {"task_ids": self.ids, "final_evaluation_overlap": True}

    def _spawn(self, args, log, seconds):
        if seconds <= 0:
            raise TimeoutError("No process budget remains")
        command = ["timeout", "--signal=TERM", "--kill-after=30s", f"{int(seconds)}s"]
        if list(args[:2]) == ["sudo", "-n"]:
            command = ["sudo", "-n", *command, *map(str, args[2:])]
        else:
            command += list(map(str, args))
        stream = Path(log).open("x")
        process = subprocess.Popen(
            command,
            cwd=self.root,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        write(
            Path(log).with_suffix(".process.json"),
            {
                "pid": process.pid,
                "pgid": process.pid,
                "command": command,
                "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
            },
        )
        return process

    def _stop(self, process):
        # A still-unreaped Popen child cannot have its PID reused. GNU timeout
        # forwards TERM to its private process group, with a bounded KILL fallback.
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=100)

    def evaluate(self, step, checkpoint, task_ids, config, deadline):
        if list(task_ids) != self.ids or any(sha(p) != value for p, value in self.pins.items()):
            raise ValueError("Frozen inputs changed")
        if step == 0:
            return {
                "state": "completed",
                "step": 0,
                "task_ids": self.ids,
                "solved_task_ids": self.baseline_solved,
                "cleanup_complete": True,
            }
        stage = self.output / f"step-{step:06d}"
        stage.mkdir()
        adapter = Path(checkpoint) / "adapter"
        from safetensors import safe_open

        with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
            if not tensors.keys() or any(tensors.get_slice(k).get_dtype() != "BF16" for k in tensors.keys()):
                raise ValueError("Evaluation must use the actual BF16 export")
        bundle = stage / "submission"
        shutil.copytree(
            self.root / "submission/sft_e4b_thinking_official_dp8", bundle, ignore=shutil.ignore_patterns("adapters")
        )
        name = f"sft_step_{step}"
        (bundle / "adapters").mkdir()
        shutil.copytree(adapter, bundle / "adapters" / name)
        agent = bundle / "agent.yaml"
        agent.write_text(agent.read_text().replace("adapter: sft_v3_thinking", f"adapter: {name}"))
        adapter_pin = sha(adapter / "adapter_model.safetensors")
        log_pin = {
            "step": step,
            "adapter_sha256": adapter_pin,
            "dtype": "BF16",
            "task_ids": self.ids,
            "excluded_ids": self.excluded,
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 42},
            "raw_model_log": str(stage / "model-requests-responses.jsonl"),
            "dev_set_used_for_selection": True,
            "judge_is_diagnostic_only": True,
        }
        write(stage / "protocol.json", log_pin)
        server = proxy = None
        outcomes = {}
        try:
            inference = config.generation_inference
            command = [
                self.python,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                self.model,
                "--host",
                "127.0.0.1",
                "--port",
                "8400",
                "--data-parallel-size",
                "8",
                "--tensor-parallel-size",
                "1",
                "--data-parallel-backend",
                "mp",
                "--api-server-count",
                "8",
                "--language-model-only",
                "--dtype",
                "bfloat16",
                "--lora-dtype",
                "bfloat16",
                "--gpu-memory-utilization",
                str(inference.gpu_memory_utilization),
                "--max-model-len",
                "32768",
                "--max-num-seqs",
                str(inference.max_num_seqs),
                "--max-num-batched-tokens",
                str(inference.max_num_batched_tokens),
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "gemma4",
                "--reasoning-parser",
                "gemma4",
                "--reasoning-config",
                '{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}',
                "--no-scheduler-reserve-full-isl",
                "--served-model-name",
                MODEL,
                "--override-generation-config",
                '{"temperature":1.0,"top_p":0.95,"top_k":64}',
                "--enable-lora",
                "--max-loras",
                "1",
                "--max-lora-rank",
                "32",
                "--lora-modules",
                f"{name}={adapter}",
            ]
            server = self._spawn(command, stage / "server.log", deadline - time.monotonic() - 120)
            proxy = self._spawn(
                [self.python, __file__, "--proxy", "--log", stage / "model-requests-responses.jsonl"],
                stage / "proxy.log",
                deadline - time.monotonic() - 120,
            )
            ready_until = min(deadline - 1200, time.monotonic() + 1200)
            while time.monotonic() < ready_until:
                if server.poll() is not None or proxy.poll() is not None:
                    raise RuntimeError("Inference startup failed; inspect logs. No automatic setting reduction")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8401/v1/models", timeout=3) as response:
                        if name in {v["id"] for v in json.load(response)["data"]}:
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("vLLM startup deadline exceeded")
            with (
                cf.ThreadPoolExecutor(max_workers=8) as graders,
                cf.ThreadPoolExecutor(max_workers=min(221, int(inference.episode_concurrency))) as generators,
            ):
                pending = {generators.submit(self._generate, t, stage, bundle, deadline): t for t in self.ids}
                grading = {}
                for future in cf.as_completed(pending):
                    task = pending[future]
                    prediction = future.result()
                    grading[graders.submit(self._grade, task, prediction, stage, deadline)] = task
                # Free all GPU contexts while independent CPU grading finishes.
                self._stop(server)
                server = None
                self._stop(proxy)
                proxy = None
                for future in cf.as_completed(grading):
                    outcomes[grading[future]] = future.result()
        finally:
            if server is not None:
                self._stop(server)
            if proxy is not None:
                self._stop(proxy)
            self.assert_gpus_idle(config, deadline)
            extra = self.containers() - self.initial_containers
            write(stage / "cleanup.json", {"gpu_contexts_released": True, "unexpected_new_containers": sorted(extra)})
            if extra:
                raise RuntimeError("New containers remain; no training resume or foreign deletion")
        if set(outcomes) != set(self.ids) or sha(adapter / "adapter_model.safetensors") != adapter_pin:
            raise ValueError("Incomplete cohort or changed adapter")
        result = {
            "state": "completed",
            "step": step,
            "task_ids": self.ids,
            "solved_task_ids": [t for t in self.ids if outcomes[t]["resolved"]],
            "cleanup_complete": True,
            "outcomes": outcomes,
        }
        write(stage / "results.json", result)
        write(
            stage / "judge-index.json",
            {
                "purpose": "Diagnostic LLM review; executable tests retain scoring authority",
                "model_requests_responses": str(stage / "model-requests-responses.jsonl"),
                "episodes": str(stage / "episodes"),
                "results": str(stage / "results.json"),
                "suggested_labels": [
                    "format_or_tool_error",
                    "context_overflow",
                    "timeout",
                    "wrong_repair",
                    "test_regression",
                    "empty_patch",
                    "unsupported",
                    "resolved",
                ],
            },
        )
        return result

    def _generate(self, task, stage, bundle, deadline):
        slug = task + "-" + hashlib.sha256(task.encode()).hexdigest()[:12]
        job = stage / "episodes" / slug
        job.mkdir(parents=True)
        write(job / "fixture-v4.json", self.fixture)
        selected = {k: v for k, v in self.images.items() if k != "tasks"}
        selected["tasks"] = {task: self.images["tasks"][task]}
        write(job / "image-map.json", selected)
        allowed = (
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHON_DOTENV_DISABLED",
            "LITELLM_LOCAL_MODEL_COST_MAP",
            "LITELLM_TELEMETRY",
            "OTEL_SDK_DISABLED",
            "OPENAI_API_KEY",
            "LOCAL_API_KEY",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "VLLM_NO_USAGE_STATS",
            "DOCKER_HOST",
        )
        args = [
            "sudo",
            "-n",
            "env",
            "-i",
            *[f"{k}={self.env[k]}" for k in allowed if k in self.env],
            self.python,
            "-u",
            self.root / "scripts/run_swebench_solver_v4.py",
            "--binding-json",
            job / "fixture-v4.json",
            "--policy-authorization",
            self.policy,
            "--solver-jsonl",
            self.solver,
            "--image-map",
            job / "image-map.json",
            "--adapter-dir",
            job / "adapter",
            "--task-id",
            task,
            "--submission-dir",
            bundle,
            "--model-path",
            self.model,
            "--results-dir",
            job / "generation",
            "--endpoint",
            "http://127.0.0.1:8401/v1",
            "--served-model",
            MODEL,
            "--execute",
        ]
        process = self._spawn(args, job / "generation.log", min(900, deadline - time.monotonic() - 150))
        code = process.wait(timeout=min(950, max(1, deadline - time.monotonic())))
        pred = job / "generation/predictions.jsonl"
        if not pred.exists():
            raise RuntimeError(f"Generation infrastructure failed for {task}, exit {code}; no reroll")
        rows = [json.loads(x) for x in pred.read_text().splitlines() if x.strip()]
        if len(rows) != 1 or rows[0]["instance_id"] != task:
            raise ValueError("Invalid prediction identity")
        write(job / "generation-complete.json", {"returncode": code, "prediction_sha256": sha(pred)})
        return pred

    def _grade(self, task, prediction, stage, deadline):
        pin = sha(prediction)
        patch = json.loads(prediction.read_text())["model_patch"]
        if not patch.strip():
            return {"resolved": False, "reason": "empty_patch", "prediction_sha256": pin}
        job = prediction.parent.parent
        image = self.images["tasks"][task]
        run_hash = hashlib.sha256(str(stage).encode()).hexdigest()[:12]
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:12]
        last = None
        for attempt in range(1, 4):
            if deadline - time.monotonic() < 100:
                break
            run_id = f"v4-{run_hash}-{task_hash}-g{attempt}"
            report = self.root / "outputs/swebench-grader" / run_id / "worker_result.json"
            if report.parent.exists():
                raise ValueError("Existing grading attempt; immutable history must be reconciled")
            args = [
                self.grader_python,
                self.root / "scripts/run_swebench_grader_v4.py",
                "--binding-json",
                job / "fixture-v4.json",
                "--policy-authorization",
                self.policy,
                "--predictions",
                prediction,
                "--task-id",
                task,
                "--attempt",
                str(attempt),
                "--instance-image-tag",
                image["image"].rsplit(":", 1)[1],
                "--run-id",
                run_id,
                "--private-docker-receipt",
                self.daemon_receipt,
                "--execute",
            ]
            process = self._spawn(args, job / f"grading-{attempt}.log", min(900, deadline - time.monotonic() - 90))
            code = process.wait(timeout=min(950, max(1, deadline - time.monotonic())))
            if sha(prediction) != pin:
                raise ValueError("Prediction changed during grading")
            if code == 0 and report.is_file():
                return {
                    "resolved": read_report(report, task),
                    "prediction_sha256": pin,
                    "report": str(report),
                    "report_sha256": sha(report),
                    "grade_attempt": attempt,
                }
            last = code
        raise RuntimeError(f"Independent grading incomplete for {task}, last exit {last}")


def proxy(log):
    import asyncio

    from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

    async def serve():
        stream = Path(log).open("x", buffering=1)
        async with ClientSession(timeout=ClientTimeout(total=650), connector=TCPConnector(limit=512)) as client:

            async def forward(request):
                body = await request.read()
                start = time.time()
                try:
                    async with client.request(
                        request.method,
                        "http://127.0.0.1:8400" + request.path_qs,
                        data=body,
                        headers={
                            "Content-Type": request.headers.get("Content-Type", "application/json"),
                            "Authorization": "Bearer local-only",
                        },
                    ) as response:
                        content = await response.read()
                        status = response.status
                        content_type = response.headers.get("Content-Type", "application/json")
                except Exception as exc:
                    status, content_type = 502, "application/json"
                    content = json.dumps({"error": type(exc).__name__ + ": " + str(exc)}).encode()
                # Headers/credentials are deliberately excluded. Neither body is clipped.
                stream.write(
                    json.dumps(
                        {
                            "started_unix": start,
                            "duration_seconds": time.time() - start,
                            "path": request.path_qs,
                            "request": body.decode(),
                            "status": status,
                            "response": content.decode(),
                        }
                    )
                    + "\n"
                )
                stream.flush()
                return web.Response(body=content, status=status, headers={"Content-Type": content_type})

            app = web.Application(client_max_size=64 * 1024 * 1024)
            app.router.add_route("*", "/{tail:.*}", forward)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", 8401).start()
            try:
                await asyncio.Event().wait()
            finally:
                await runner.cleanup()
                stream.close()

    asyncio.run(serve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if not args.proxy:
        parser.error("Only the raw-response recorder has a standalone entrypoint")
    proxy(args.log)
