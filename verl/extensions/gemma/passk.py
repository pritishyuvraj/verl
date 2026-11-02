"""Batched SWE-rebench pass@k; native vLLM serving and independent upstream grading.

No optimizer or judge. Incomplete infrastructure is never a model failure.
Run this file with --config gemma_pass8.yaml --mode run on the experiment host.
"""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

DOCKER = "unix:///dev/shm/pritish-gemma4-20261025/eval-docker/docker.sock"
DAEMON = "9469315a-6cc5-4032-a513-6427b656879d"
NEUTRAL = "diff --git a/.neutral_probe b/.neutral_probe\nnew file mode 100644\n--- /dev/null\n+++ b/.neutral_probe\n@@ -0,0 +1 @@\n+neutral\n"
_process_cache = {"time": 0, "rows": []}
_process_lock = threading.Lock()


class QualificationRejected(RuntimeError):
    """A task-specific control failed; retain it in coverage and continue other tasks."""


def private_snapshot(pgid, retained=(), deadline=None):
    """Share one cheap ps inventory; read /proc details only for this owned session."""
    deadline = time.monotonic()+5 if deadline is None else deadline
    def remaining():
        seconds = deadline-time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Private process observation deadline exhausted")
        return seconds
    if not _process_lock.acquire(timeout=remaining()):
        raise TimeoutError("Private process inventory lock deadline exhausted")
    try:
        if time.monotonic()-_process_cache["time"] > .5:
            result = subprocess.run(["ps", "-e", "-o", "pid=,pgid=,sid="],
                                    capture_output=True, text=True, check=True, timeout=min(5,remaining()))
            _process_cache.update(time=time.monotonic(), rows=[tuple(map(int,l.split())) for l in result.stdout.splitlines()])
        pids = {pgid} | {p for p,g,s in _process_cache["rows"] if g == pgid or s == pgid} | set(map(int, retained))
    finally:
        _process_lock.release()
    observed = {}
    for pid in pids:
        remaining()
        directory = Path("/proc")/str(pid)
        try:
            fields = (directory/"stat").read_text().rsplit(")",1)[1].split()
            uid = int(next(l.split()[1] for l in (directory/"status").read_text().splitlines() if l.startswith("Uid:")))
        except (FileNotFoundError, ProcessLookupError):
            continue  # Actual ESRCH; permission/read failures propagate.
        observed[pid] = {"pid":pid,"ppid":int(fields[1]),"pgid":int(fields[2]),"sid":int(fields[3]),
                         "uid":uid,"state":fields[0],"start_ticks":fields[19]}
    return observed


def live_retention(base):
    class LiveRetainedGroup(base):
        def _remember(self, observed):
            # Keep historical identities, but bound concurrently live members, not lifetime forks.
            for pid, previous in list(self.record["known_processes"].items()):
                current = observed.get(int(pid))
                dead = current is not None and current["state"] == "Z" and all(
                    current[k] == previous[k] for k in ("pid","uid","sid","pgid","start_ticks"))
                if current is None:
                    try:
                        (Path("/proc")/pid).stat()
                    except FileNotFoundError:
                        dead = True
                if dead:
                    self.record.setdefault("retired_processes", []).append(previous)
                    del self.record["known_processes"][pid]
            super()._remember(observed)
    return LiveRetainedGroup


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def docker_client():
    import docker
    c = docker.DockerClient(base_url=DOCKER, timeout=120)
    if c.info()["ID"] != DAEMON:
        raise RuntimeError("Private daemon identity changed")
    return c


def published_test_spec(row, make_spec, python_spec):
    """Render prebuilt-image metadata from its published requirements snapshot."""
    install = row.get("install_config", {})
    if install.get("packages") != "requirements.txt" or install.get("reqs_path") is not None:
        return make_spec(row, namespace="swerebench"), None
    requirements = row.get("requirements")
    if not isinstance(requirements, str) or not requirements.strip():
        raise RuntimeError("Missing published requirements; refuse an empty dependency fallback")
    original = python_spec.get_requirements
    # Each grader is an isolated child. Only unused environment-build rendering changes;
    # run_instance uses the already verified image, original test script and test labels.
    python_spec.get_requirements = lambda instance: requirements if instance is row else original(instance)
    try:
        spec = make_spec(row, namespace="swerebench")
    finally:
        python_spec.get_requirements = original
    return spec, {"reason":"published_null_requirements_path", "uses_existing_image":True,
                  "requirements_sha256":hashlib.sha256(requirements.encode()).hexdigest()}


def grade(config, request):
    """Child process: import only the pinned Nebius fork, never the model runtime."""
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.test_spec import python as python_spec
    row = json.loads(Path(request["row"]).read_text())
    out = Path(request["output"])
    out.mkdir(parents=True, exist_ok=False)
    patch = Path(request["patch"]).read_text()
    client = docker_client()
    image = client.images.get(request["image"])
    if image.id != request["image_id"]:
        raise RuntimeError("Image drift")
    spec, compatibility = published_test_spec(row, make_test_spec, python_spec)
    if compatibility:
        write(out/"spec-compatibility.json", compatibility | {"image_id":image.id,
              "eval_script_sha256":hashlib.sha256(spec.eval_script.encode()).hexdigest()})
    if spec.instance_image_key != request["image"]:
        raise RuntimeError("Published image and official test specification disagree")
    os.chdir(out)
    run_id = "pass8-" + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    name = spec.get_instance_container_name(run_id)
    import docker
    try:
        client.containers.get(name)
    except docker.errors.NotFound:
        pass
    else:
        raise RuntimeError("Grader container already exists; refusing reuse")
    if not patch.strip():
        write(out / "outcome.json", {"resolved": False, "empty_patch": True, "tests_executed": False})
        return
    started = time.monotonic()
    try:
        result = run_instance(spec, {"instance_id": row["instance_id"], "model_name_or_path": "pass8",
                              "model_patch": patch}, False, False, client, run_id,
                              config["environment"]["test_seconds"])
        reports = list(out.glob("logs/run_evaluation/**/report.json"))
        if result is None or len(reports) != 1:
            raise RuntimeError("No authoritative executable-test report; infrastructure invalid")
        report = json.loads(reports[0].read_text())[row["instance_id"]]
        if type(report.get("resolved")) is not bool:
            raise RuntimeError("Invalid test report")
        write(out / "outcome.json", {"resolved": report["resolved"], "empty_patch": False,
                                    "tests_executed": True, "seconds": time.monotonic()-started,
                                    "report": str(reports[0]), "report_sha256": digest(reports[0]),
                                    "image_id": image.id})
    finally:
        try:
            own = client.containers.get(name)
        except docker.errors.NotFound:
            pass
        else:
            if own.image.id != image.id:
                raise RuntimeError("Refusing cleanup: image identity differs")
            own.remove(force=True)


async def proxy(config):
    import asyncio
    from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
    out = Path(config["output"])
    async with ClientSession(timeout=ClientTimeout(total=3750),
                             connector=TCPConnector(limit=512, force_close=True)) as client:
        async def forward(request):
            raw = await request.read()
            episode = request.headers.get("X-Episode-ID", "readiness")
            rank = request.headers.get("X-data-parallel-rank")
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", episode) or (rank is not None and rank not in map(str, range(8))):
                raise web.HTTPBadRequest(text="Invalid episode routing")
            headers = {"Content-Type": "application/json", "Authorization": "Bearer local-only"}
            if rank is not None:
                headers["X-data-parallel-rank"] = rank
            started = time.time()
            log_path = out / "readiness-http.jsonl"
            if episode != "readiness":
                task, arm, seed = episode.rsplit(".", 2)
                if arm not in config["arms"] or int(seed) not in config["seeds"]:
                    raise web.HTTPBadRequest(text="Unknown arm or seed")
                log_path = out/"episodes"/task/arm/seed/"http-metrics.jsonl"
            if request.path.endswith("chat/completions"):
                body = json.loads(raw)
                # Ordinary calls; official auxiliary compaction calls remain identifiable.
                if body.get("max_completion_tokens", body.get("max_tokens")) == 16384:
                    expected = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "thinking_token_budget": 4096}
                    if any(body.get(k) != v for k, v in expected.items()):
                        raise web.HTTPBadRequest(text="Sampling policy mismatch")
            try:
                async with client.request(request.method, "http://127.0.0.1:8400"+request.path_qs,
                                          data=raw, headers=headers) as response:
                    data = await response.read()
                    try:
                        usage = json.loads(data).get("usage")
                    except ValueError:
                        usage = None
                    event = {"episode": episode, "rank": rank, "started": started, "ended": time.time(),
                             "status": response.status, "usage": usage}
                    with log_path.open("a") as log:
                        log.write(json.dumps(event)+"\n")
                    sample = int(hashlib.sha256(episode.encode()).hexdigest()[:8], 16)/2**32
                    if sample < config["raw_http_sample_fraction"] or response.status != 200:
                        path = out / "http-samples" / (episode+".jsonl.gz")
                        path.parent.mkdir(exist_ok=True)
                        with gzip.open(path, "at") as f:
                            f.write(json.dumps(event | {"request": raw.decode(), "response": data.decode()})+"\n")
                    return web.Response(body=data, status=response.status,
                                        headers={"Content-Type": response.headers.get("Content-Type", "application/json")})
            except BaseException as e:
                with log_path.open("a") as log:
                    log.write(json.dumps({"episode": episode, "error": type(e).__name__, "started": started,
                                         "ended": time.time()})+"\n")
                raise
        app = web.Application(client_max_size=32*2**20)
        app.router.add_route("*", "/{path:.*}", forward)
        runner = web.AppRunner(app, handler_cancellation=True)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 8401).start()
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()


def histogram(records, k=8):
    bins = [0] * (k+1)
    incomplete = 0
    for results in records:
        if len(results) != k or any(type(r.get("resolved")) is not bool for r in results):
            incomplete += 1
        else:
            bins[sum(r["resolved"] for r in results)] += 1
    n = sum(bins)
    return {"histogram": bins, "complete_issues": n, "incomplete_issues": incomplete,
            "pass_at_8": (n-bins[0])/n if n else None,
            "mean_attempt_success": sum(i*v for i,v in enumerate(bins))/(k*n) if n else None,
            "all_zero_fraction": bins[0]/n if n else None, "all_pass_fraction": bins[k]/n if n else None}


def next_completed_issue(pending, check_services=lambda: None):
    """Release any completed issue; one slow episode must not idle the other ranks."""
    while pending:
        check_services()
        for index, (_, _, jobs) in enumerate(pending):
            if all(job.done() for job in jobs):
                return pending.pop(index)
        cf.wait([job for _, _, jobs in pending for job in jobs if not job.done()],
                return_when=cf.FIRST_COMPLETED, timeout=5)
    raise ValueError("No pending issues")


class Experiment:
    def __init__(self, config, config_path):
        self.c = config
        self.config_path = str(Path(config_path).resolve())
        self.root = Path(config["project"])
        self.out = Path(config["output"])
        self.python = config["python"]
        self.script = str(Path(__file__).resolve())
        self.worker = Path(__file__).with_name("rl_episode.py")
        self.owner = "pass8-"+uuid.uuid4().hex
        self.deadline = time.monotonic()+config["wall_seconds"]
        self.env = dict(os.environ, DOCKER_HOST=DOCKER, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                        VLLM_NO_USAGE_STATS="1", PYTHON_DOTENV_DISABLED="1", LITELLM_TELEMETRY="False",
                        LITELLM_LOCAL_MODEL_COST_MAP="True", OTEL_SDK_DISABLED="true",
                        OPENAI_API_KEY="local-only", LOCAL_API_KEY="local-only", GEMMA_RL_OWNER=self.owner)
        self.client = docker_client()
        sys.path.insert(0, str(self.root / "scripts"))
        from owned_process_group_v4 import RetainedGroup
        self.retain = live_retention(RetainedGroup)
        self.attempt = self.out/"attempts"/str(time.time_ns())
        self.attempt.mkdir(parents=True, exist_ok=False)
        self.services = []
        self.acquired_images = {}
        self.initial_images = {i.id for i in self.client.images.list()}
        self.grade_slots = threading.Semaphore(config["environment"]["grading_workers"])
        self.rank_slots = [threading.Semaphore(config["inference"]["concurrent_episodes"]//8) for _ in range(8)]
        self.lock = threading.Lock()
        self.results = {}

    def spawn(self, command, path, *, grading=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(self.env)
        if grading:
            env["PYTHONPATH"] = str(self.out/"harness")+":"+str(self.out/"deps")
        with path.open("xb") as log:
            p = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True, cwd=self.out)
        holder = {}
        def snapshot(deadline=None):
            keys = holder["group"].record["known_processes"] if "group" in holder else ()
            return private_snapshot(p.pid, keys, deadline=deadline)
        retained = self.retain(p, path.with_suffix(".ownership.json"), role="client", snapshot=snapshot)
        holder["group"] = retained
        return p, retained

    def bounded(self, command, path, seconds, *, grading=False):
        p, group = self.spawn(command, path, grading=grading)
        try:
            code = p.wait(timeout=min(seconds, max(1, self.deadline-time.monotonic())))
            if code:
                raise RuntimeError(f"Child failed ({code}); see {path}")
        finally:
            group.cleanup()

    def capacity(self):
        e = self.c["environment"]
        if shutil.disk_usage("/dev/shm").free < e["scratch_floor_gib"]*2**30:
            raise RuntimeError("Private scratch floor reached; preserved outputs, no broad cleanup")
        if shutil.disk_usage(self.out).free < e["disk_floor_gib"]*2**30:
            raise RuntimeError("Output disk floor reached; preserve trajectories")

    def check_services(self):
        for process, _ in self.services:
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"Inference service PID {process.pid} exited ({code}); stop new dispatch")

    def run_grade(self, row_path, patch, out, image):
        request = {"row": str(row_path), "patch": str(patch), "output": str(out),
                   "image": image["image"], "image_id": image["image_id"]}
        request_path = out.with_name(out.name+"-request.json")
        write(request_path, request)
        with self.grade_slots:
            self.bounded([self.python, self.script, "--config", self.config_path, "--mode", "grade",
                          "--request", request_path], out.with_name(out.name+".log"),
                         self.c["environment"]["test_seconds"]+180, grading=True)
        return json.loads((out/"outcome.json").read_text())

    def release_image(self, task):
        entry = self.acquired_images.get(task)
        if entry is None or not self.c["environment"]["release_owned_images"]:
            return
        image = self.client.images.get(entry["image"])
        if image.id != entry["image_id"]:
            raise RuntimeError("Owned image changed; refuse removal")
        if self.client.containers.list(all=True, filters={"ancestor":image.id}):
            raise RuntimeError("Owned image has a container; refuse removal")
        self.client.images.remove(entry["image"], force=False, noprune=True)
        write(self.out/"qualification"/task/"image-release.json", entry | {"released_unix":time.time()})
        del self.acquired_images[task]

    def qualify(self, row):
        task = row["instance_id"]
        directory = self.out/"qualification"/task
        if (directory/"admission.json").exists():
            entry = json.loads((directory/"admission.json").read_text())
            if self.client.images.get(entry["image"]).id != entry["image_id"]:
                raise RuntimeError("Previously qualified image changed")
            for label, expected in (("gold", True), ("neutral", False)):
                path = directory/label/"outcome.json"
                if digest(path) != entry[label+"_outcome_sha256"]:
                    raise RuntimeError("Environment control changed")
                if json.loads(path.read_text())["resolved"] != expected:
                    raise RuntimeError("Environment control failed")
            if json.loads(Path(entry["row"]).read_text()) != row:
                raise RuntimeError("Task row differs from qualification")
            return entry
        directory.mkdir(parents=True, exist_ok=False)
        row_path = directory/"private-row.json"
        write(row_path, row)
        name = row.get("image_name")
        if not name:
            raise RuntimeError("No published image: environment build remains unqualified")
        name = name if ":" in name else name+":latest"
        self.capacity()
        import docker
        try:
            image = self.client.images.get(name)
        except docker.errors.ImageNotFound:
            self.bounded(["docker", "-H", DOCKER, "pull", name], directory/"pull.log",
                         self.c["environment"]["pull_seconds"])
            image = self.client.images.get(name)
            if image.id not in self.initial_images:
                self.acquired_images[task] = {"image":name,"image_id":image.id}
                write(directory/"acquired-image.json", self.acquired_images[task])
        self.capacity()
        entry = {"image": name, "image_id": image.id, "python_prefix": "/opt/miniconda3/envs/testbed"}
        for label, patch in (("gold", row["patch"]), ("neutral", NEUTRAL)):
            file = directory/(label+".patch")
            file.write_text(patch)
            outcome = self.run_grade(row_path, file, directory/label, entry)
            if outcome["resolved"] != (label == "gold") or not outcome["tests_executed"]:
                raise QualificationRejected(f"Environment {label} control failed")
            entry[label+"_outcome_sha256"] = digest(directory/label/"outcome.json")
        prepare = directory/"solver-preparation"
        prepare.mkdir()
        public_request = {"project_root":str(self.root),"output":str(prepare),"task":{
            k:row[k] for k in ("instance_id","repo","base_commit","problem_statement","version")},
            "image":entry,"prepare_only":True}
        write(directory/"prepare-request.json", public_request)
        self.bounded([self.python,self.worker,"--request",directory/"prepare-request.json"],directory/"prepare.log",300)
        entry["row"] = str(row_path)
        write(directory/"admission.json", entry)
        return entry

    def bundles(self):
        source = Path(self.c["bundle"])
        for arm in self.c["arms"]:
            for seed in self.c["seeds"]:
                dest = self.out/"bundles"/arm/str(seed)
                shutil.copytree(source, dest, ignore=shutil.ignore_patterns("adapters"))
                agent = (dest/"agent.yaml").read_text()
                if arm == "baseline":
                    agent = "\n".join(l for l in agent.splitlines() if not l.startswith("adapter:"))+"\n"
                else:
                    shutil.copytree(self.c["adapter"], dest/"adapters"/"hor3_step25", copy_function=os.link)
                (dest/"agent.yaml").write_text(agent)
                sampling = dest/"configs"/"sampling.yaml"
                sampling.write_text(re.sub(r"(?m)^seed:.*$", f"seed: {seed}", sampling.read_text()))

    def start_server(self):
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            raise RuntimeError("GPU contexts are occupied; refusing allocation")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        if len(gpu.splitlines()) != 8 or any(int(a)>4 or int(b)>0 for a,b in (l.split(",") for l in gpu.splitlines())):
            raise RuntimeError("Eight idle GPUs required")
        torchlib = subprocess.check_output([self.python,"-c","import importlib.metadata;print(importlib.metadata.distribution('torch').locate_file('torch/lib'))"], text=True).strip()
        self.env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", NCCL_TOPO_FILE=str(self.root/".runtime/nccl/autodetect.xml"),
                        LD_LIBRARY_PATH=torchlib+":"+str(self.root/".runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat"))
        cfg = self.c["inference"]
        command = [self.python,"-m","vllm.entrypoints.cli.main","serve",self.c["model"],"--host","127.0.0.1","--port","8400",
                   "--data-parallel-size","8","--tensor-parallel-size","1","--data-parallel-backend","mp","--api-server-count","8",
                   "--language-model-only","--dtype","bfloat16","--lora-dtype","bfloat16",
                   "--gpu-memory-utilization",str(cfg["gpu_memory_utilization"]),"--max-model-len",str(cfg["context"]),
                   "--max-num-seqs",str(cfg["max_num_seqs"]),"--max-num-batched-tokens",str(cfg["max_num_batched_tokens"]),
                   "--enable-auto-tool-choice","--tool-call-parser","gemma4","--reasoning-parser","gemma4",
                   "--reasoning-config",'{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}',
                   "--no-scheduler-reserve-full-isl","--served-model-name","gemma-rl",
                   "--override-generation-config",'{"temperature":1.0,"top_p":0.95,"top_k":64}',
                   "--enable-lora","--max-loras","1","--max-cpu-loras","1","--max-lora-rank","32",
                   "--lora-modules","hor3_step25="+self.c["adapter"]]
        write(self.attempt/"server-command.json", command)
        server = self.spawn(command, self.attempt/"server.log")
        self.services.append(server)
        until = time.monotonic()+1200
        while time.monotonic()<until:
            if server[0].poll() is not None:
                raise RuntimeError("vLLM startup failed; no automatic memory reduction")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8400/v1/models",timeout=3) as r:
                    names = {x["id"] for x in json.load(r)["data"]}
                if {"gemma-rl","hor3_step25"} <= names:
                    break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("vLLM readiness deadline")
        proxy_process = self.spawn([self.python,self.script,"--config",self.config_path,"--mode","proxy"],self.attempt/"proxy.log")
        self.services.append(proxy_process)
        for _ in range(60):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8401/v1/models", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise TimeoutError("Recording proxy did not start")

    def episode(self, row, image, arm, seed):
        task = row["instance_id"]
        eid = f"{task}.{arm}.{seed}"
        rank = (seed-self.c["seeds"][0]) % 8
        out = self.out/"episodes"/task/arm/str(seed)
        try:
            with self.rank_slots[rank]:
                self.check_services()
                if self.deadline-time.monotonic()<self.c["episode"]["minutes"]*60+900:
                    raise TimeoutError("Insufficient slice budget; no generation dispatched")
                self.capacity()
                out.mkdir(parents=True, exist_ok=False)
                request = {"project_root":str(self.root),"output":str(out),
                           "task":{k:row[k] for k in ("instance_id","repo","base_commit","problem_statement","version")},
                           "image":image,"bundle":str(self.out/"bundles"/arm/str(seed)),
                           "endpoint":"http://127.0.0.1:8401/v1","episode":self.c["episode"],"rank":rank,"episode_id":eid}
                # Private test labels and reference patch never enter the solver request.
                request["image"] = {k:v for k,v in image.items() if k!="row"}
                path = out/"request.json"
                write(path,request)
                self.bounded([self.python,self.worker,"--request",path],out/"generation.log",
                             self.c["episode"]["minutes"]*60+300)
                if not (out/"generation.json").exists():
                    raise RuntimeError("Missing generation completion")
                with (out/"trace.json").open("rb") as src, gzip.open(out/"trace.json.gz","xb") as dst:
                    shutil.copyfileobj(src,dst)
                with gzip.open(out/"trace.json.gz","rb") as src:
                    if hashlib.sha256(src.read()).hexdigest()!=digest(out/"trace.json"):
                        raise RuntimeError("Trajectory compression mismatch")
                (out/"trace.json").unlink()
                events = [json.loads(l) for l in (out/"http-metrics.jsonl").read_text().splitlines()]
                if not events or any(e.get("status") != 200 for e in events):
                    raise RuntimeError("Missing/failed model responses: infrastructure invalid, not zero")
            outcome = self.run_grade(image["row"],out/"agent.patch",out/"grading",image)
            write(out/"result.json", outcome | {"episode":eid,"seed":seed,"arm":arm,"patch_sha256":digest(out/"agent.patch")})
            with self.lock:
                self.results.setdefault((task,arm),[]).append(outcome)
        except Exception as e:
            write(out/"failure.json",{"episode":eid,"stage":"episode_or_grading","error":repr(e),"model_rerun":False})
        self.status("running")

    def status(self, state):
        with self.lock:
            summary = {arm:histogram([v for (t,a),v in self.results.items() if a==arm],len(self.c["seeds"])) for arm in self.c["arms"]}
            write(self.out/"status.json",{"state":state,"owner":self.owner,"updated_unix":time.time(),
                  "graded_episodes":sum(len(v) for v in self.results.values()),"arms":summary,
                  "eligible_issues":self.cohort["eligible_tasks"] if hasattr(self,"cohort") else None,
                  "unqualified_build_issues":self.cohort["build_required"] if hasattr(self,"cohort") else None,
                  "qualification_failures":len(list((self.out/"qualification").glob("*/failure.json"))),
                  "episode_failures":len(list((self.out/"episodes").glob("*/*/*/failure.json")))})

    def run(self):
        import fcntl
        self.run_lock = (self.out/"controller.lock").open("a")
        fcntl.flock(self.run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous = json.loads((self.out/"launch.json").read_text()) if (self.out/"launch.json").exists() else None
        if previous:
            if previous["config"] != self.c:
                raise RuntimeError("Continuation must preserve the exact experiment configuration")
            self.deadline = time.monotonic()+previous["started_unix"]+self.c["wall_seconds"]-time.time()
            for path in self.out.glob("episodes/*/*/*/result.json"):
                result=json.loads(path.read_text())
                if digest(path.parent/"agent.patch") != result["patch_sha256"]:
                    raise RuntimeError("Saved prediction changed")
                if result.get("tests_executed") and digest(result["report"])!=result["report_sha256"]:
                    raise RuntimeError("Saved grading report changed")
                self.results.setdefault((path.parents[2].name,path.parents[1].name),[]).append(result)
        if digest(Path(self.c["adapter"])/"adapter_model.safetensors")!=self.c["adapter_sha256"]:
            raise RuntimeError("Wrong SFT checkpoint")
        fork_revision = subprocess.check_output(["git","-C",str(self.out/"harness"),"rev-parse","HEAD"],text=True).strip()
        if fork_revision!=self.c["harness_revision"]:
            raise RuntimeError("Grader revision changed")
        cohort = json.loads((self.out/"cohort.json").read_text())
        self.cohort = cohort
        if digest(self.out/"private/tasks.jsonl")!=cohort["tasks_sha256"]:
            raise RuntimeError("Cohort changed")
        write(self.attempt/"launch.json" if previous else self.out/"launch.json",{"owner":self.owner,"config":self.c,"started_unix":time.time(),
                                    "sources":{str(p):digest(p) for p in (Path(self.script),self.worker)}})
        if not previous:
            self.bundles()
        ready = []
        consecutive_failures = 0
        try:
            def finish(item):
                row,image,jobs=item
                for f in cf.as_completed(jobs):
                    f.result()
                self.release_image(row["instance_id"])
            with cf.ThreadPoolExecutor(max_workers=self.c["inference"]["concurrent_episodes"]) as pool:
                for line in (self.out/"private/tasks.jsonl").open():
                    row = json.loads(line)
                    if not row.get("image_name") or self.deadline-time.monotonic()<5400:
                        break  # Build-only coverage remains explicitly pending.
                    if previous and (self.out/"qualification"/row["instance_id"]).exists():
                        continue  # Preserve every prior attempt; never regenerate an attempted issue.
                    self.check_services()
                    if len(ready)>=self.c["environment"]["issues_per_batch"]:
                        finish(next_completed_issue(ready, self.check_services))
                    self.status("running_and_qualifying" if self.services else "environment_qualification")
                    try:
                        image = self.qualify(row)
                        consecutive_failures = 0
                    except QualificationRejected as e:
                        write(self.out/"qualification"/row["instance_id"]/"failure.json",{
                            "error":repr(e),"model_attempts":0,"classification":"failed_reference_control"})
                        self.release_image(row["instance_id"])
                        consecutive_failures = 0
                        continue
                    except Exception as e:
                        write(self.out/"qualification"/row["instance_id"]/"failure.json",{"error":repr(e),"model_attempts":0})
                        self.release_image(row["instance_id"])
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            raise RuntimeError("Three consecutive environment failures; inspect before further launches") from e
                        continue
                    if not self.services:
                        self.start_server()
                    self.check_services()
                    jobs=[pool.submit(self.episode,row,image,arm,seed)
                          for arm in self.c["arms"] for seed in self.c["seeds"]]
                    ready.append((row,image,jobs))
                    self.status("running_and_qualifying")
                for item in ready:
                    finish(item)
            self.status("slice_complete_coverage_pending")
        except BaseException as e:
            write(self.attempt/"failure.json",{"error":repr(e),"predictions_preserved":True})
            self.status("failed")
            raise
        finally:
            cleanup_errors=[]
            # The executor has joined all episode workers before this point.
            for task in list(self.acquired_images):
                try:
                    self.release_image(task)
                except Exception as error:
                    cleanup_errors.append({"task":task,"error":repr(error)})
            for p,group in reversed(self.services):
                try:
                    group.cleanup()
                except Exception as error:
                    cleanup_errors.append({"pid":p.pid,"error":repr(error)})
            write(self.attempt/"cleanup.json",{"clean":not cleanup_errors,"errors":cleanup_errors})
            if cleanup_errors:
                raise RuntimeError(f"Owned service cleanup failed: {cleanup_errors}")


def main():
    import yaml
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--mode",choices=["run","qualify","grade","proxy"],default="run")
    p.add_argument("--request")
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    if args.mode=="grade":
        grade(config,json.loads(Path(args.request).read_text()))
    elif args.mode=="proxy":
        import asyncio
        asyncio.run(proxy(config))
    elif args.mode=="qualify":
        row=json.loads(next((Path(config["output"])/"private/tasks.jsonl").open()))
        print(json.dumps(Experiment(config,args.config).qualify(row)),flush=True)
    else:
        Experiment(config,args.config).run()


if __name__=="__main__":
    def stop(number,frame):
        raise KeyboardInterrupt("Stop requested; preserve predictions and clean owned workers")
    signal.signal(signal.SIGTERM,stop)
    main()
