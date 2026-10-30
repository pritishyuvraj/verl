"""Finite, isolated hor-1 pilot launcher; never attaches to an existing Ray cluster."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def main():
    project = Path(os.environ["GEMMA_PROJECT_ROOT"]).resolve()
    root = Path(os.environ["GEMMA_RL_ROOT"]).resolve()
    admission = json.loads((root / "environment-admission.json").read_text())
    assert admission["status"] == "qualified" and len(admission["tasks"]) == 8
    assert sha(root / "train.parquet") == admission["train_sha256"]
    assert sha(root / "initial-adapter/adapter_model.safetensors") == (
        "b545e114ce31262cf455e8545b7d405b9a63fc1e9c78b3ffb4027a5bed77cf79")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = [list(map(int, line.split(","))) for line in gpu.strip().splitlines()]
    assert len(rows) == 8 and all(memory <= 4 and util == 0 for _, memory, util in rows)
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                        "--format=csv,noheader"], text=True).strip()
    token = uuid.uuid4().hex
    attempt = root / "launches" / token
    attempt.mkdir(parents=True)
    env = {**os.environ, "GEMMA_RL_OWNER": token, "RAY_ADDRESS": "local",
           "VERL_USE_EXTERNAL_PLUGINS": "none", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", "HYDRA_FULL_ERROR": "1",
           "LD_LIBRARY_PATH": str(project / ".runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat"),
           "NCCL_TOPO_FILE": str(project / ".runtime/nccl/autodetect.xml"),
           "TENSORBOARD_DIR": str(root / "tensorboard")}
    command = [sys.executable, "-u", "-m", "verl.trainer.main_ppo", "--config-name", "gemma_grpo_hor1",
               "++ray_kwargs.ray_init._temp_dir=/tmp/gemma-rl-" + token[:12],
               "++ray_kwargs.ray_init.runtime_env.env_vars.GEMMA_RL_OWNER=" + token]
    known = {}

    def owned():
        found = {}
        for directory in Path("/proc").glob("[0-9]*"):
            try:
                if directory.stat().st_uid != os.getuid():
                    continue
                if ("GEMMA_RL_OWNER=" + token).encode() not in (directory / "environ").read_bytes().split(b"\0"):
                    continue
                stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                if stat[0] != "Z":
                    found[int(directory.name)] = stat[19]
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        known.update(found)
        return found

    def terminate(sig):
        for pid, start in owned().items():
            try:
                fd = os.pidfd_open(pid)
                now = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
                if now != start:
                    raise RuntimeError("Owned process identity changed")
                signal.pidfd_send_signal(fd, sig)
                os.close(fd)
            except (FileNotFoundError, ProcessLookupError):
                continue

    started = time.time()
    receipt = {"state": "starting", "command": command, "started": started,
               "deadline": started + 21600, "owner": token,
               "config_sha256": sha(Path(__file__).parents[2] / "trainer/config/gemma_grpo_hor1.yaml")}
    write(attempt / "status.json", receipt)
    code = None
    try:
        with (attempt / "trainer.log").open("x") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
            receipt.update(state="initializing", pid=proc.pid)
            write(attempt / "status.json", receipt)
            print(json.dumps({"launch": str(attempt), "pid": proc.pid}), flush=True)
            while proc.poll() is None and time.time() < receipt["deadline"]:
                owned()
                time.sleep(3)
            code = proc.poll()
    finally:
        terminate(signal.SIGTERM)
        cutoff = time.monotonic() + 45
        while owned() and time.monotonic() < cutoff:
            time.sleep(1)
        terminate(signal.SIGKILL)
        time.sleep(2)
        survivors = owned()
        receipt.update(state="completed" if code == 0 and not survivors else "failed",
                       returncode=code, finished=time.time(), remaining_owned=survivors,
                       observed_processes=known)
        write(attempt / "status.json", receipt)
        print(json.dumps(receipt), flush=True)
    if receipt["state"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Finite pilot interrupted; clean owned processes")
    signal.signal(signal.SIGTERM, interrupt)
    main()
