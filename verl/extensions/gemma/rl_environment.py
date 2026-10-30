"""Finite SWE environment admission and independent executable reward worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

DOCKER_HOST = "unix:///dev/shm/pritish-gemma4-20261025/eval-docker/docker.sock"
DAEMON = "9469315a-6cc5-4032-a513-6427b656879d"
REVISION = "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def client():
    import docker
    if os.environ.get("DOCKER_HOST") != DOCKER_HOST:
        raise RuntimeError("Explicit private Docker endpoint required")
    c = docker.from_env(timeout=120)
    if c.info()["ID"] != DAEMON:
        raise RuntimeError("Private daemon identity changed")
    if shutil.disk_usage("/dev/shm").free < 96 * 2**30:
        raise RuntimeError("Insufficient grading scratch: retain 96 GiB")
    return c


def grade(row, patch, output, expected_image=None):
    """Uses the pinned upstream SWE-bench grader; never exposes gold to the solver."""
    from swebench.harness.run_evaluation import run_instances
    from swebench.harness.test_spec.test_spec import make_test_spec
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    c = client()
    spec = make_test_spec(row, namespace="swebench")
    image = c.images.get(spec.instance_image_key)
    if expected_image and image.id != expected_image:
        raise RuntimeError("Grading image drift")
    task = row["instance_id"]
    # Empty generated patches are genuine failures, not an infrastructure exception.
    if not patch.strip():
        result = {"instance_id": task, "resolved": False, "empty_patch": True,
                  "official_tests_executed": False, "image_id": image.id}
        write(output / "reward.json", result)
        return result
    run_id = "grpo-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    owned_name = spec.get_instance_container_name(run_id)
    try:
        c.containers.get(owned_name)
    except __import__("docker").errors.NotFound:
        pass
    else:
        raise RuntimeError("Refusing to reuse an existing grading container")
    cwd = Path.cwd()
    started = time.monotonic()
    os.chdir(output)
    try:
        predictions = {task: {"instance_id": task, "model_name_or_path": "gemma-grpo",
                              "model_patch": patch}}
        run_instances(predictions, [row], "instance", False, False, 1, run_id, 600,
                      namespace="swebench", rewrite_reports=False)
        reports = list(Path("logs/run_evaluation").glob("**/report.json"))
        if len(reports) != 1:
            raise RuntimeError("Official grading report missing or ambiguous")
        report = json.loads(reports[0].read_text())[task]
        if type(report.get("resolved")) is not bool:
            raise RuntimeError("Invalid official grading report")
        result = {"instance_id": task, "resolved": report["resolved"], "empty_patch": False,
                  "official_tests_executed": True, "report": str(output / reports[0]),
                  "report_sha256": digest(reports[0]), "image_id": image.id,
                  "patch_applied": report.get("patch_successfully_applied", False),
                  "seconds": time.monotonic() - started}
        write(output / "reward.json", result)
        return result
    finally:
        os.chdir(cwd)
        # This exact name was absent before this unique run. Never touch other containers.
        try:
            own = c.containers.get(owned_name)
        except __import__("docker").errors.NotFound:
            pass
        else:
            if own.image.id != image.id:
                raise RuntimeError("Owned grading container identity changed")
            own.remove(force=True)


def prepare(project, root, count=8):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from swebench.harness.test_spec.test_spec import make_test_spec
    project, root = Path(project).resolve(), Path(root).resolve()
    source = root / "source/data/test-00000-of-00001.parquet"
    rows = pq.read_table(source).to_pylist()
    exclusions = json.loads((root / "exclusions.json").read_text())
    excluded = set(exclusions["instance_ids"])
    excluded_patches = set(exclusions["patch_sha256"])
    if len(excluded) != 354:
        raise RuntimeError("Expected all 354 known heldout tasks")
    candidates = [r for r in rows if r["repo"] == "pytest-dev/pytest" and r["instance_id"] not in excluded
                  and hashlib.sha256(r["patch"].strip().encode()).hexdigest() not in excluded_patches]
    selected = sorted(candidates, key=lambda x: hashlib.sha256(("42:" + x["instance_id"]).encode()).hexdigest())[:count]
    if len(selected) != count:
        raise RuntimeError("Insufficient disjoint training issues")
    manifest = {"status": "preparing", "dataset": "princeton-nlp/SWE-bench", "revision": REVISION,
                "split": "test", "use": "training; public-test benchmark claims forfeited",
                "source_sha256": digest(source), "selection": "pytest repository; hash order seed42",
                "excluded_issue_count": len(excluded),
                "exclusion_files": exclusions["source_sha256"],
                "exclusions_sha256": digest(root / "exclusions.json"),
                "tasks": [r["instance_id"] for r in selected], "judge": "offline"}
    write(root / "cohort.json", manifest)
    private = root / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    admitted = {}
    start = time.monotonic()
    for row in selected:
        if time.monotonic() - start > 5400:
            raise TimeoutError("Environment preparation stage exhausted")
        c = client()
        task = row["instance_id"]
        spec = make_test_spec(row, namespace="swebench")
        try:
            image = c.images.get(spec.instance_image_key)
        except __import__("docker").errors.ImageNotFound:
            subprocess.run(["timeout", "--kill-after=20s", "300s", "docker", "-H", DOCKER_HOST,
                            "pull", spec.instance_image_key], check=True, timeout=325)
            image = c.images.get(spec.instance_image_key)
        client()  # capacity after pull
        write(private / f"{task}.json", row)
        gold = grade(row, row["patch"], root / "qualification" / task / "gold", image.id)
        neutral = "diff --git a/.rl_neutral_probe b/.rl_neutral_probe\nnew file mode 100644\n--- /dev/null\n+++ b/.rl_neutral_probe\n@@ -0,0 +1 @@\n+neutral\n"
        negative = grade(row, neutral, root / "qualification" / task / "neutral", image.id)
        if not gold["resolved"] or negative["resolved"]:
            raise RuntimeError(f"Environment controls failed for {task}; cohort will not be silently replaced")
        admitted[task] = {"image": spec.instance_image_key, "image_id": image.id,
                          "python_prefix": "/opt/miniconda3/envs/testbed", "base_commit": row["base_commit"],
                          "gold": gold, "neutral": negative}
        print(json.dumps({"qualified": len(admitted), "target": count, "task": task}), flush=True)
    public = [{"prompt": [{"role": "user", "content": r["problem_statement"]}],
               "agent_name": "gemma_swe", "data_source": "swebench_rl_train",
               "reward_model": {"style": "rule", "ground_truth": ""},
               "extra_info": {k: r[k] for k in ("instance_id", "repo", "base_commit", "problem_statement", "version")}}
              for r in selected]
    pq.write_table(pa.Table.from_pylist(public), root / "train.parquet")
    manifest.update(status="qualified", images=admitted, train_sha256=digest(root / "train.parquet"),
                    seconds=time.monotonic()-start)
    write(root / "environment-admission.json", manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "grade"])
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--row", type=Path)
    p.add_argument("--patch", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--image-id")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.project, args.root)
    else:
        grade(json.loads(args.row.read_text()), args.patch.read_text(), args.output, args.image_id)


if __name__ == "__main__":
    main()
