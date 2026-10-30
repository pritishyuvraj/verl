"""Recover only unchanged step-zero validation generations, never training samples."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_cap(payload):
    # The official LiteLLM transport uses the newer OpenAI field.
    return payload.get("max_completion_tokens") or payload.get("max_tokens")


def context_budget_error(error):
    text = str(error).lower()
    return "token" in text and "context length" in text and ("exceed" in text or "maximum" in text)


def grade_run_id(out, task, attempt=1):
    if attempt not in (1, 2, 3):
        raise ValueError("At most three grade-only attempts")
    return f"v4-{hashlib.sha256(str(out).encode()).hexdigest()[:12]}-{hashlib.sha256(task.encode()).hexdigest()[:12]}-g{attempt}"


def recover_initial(root, task, out, *, validate, step, config_path):
    manifest_path = root / "initial-validation-recovery.json"
    if not validate or int(step) != 0 or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != sha(config_path):
        raise ValueError("Initial validation recovery config changed")
    for path, digest in manifest["inputs"].items():
        if sha(root / path) != digest:
            raise ValueError("Initial validation recovery input changed")
    record = manifest["tasks"].get(task["instance_id"])
    if record is None:
        return None
    source = root / record["path"]
    for name, digest in record["files"].items():
        if Path(name).name != name or sha(source / name) != digest:
            raise ValueError("Saved initial generation changed")
    request = json.loads((source / "episode-request.json").read_text())
    if request["task"] != task or request["launch_owner"] != manifest["source_owner"]:
        raise ValueError("Recovery task/owner mismatch")
    turns = []
    for name in sorted(record["files"]):
        if name.startswith("response-"):
            turn = json.loads((source / name).read_text())
            payload = json.loads((source / name.replace("response-", "request-")).read_text())
            if turn["extra_fields"]["min_global_steps"] != 0 or turn["extra_fields"]["max_global_steps"] != 0:
                raise ValueError("Only unchanged initial policy generations can be reused")
            if not turn["response_ids"] or len(turn["response_ids"]) != len(turn["log_probs"]):
                raise ValueError("Saved token/logprob alignment is invalid")
            turn["ordinary"] = output_cap(payload) == 16384
            turns.append(turn)
    if not turns or not any(t["ordinary"] for t in turns):
        raise ValueError("Missing saved policy actions")
    for name in record["files"]:
        shutil.copyfile(source / name, out / name)
    (out / "recovered-generation.json").write_text(json.dumps({"source":str(source),
        "source_owner":manifest["source_owner"], "manifest_sha256":sha(manifest_path)}))
    return turns
