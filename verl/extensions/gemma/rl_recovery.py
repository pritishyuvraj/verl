"""Recover immutable generations with their original behavior policy and test rewards."""
import hashlib
import json
from pathlib import Path
import shutil


class CompletedClaims:
    """One async actor assigns whole saved groups, never independently sampled turns."""
    def __init__(self):
        self.groups = {}
        self.used = {}
        self.sessions = set()

    def claim(self, manifest, *, task, uid, session, validate, step):
        if validate:
            return manifest['dev'][task] if int(step) == 0 else None
        if int(step) > 1:
            # Native async sampling may evict prefetched initial-policy groups
            # after an update. Its refill must generate using the updated policy,
            # never relabel or replay initial-policy data as fresh experience.
            return None
        if int(step) not in (0, 1):
            raise ValueError('Saved initial-policy training groups exceed allowed version lag')
        key = (task, uid)
        if key not in self.groups:
            index = self.used.get(task, 0)
            groups = manifest['train'][task]
            if index >= len(groups):
                raise ValueError('Saved training groups exhausted; refusing duplicate generation')
            self.groups[key] = groups[index]
            self.used[task] = index + 1
        session_key = (task, uid, int(session))
        if session_key in self.sessions:
            raise ValueError('Training session already recovered')
        record = self.groups[key][str(int(session))]
        self.sessions.add(session_key)
        return record


def recover_completed(root, task, out, record, *, config_path):
    """Reuse independently graded episodes without another generation or test run."""
    manifest_path = root / 'completed-episode-recovery.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['config_sha256'] != sha(config_path):
        raise ValueError('Completed recovery config changed')
    for name, digest in manifest['inputs'].items():
        if sha(root / name) != digest:
            raise ValueError('Completed recovery input changed')
    source = (root / record['path']).resolve()
    if not source.is_relative_to(root.resolve() / 'episodes'):
        raise ValueError('Recovery source outside episode archive')
    for name, digest in record['files'].items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source) or sha(path) != digest:
            raise ValueError('Completed episode artifact changed')
    for name, digest in record['evidence'].items():
        if sha(name) != digest:
            raise ValueError('Independent grading evidence changed')
    completed = json.loads((source / 'completed.json').read_text())
    request = json.loads((source / 'episode-request.json').read_text())
    reward = json.loads((source / 'grade/reward.json').read_text())
    if (completed['launch_owner'] != manifest['source_owner'] or request['task'] != task
            or completed['task'] != task['instance_id'] or type(reward['resolved']) is not bool
            or completed['reward'] != float(reward['resolved'])):
        raise ValueError('Completed task/reward/owner mismatch')
    turns = []
    for name in sorted(record['files']):
        if not name.startswith('response-'):
            continue
        turn = json.loads((source / name).read_text())
        payload = json.loads((source / name.replace('response-', 'request-')).read_text())
        if any(turn['extra_fields'][key] != 0 for key in ('min_global_steps', 'max_global_steps')):
            raise ValueError('Recovery requires the unchanged initial behavior policy')
        if not turn['response_ids'] or len(turn['response_ids']) != len(turn['log_probs']):
            raise ValueError('Saved token/logprob alignment is invalid')
        turn['ordinary'] = output_cap(payload) == 16384
        turns.append(turn)
    if not any(t['ordinary'] for t in turns):
        raise ValueError('No saved ordinary actions')
    for name in record['files']:
        target = out / ('saved-completed.json' if name == 'completed.json' else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (out / 'completed-recovery.json').write_text(json.dumps({
        'source':str(source), 'manifest_sha256':sha(manifest_path),
        'source_uid':completed['uid'], 'source_session':completed['session'],
        'behavior_policy_version':0, 'generation_rerun':False, 'grading_rerun':False}))
    return turns, completed['seconds'], completed['context_errors']


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
