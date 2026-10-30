"""One official ADK episode, invoked with the scorer interpreter and public inputs only."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    a = p.parse_args()
    request = json.loads(a.request.read_text())
    root = Path(request["project_root"])
    out = Path(request["output"])
    helper_path = root / "outputs/runtime/gemma-relaxed-128k-60m-20261027-attempt004-baseline-continuation/diagnostic_agent_prepare.py"
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != "02cd1364fc54eb2006addf64fc04cd3746c4ddbba907aabff9b6102c7e758fbe":
        raise RuntimeError("Qualified environment adapter changed")
    helper = module(helper_path, "swe_rl_environment_adapter")
    from swegemma.sandbox import ContainerConfig, ContainerManager
    from swegemma.config import EvalConfig, build_submission_limits, ALLOWED_ADAPTER_EXTENSIONS
    from swegemma.harness.agent_runner import run_agent_sandbox
    from swegemma.models import Task
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from google.adk.apps._configs import EventsCompactionConfig
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.models.lite_llm import LiteLlm
    from urllib.parse import urlsplit
    task, image = request["task"], request["image"]
    if set(task) != {"instance_id", "repo", "base_commit", "problem_statement", "version"}:
        raise RuntimeError("Solver inputs must contain only five public fields")
    fixture_helper = None
    profile = None
    if request.get("fixture_index"):
        fixture_helper = module(helper.FIXTURE_HELPER, "swe_rl_public_fixtures")
        profile = fixture_helper.load_profile(Path(request["fixture_index"]))

    def factory(image):
        manager = ContainerManager(ContainerConfig(image=image["image"], network_mode="none", mem_limit="4g",
                    cpu_quota=200000, working_dir="/workspace", timeout_seconds=300,
                    environment=helper.environment(image["python_prefix"])))
        return fixture_helper.FixtureManager(manager, profile) if profile else manager

    prepared = out / "adapter"
    prepared.mkdir()
    snapshot, setup, _ = helper.prepare_snapshot(task, image, prepared, factory,
                                                profile["identity"] if profile else None)
    bundle = Path(request["bundle"])
    adapters = discover_adapters(str(bundle), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    endpoint = urlsplit(request["endpoint"])
    if endpoint.hostname != "127.0.0.1":
        raise RuntimeError("Only the local native-verl bridge may serve rollouts")
    server = VllmServer(VllmConfig(model="gemma-rl", host="127.0.0.1", port=endpoint.port,
                                  tool_call_parser="gemma4", reasoning_parser="gemma4", max_model_len=32768),
                         adapter_manifest=adapters)
    models = server.create_model_registry(aliases=[helper.MODEL_NAME], model_prefix="openai/", api_key="local-only")
    for alias in models.list_available():
        model = models.get(alias)
        if not isinstance(model, LiteLlm):
            raise RuntimeError("Expected the official LiteLLM transport")
        model._additional_args["timeout"] = 330
        model._additional_args["num_retries"] = 0
    limits, generation = build_submission_limits()
    wheels = prepared / "empty_wheels"
    wheels.mkdir()
    config = EvalConfig(tasks_path=a.request, snapshots_dir=snapshot.parent, results_dir=out,
        submission_dir=bundle, models=models, sandbox="docker", image=image["image"], wheels_dir=wheels,
        graph_dir=str(prepared/"absent_graph"), embeddings_dir=str(prepared/"absent_embeddings"),
        max_time_minutes=5, max_tool_calls=60, max_turns=60, timeout_seconds=300,
        verification_timeout_seconds=300, limits=limits, generation_constraints=generation,
        adapter_manifest=adapters, display_mode="quiet",
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=1, overlap_size=2,
                                                        token_threshold=24576, event_retention_size=5))
    manager = helper.EnvironmentAdapter(factory(image), image)
    try:
        with helper.checked_setup_directory(setup):
            patch, error, trace = asyncio.run(run_agent_sandbox(manager, config,
                Task(**{k:task[k] for k in ("instance_id","repo","base_commit","problem_statement")}), snapshot))
        trace.save(out/"trace.json")
        (out/"agent.patch").write_text(patch or "")
        (out/"generation.json").write_text(json.dumps({"instance_id":task["instance_id"],"error":error,
                                                        "empty_patch":not bool((patch or "").strip())},indent=2))
    finally:
        manager.cleanup_all()


if __name__ == "__main__":
    import signal
    def stop(signum, frame):
        raise KeyboardInterrupt("Owned episode deadline; unwind official sandbox cleanup")
    signal.signal(signal.SIGTERM, stop)
    main()
