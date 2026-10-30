"""Official SWE tool-loop episodes connected to native verl token-in/token-out serving."""
import asyncio
import hashlib
import json
import os
import signal
from pathlib import Path
import time
import uuid

from aiohttp import web
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics


@ray.remote(num_cpus=0, max_concurrency=1024)
class EpisodeCapacity:
    def __init__(self, episodes, graders):
        self.slots = {"episode": asyncio.Semaphore(episodes), "grade": asyncio.Semaphore(graders)}
    async def acquire(self, kind):
        await self.slots[kind].acquire()
    def release(self, kind):
        self.slots[kind].release()


def make_renderer(model, tokenizer):
    from vllm.config import VllmConfig, ModelConfig, DeviceConfig
    from vllm.renderers import renderer_from_config
    from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender
    config = ModelConfig(model=model, tokenizer=tokenizer, max_model_len=32768, dtype="bfloat16")
    renderer = renderer_from_config(VllmConfig(model_config=config, device_config=DeviceConfig(device="cpu")))
    registry = OpenAIModelRegistry(config, [BaseModelPath(name="gemma-rl", model_path=model)])
    return OpenAIServingRender(config, renderer, None, registry, request_logger=None, chat_template=None,
        chat_template_content_format="auto", enable_auto_tools=True, tool_parser="gemma4", reasoning_parser="gemma4",
        default_chat_template_kwargs={"enable_thinking":True,"include_thoughts":True})


class GemmaSWEAgentLoop(AgentLoopBase):
    _renderer = None

    async def run(self, sampling_params, **kwargs):
        cfg = self.config.gemma_swe
        if cfg.judge_mode != "offline":
            raise ValueError("This pilot does not authorize online judging")
        root, project = Path(cfg.root), Path(cfg.project_root)
        admission = json.loads(Path(cfg.environment_admission).read_text())
        if admission["status"] != "qualified":
            raise RuntimeError("Training environments are not admitted")
        task = kwargs["extra_info"]
        task_id = task["instance_id"]
        # Native AgentLoopWorker consumes the trajectory argument before dispatch.
        # Membership, not a mutable caller flag, determines the data partition.
        validate = task_id not in admission["tasks"]
        partition = "dev" if validate else "train"
        metadata = json.loads((root/"dev-admission.json").read_text()) if validate else admission
        if task_id not in metadata["images"]:
            raise RuntimeError("Task is outside the admitted cohort")
        if not validate and task_id not in admission["tasks"]:
            raise RuntimeError("Heldout task cannot enter training")
        capacity = EpisodeCapacity.options(name="gemma-swe-capacity", get_if_exists=True).remote(
            cfg.max_active_episodes, cfg.max_active_graders)
        identifier = uuid.uuid4().hex
        out = root / "episodes" / partition / identifier
        out.mkdir(parents=True, exist_ok=False)
        token_turns, faults, budget_errors = [], [], []
        if GemmaSWEAgentLoop._renderer is None:
            GemmaSWEAgentLoop._renderer = make_renderer(self.config.actor_rollout_ref.model.path,
                                                       self.config.actor_rollout_ref.model.tokenizer_path)
        renderer = GemmaSWEAgentLoop._renderer
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser
        from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

        async def generate_chat(http):
            payload = await http.json()
            number = len(token_turns)
            (out/f"request-{number:03d}.json").write_text(json.dumps(payload))
            if payload.get("stream"):
                faults.append("Unexpected streaming request")
                return web.json_response({"error":"Streaming transport not qualified"},status=400)
            request = ChatCompletionRequest(**{**payload,"model":"gemma-rl"})
            rendered = await renderer.render_chat_request(request)
            if not hasattr(rendered,"token_ids"):
                error = str(rendered)
                if "token" in error.lower() and any(x in error.lower() for x in ("context", "maximum", "exceed")):
                    budget_errors.append(error)
                else:
                    faults.append(error)
                return web.json_response({"error":str(rendered)},status=400)
            ordinary = request.max_tokens == cfg.output_tokens
            if ordinary and (request.temperature,request.top_p,request.top_k,request.thinking_token_budget) != (1.0,0.95,64,4096):
                raise RuntimeError("Official policy sampling mismatch")
            if len(rendered.token_ids) + rendered.sampling_params.max_tokens > cfg.context_tokens:
                raise RuntimeError("Renderer violated total context reservation")
            params = dict(sampling_params)
            params.update(max_tokens=rendered.sampling_params.max_tokens,
                          thinking_token_budget=request.thinking_token_budget,
                          skip_special_tokens=False, logprobs=True,
                          stop_token_ids=rendered.sampling_params.stop_token_ids)
            for key in ("temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty", "frequency_penalty"):
                params[key] = getattr(rendered.sampling_params, key)
            response = await self.server_manager.generate(identifier,prompt_ids=rendered.token_ids,sampling_params=params)
            if response.stop_reason == "aborted":
                raise RuntimeError("Native generation aborted; infrastructure failure, not reward zero")
            if not response.token_ids or response.log_probs is None or len(response.token_ids)!=len(response.log_probs):
                raise RuntimeError("Native sampled token/logprob alignment missing")
            display_ids = list(response.token_ids)
            terminal = {self.tokenizer.eos_token_id, *(rendered.sampling_params.stop_token_ids or [])}
            while display_ids and display_ids[-1] in terminal:
                display_ids.pop()
            text = self.tokenizer.decode(display_ids,skip_special_tokens=False)
            reasoning, content = Gemma4ReasoningParser(renderer.renderer.tokenizer).extract_reasoning(text,request)
            parsed = Gemma4ToolParser(renderer.renderer.tokenizer).extract_tool_calls(content or "",request)
            message = {"role":"assistant","content":parsed.content,"reasoning_content":reasoning}
            if parsed.tools_called:
                message["tool_calls"]=[t.model_dump() for t in parsed.tool_calls]
            record = {"prompt_ids":rendered.token_ids,"response_ids":response.token_ids,
                      "log_probs":response.log_probs,"extra_fields":response.extra_fields,
                      "message":message,"ordinary":ordinary,"stop_reason":response.stop_reason}
            token_turns.append(record)
            (out/f"response-{number:03d}.json").write_text(json.dumps(record))
            return web.json_response({"id":"chatcmpl-"+uuid.uuid4().hex,"object":"chat.completion",
                "created":int(time.time()),"model":payload["model"],
                "choices":[{"index":0,"message":message,"finish_reason":
                    "tool_calls" if parsed.tools_called else
                    ("length" if len(response.token_ids) >= rendered.sampling_params.max_tokens else "stop")}],
                "usage":{"prompt_tokens":len(rendered.token_ids),"completion_tokens":len(response.token_ids),
                         "total_tokens":len(rendered.token_ids)+len(response.token_ids)}})

        async def chat(http):
            try:
                return await generate_chat(http)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                faults.append(repr(error))
                return web.json_response({"error":repr(error)},status=500)

        async def models(http):
            return web.json_response({"object":"list","data":[{"id":"gemma-rl","object":"model"}]})
        app = web.Application(client_max_size=16*1024**2)
        app.router.add_post("/v1/chat/completions",chat)
        app.router.add_get("/v1/models",models)
        runner = web.AppRunner(app,handler_cancellation=True)
        start = time.monotonic()
        env = {**os.environ,"DOCKER_HOST":"unix:///dev/shm/pritish-gemma4-20261025/eval-docker/docker.sock",
               "OPENAI_API_KEY":"local-only","PYTHON_DOTENV_DISABLED":"1","OTEL_SDK_DISABLED":"true",
               "LITELLM_TELEMETRY":"False","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
        await capacity.acquire.remote("episode")
        try:
            await runner.setup()
            site = web.TCPSite(runner,"127.0.0.1",0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            request = {"project_root":str(project),"output":str(out),"task":task,
                       "image":metadata["images"][task_id],"bundle":str(root/"bundle"),
                       "endpoint":f"http://127.0.0.1:{port}/v1","fixture_index":metadata.get("fixture_index")}
            (out/"episode-request.json").write_text(json.dumps(request))
            await self._subprocess([str(project/".runtime/venvs/scorer-v31/bin/python"),
                str(Path(__file__).with_name("rl_episode.py")),"--request",str(out/"episode-request.json")],out/"episode.log",env,420)
            if faults or not token_turns:
                raise RuntimeError(f"Invalid episode transport: {faults}; generated turns={len(token_turns)}")
            await capacity.acquire.remote("grade")
            try:
                if validate:
                    await self._grade_dev(project,root,out,task_id,metadata,env)
                else:
                    await self._subprocess([str(project/"downloads/.venv-grader/bin/python"),
                        str(Path(__file__).with_name("rl_environment.py")),"grade","--project",str(project),
                        "--root",str(root),"--row",str(root/"private"/f"{task_id}.json"),"--patch",str(out/"agent.patch"),
                        "--output",str(out/"grade"),"--image-id",metadata["images"][task_id]["image_id"]],out/"grade.log",env,900)
                grade = json.loads((out/"grade/reward.json").read_text())
            finally:
                await capacity.release.remote("grade")
            if type(grade["resolved"]) is not bool:
                raise RuntimeError("Executable reward missing")
            generation = json.loads((out/"generation.json").read_text())
            reward = float(grade["resolved"])
            outputs = []
            for turn in token_turns:
                if not turn["ordinary"]:
                    # Official auxiliary summaries remain logged, but are not policy actions.
                    continue
                extra = dict(turn["extra_fields"])
                extra["reward_extra_info"] = {"acc":reward,"empty_patch":float(generation["empty_patch"]),
                    "timeout":float("time" in str(generation.get("error","")).lower()),"turns":len(token_turns)}
                outputs.append(AgentLoopOutput(prompt_ids=turn["prompt_ids"],response_ids=turn["response_ids"],
                    response_mask=[1]*len(turn["response_ids"]),response_logprobs=turn["log_probs"],reward_score=reward,
                    num_turns=len(token_turns),metrics=AgentLoopMetrics(generate_sequences=time.monotonic()-start),extra_fields=extra))
            if not outputs:
                raise RuntimeError("No ordinary policy actions were generated")
            audit_hash = hashlib.sha256(f"{cfg.audit_seed}:{partition}:{task_id}:{kwargs['session_id']}".encode()).digest()
            (out/"completed.json").write_text(json.dumps({"task":task_id,"uid":kwargs["uid"],"session":kwargs["session_id"],
                "reward":reward,"turns":len(token_turns),"seconds":time.monotonic()-start,"judge":"offline",
                "context_errors":budget_errors,"offline_audit_selected":
                    int.from_bytes(audit_hash[:8],"big") / 2**64 < cfg.audit_fraction_per_outcome,
                "policy_versions":[t["extra_fields"] for t in token_turns]}))
            return outputs
        finally:
            await runner.cleanup()
            await capacity.release.remote("episode")

    async def _subprocess(self, command, log, env, seconds):
        with log.open("x") as stream:
            proc = await asyncio.create_subprocess_exec("timeout","--signal=TERM","--kill-after=30s",f"{seconds}s",
                                                       *command,env=env,stdout=stream,stderr=stream,start_new_session=True)
            try:
                code = await proc.wait()
            except asyncio.CancelledError:
                # The unreaped child remains the private session leader, preventing PID reuse.
                if proc.returncode is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(asyncio.shield(proc.wait()),35)
                    except asyncio.TimeoutError:
                        if proc.returncode is None:
                            os.killpg(proc.pid,signal.SIGKILL)
                        await proc.wait()
                raise
        if code:
            raise RuntimeError(f"Episode/grading process exit {code}; see {log}")

    async def _grade_dev(self, project, root, out, task_id, metadata, env):
        from verl.extensions.gemma.swe_dev import read_report
        patch = (out/"agent.patch").read_text()
        grade = out/"grade"
        grade.mkdir()
        if not patch.strip():
            result = {"resolved":False,"empty_patch":True}
        else:
            prediction = out/"predictions.jsonl"
            prediction.write_text(json.dumps({"instance_id":task_id,"model_name_or_path":"gemma-rl","model_patch":patch})+"\n")
            run_id = "grpo-"+out.name
            await self._subprocess([str(project/"downloads/.venv-grader/bin/python"),str(project/"scripts/run_swebench_grader_v4.py"),
                "--binding-json",str(root/"fixture-v4.json"),"--policy-authorization",str(project/"outputs/runtime/context-policy-admission.json"),
                "--predictions",str(prediction),"--task-id",task_id,"--attempt","1","--instance-image-tag",
                metadata["images"][task_id]["image"].rsplit(":",1)[1],"--run-id",run_id,"--private-docker-receipt",
                str(project/"outputs/transfer/hor1-image-plan/private-daemon-ready.json"),"--execute"],out/"grade.log",env,900)
            report = project/"outputs/swebench-grader"/run_id/"worker_result.json"
            result = {"resolved":read_report(report,task_id),"report":str(report)}
        (grade/"reward.json").write_text(json.dumps(result))
