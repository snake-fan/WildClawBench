from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class MementoRunResult:
    task_id: str
    output_dir: Path
    workspace_dir: Path
    transcript_path: Path
    raw_events_path: Path
    result: dict[str, Any]
    usage: dict[str, Any]


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _patched_environ(updates: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _copy_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def _safe_model_label(model: str | None) -> str:
    return re.sub(r"[^a-zA-Z0-9.\-_]", "_", model or "configured-profile")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MementoSAgent:
    """Local no-Docker WildClawBench harness for Memento-S.

    The official WildClawBench runners create a fresh container and expose the
    task files at /tmp_workspace. This runner mirrors that layout in a local
    directory and rewrites task prompts plus grading code to point at the local
    path.
    """

    def __init__(
        self,
        *,
        bench_root: Path,
        project_root: Path,
        output_root: Path,
        runs_root: Path,
        home_dir: Path,
        model: str | None = None,
        host_llm_config: Path | None = None,
        sync_host_llm_config: bool = True,
    ) -> None:
        self.bench_root = bench_root.resolve()
        self.project_root = project_root.resolve()
        self.output_root = output_root.resolve()
        self.runs_root = runs_root.resolve()
        self.home_dir = home_dir.resolve()
        self.model = model
        self.host_llm_config = host_llm_config.resolve() if host_llm_config else None
        self.sync_host_llm_config = sync_host_llm_config
        self._agent: Any | None = None
        self._warmup_procs: list[tuple[subprocess.Popen[str], str]] = []

    async def ensure_ready(self) -> None:
        if self._agent is not None:
            return

        self.home_dir.mkdir(parents=True, exist_ok=True)
        self._sync_host_llm_config()
        os.environ["HOME"] = str(self.home_dir)
        os.environ.setdefault("MEMENTOS_BENCHMARK_LOCAL", "1")

        self._install_model_override_profile()

        from experimental.benchmark.praxis.agent import build_agent, ensure_bootstrapped

        await ensure_bootstrapped()
        # Bootstrap may discover remote profiles and reconcile active_profile.
        # Re-apply the explicit benchmark override immediately before the
        # MementoSAgent constructs its LLMClient.
        self._install_model_override_profile()
        self._agent = await build_agent()

    def _sync_host_llm_config(self) -> None:
        if not self.sync_host_llm_config or self.host_llm_config is None:
            return
        if not self.host_llm_config.is_file():
            logger.warning("Host Memento LLM config not found: %s", self.host_llm_config)
            return

        target = self.home_dir / "memento_s" / "llm.json"
        if self.host_llm_config == target.resolve():
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.host_llm_config, target)
        logger.info("Synced host Memento LLM config into benchmark HOME: %s", target)

    def _install_model_override_profile(self) -> None:
        if not self.model:
            return

        from middleware.config.llm_config_manager import g_llm_config_manager
        from middleware.llm.llm_client import invalidate_llm_profile_cache

        try:
            g_llm_config_manager.load()
        except Exception:
            logger.exception("Failed to load Memento LLM config before benchmark model override")
            raise

        base_profile = g_llm_config_manager.get_current_profile() or {}
        profile_name = "wildclawbench-model-override"
        max_tokens = os.environ.get("MEMENTOS_MAX_TOKENS", base_profile.get("max_tokens") or 8192)
        temperature = os.environ.get("MEMENTOS_TEMPERATURE", base_profile.get("temperature") or 0)
        timeout = os.environ.get("MEMENTOS_LLM_TIMEOUT", base_profile.get("timeout") or 600)
        profile = {
            **base_profile,
            "model": self.model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "timeout": int(timeout),
        }

        # Keep explicit benchmark overrides in memory only. Credentials and
        # routing still come from the active llm.json profile.
        g_llm_config_manager._backend_profiles = {profile_name: profile}  # noqa: SLF001
        g_llm_config_manager._active_profile = profile_name  # noqa: SLF001
        if g_llm_config_manager._data is not None:  # noqa: SLF001
            g_llm_config_manager._data["active_profile"] = profile_name  # noqa: SLF001
        invalidate_llm_profile_cache()

    async def run_task(self, task: dict[str, Any]) -> MementoRunResult:
        await self.ensure_ready()
        assert self._agent is not None

        task_id_ori = task["task_id"]
        model_label = _safe_model_label(self.model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_id = uuid.uuid4().hex[:6]
        match = re.match(r"(\d+)_.*?(task_\d+)", task_id_ori)
        short_task_id = f"{match.group(1)}_{match.group(2)}" if match else task_id_ori
        suffix = f"{model_label}_{timestamp}_{run_id}"
        task_id = f"{short_task_id}_{suffix}"

        output_dir = self.output_root / task["category"] / task_id_ori / suffix
        run_root = self.runs_root / task["category"] / task_id_ori / suffix
        workspace_dir = run_root / "tmp_workspace"
        output_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        status_path = output_dir / "execution_status.json"
        raw_events_path = output_dir / "mementos_events.jsonl"
        transcript_path = output_dir / "chat.jsonl"
        raw_events: list[dict[str, Any]] = []
        transcript: list[dict[str, Any]] = []
        assistant_text: list[str] = []
        usage = self._empty_usage()
        error: str | None = None

        self._write_status(
            status_path,
            task_id=task_id,
            status="preparing_workspace",
            model=self.model,
            started_at=_now_iso(),
            timeout_seconds=task["timeout_seconds"],
        )

        self._prepare_workspace(task, workspace_dir)
        warmup_errors = self._run_warmup(task, workspace_dir, output_dir)
        if warmup_errors:
            error = "Warmup failed: " + " | ".join(warmup_errors)
            logger.error("[%s] %s", task_id, error)
            self._stop_warmup_processes()
            self._write_jsonl(raw_events_path, raw_events)
            self._write_jsonl(transcript_path, transcript)
            self._write_status(
                status_path,
                status="error",
                error=error,
                elapsed_time=0.0,
                transcript_path=str(transcript_path),
                workspace_dir=str(workspace_dir),
            )
            (output_dir / "usage.json").write_text(
                json.dumps(usage, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return MementoRunResult(
                task_id=task_id,
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                transcript_path=transcript_path,
                raw_events_path=raw_events_path,
                result={"task_id": task_id, "scores": {}, "error": error},
                usage=usage,
            )

        prompt = self._build_prompt(task, workspace_dir)
        started = time.perf_counter()

        self._write_status(status_path, status="mementos_running", updated_at=_now_iso())

        from shared.chat import ChatManager

        created = await ChatManager.create_session(
            title=f"wildclawbench {task_id_ori}",
            metadata={"channel": "wildclawbench-local", "task_id": task_id_ori},
            source="cli",
        )
        session_id = created.id

        task_env = {
            "TMP_WORKSPACE": str(workspace_dir),
            "WILDCLAWBENCH_TMP_WORKSPACE": str(workspace_dir),
            "HOME": str(self.home_dir),
        }

        try:
            with _patched_environ(task_env), _pushd(workspace_dir):
                async with asyncio.timeout(int(task["timeout_seconds"])):
                    async for event in self._agent.reply_stream(
                        session_id=session_id,
                        user_content=prompt,
                    ):
                        if isinstance(event, dict):
                            raw_events.append(event)
                            self._consume_event(event, transcript, assistant_text, usage)
        except asyncio.TimeoutError:
            error = "Memento-S run timed out"
            try:
                self._agent.cancel(session_id)
            except Exception:
                pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] Memento-S execution failed", task_id)
        finally:
            self._stop_warmup_processes()

        elapsed = time.perf_counter() - started
        usage["elapsed_time"] = round(elapsed if error != "Memento-S run timed out" else task["timeout_seconds"], 2)

        if assistant_text:
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "".join(assistant_text).strip()}],
                    },
                }
            )

        self._write_jsonl(raw_events_path, raw_events)
        self._write_jsonl(transcript_path, transcript)

        self._write_status(
            status_path,
            status="timed_out" if error == "Memento-S run timed out" else ("error" if error else "finished"),
            error=error,
            elapsed_time=usage["elapsed_time"],
            transcript_path=str(transcript_path),
            workspace_dir=str(workspace_dir),
        )

        result = {"task_id": task_id, "scores": {}, "error": error}
        (output_dir / "usage.json").write_text(
            json.dumps(usage, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return MementoRunResult(
            task_id=task_id,
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            transcript_path=transcript_path,
            raw_events_path=raw_events_path,
            result=result,
            usage=usage,
        )

    def _prepare_workspace(self, task: dict[str, Any], workspace_dir: Path) -> None:
        task_workspace = Path(task["workspace_path"])
        exec_dir = task_workspace / "exec"
        tmp_dir = task_workspace / "tmp"

        _copy_contents(exec_dir, workspace_dir)
        if tmp_dir.exists():
            _copy_contents(tmp_dir, workspace_dir / "tmp")
        (workspace_dir / "results").mkdir(exist_ok=True)

    def _run_warmup(self, task: dict[str, Any], workspace_dir: Path, output_dir: Path) -> list[str]:
        warmup = (task.get("warmup") or "").strip()
        if not warmup:
            return []

        env = os.environ.copy()
        env["TMP_WORKSPACE"] = str(workspace_dir)
        env["WILDCLAWBENCH_TMP_WORKSPACE"] = str(workspace_dir)
        env["HOME"] = str(self.home_dir)
        log_path = output_dir / "warmup.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        commands = [
            line.strip().replace("/tmp_workspace", str(workspace_dir))
            for line in warmup.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        with log_path.open("a", encoding="utf-8") as log_file:
            for idx, cmd in enumerate(commands, start=1):
                log_file.write(f"$ {cmd}\n")
                log_file.flush()
                if cmd.endswith("&"):
                    try:
                        proc = subprocess.Popen(
                            ["/bin/bash", "-lc", cmd[:-1].strip()],
                            cwd=workspace_dir,
                            env=env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    except Exception as exc:
                        message = f"background command {idx} failed to start: {cmd} ({type(exc).__name__}: {exc})"
                        log_file.write(f"[warmup-error] {message}\n")
                        errors.append(message)
                        break
                    self._warmup_procs.append((proc, cmd))
                    continue
                try:
                    completed = subprocess.run(
                        ["/bin/bash", "-lc", cmd],
                        cwd=workspace_dir,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=300,
                    )
                except subprocess.TimeoutExpired:
                    message = f"command {idx} timed out after 300s: {cmd}"
                    log_file.write(f"[warmup-error] {message}\n")
                    errors.append(message)
                    break
                except Exception as exc:
                    message = f"command {idx} failed to start: {cmd} ({type(exc).__name__}: {exc})"
                    log_file.write(f"[warmup-error] {message}\n")
                    errors.append(message)
                    break
                if completed.returncode != 0:
                    message = f"command {idx} exited {completed.returncode}: {cmd}"
                    log_file.write(f"[warmup-error] {message}\n")
                    errors.append(message)
                    break
                background_errors = self._warmup_background_errors()
                if background_errors:
                    for message in background_errors:
                        log_file.write(f"[warmup-error] {message}\n")
                    errors.extend(background_errors)
                    break
        return errors

    def _stop_warmup_processes(self) -> None:
        for proc, _cmd in self._warmup_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._warmup_procs.clear()

    def _warmup_background_errors(self) -> list[str]:
        errors: list[str] = []
        for proc, cmd in self._warmup_procs:
            returncode = proc.poll()
            if returncode is not None:
                errors.append(f"background command exited early with code {returncode}: {cmd}")
        return errors

    def _build_prompt(self, task: dict[str, Any], workspace_dir: Path) -> str:
        task_prompt = (task.get("prompt") or "").replace("/tmp_workspace", str(workspace_dir))
        timeout_seconds = int(task.get("timeout_seconds", 0) or 0)
        parts = [
            "You are running WildClawBench locally without Docker.",
            f"The benchmark /tmp_workspace path for this task is: {workspace_dir}",
            "Use that local path literally for every file the task asks you to read or write.",
            f"Your HOME for this benchmark run is isolated at: {self.home_dir}",
            "Do not write to the real user home directory.",
            f"Solve the task efficiently before the timeout ({timeout_seconds}s).",
        ]

        skill_docs = self._load_skill_documents(task, workspace_dir)
        if skill_docs:
            parts.append("## Local skill references")
            parts.extend(skill_docs)

        parts.append("## Task")
        parts.append(task_prompt)
        return "\n\n".join(parts).strip() + "\n"

    def _load_skill_documents(self, task: dict[str, Any], workspace_dir: Path) -> list[str]:
        skills_text = task.get("skills") or ""
        skills_path = Path(task.get("skills_path") or self.bench_root / "skills")
        docs: list[str] = []
        for raw in skills_text.splitlines():
            rel = raw.strip().strip("/")
            if not rel:
                continue
            skill_dir = skills_path / rel
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                logger.warning("Skill file not found: %s", skill_file)
                continue
            content = skill_file.read_text(encoding="utf-8", errors="ignore")
            content = content.replace("{baseDir}", str(skill_dir))
            content = content.replace("/tmp_workspace", str(workspace_dir))
            docs.append(f"### Skill: {rel}\n\n{content.strip()}")
        return docs

    def _consume_event(
        self,
        event: dict[str, Any],
        transcript: list[dict[str, Any]],
        assistant_text: list[str],
        usage: dict[str, Any],
    ) -> None:
        event_type = str(event.get("type") or event.get("event_type") or "")

        if event_type in {"TEXT_MESSAGE_CONTENT", "text_message_content"}:
            delta = event.get("delta") or event.get("content") or event.get("payload", {}).get("delta")
            if isinstance(delta, str):
                assistant_text.append(delta)
            return

        if event_type in {"TEXT_MESSAGE_END", "text_message_end"}:
            content = event.get("content") or event.get("payload", {}).get("content")
            if isinstance(content, str) and not assistant_text:
                assistant_text.append(content)
            return

        if event_type in {"TOOL_CALL_START", "tool_call_start"}:
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
            name = payload.get("toolName") or payload.get("tool_name") or payload.get("name") or ""
            args = payload.get("arguments") or payload.get("args") or {}
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "name": str(name),
                                "arguments": args,
                            }
                        ],
                    },
                }
            )
            return

        if event_type in {"TOOL_CALL_RESULT", "tool_call_result"}:
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
            result = payload.get("result") or payload.get("output") or ""
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "toolResult",
                                "content": str(result)[:4000],
                            }
                        ],
                    },
                }
            )
            return

        if event_type in {"RUN_FINISHED", "run_finished"}:
            raw_usage = event.get("usage") or event.get("payload", {}).get("usage")
            self._merge_usage(usage, raw_usage)

    @staticmethod
    def _empty_usage() -> dict[str, Any]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
            "elapsed_time": 0.0,
        }

    @staticmethod
    def _merge_usage(target: dict[str, Any], raw_usage: Any) -> None:
        if not isinstance(raw_usage, dict):
            return
        input_tokens = int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)) or 0)
        output_tokens = int(raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)) or 0)
        cache_read = int(raw_usage.get("cache_read_tokens", raw_usage.get("cached_tokens", 0)) or 0)
        total = int(raw_usage.get("total_tokens", input_tokens + output_tokens + cache_read) or 0)
        target["input_tokens"] += input_tokens
        target["output_tokens"] += output_tokens
        target["cache_read_tokens"] += cache_read
        target["total_tokens"] += total
        target["request_count"] += 1 if total or input_tokens or output_tokens else 0

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _write_status(path: Path, **updates: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        current: dict[str, Any] = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current.update(updates)
        current["updated_at"] = _now_iso()
        path.write_text(json.dumps(current, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
