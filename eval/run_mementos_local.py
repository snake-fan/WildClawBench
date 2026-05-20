from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

BENCH_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCH_ROOT.parent

for path in (str(PROJECT_ROOT), str(BENCH_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.agents.mementos import MementoSAgent
from src.utils.grading import print_global_summary, print_summary
from src.utils.task_parser import parse_task_md


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


ALL_CATEGORIES = [
    "01_Productivity_Flow",
    "02_Code_Intelligence",
    "03_Social_Interaction",
    "04_Search_Retrieval",
    "05_Creative_Synthesis",
    "06_Safety_Alignment",
]


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_env_files() -> None:
    for env_path in [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.local",
        BENCH_ROOT / ".env",
        BENCH_ROOT / ".env.local",
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WildClawBench tasks with local Memento-S, without Docker.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", "-t", help="Path to a single task.md file")
    mode.add_argument(
        "--category",
        "-c",
        help="Category name, or all",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Optional explicit Memento LLM model override. If omitted, uses the active Memento LLM profile.",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="Accepted for CLI compatibility; local Memento runs are executed sequentially.",
    )
    parser.add_argument(
        "--output-root",
        default=str(BENCH_ROOT / os.environ.get("OUTPUT_SUBDIR", "output") / "mementos-local"),
        help="Directory for reports and per-task artifacts.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(BENCH_ROOT / ".local_runs" / "mementos"),
        help="Directory for local task workspaces.",
    )
    parser.add_argument(
        "--memento-home",
        default=str(BENCH_ROOT / ".local_runs" / "mementos_home"),
        help="Isolated HOME used by Memento-S during benchmark execution.",
    )
    parser.add_argument(
        "--sync-host-llm-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mirror ~/memento_s/llm.json into the isolated benchmark HOME before "
            "bootstrapping Memento-S, so the normal profile loader is used."
        ),
    )
    parser.add_argument(
        "--grade-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run grading even if the agent timed out or raised an error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse and list selected tasks; do not start Memento-S.",
    )
    return parser


def load_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    if args.task:
        task_file = Path(args.task)
        if not task_file.is_absolute():
            candidates = [
                task_file,
                BENCH_ROOT / task_file,
                PROJECT_ROOT / task_file,
            ]
            task_file = next((candidate.resolve() for candidate in candidates if candidate.exists()), task_file.resolve())
        return [parse_task_md(task_file)], [task_file.parent.name]

    categories = ALL_CATEGORIES if args.category.lower() == "all" else [args.category]
    tasks: list[dict[str, Any]] = []
    loaded_categories: list[str] = []
    for category in categories:
        category_dir = BENCH_ROOT / "tasks" / category
        if not category_dir.exists():
            logger.error("Category directory not found: %s", category_dir)
            continue
        category_tasks = []
        for task_file in sorted(category_dir.glob("*task_*.md")):
            try:
                category_tasks.append(parse_task_md(task_file))
            except Exception as exc:
                logger.error("Failed to parse %s: %s", task_file, exc)
        tasks.extend(category_tasks)
        if category_tasks:
            loaded_categories.append(category)
    return tasks, loaded_categories


def copy_gt_for_grading(task: dict[str, Any], workspace_dir: Path) -> None:
    gt_src = Path(task["workspace_path"]) / "gt"
    gt_dst = workspace_dir / "gt"
    if gt_dst.exists():
        shutil.rmtree(gt_dst)
    if gt_src.is_dir():
        shutil.copytree(gt_src, gt_dst)


def load_transcript(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def rewrite_grading_code(
    code: str,
    *,
    workspace_dir: Path,
    transcript_path: Path,
    home_dir: Path,
) -> str:
    replacements = {
        "/tmp_workspace": workspace_dir.as_posix(),
        "/root/.openclaw/agents/main/sessions/chat.jsonl": transcript_path.as_posix(),
        "/claude_code/log/chat.json": transcript_path.as_posix(),
        "/root/skills": (home_dir / "skills").as_posix(),
    }
    rewritten = code
    for old, new in replacements.items():
        rewritten = rewritten.replace(old, new)
    return rewritten


def write_score(output_dir: Path, scores: dict[str, Any]) -> None:
    score_path = output_dir / "score.json"
    score_path.write_text(
        json.dumps(scores, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def error_score(output_dir: Path, message: str) -> dict[str, Any]:
    scores = {"overall_score": 0.0, "error": message}
    write_score(output_dir, scores)
    return scores


def empty_usage() -> dict[str, Any]:
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


def record_task_failure(
    task: dict[str, Any],
    *,
    output_root: Path,
    runs_root: Path,
    model: str | None,
    message: str,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"{_safe_summary_label(model)}_{timestamp}_failed"
    output_dir = output_root / task["category"] / task["task_id"] / suffix
    workspace_dir = runs_root / task["category"] / task["task_id"] / suffix / "tmp_workspace"
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    usage = empty_usage()
    (output_dir / "usage.json").write_text(
        json.dumps(usage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "mementos_events.jsonl").write_text("", encoding="utf-8")
    (output_dir / "chat.jsonl").write_text("", encoding="utf-8")
    (output_dir / "execution_status.json").write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "status": "error",
                "error": message,
                "workspace_dir": str(workspace_dir),
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scores = error_score(output_dir, message)
    return {
        "task_id": f"{task['task_id']}_{suffix}",
        "scores": scores,
        "error": message,
        "usage": usage,
    }


def run_local_grading(
    task: dict[str, Any],
    *,
    workspace_dir: Path,
    transcript_path: Path,
    output_dir: Path,
    home_dir: Path,
) -> dict[str, Any]:
    automated_checks = (task.get("automated_checks") or "").strip()
    if not automated_checks:
        logger.info("[%s] No automated checks, skipping grading", task["task_id"])
        return {}

    code = rewrite_grading_code(
        automated_checks,
        workspace_dir=workspace_dir,
        transcript_path=transcript_path,
        home_dir=home_dir,
    )
    namespace: dict[str, Any] = {"__name__": "__wildclawbench_local_grade__"}
    try:
        exec(code, namespace)
        grade_fn = namespace.get("grade")
        if not callable(grade_fn):
            return error_score(output_dir, "automated checks did not define grade()")
        transcript = load_transcript(transcript_path)
        with pushd(workspace_dir):
            scores = grade_fn(transcript=transcript, workspace_path=str(workspace_dir))
        if not isinstance(scores, dict):
            scores = {"overall_score": 0.0, "error": f"grade() returned {type(scores).__name__}"}
        write_score(output_dir, scores)
        return scores
    except Exception as exc:
        logger.error("[%s] Local grading failed: %s", task["task_id"], exc)
        return error_score(output_dir, traceback.format_exc())


def collect_local_output(workspace_dir: Path, output_dir: Path) -> None:
    task_output = output_dir / "task_output"
    workspace_out = task_output / "workspace"
    if workspace_out.exists():
        shutil.rmtree(workspace_out)
    workspace_out.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(
        "gt",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".cache",
    )
    shutil.copytree(workspace_dir, workspace_out, ignore=ignore)


async def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.parallel != 1:
        logger.warning("Local Memento runner is sequential; ignoring --parallel=%s", args.parallel)

    tasks, categories = load_tasks(args)
    if not tasks:
        return []

    if args.dry_run:
        print(f"Selected {len(tasks)} task(s):")
        for task in tasks:
            print(f"- {task['task_id']}  workspace={task['workspace_path']}")
        return []

    output_root = Path(args.output_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    host_home = Path.home().resolve()
    home_dir = Path(args.memento_home).expanduser().resolve()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    agent = MementoSAgent(
        bench_root=BENCH_ROOT,
        project_root=PROJECT_ROOT,
        output_root=output_root,
        runs_root=runs_root,
        home_dir=home_dir,
        model=args.model,
        host_llm_config=host_home / "memento_s" / "llm.json",
        sync_host_llm_config=args.sync_host_llm_config,
    )

    grouped_results: dict[str, list[dict[str, Any]]] = {category: [] for category in categories}
    all_results: list[dict[str, Any]] = []

    for task in tasks:
        logger.info("Running task: %s", task["task_id"])
        try:
            run = await agent.run_task(task)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] Task failed before result creation", task["task_id"])
            result = record_task_failure(
                task,
                output_root=output_root,
                runs_root=runs_root,
                model=args.model,
                message=message,
            )
            all_results.append(result)
            grouped_results.setdefault(task["category"], []).append(result)
            continue

        result = run.result
        result["usage"] = run.usage

        should_grade = bool(task.get("automated_checks")) and (
            not result.get("error") or args.grade_on_error
        )
        if should_grade:
            copy_gt_for_grading(task, run.workspace_dir)
            result["scores"] = run_local_grading(
                task,
                workspace_dir=run.workspace_dir,
                transcript_path=run.transcript_path,
                output_dir=run.output_dir,
                home_dir=home_dir,
            )
        elif result.get("error"):
            result["scores"] = error_score(run.output_dir, result["error"])

        collect_local_output(run.workspace_dir, run.output_dir)
        all_results.append(result)
        grouped_results.setdefault(task["category"], []).append(result)

    summary_label = _safe_summary_label(args.model)
    for category, results in grouped_results.items():
        if results:
            print_summary(results, category, output_root, summary_label, run_timestamp)
    if len([c for c, results in grouped_results.items() if results]) > 1:
        print_global_summary(all_results, output_root, summary_label, run_timestamp)

    return all_results


def _safe_summary_label(model: str | None) -> str:
    return re.sub(r"[^a-zA-Z0-9.\-_]", "_", model or "configured-profile")


def main() -> None:
    load_env_files()
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
