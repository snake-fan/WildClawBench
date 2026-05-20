#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_root="$(cd "$script_dir/.." && pwd)"
cd "$bench_root"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash script/run.sh openclaw    [run_batch args...]
  bash script/run.sh claudecode  [run_batch args...]
  bash script/run.sh codex       [run_batch args...]
  bash script/run.sh hermesagent [run_batch args...]
  bash script/run.sh mementos    [local no-Docker args...]

Examples:
  bash script/run.sh openclaw --category all --parallel 4 --model openrouter/openai/gpt-5.5
  bash script/run.sh claudecode --category all --parallel 4 --model openai/gpt-5.5
  bash script/run.sh codex --category all --parallel 4 --model openrouter/openai/gpt-5.5
  bash script/run.sh hermesagent --category all --parallel 4 --model openai/gpt-5.5
  bash script/run.sh mementos --category 06_Safety_Alignment

  bash script/run.sh openclaw --task tasks/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite.md --model openrouter/openai/gpt-5.5
  bash script/run.sh mementos --task tasks/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite.md
EOF
  exit 1
fi

backend="$1"
shift || true

case "$backend" in
  openclaw)
    exec python3 eval/run_batch.py --agent-backend openclaw "$@"
    ;;
  claudecode)
    exec python3 eval/run_batch.py --agent-backend claudecode "$@"
    ;;
  codex)
    exec python3 eval/run_batch.py --agent-backend codex "$@"
    ;;
  hermesagent)
    exec python3 eval/run_batch.py --agent-backend hermesagent "$@"
    ;;
  mementos|mementos-local)
    exec python3 eval/run_mementos_local.py "$@"
    ;;
  *)
    echo "Unknown backend: $backend"
    echo "Expected one of: openclaw, claudecode, codex, hermesagent, mementos"
    exit 1
    ;;
esac
