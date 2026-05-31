#!/usr/bin/env bash
# Local one-command start: scheduler in background, Streamlit in foreground.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p data logs

# Activate venv if present.
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

cleanup() {
  if [[ -n "${SCHED_PID:-}" ]] && kill -0 "$SCHED_PID" 2>/dev/null; then
    echo "Stopping scheduler (pid=$SCHED_PID)..."
    kill "$SCHED_PID" 2>/dev/null || true
    wait "$SCHED_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting scheduler..."
python -m src.scheduler.runner > logs/scheduler.log 2>&1 &
SCHED_PID=$!
echo "Scheduler pid=$SCHED_PID (logs: logs/scheduler.log)"

echo "Starting Streamlit on :8501..."
exec streamlit run app.py --server.port 8501
