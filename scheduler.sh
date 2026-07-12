#!/usr/bin/env bash
# Headless always-on scheduler (no Streamlit UI).
#
# Runs the standalone APScheduler process that owns all agent ticking, using
# the REAL sibling MCP servers (short-term-trader / stock-recommender) when
# their paths are configured in .streamlit/secrets.toml or .env. If a sibling
# server can't start, that leg transparently falls back to the built-in
# yfinance provider (see src/signals/local.py).
#
# The process is restarted automatically if it exits. Stop with Ctrl-C.
#
# Usage:
#   ./scheduler.sh              # run in the foreground with auto-restart
#   nohup ./scheduler.sh &      # detach; logs to logs/scheduler.log
#
# View it live:  tail -f logs/scheduler.log
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data logs

# Activate venv if present.
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# A stale lock from a hard-killed previous run would block startup; the runner
# itself checks whether the recorded pid is still alive, so we leave it be.
echo "Starting always-on scheduler (Ctrl-C to stop)..."
echo "Logs: logs/scheduler.log"

trap 'echo; echo "Stopping scheduler..."; exit 0' INT TERM

while true; do
  python -m src.scheduler.runner >> logs/scheduler.log 2>&1 || true
  code=$?
  echo "$(date -u +%FT%TZ) scheduler exited (code=$code); restarting in 5s" \
    | tee -a logs/scheduler.log
  sleep 5
done
