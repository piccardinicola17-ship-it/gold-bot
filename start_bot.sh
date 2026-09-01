#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BOT_DIR="${BOT_DIR:-/data}"
export DB_PATH="${DB_PATH:-$BOT_DIR/goldbot.db}"
export DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
export DASHBOARD_PORT="${PORT:-5050}"

mkdir -p "$BOT_DIR"

PYTHONPATH="$PROJECT_DIR" python3 -c "from trade_manager import init_db; init_db()"

python3 "$PROJECT_DIR/dashboard.py" &
DASHBOARD_PID=$!

cleanup() {
  kill "$DASHBOARD_PID" 2>/dev/null || true
  if [[ -n "${BOT_PID:-}" ]]; then
    kill "$BOT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python3 "$PROJECT_DIR/gold_bot.py" &
BOT_PID=$!

set +e
wait -n "$DASHBOARD_PID" "$BOT_PID"
EXIT_CODE=$?
set -e
cleanup
wait "$DASHBOARD_PID" "$BOT_PID" 2>/dev/null || true
exit "$EXIT_CODE"
