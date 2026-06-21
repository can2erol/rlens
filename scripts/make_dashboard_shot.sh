#!/usr/bin/env bash
# Capture a screenshot of the live dashboard for the README (macOS + Google Chrome).
#
#   scripts/make_dashboard_shot.sh [runs_dir] [out.png]
#
# Serves the dashboard against a runs dir, then uses headless Chrome to screenshot it.
# Defaults to runs_lander (PPO + DQN on LunarLander-v3) for a clean multi-run overlay.
set -euo pipefail

RUNS="${1:-runs_lander}"
OUT="${2:-docs/dashboard.png}"
PORT=8099
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

rlens dashboard --runs-dir "$RUNS" --port "$PORT" >/tmp/rlens_dash_shot.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
sleep 5

"$CHROME" --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=1440,1000 --virtual-time-budget=15000 \
  --screenshot="$OUT" "http://localhost:$PORT/"

echo "wrote $OUT"
