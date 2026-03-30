#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_bootstrap_profile.sh <bootstrap_smoke|bootstrap_pilot|bootstrap_full>"
  exit 1
fi

PROFILE="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
export CURE_PROFILE="$PROFILE"
LOG_FILE="logs/${PROFILE}_$(date +%Y%m%d_%H%M%S).log"
export CURE_LOG_FILE="$LOG_FILE"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" run.py 2>&1 | tee "$LOG_FILE"
