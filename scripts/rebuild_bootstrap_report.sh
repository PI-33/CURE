#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/rebuild_bootstrap_report.sh experiments/<exp_name>"
  exit 1
fi

EXP_DIR="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" analysis/generate_report.py --exp_dir "$EXP_DIR" --output_dir "$EXP_DIR/report"
