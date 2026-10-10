#!/usr/bin/env bash
# Aggregate status monitor for wave1/wave2/wave3 jobs.
# Reads provenance.json + tail of log per run directory.

set -eu -o pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")"/../../.. && pwd)}"
python3 "$REPO_ROOT/scripts/rebuttal/monitor.py" "$@"
