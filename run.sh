#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build.py
python3 -m http.server "${PORT:-8647}" --bind 127.0.0.1 --directory webapp
