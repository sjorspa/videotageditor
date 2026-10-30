#!/usr/bin/env bash
# Start the video editor; creates the virtualenv on first run.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$DIR/.venv/bin/python" ]; then
    python3 -m venv "$DIR/.venv"
    "$DIR/.venv/bin/pip" install -r "$DIR/requirements.txt"
fi
exec "$DIR/.venv/bin/python" "$DIR/videoedit.py" "$@"
