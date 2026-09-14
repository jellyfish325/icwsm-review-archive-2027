#!/bin/sh
set -eu

ARCHIVE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ARCHIVE_DIR"

if command -v uv >/dev/null 2>&1; then
    UV_COMMAND=$(command -v uv)
elif [ -x "${HOME}/.local/bin/uv" ]; then
    UV_COMMAND="${HOME}/.local/bin/uv"
else
    echo "FAIL: uv is required. Install it from https://docs.astral.sh/uv/ and run this command again." >&2
    exit 1
fi

echo "[1/2] Verifying archive integrity..."
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c MANIFEST.sha256
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c MANIFEST.sha256
else
    echo "FAIL: no SHA-256 checker was found (expected sha256sum or shasum)." >&2
    exit 1
fi

echo "[2/2] Reproducing and checking the registered results..."
echo "This normally takes 2-10 minutes. Progress appears below."
"$UV_COMMAND" run --python 3.13.5 --with-requirements requirements.txt reproduce.py
