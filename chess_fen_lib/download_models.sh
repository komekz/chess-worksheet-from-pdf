#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

URL="https://github.com/tsoj/Chess_diagram_to_FEN/releases/download/1.0/models.zip"
ZIP_FILE="models.zip"

echo "Downloading models..."
wget -O "$ZIP_FILE" "$URL"

echo "Extracting models..."
unzip -o "$ZIP_FILE"

rm "$ZIP_FILE"

echo "Done. Models extracted to models/"
