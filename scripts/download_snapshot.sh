#!/usr/bin/env bash
set -euo pipefail

RELEASE="${1:-v1}"
HF_REPO="${HF_SNAPSHOT_REPO:-EmbeddingsOG/propertylm-uk-qdrant}"
OUTPUT_DIR="${2:-artifacts}"

mkdir -p "$OUTPUT_DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required to download the private snapshot."
  echo "Install it with: brew install hf"
  exit 1
fi

echo "Downloading PropertyLM snapshot release: $RELEASE"

hf download \
  "$HF_REPO" \
  --repo-type dataset \
  --include "${RELEASE}/*" \
  --local-dir "$OUTPUT_DIR"

SNAPSHOT="$OUTPUT_DIR/$RELEASE/uk_properties-$RELEASE.snapshot"
CHECKSUM="$SNAPSHOT.sha256"

if [ ! -f "$SNAPSHOT" ]; then
  echo "Snapshot was not downloaded: $SNAPSHOT"
  exit 1
fi

if [ -f "$CHECKSUM" ]; then
  (
    cd "$(dirname "$SNAPSHOT")"
    shasum -a 256 -c "$(basename "$CHECKSUM")"
  )
fi

echo
echo "Snapshot downloaded:"
echo "$SNAPSHOT"
