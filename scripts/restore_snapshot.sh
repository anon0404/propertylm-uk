#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/uk_properties.snapshot"
  exit 1
fi

SNAPSHOT="$1"
COLLECTION="${QDRANT_COLLECTION:-uk_properties}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

if [ ! -f "$SNAPSHOT" ]; then
  echo "Snapshot not found: $SNAPSHOT"
  exit 1
fi

echo "Restoring $SNAPSHOT into collection $COLLECTION..."

curl --fail --show-error \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@${SNAPSHOT}" \
  "${QDRANT_URL}/collections/${COLLECTION}/snapshots/upload?priority=snapshot"

echo
echo "Restore requested."
