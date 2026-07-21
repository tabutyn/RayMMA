#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Thomas Butyn
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/build/scenes/coastal-cliff-01}"
BASE="https://dl.polyhaven.org/file/ph-assets/Models"

if [[ -e "$DEST" ]]; then
    printf 'Destination already exists: %s\n' "$DEST" >&2
    exit 1
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/raymma-open-model.XXXXXX")"

cleanup() {
    if [[ -n "${STAGE:-}" && -d "$STAGE" ]]; then
        rm -rf -- "$STAGE"
    fi
}
trap cleanup EXIT

mkdir -p "$STAGE/textures"

fetch() {
    local url="$1"
    local output="$2"
    curl --fail --location --silent --show-error "$url" --output "$output"
}

fetch "$BASE/gltf/1k/coastal_cliff_01/coastal_cliff_01_1k.gltf" \
    "$STAGE/coastal_cliff_01_1k.gltf"
fetch "$BASE/gltf/8k/coastal_cliff_01/coastal_cliff_01.bin" \
    "$STAGE/coastal_cliff_01.bin"
fetch "$BASE/jpg/1k/coastal_cliff_01/coastal_cliff_01_diff_1k.jpg" \
    "$STAGE/textures/coastal_cliff_01_diff_1k.jpg"
fetch "$BASE/jpg/1k/coastal_cliff_01/coastal_cliff_01_arm_1k.jpg" \
    "$STAGE/textures/coastal_cliff_01_arm_1k.jpg"
fetch "$BASE/jpg/1k/coastal_cliff_01/coastal_cliff_01_nor_gl_1k.jpg" \
    "$STAGE/textures/coastal_cliff_01_nor_gl_1k.jpg"

(
    cd "$STAGE"
    sha256sum --check <<'EOF'
2e6bb1f77965b314208b309ac89b94b8148c6a08ee6c9808cd71455e5475602d  coastal_cliff_01_1k.gltf
6833dfaf75e0d039e64bc09ad642c4be3361739f20048af32ad50fd8e6471a65  coastal_cliff_01.bin
7f287d0b44a3162b5e360fe45ccc8b6009c189d21bd33c1094060eef613c6fa6  textures/coastal_cliff_01_arm_1k.jpg
6c9626fdb5550ba260b195638c8c89053faedc22005f4ca94d3be5df72ae525a  textures/coastal_cliff_01_diff_1k.jpg
afbe8c2f9ff62dd616f776c074b996d2d275eef982a40aee1f28f4f3b3a3cf0e  textures/coastal_cliff_01_nor_gl_1k.jpg
EOF
)

mkdir -p "$(dirname "$DEST")"
mv -- "$STAGE" "$DEST"
STAGE=""

printf '%s\n' \
    "Fetched Poly Haven Coastal Cliff 01 (CC0) to $DEST" \
    "Configure with:" \
    "  -DRAYMMA_ENABLE_EXTERNAL_SCENES=ON" \
    "  -DRAYMMA_OPEN_MODEL_ASSET_DIR=$DEST"
