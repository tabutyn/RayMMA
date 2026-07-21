#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ "${1:-}" != "--accept-separate-scene-licenses" ]]; then
    printf '%s\n' \
        "This downloads separately licensed benchmark data." \
        "Sibenik is reported as CC BY-NC; Sponza is CC BY 3.0." \
        "Read THIRD_PARTY.md and the original archive terms first." \
        "Rerun with: $0 --accept-separate-scene-licenses [destination]"
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${2:-$ROOT/build/scenes}"
CACHE="$ROOT/build/cuda-lbvh-scenes"
REVISION="605802671beb6473b74a43552168f61e63af46db"

if [[ ! -d "$CACHE/.git" ]]; then
    mkdir -p "$CACHE"
    git -C "$CACHE" init
    git -C "$CACHE" remote add origin \
        https://github.com/nolmoonen/cuda-lbvh.git
    git -C "$CACHE" fetch --depth 1 origin "$REVISION"
    git -C "$CACHE" checkout --detach FETCH_HEAD
fi

if [[ "$(git -C "$CACHE" rev-parse HEAD)" != "$REVISION" ]]; then
    printf 'Unexpected scene-source revision in %s\n' "$CACHE" >&2
    exit 1
fi

mkdir -p "$DEST"
unzip -n "$CACHE/scenes/sibenik.zip" -d "$DEST"
unzip -n "$CACHE/scenes/sponza.zip" -d "$DEST"

printf '%s  %s\n' \
    "40494f9fa83771e3c4ea4442399042b2fa098d10206db6a6d6da7e48e2182289" \
    "$DEST/sibenik.obj" | sha256sum --check -
printf '%s  %s\n' \
    "eee3e272e2c3fc6ab5b7a3868191e07ff7e363af88d3352747112fe58a8c36d4" \
    "$DEST/sponza.obj" | sha256sum --check -

printf '\nConfigure with:\n'
printf '  -DRAYMMA_ENABLE_EXTERNAL_SCENES=ON \\\n'
printf '  -DRAYMMA_SIBENIK_PATH=%q \\\n' "$DEST/sibenik.obj"
printf '  -DRAYMMA_SPONZA_PATH=%q\n' "$DEST/sponza.obj"
