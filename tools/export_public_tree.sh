#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s DESTINATION\n' "$0" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$1"

if [[ -e "$DEST" ]]; then
    printf 'Destination already exists: %s\n' "$DEST" >&2
    exit 1
fi

mkdir -p "$DEST"

FILES=(
    .gitignore
    .github/ISSUE_TEMPLATE/benchmark-result.yml
    .github/ISSUE_TEMPLATE/config.yml
    .github/workflows/cpu-algebra.yml
    CHANGELOG.md
    CITATION.cff
    CMakeLists.txt
    CMakePresets.json
    CONTRIBUTING.md
    LICENSE
    README.md
    SECURITY.md
    THIRD_PARTY.md
    docs/GITHUB_POST.md
    docs/METHOD.md
    docs/REPOSITORY_PROFILE.md
    docs/RELATED_WORK.md
    docs/RELEASE_CHECKLIST.md
    docs/REPRODUCIBILITY.md
    docs/RESULTS_RTX3050TI.md
    docs/checker-high-crossover.svg
    prototype/ALGEBRA.md
    prototype/README.md
    prototype/tensor_ray_triangle_wmma.cu
    prototype/verify_algebra.py
    results/README.md
    results/rtx3050ti-grid-2026-07-19/README.md
    results/rtx3050ti-grid-2026-07-19/environment.txt
    results/rtx3050ti-grid-2026-07-19/raw.csv
    results/rtx3050ti-grid-2026-07-19/stdout.txt
    src/production_bvh.h
    src/live_viewer.cu
    src/research_benchmark.cu
    src/tinybvh_builder.cpp
    src/viewer.cpp
    tools/capture_environment.sh
    tools/export_public_tree.sh
    tools/extract_checker_triangles.py
    tools/fetch_benchmark_scenes.sh
)

for file in "${FILES[@]}"; do
    mkdir -p "$DEST/$(dirname "$file")"
    cp -- "$ROOT/$file" "$DEST/$file"
done

printf '%s\n' \
    "Exported ${#FILES[@]} allowlisted RayMMA files to $DEST" \
    "Excluded: Ball Roller history/assets, build outputs, and third_party SDKs." \
    "New source/docs/results files require an explicit manifest update." \
    "Inspect and scan this directory before creating the public Git repository."
