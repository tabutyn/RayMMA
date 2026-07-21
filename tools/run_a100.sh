#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="archive"
cd "$ROOT"

usage() {
    cat <<'EOF'
usage: ./tools/run_a100.sh [--profile quick|archive]

Build RayMMA for SM 80, run the procedural Grid evidence suite, and write:
  build/raymma-a100-results.tar.gz
  build/raymma-a100-results.tar.gz.sha256

quick    128x72, five timing samples, primary rays
archive  256x144, nine timing samples, primary and secondary rays (default)

Set RAYMMA_ALLOW_NON_A100=1 only for workflow testing on another SM 80 GPU.
EOF
}

while (($#)); do
    case "$1" in
        --profile)
            (($# >= 2)) || { printf 'Missing --profile value\n' >&2; exit 2; }
            PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$PROFILE" != quick && "$PROFILE" != archive ]]; then
    printf 'Profile must be quick or archive\n' >&2
    exit 2
fi

for command_name in git cmake ctest nvcc nvidia-smi python3 tar sha256sum \
        tee find sort xargs cp ldd; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Required command is unavailable: %s\n' "$command_name" >&2
        exit 2
    fi
done

if ! git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'Run from a Git checkout so the archive has source provenance.\n' >&2
    exit 2
fi

dirty="$(git -C "$ROOT" status --porcelain --untracked-files=normal)"
if [[ -n "$dirty" ]]; then
    printf 'Refusing a dirty checkout; commit or remove these files first:\n%s\n' \
        "$dirty" >&2
    exit 2
fi

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if ((${#gpu_names[@]} != 1)); then
    printf 'Expected a 1xA100 host; nvidia-smi reports %d physical GPUs.\n' \
        "${#gpu_names[@]}" >&2
    exit 2
fi
if [[ "${gpu_names[0]}" != *A100* && "${RAYMMA_ALLOW_NON_A100:-0}" != 1 ]]; then
    printf 'Expected NVIDIA A100 as CUDA device 0, found: %s\n' \
        "${gpu_names[0]}" >&2
    exit 2
fi

BUILD_DIR="$ROOT/build/a100"
RUN_STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
COMMIT="$(git -C "$ROOT" rev-parse --short=12 HEAD)"
RESULT_DIR="$ROOT/build/raymma-a100-$RUN_STAMP-$COMMIT"
ARCHIVE="$ROOT/build/raymma-a100-results.tar.gz"
DIGEST="$ARCHIVE.sha256"
mkdir -p "$RESULT_DIR"

finalize() {
    local status=$?
    local package_status=0
    trap - EXIT
    set +e
    printf '%s\n' "$status" > "$RESULT_DIR/exit-code.txt"
    (
        cd "$RESULT_DIR" || exit 1
        find . -type f ! -name SHA256SUMS -print0 |
            sort -z | xargs -0 sha256sum > SHA256SUMS
    ) || package_status=1
    tar -C "$RESULT_DIR" -czf "$ARCHIVE" . || package_status=1
    (
        cd "$(dirname "$ARCHIVE")" || exit 1
        sha256sum "$(basename "$ARCHIVE")" > "$(basename "$DIGEST")"
    ) || package_status=1
    if [[ $package_status -eq 0 ]]; then
        printf '%s\n%s\n' "$ARCHIVE" "$DIGEST"
    else
        printf 'Failed to package the result archive.\n' >&2
        [[ $status -ne 0 ]] || status=70
    fi
    exit "$status"
}
trap finalize EXIT

COMMANDS="$RESULT_DIR/commands.txt"
record_command() {
    printf '$' >> "$COMMANDS"
    printf ' %q' "$@" >> "$COMMANDS"
    printf '\n' >> "$COMMANDS"
}

run_logged() {
    local label="$1"
    shift
    record_command "$@"
    "$@" 2>&1 | tee "$RESULT_DIR/$label.log"
}

printf 'RayMMA A100 evidence run\n'
printf 'profile=%s\ncommit=%s\nresult_dir=%s\n' \
    "$PROFILE" "$COMMIT" "$RESULT_DIR"

git -C "$ROOT" show -s --format=fuller HEAD > "$RESULT_DIR/commit.txt"
git -C "$ROOT" status --short > "$RESULT_DIR/git-status.txt"
(
    cd "$ROOT"
    git ls-files -z | sort -z | xargs -0 sha256sum
) > "$RESULT_DIR/source-sha256.txt"

run_logged configure cmake --preset a100
run_logged build cmake --build --preset a100 --parallel
mkdir -p "$RESULT_DIR/bin"
cp "$BUILD_DIR/tensor-wide-bvh-bench" "$RESULT_DIR/bin/"
cp "$BUILD_DIR/tensor-ray-correctness" "$RESULT_DIR/bin/"
ldd "$BUILD_DIR/tensor-wide-bvh-bench" > "$RESULT_DIR/benchmark-ldd.txt"
"$ROOT/tools/capture_environment.sh" "$BUILD_DIR" \
    > "$RESULT_DIR/environment.txt" 2>&1
run_logged tests ctest --preset a100

BENCH="$BUILD_DIR/tensor-wide-bvh-bench"
COMMON=(--scene Grid --leaf-sweep)
if [[ "$PROFILE" == quick ]]; then
    COMMON+=(--quick)
fi

for variant in validated uvt-depthsorted e0e1e2; do
    label="grid-primary-$variant"
    run_logged "$label" "$BENCH" "${COMMON[@]}" --variant "$variant" \
        --raw-csv "$RESULT_DIR/$label.csv"
done

if [[ "$PROFILE" == archive ]]; then
    for variant in validated uvt-depthsorted e0e1e2; do
        label="grid-secondary-$variant"
        run_logged "$label" "$BENCH" "${COMMON[@]}" \
            --ray-mode secondary --variant "$variant" \
            --raw-csv "$RESULT_DIR/$label.csv"
    done
fi
