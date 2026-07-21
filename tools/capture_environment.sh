#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/build/core}"

run_if_available() {
    local command_name="$1"
    shift
    if command -v "$command_name" >/dev/null 2>&1; then
        "$command_name" "$@" 2>&1
    else
        printf '%s\n' "$command_name: unavailable"
    fi
}

printf 'captured_utc='
date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'repository=RayMMA\n'
printf 'commit='
git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'uncommitted-export\n'
printf 'describe='
git -C "$ROOT" describe --always 2>/dev/null || printf 'unavailable\n'
printf 'status_begin\n'
git -C "$ROOT" status --short -- . 2>/dev/null || true
printf 'status_end\n'

printf '\n[system]\n'
run_if_available uname -srmo
if command -v lscpu >/dev/null 2>&1; then
    lscpu | sed -n \
        -e '/^Architecture:/p' \
        -e '/^CPU(s):/p' \
        -e '/^Model name:/p'
fi

printf '\n[gpu]\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,pstate,power.limit,clocks.sm,clocks.mem \
        --format=csv,noheader 2>&1
else
    printf 'nvidia-smi: unavailable\n'
fi

printf '\n[toolchain]\n'
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>&1
elif [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CUDA_COMPILER="$(
        sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
            "$BUILD_DIR/CMakeCache.txt" | head -n 1
    )"
    if [[ -n "$CUDA_COMPILER" && -x "$CUDA_COMPILER" ]]; then
        "$CUDA_COMPILER" --version 2>&1
    else
        printf 'nvcc: unavailable\n'
    fi
else
    printf 'nvcc: unavailable\n'
fi
run_if_available cmake --version
run_if_available c++ --version
run_if_available python3 --version

printf '\n[cmake-cache]\n'
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    sed -n \
        -e '/^CMAKE_BUILD_TYPE:/p' \
        -e '/^CMAKE_CUDA_ARCHITECTURES:/p' \
        -e '/^CMAKE_CUDA_COMPILER:/p' \
        -e '/^CMAKE_CXX_COMPILER:/p' \
        -e '/^RAYMMA_BUILD_/p' \
        -e '/^RAYMMA_ENABLE_/p' \
        "$BUILD_DIR/CMakeCache.txt"
else
    printf 'No CMake cache at %s\n' "$BUILD_DIR"
fi
