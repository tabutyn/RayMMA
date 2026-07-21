#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOURS=12
MIN_INTERVAL=180
MAX_INTERVAL=420
OUTPUT_DIR="$ROOT/lambda-results"
KEY_FILE="${LAMBDA_API_KEY_FILE:-}"
PRIVATE_KEY="${RAYMMA_LAMBDA_PRIVATE_KEY:-$HOME/.ssh/raymma_lambda}"
PUBLIC_KEY="${RAYMMA_LAMBDA_PUBLIC_KEY:-$PRIVATE_KEY.pub}"
LOCK_FILE="/tmp/raymma-b200-watch.lock"

usage() {
    cat <<'EOF'
usage: LAMBDA_API_KEY_FILE=/secure/key ./tools/watch_lambda_b200.sh [options]

Options:
  --hours N          availability window (default: 12)
  --min-interval N   minimum seconds between checks (default: 180)
  --max-interval N   maximum seconds between checks (default: 420)
  --output-dir PATH  downloaded run directory root

The watcher polls only for a single B200. If one appears, it runs the archive
profile once, verifies retrieval, terminates the rental, and exits. It never
falls back to another GPU.
EOF
}

while (($#)); do
    case "$1" in
        --hours) HOURS="$2"; shift 2 ;;
        --min-interval) MIN_INTERVAL="$2"; shift 2 ;;
        --max-interval) MAX_INTERVAL="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$HOURS" "$MIN_INTERVAL" "$MAX_INTERVAL"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        printf 'Hours and intervals must be positive integers.\n' >&2
        exit 2
    }
done
((MIN_INTERVAL <= MAX_INTERVAL)) || {
    printf 'Minimum interval exceeds maximum interval.\n' >&2
    exit 2
}
[[ -n "$KEY_FILE" && -f "$KEY_FILE" ]] || {
    printf 'LAMBDA_API_KEY_FILE must name a readable credential file.\n' >&2
    exit 2
}
[[ -f "$PRIVATE_KEY" && -f "$PUBLIC_KEY" ]] || {
    printf 'Dedicated Lambda SSH keypair is missing.\n' >&2
    exit 2
}
for command_name in curl flock python3 tr awk date mkdir; do
    command -v "$command_name" >/dev/null || {
        printf 'Required command is unavailable: %s\n' "$command_name" >&2
        exit 2
    }
done

mkdir -p "$OUTPUT_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || {
    printf 'Another B200 watcher already holds %s.\n' "$LOCK_FILE" >&2
    exit 2
}

STATE_FILE="$OUTPUT_DIR/b200-watch.state"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
ATTEMPT=0
printf 'status=watching\nstarted_utc=%s\ndeadline_epoch=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$DEADLINE" > "$STATE_FILE"

cd "$ROOT"
while (( $(date +%s) < DEADLINE )); do
    ATTEMPT=$((ATTEMPT + 1))
    printf '[%s] B200 capacity check %d\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ATTEMPT"
    export LAMBDA_API_KEY
    LAMBDA_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
    INVENTORY="$(python3 tools/lambda_cloud.py inventory 2>&1)" || {
        printf '%s\n' "$INVENTORY" >&2
        printf 'status=inventory-error\nattempt=%d\nupdated_utc=%s\n' \
            "$ATTEMPT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_FILE"
        exit 1
    }
    printf '%s\n' "$INVENTORY"
    B200_TYPE="$(printf '%s\n' "$INVENTORY" |
        awk '$1 ~ /^gpu_1x_b200/ {print $1; exit}')"
    if [[ -n "$B200_TYPE" ]]; then
        PUBLIC_IP="$(curl -4fsS https://icanhazip.com | tr -d '\r\n')"
        printf 'status=launching\nattempt=%d\ninstance_type=%s\nupdated_utc=%s\n' \
            "$ATTEMPT" "$B200_TYPE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            > "$STATE_FILE"
        printf 'Eligible B200 found: %s; starting archive run.\n' "$B200_TYPE"
        if PYTHONUNBUFFERED=1 python3 tools/lambda_cloud.py run \
                --instance-type "$B200_TYPE" \
                --ssh-private-key "$PRIVATE_KEY" \
                --ssh-public-key "$PUBLIC_KEY" \
                --ssh-cidr "$PUBLIC_IP/32" \
                --profile archive \
                --output-dir "$OUTPUT_DIR" \
                --yes; then
            STATUS=0
        else
            STATUS=$?
        fi
        printf 'status=%s\nattempt=%d\nfinished_utc=%s\n' \
            "$([[ $STATUS -eq 0 ]] && printf complete || printf run-failed)" \
            "$ATTEMPT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_FILE"
        exit "$STATUS"
    fi

    printf 'status=watching\nattempt=%d\nupdated_utc=%s\n' \
        "$ATTEMPT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_FILE"
    REMAINING=$((DEADLINE - $(date +%s)))
    ((REMAINING > 0)) || break
    DELAY=$((MIN_INTERVAL + RANDOM % (MAX_INTERVAL - MIN_INTERVAL + 1)))
    ((DELAY < REMAINING)) || DELAY=$REMAINING
    printf 'No B200; next randomized check in %ds.\n' "$DELAY"
    sleep "$DELAY"
done

printf 'status=expired\nattempt=%d\nfinished_utc=%s\n' \
    "$ATTEMPT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_FILE"
printf 'Monitoring window expired; no single B200 appeared.\n'
