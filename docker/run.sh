#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker/compose.yml)
G1_COMPOSE=(docker compose -f docker/compose.yml -f docker/compose.g1.yml)
LOCAL_COMPOSE=(docker compose -f docker/compose.yml -f docker/compose.local.yml)

require_env_local() {
  if [[ ! -f teleop/.env.local ]]; then
    echo "missing teleop/.env.local — copy teleop/.env.local.example first" >&2
    exit 1
  fi
}

usage() {
  echo "usage: $0 {build|build-g1|operator|robot|robot-g1|mock|local|local-livekit|local-robot|local-operator} [args...]"
  echo "  build           portal-wheel + mock + operator + robot (PC / software encode)"
  echo "  build-g1        robot-g1 (Jetson MMAPI FFI; build on the G1)"
  echo "  operator        interactive operator (production .env / portal.yaml)"
  echo "  robot           PC robot loop (no --require-hw-encode)"
  echo "  robot-g1        G1 robot; pass --require-hw-encode to fail-closed HW encode"
  echo "  mock            LiveKit echo robot"
  echo "  local           LiveKit --dev + mock robot + mock operator"
  echo "  local-livekit   LiveKit --dev only"
  echo "  local-robot     robot --sim + portal_local.yaml (Isaac Lab DDS)"
  echo "  local-operator  operator + portal_local.yaml (TeleVuer :8012)"
  exit 1
}

cmd="${1:-}"
shift || true
case "$cmd" in
  build)
    "${COMPOSE[@]}" --profile wheel build portal-wheel
    "${COMPOSE[@]}" build mock operator robot "$@"
    ;;
  build-g1)
    "${G1_COMPOSE[@]}" build robot-g1 "$@"
    ;;
  operator)
    "${COMPOSE[@]}" run --rm -it operator "$@"
    ;;
  robot)
    "${COMPOSE[@]}" run --rm robot "$@"
    ;;
  robot-g1)
    if [[ $# -eq 0 ]]; then
      "${G1_COMPOSE[@]}" run --rm --no-deps robot-g1
    elif [[ "${1:-}" == python || "${1:-}" == bash ]]; then
      "${G1_COMPOSE[@]}" run --rm --no-deps robot-g1 "$@"
    else
      "${G1_COMPOSE[@]}" run --rm --no-deps robot-g1 \
        python teleop_robot.py --ee dex3 --arm G1_29 "$@"
    fi
    ;;
  mock)
    "${COMPOSE[@]}" run --rm mock "$@"
    ;;
  local)
    require_env_local
    "${LOCAL_COMPOSE[@]}" up livekit mock operator-mock "$@"
    ;;
  local-livekit)
    require_env_local
    "${LOCAL_COMPOSE[@]}" up livekit "$@"
    ;;
  local-robot)
    require_env_local
    "${LOCAL_COMPOSE[@]}" run --rm robot "$@"
    ;;
  local-operator)
    require_env_local
    "${LOCAL_COMPOSE[@]}" run --rm -it operator "$@"
    ;;
  *)
    usage
    ;;
esac
