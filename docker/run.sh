#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker/compose.yml)
LOCAL_COMPOSE=(docker compose -f docker/compose.yml -f docker/compose.local.yml)

require_env_local() {
  if [[ ! -f teleop/.env.local ]]; then
    echo "missing teleop/.env.local — copy teleop/.env.local.example first" >&2
    exit 1
  fi
}

usage() {
  echo "usage: $0 {build|operator|robot|mock|local|local-livekit|local-robot|local-operator} [args...]"
  echo "  build           docker compose -f docker/compose.yml build mock operator robot"
  echo "  operator        interactive operator (production .env / portal.yaml)"
  echo "  robot           G1 DDS robot loop (production .env / portal.yaml)"
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
    "${COMPOSE[@]}" build mock operator robot "$@"
    ;;
  operator)
    "${COMPOSE[@]}" run --rm -it operator "$@"
    ;;
  robot)
    "${COMPOSE[@]}" run --rm robot "$@"
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
