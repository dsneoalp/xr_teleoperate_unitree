#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f docker/compose.yml)

usage() {
  echo "usage: $0 {build|operator|robot|mock} [extra docker compose args...]"
  echo "  build     docker compose -f docker/compose.yml build mock operator robot"
  echo "  operator  interactive operator (TeleVuer :8012, press r/q)"
  echo "  robot     G1 DDS robot loop (host network)"
  echo "  mock      LiveKit echo robot; prints incoming actions at 1 Hz"
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
  *)
    usage
    ;;
esac
