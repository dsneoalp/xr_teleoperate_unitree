#!/bin/sh
set -e
if [ ! -f "${XR_TELEOP_CERT:-/certs/cert.pem}" ] || [ ! -f "${XR_TELEOP_KEY:-/certs/key.pem}" ]; then
  echo "[operator] generating self-signed TLS cert in /certs"
  mkdir -p /certs
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /certs/key.pem -out /certs/cert.pem \
    -subj "/CN=localhost"
fi
export XR_TELEOP_CERT="${XR_TELEOP_CERT:-/certs/cert.pem}"
export XR_TELEOP_KEY="${XR_TELEOP_KEY:-/certs/key.pem}"
cd /app/teleop
exec python teleop_operator.py "$@"
