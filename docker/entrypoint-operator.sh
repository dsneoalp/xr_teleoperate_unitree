#!/bin/sh
set -e
CERT="${XR_TELEOP_CERT:-/certs/cert.pem}"
KEY="${XR_TELEOP_KEY:-/certs/key.pem}"
mkdir -p "$(dirname "$CERT")"

need_cert=0
if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
  need_cert=1
elif ! openssl x509 -in "$CERT" -noout -text 2>/dev/null | grep -q "Subject Alternative Name"; then
  echo "[operator] existing cert has no SAN; regenerating for LAN headset access"
  need_cert=1
fi

if [ "$need_cert" -eq 1 ]; then
  SAN="DNS:localhost,IP:127.0.0.1"
  for ip in $(hostname -I 2>/dev/null); do
    case "$ip" in
      *:*) continue ;;
      127.*) continue ;;
    esac
    SAN="${SAN},IP:${ip}"
  done
  echo "[operator] generating self-signed TLS cert SAN=${SAN}"
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=localhost" \
    -addext "subjectAltName=${SAN}"
fi

export XR_TELEOP_CERT="$CERT"
export XR_TELEOP_KEY="$KEY"
cd /app/teleop
exec python teleop_operator.py "$@"
