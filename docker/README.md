# Docker images

Three images from [`Dockerfile`](Dockerfile). Compose file is [`compose.yml`](compose.yml). Run from the **repo root**.

`.dockerignore` stays at the repo root because the build context is the repo (`context: ..`). Docker only reads ignore files from the context root.

## Images

| Service | What it is |
|---|---|
| `operator` | README 1.1 conda env `tv`: `python=3.10 pinocchio=3.1.0 numpy=1.26.4` (conda-forge), then `pip install -e` teleimager / televuer / dex-retargeting, then `requirements.txt`. Plus Portal extras (`livekit-api`, FFI) and `params-proto==2.13.2` so `vuer==0.0.60` still imports. No Unitree SDK. |
| `robot` | README 1.2 `unitree_sdk2_python` + cyclonedds. DDS controllers. |
| `mock` | LiveKit echo robot. Prints latest action at 1 Hz. No SDK, no Pinocchio. |

Operator default command: `--arm G1_29 --ee dex3 --input-mode hand`.

## Build / run

```bash
docker compose -f docker/compose.yml build mock operator robot
# or: ./docker/run.sh build

docker compose -f docker/compose.yml up mock
docker compose -f docker/compose.yml run --rm -it operator
# extra argparse after the service name replaces CMD, entrypoint stays:
docker compose -f docker/compose.yml run --rm -it operator --headless --ipc
```

`teleop/.env` is mounted at runtime (`env_file`). Python under `teleop/` is copied at **build** time; code edits need a rebuild.

## TLS (TeleVuer :8012)

The operator entrypoint generates a Pico/Quest-style self-signed cert in `/certs` if none is present (`XR_TELEOP_CERT` / `XR_TELEOP_KEY`). For Apple Vision Pro, generate SAN certs on the host (README 1.1.2) and bind-mount them over `/certs`.
