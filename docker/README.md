# Docker images

Three images from [`Dockerfile`](Dockerfile). Compose file is [`compose.yml`](compose.yml). Run from the **repo root**.

`.dockerignore` stays at the repo root because the build context is the repo (`context: ..`). Docker only reads ignore files from the context root.

## Host prerequisites

No conda/`tv` and no extra checkouts. Needed on the host:

- Docker + Compose
- this repo with submodules (`git submodule update --init --depth 1`)
- `teleop/.env` (copy from `teleop/.env.example`)
- a LiveKit server reachable at `LIVEKIT_URL` (default `ws://127.0.0.1:7880`), because services use `network_mode: host`

PyPI `livekit-portal` wheels are Python 3.12 only. The image stays on Python 3.10 (README `tv`) and builds `livekit-portal` from git commit `4fb4385` (same as conda `tv`: `0.2.6.dev5+g4fb4385`).

## Images

| Service | What it is |
|---|---|
| `operator` | README 1.1 conda env `tv`: `python=3.10 pinocchio=3.1.0 numpy=1.26.4 tk` (conda-forge), then `pip install -e` teleimager / televuer / dex-retargeting, then `requirements.txt`. Plus `livekit-portal` built from git `4fb4385` (PyPI wheels are Python 3.12 only), `livekit-api`, and `params-proto==2.13.2` so `vuer==0.0.60` still imports. No Unitree SDK. Tkinter for `--custom_mapping` GUI. |
| `robot` | README 1.2 `unitree_sdk2_python` at commit `65691c8` (`git+https` during build) + cyclonedds. DDS controllers. |
| `mock` | LiveKit echo robot. Prints latest action at 1 Hz. No SDK, no Pinocchio. |

Operator default command: `--arm G1_29 --ee dex3 --input-mode hand`.

`teleop/` is bind-mounted into every service (`assets/` into operator). `.py` / YAML edits apply on container restart. New pip dependencies or Dockerfile changes still need a rebuild.

## Portal sync and recording

Robot and operator share [`teleop/portal.yaml`](../teleop/portal.yaml) (`fps: 30`, `slack: 2`). Each robot tick stamps **state and video with the same `timestamp_us`** (Portal unified sampling). The video thread holds one latest frame only (drop-oldest); there is no action/state queue between Operator and Robot.

Live XR uses unmatched `on_video_frame` (lowest latency). Recordings (`--record`, toggle `s`) join Portal-matched observations with the action that has `in_reply_to_ts_us == obs.timestamp_us` (`action_subscription` on the operator). Target rate is 30 Hz; a rolling window below 20 Hz is logged as an error (`--record-min-hz`).

## Tests

On the host, from the repo root (conda env `tv`):

```bash
conda run -n tv python -m pytest tests/
# unittest fallback:
conda run -n tv python -m unittest discover -s tests -v
```

In the operator image:

```bash
docker compose -f docker/compose.yml run --rm --no-deps --entrypoint python -w /app \
  operator -m unittest discover -s tests -v
```

See [`tests/Test_ReadMe.md`](../tests/Test_ReadMe.md). LiveKit round-trip tests skip unless `teleop/.env` has `LIVEKIT_URL`.

## Build / run

```bash
docker compose -f docker/compose.yml build mock operator robot
# or: ./docker/run.sh build

docker compose -f docker/compose.yml up mock
docker compose -f docker/compose.yml run --rm -it operator
# extra argparse after the service name replaces CMD, entrypoint stays:
docker compose -f docker/compose.yml run --rm -it operator --headless --ipc

# Dex3 custom pose GUI (WORKDIR is /app/teleop; YAML lands on the host bind-mount):
xhost +local:docker
docker compose -f docker/compose.yml run --rm -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  operator --ee dex3 --input-mode controller --custom_mapping --hand-pose-yaml my_poses.yaml
```

If `my_poses.yaml` already exists, the operator loads it and skips the GUI. `--custom_mapping --headless` without that file exits with an error.

`teleop/.env` is loaded at runtime (`env_file`). It is not copied into the image.

## TLS (TeleVuer :8012)

The operator entrypoint generates a Pico/Quest-style self-signed cert in `/certs` if none is present (`XR_TELEOP_CERT` / `XR_TELEOP_KEY`). For Apple Vision Pro, generate SAN certs on the host (README 1.1.2) and bind-mount them over `/certs`.
