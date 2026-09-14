# Docker images

Separate Dockerfiles under `docker/`. Compose file is `[compose.yml](compose.yml)`. Run from the **repo root**.

`.dockerignore` stays at the repo root because the build context is the repo (`context: ..`). Docker only reads ignore files from the context root.

## Host prerequisites

No conda/`tv` and no extra checkouts. Needed on the host:

- Docker + Compose
- this repo with submodules (`git submodule update --init --depth 1`)
- `teleop/.env` (copy from `teleop/.env.example`)
- a LiveKit server reachable at `LIVEKIT_URL` (default `ws://127.0.0.1:7880`), because services use `network_mode: host`

PyPI `livekit-portal` wheels are Python 3.12 only. Mock, operator, and the **PC** robot stay on Python 3.10 and share the build-only wheel image `xr-teleop:portal-wheel` (`[Dockerfile.portal-wheel](Dockerfile.portal-wheel)`). That wheel image is not a runtime.

## Images

| Service / image | Dockerfile | What it is |
| --- | --- | --- |
| `portal-wheel` (`xr-teleop:portal-wheel`) | `[Dockerfile.portal-wheel](Dockerfile.portal-wheel)` | Build-only Python 3.10 livekit-portal FFI wheel. Profile `wheel`. |
| `operator` (`xr-teleop:operator`) | `[Dockerfile.operator](Dockerfile.operator)` | README 1.1 conda env `tv`: `python=3.10 pinocchio=3.1.0 numpy=1.26.4`, then `pip install -e` teleimager / televuer / dex-retargeting, `requirements.txt`, Portal wheel, `params-proto==2.13.2`. No Unitree SDK. |
| `robot` (`xr-teleop:robot`) | `[Dockerfile.robot](Dockerfile.robot)` | PC / sim: Python 3.10 software Portal + CycloneDDS + `unitree_sdk2_python`. No Jetson MMAPI. |
| `robot-g1` (`xr-teleop:robot-g1`) | `[Dockerfile.robot-g1](Dockerfile.robot-g1)` + `[compose.g1.yml](compose.g1.yml)` | G1/Jetson: Python 3.12, FFI copied from `neox/portal-robot:lab-jetson`. Do not `pip install livekit-portal`. |
| `mock` (`xr-teleop:mock`) | `[Dockerfile.mock](Dockerfile.mock)` | LiveKit echo robot (and mock operator via a different CMD). No SDK, no Pinocchio. |

Operator default command: `--arm G1_29 --ee dex3 --input-mode hand`.

`teleop/` is bind-mounted into every service (`assets/` into operator). `.py` / YAML edits apply on container restart. New pip dependencies or Dockerfile changes still need a rebuild.

HW encode is **not** implied by the G1 image. Pass `--require-hw-encode` at run time (see `[HW_ENCODE_G1_PROTOCOL.md](HW_ENCODE_G1_PROTOCOL.md)`). Compose does not set that flag.

## Build / run

```bash
./docker/run.sh build
# portal-wheel, then mock + operator + robot (PC)

docker compose -f docker/compose.yml up mock
docker compose -f docker/compose.yml run --rm -it operator
# extra argparse after the service name replaces CMD, entrypoint stays:
docker compose -f docker/compose.yml run --rm -it operator --headless --ipc

# G1 image (on the robot, needs lab-jetson locally):
./docker/run.sh build-g1
./docker/run.sh robot-g1 --require-hw-encode

# Dex3 custom pose GUI (WORKDIR is /app/teleop; YAML lands on the host bind-mount):
xhost +local:docker
docker compose -f docker/compose.yml run --rm -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  operator --ee dex3 --input-mode controller --custom_mapping --hand-pose-yaml my_poses.yaml
```

If `my_poses.yaml` already exists, the operator loads it and skips the GUI. `--custom_mapping --headless` without that file exits with an error.

`teleop/.env` is loaded at runtime (`env_file`). It is not copied into the image.

## Local stack (LiveKit + mock robot + mock operator)

Linux only (`network_mode: host`). No G1, no XR headset. This is the loopback check for Portal send/receive with `[portal_local.yaml](../teleop/portal_local.yaml)`.

### 1. Config

```bash
cp teleop/.env.local.example teleop/.env.local
```

`.env.local` matches LiveKit `--dev`:

| Key                  | Value                 |
| -------------------- | --------------------- |
| `LIVEKIT_URL`        | `ws://127.0.0.1:7880` |
| `LIVEKIT_API_KEY`    | `devkey`              |
| `LIVEKIT_API_SECRET` | `secret`              |
| `LIVEKIT_ROOM`       | `g1-portal-local`     |

`portal_local.yaml` is the wire contract the containers pass via `--portal-yaml`. Same field names as `portal.yaml` / `portal_mapping.yaml`. Edit fps/tolerance there without touching production.

### 2. Build the mock image (once)

```bash
./docker/run.sh build   # includes portal-wheel + mock
```

### 3. Start LiveKit, echo robot, mock operator

```bash
./docker/run.sh local
# same as:
# docker compose -f docker/compose.yml -f docker/compose.local.yml up livekit mock operator-mock
```

Order: LiveKit on `:7880`, then mock robot echoes actions as state + a test-pattern `head_camera` frame, then mock operator sends dummy arm/hand at 30 Hz.

### 4. What success looks like

Mock robot (1 Hz):

```
[mock-robot] 1Hz: count=... fsm=1 vx=0.120 ... L_SHOULDER_PITCH=0.0500
```

Mock operator (1 Hz):

```
[mock-op] 1Hz: echoed=True xr_frame=True obs_frames=True ...
```

`echoed=True` means the robot published the same joints back. `xr_frame=True` means `get_head_frame()` has a decoded image for TeleVuer. `obs_frames=True` means Portal matched a video frame to that state tick.

Local video is MJPEG (`portal_local.yaml`). H264 on LiveKit `--dev` drops `user_timestamp` and panics the operator.

One-shot (exit 0 after echo + matched frame):

```bash
docker compose -f docker/compose.yml -f docker/compose.local.yml run --rm operator-mock \
  python tests/portal_operator_mock.py \
  --portal-yaml portal_local.yaml --env-file .env.local \
  --expect-echo --duration 20
```

Stop with Ctrl-C, or `docker compose -f docker/compose.yml -f docker/compose.local.yml down`.

### Optional: Isaac Lab (own Docker) + robot + operator

Do **not** run `./docker/run.sh local` — that starts mock robot/operator in the same LiveKit room. Isaac Lab is the simulated G1; our `robot` container is the DDS/Portal bridge (`--sim`); `operator` is XR.

All xr-teleop services use host network, so they share `127.0.0.1` with Isaac Lab: LiveKit `:7880`, CycloneDDS domain **1**, and teleimager ZMQ if Isaac publishes cameras on localhost.

```bash
./docker/run.sh build
./docker/run.sh local-livekit
# Isaac Lab — your image, host network
./docker/run.sh local-robot
./docker/run.sh local-operator
```

Start Isaac Lab before `local-robot`. Extra args after `local-robot` / `local-operator` **replace** the compose command and drop `--portal-yaml`. Repeat the flags if you need extras:

```bash
./docker/run.sh local-operator --arm G1_29 --ee dex3 --input-mode hand \
  --portal-yaml portal_local.yaml --env-file .env.local --headless --ipc
```

No sim video: pass a full `local-robot` command that includes `--no-img`. Default ImageClient host is `127.0.0.1`.
The vuer should then be visible if you enter the website: `https://127.0.0.1:8012/?ws=wss://127.0.0.1:8012`

Stop LiveKit with Ctrl-C in that terminal, or `docker compose -f docker/compose.yml -f docker/compose.local.yml down`.

## G1 hardware encode

Hardware accept on the Unitree G1: follow [`HW_ENCODE_G1_PROTOCOL.md`](HW_ENCODE_G1_PROTOCOL.md). Primary check is `python tests/portal_hw_encode_check.py` inside `robot-g1` (no DDS). Device presence is not encode.

H.264 is encoded in the **livekit-portal FFI / Jetson MMAPI**, not in Python. Python still sends RGB via `send_video_frame`. Only `robot-g1` gets the encoder; mock, operator, and PC `robot` stay software.

Fail-closed encode is opt-in: `teleop_robot.py --require-hw-encode` or the check script (always requires HW). Compose does **not** pass the flag.

Offline probe tests: `pytest teleop/tests/robot_control/test_portal_hw_encode.py`.

## TLS (TeleVuer :8012)

The operator entrypoint generates a Pico/Quest-style self-signed cert in `/certs` if none is present (`XR_TELEOP_CERT` / `XR_TELEOP_KEY`). For Apple Vision Pro, generate SAN certs on the host (README 1.1.2) and bind-mount them over `/certs`.
