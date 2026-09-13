# Docker images

Three images from `[Dockerfile](Dockerfile)`. Compose file is `[compose.yml](compose.yml)`. Run from the **repo root**.

`.dockerignore` stays at the repo root because the build context is the repo (`context: ..`). Docker only reads ignore files from the context root.

## Host prerequisites

No conda/`tv` and no extra checkouts. Needed on the host:

- Docker + Compose
- this repo with submodules (`git submodule update --init --depth 1`)
- `teleop/.env` (copy from `teleop/.env.example`)
- a LiveKit server reachable at `LIVEKIT_URL` (default `ws://127.0.0.1:7880`), because services use `network_mode: host`

PyPI `livekit-portal` wheels are Python 3.12 only. The image stays on Python 3.10 (README `tv`) and builds `livekit-portal` from git commit `4fb4385` (same as conda `tv`: `0.2.6.dev5+g4fb4385`).

## Images


| Service    | What it is                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `operator` | README 1.1 conda env `tv`: `python=3.10 pinocchio=3.1.0 numpy=1.26.4 tk` (conda-forge), then `pip install -e` teleimager / televuer / dex-retargeting, then `requirements.txt`. Plus `livekit-portal` built from git `4fb4385` (PyPI wheels are Python 3.12 only), `livekit-api`, and `params-proto==2.13.2` so `vuer==0.0.60` still imports. No Unitree SDK. Tkinter for `--custom_mapping` GUI. |
| `robot`    | README 1.2 `unitree_sdk2_python` at commit `65691c8` (`git+https` during build) + cyclonedds. DDS controllers.                                                                                                                                                                                                                                                                                    |
| `mock`     | LiveKit echo robot. Prints latest action at 1 Hz. No SDK, no Pinocchio.                                                                                                                                                                                                                                                                                                                           |


Operator default command: `--arm G1_29 --ee dex3 --input-mode hand`.

`teleop/` is bind-mounted into every service (`assets/` into operator). `.py` / YAML edits apply on container restart. New pip dependencies or Dockerfile changes still need a rebuild.

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
docker compose -f docker/compose.yml -f docker/compose.local.yml build mock
# or: ./docker/run.sh build   # also builds operator + robot images
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

Build operator + robot images once:

```bash
./docker/run.sh build
```

Four processes, repo root:

```bash
# 1) LiveKit
./docker/run.sh local-livekit

# 2) Isaac Lab — your image, host network (or the same DDS multicast domain).
#    After start: click the sim window until "controller started, start main loop..."

# 3) Portal robot bridge (overlay already passes --sim + portal_local.yaml)
./docker/run.sh local-robot

# 4) Portal operator (TeleVuer :8012)
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

## TLS (TeleVuer :8012)

The operator entrypoint generates a Pico/Quest-style self-signed cert in `/certs` if none is present (`XR_TELEOP_CERT` / `XR_TELEOP_KEY`). For Apple Vision Pro, generate SAN certs on the host (README 1.1.2) and bind-mount them over `/certs`.