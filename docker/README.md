# Docker images

Three images from [`Dockerfile`](Dockerfile). Compose file is [`compose.yml`](compose.yml). Run from the **repo root**.

`.dockerignore` stays at the repo root because the build context is the repo (`context: ..`). Docker only reads ignore files from the context root.

## Host prerequisites

No conda/`tv` and no extra checkouts. Needed on the host:

- Docker + Compose
- this repo with submodules (`git submodule update --init --depth 1`)
- `teleop/.env` (copy from `teleop/.env.example`)
- a LiveKit server reachable at `LIVEKIT_URL` (default `ws://127.0.0.1:7880`), because services use `network_mode: host`

PyPI `livekit-portal` wheels are Python 3.12 only. Mock and operator stay on Python 3.10 and build `livekit-portal` from git commit `4fb4385` (same as conda `tv`: `0.2.6.dev5+g4fb4385`). The robot image is Python 3.12 and copies the Jetson MMAPI FFI from `neox/portal-robot:lab-jetson` — do not `pip install livekit-portal` there (PyPI / `lab-jetson-v` are OpenH264-only).

## Images

| Service | What it is |
|---|---|
| `operator` | README 1.1 conda env `tv`: `python=3.10 pinocchio=3.1.0 numpy=1.26.4 tk` (conda-forge), then `pip install -e` teleimager / televuer / dex-retargeting, then `requirements.txt`. Plus `livekit-portal` built from git `4fb4385` (PyPI wheels are Python 3.12 only), `livekit-api`, and `params-proto==2.13.2` so `vuer==0.0.60` still imports. No Unitree SDK. Tkinter for `--custom_mapping` GUI. |
| `robot` | Python 3.12 (`python:3.12-slim-trixie`, glibc 2.41). README 1.2 `unitree_sdk2_python` at commit `65691c8` + CycloneDDS from source (`CYCLONEDDS_HOME`). Portal FFI copied from `neox/portal-robot:lab-jetson` (Jetson MMAPI). DDS controllers. |
| `mock` | LiveKit echo robot. Prints latest action at 1 Hz. No SDK, no Pinocchio. |

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

## G1 hardware encode

Hardware accept (G1 agent): follow [`HW_ENCODE_G1_PROTOCOL.md`](HW_ENCODE_G1_PROTOCOL.md). Primary check is `portal_hw_encode_check.py` (no DDS). Device presence is not encode.

H.264 is encoded in the **livekit-portal FFI / Jetson MMAPI**, not in Python. Python still sends RGB via `send_video_frame`. Only the **robot** service gets the encoder; mock and operator stay software.

**Image:** `python:3.12-slim-trixie` (glibc 2.41; there is no official `*-noble` Python tag). The MMAPI `.so` is copied from local `neox/portal-robot:lab-jetson`. Do not copy it onto bookworm (glibc 2.36) and do not use `:lab-jetson-v` / PyPI. The Dockerfile asserts the FFI contains `nvhost-msenc` or `Using Jetson`.

**Host:** `nvidia-container-toolkit` and Docker runtime `nvidia`. Build the robot image **on the G1 (aarch64)**. Submodules must be initialized (`git submodule update --init --depth 1`). `lab-jetson` must exist locally (`docker images neox/portal-robot:lab-jetson`).

Compose fields on `robot` only: `runtime: nvidia`, `group_add` GIDs `44/103/994` (host `video`/`render`/`debug`; names fail in Debian slim), devices `nvhost-msenc` / `nvmap` / `nvhost-ctrl` / `nvhost-vic`, curated Tegra libs under `/opt/tegra-libs`, `SAG_REQUIRE_HW_ENCODE=1`, `RUST_LOG=info`. No `privileged: true` by default. Do not add `/dev/v4l2-nvenc` in this file (missing on JetPack 5; compose would fail).

Startup (`PortalRobotTransport`) still probes **device presence** before connect (`encode_capability backend=…`). With `SAG_REQUIRE_HW_ENCODE=1` (robot Compose), the **first RGB `send_video_frame`s** must also prove Jetson MMAPI within `SAG_HW_ENCODE_CONFIRM_S` (default 15s): `/proc/self/fd` on `nvhost-msenc` **and** FFI log `Using Jetson MMAPI encoder for H264`. `[OpenH264]` without that log exits the process (`encode_unavailable`). `--no-img` never sends frames, so this gate does not run. Mock/operator do not set the env.

`teleop_robot.py` Enter Debug Mode talks DDS. Do not start the default robot CMD while another stack owns the G1.

```bash
# build only robot (does not recreate mock/operator containers, does not start DDS)
docker compose -f docker/compose.yml build robot
```

**Accept in the container** (override CMD; do not run `teleop_robot.py` next to NeoX/SAG):

```bash
ls -l /dev/nvhost-msenc
echo $NVIDIA_DRIVER_CAPABILITIES   # must include video
echo $SAG_REQUIRE_HW_ENCODE        # 1
# RGB publish + encoder FD (separate LiveKit room, no DDS):
docker compose -f docker/compose.yml run --rm --no-deps robot python portal_hw_encode_check.py
```

Default `teleop_robot.py` in this image uses the same fail-closed DoD once video frames start. PASS only when `/proc/self/fd` points at `nvhost-msenc` **and** the FFI logs `Using Jetson MMAPI encoder for H264`. The startup capability probe (device exists) is not enough (`[OpenH264]` is the software fallback).

If logs show `Could not get EGL display connection` / `bBlitMode is set to TRUE`, NVENC is still used but RGB→NVMM is a **CPU blit**. That plus 1280×720 at 30 fps (and the host `teleimager` / SAG `videoconvert` pipelines) explains high CPU even with HW encode.

**Accept on the host during a real stream:** `sudo tegrastats` → ENC/NVENC not 0.

**Fallback:** if ioctl on msenc fails, try `privileged: true` once on `robot` only. On JetPack 6, add this device line locally:

```yaml
- /dev/v4l2-nvenc:/dev/v4l2-nvenc
```

Offline probe tests (no Docker): `python -m unittest teleop.utils.test_portal_hw_encode`.

## TLS (TeleVuer :8012)

The operator entrypoint generates a Pico/Quest-style self-signed cert in `/certs` if none is present (`XR_TELEOP_CERT` / `XR_TELEOP_KEY`). For Apple Vision Pro, generate SAN certs on the host (README 1.1.2) and bind-mount them over `/certs`.
