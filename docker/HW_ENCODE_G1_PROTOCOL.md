# G1 hardware-encode test protocol

Hand-off for an agent working **on the Unitree G1 (Jetson)**. This is not an offline unit-test task. Run on the robot host. Do not start the default `teleop_robot.py` CMD while NeoX, SAG, or another stack owns DDS.

Repo: `xr_teleoperate_unitree`  
Branch: `hw_encoder`  
Compose: `docker/compose.yml` service `robot`  
Check script: `teleop/portal_hw_encode_check.py` (WORKDIR in the image is `/app/teleop`)

---

## 1. Architectural goal

Python must **not** encode H.264. Capture stays outside the Portal FFI. The robot process only:

1. Makes the Jetson encoder visible in the container (`runtime: nvidia`, `/dev/nvhost-msenc`, Tegra libs, `NVIDIA_DRIVER_CAPABILITIES` including `video`).
2. Sends **raw RGB** into `livekit.portal.Robot.send_video_frame`.
3. Lets **livekit-portal FFI + Jetson MMAPI** produce the H.264 WebRTC track (`portal.yaml`: `codec: h264`, not pre-encoded annex-B).
4. **Fail-closed** if hardware encode is not proven. Silent OpenH264 / CPU software fallback is a product failure, not a degraded mode.

That is the encode slice of ADR-023 (encode/publish in the robot runtime, HW mandatory). This repo’s **first step** keeps teleimager ZMQ as the RGB source. It does **not** implement FrameBus capture, depth/RVL, or splitting capture into a second container. Do not expand scope to those unless the operator asks.

```
teleimager (ZMQ BGR) or check-script RGB
        → teleop_robot / PortalRobotTransport
        → send_video_frame(RGB)
        → livekit-portal FFI (Jetson MMAPI / nvhost-msenc)
        → LiveKit H.264 track
```

Device presence is not encode. OpenH264 can still run if the FFI never opens `nvhost-msenc`. The DoD after the first RGB frames is:

- `/proc/self/fd` has a symlink to `nvhost-msenc`
- FFI log contains exactly: `Using Jetson MMAPI encoder for H264`
- Log `[OpenH264]` **without** that MMAPI line → fail (`encode_unavailable`, process exit 1)

Env on `robot` only: `SAG_REQUIRE_HW_ENCODE=1`. Mock and operator stay software. `--no-img` never sends frames, so the post-frame gate does not run — do not use `--no-img` as an encode accept test.

---

## 2. Safety and out of scope

**Do**

- Prefer `docker compose … run --rm --no-deps robot python portal_hw_encode_check.py` for encode accept. It does not start DDS and joins a **throwaway LiveKit room** (`$LIVEKIT_ROOM-hwcheck-<uuid>`), not the live teleop room.
- Keep the robot in a safe pose. ESTOP reachable. No walking/teleop unless the operator explicitly asks for T4.

**Do not**

- `docker compose up robot` / default CMD `python teleop_robot.py --ee dex3 --arm G1_29` while another stack owns the G1 (Enter Debug Mode / DDS).
- `pip install livekit-portal` in the robot image (overwrites Jetson FFI; PyPI and `neox/portal-robot:lab-jetson-v` are OpenH264-only here).
- Add `/dev/v4l2-nvenc` to the committed compose file (missing on JetPack 5; compose would fail). JetPack 6: local override only, see fallback.
- Change FrameBus / depth / ADR-023 capture split as part of this test.
- Treat “msenc exists at startup” or a visible WebRTC picture as PASS.

**Report, do not silently “fix”**, unless blocked: missing `lab-jetson` image, ioctl/permission on msenc, LiveKit unreachable, wrong glibc/FFI.

---

## 3. Preconditions (record all)

Work on the **G1, aarch64**. Build the robot image there.

| Check | Command / fact | Need |
|---|---|---|
| Arch | `uname -m` | `aarch64` |
| JetPack / L4T | `cat /etc/nv_tegra_release` | present |
| Encoder device | `ls -l /dev/nvhost-msenc` | exists (char device) |
| NVIDIA Docker | `docker info` → `Runtimes:` includes `nvidia` | nvidia-container-toolkit |
| Lab FFI image | `docker images neox/portal-robot:lab-jetson` | **tag `lab-jetson`**, not `lab-jetson-v` |
| Repo | `git -C <repo> rev-parse --abbrev-ref HEAD` | `hw_encoder` (or the branch you were given) |
| Submodules | `git submodule status` | initialized |
| Secrets | `teleop/.env` | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM` |
| LiveKit | URL reachable from the G1 (`network_mode: host`) | default `ws://127.0.0.1:7880` only if the SFU is on this host |

Repo root for all compose commands. Context is the repo (`docker/compose.yml` `context: ..`).

---

## 4. Protocol

Copy evidence into the report template at the bottom. Stop at the first **FAIL** unless a listed fallback is in scope.

### T0 — Host plumbing

```bash
uname -m
cat /etc/nv_tegra_release
ls -l /dev/nvhost-msenc /dev/nvmap /dev/nvhost-ctrl /dev/nvhost-vic
getent group video render debug || true
docker info 2>/dev/null | grep -A2 -E 'Runtimes|nvidia'
docker images neox/portal-robot:lab-jetson
```

**PASS:** `aarch64`, `nvhost-msenc` exists, nvidia runtime available, `lab-jetson` image present.  
**FAIL:** missing msenc, no nvidia runtime, only `lab-jetson-v` / no lab image.

### T1 — Build robot image (no DDS, no mock/operator recreate)

```bash
cd <repo>
docker compose -f docker/compose.yml build robot
```

Build must copy FFI from `neox/portal-robot:lab-jetson` and assert the `.so` contains `nvhost-msenc` or `Using Jetson`.

**PASS:** image `xr-teleop:robot` builds.  
**FAIL:** assertion `copied FFI has no Jetson MMAPI` → wrong source image. Do not pip-install a wheel to “fix” it.

### T2 — Container env (no encode yet)

```bash
docker compose -f docker/compose.yml run --rm --no-deps robot \
  bash -lc 'ls -l /dev/nvhost-msenc; echo NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES; echo SAG_REQUIRE_HW_ENCODE=$SAG_REQUIRE_HW_ENCODE; echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH; test -f /opt/tegra-libs/libnvtvmr.so && echo nvtvmr=ok'
```

**PASS:**

- `/dev/nvhost-msenc` visible
- `NVIDIA_DRIVER_CAPABILITIES` contains `video`
- `SAG_REQUIRE_HW_ENCODE=1`
- `LD_LIBRARY_PATH=/opt/tegra-libs`
- `nvtvmr=ok`

**FAIL:** device or `video` capability missing → encoder will not open; do not proceed to T3.

### T3 — Encode DoD (primary accept; no DDS)

Requires LiveKit. Does **not** run `teleop_robot.py`. Publishes a synthetic RGB pattern.

```bash
docker compose -f docker/compose.yml run --rm --no-deps robot \
  python portal_hw_encode_check.py
```

Optional: `--duration 15 --fps 30`. Confirm timeout follows `SAG_HW_ENCODE_CONFIRM_S` (compose robot does not set it; default 15s). The check script setdefaults `SAG_REQUIRE_HW_ENCODE=1` and sets confirm timeout from `--duration`.

**PASS (exit 0)** — both lines (wording may be wrapped):

- `OK  HW encoder FD …nvhost-msenc… after N RGB frames`
- `OK  FFI log 'Using Jetson MMAPI encoder for H264'`

**FAIL (exit ≠ 0)** — any of:

- `FAIL  LIVEKIT_URL missing`
- `encode_unavailable` / `FFI logged OpenH264`
- `no Jetson MMAPI after … RGB frames`
- `published N RGB frames but HW encode was not confirmed`

Save the **full** container stdout/stderr. Startup may still print `encode_capability backend=hardware` (device probe). That line alone is **not** PASS.

Note, not FAIL: `Could not get EGL display connection` / `bBlitMode is set to TRUE` means NVENC may still be used but RGB→NVMM is a **CPU blit** (high CPU at 720p30). Record it. Encode DoD can still PASS if FD + MMAPI log are present.

### T4 — Host encoder activity (during T3 or a real stream)

On the G1 host, while frames are publishing:

```bash
sudo tegrastats
```

**PASS:** `ENC` / `NVENC` not stuck at 0 while T3 is running.  
**FAIL / weak:** ENC=0 the whole time despite T3 PASS logs → treat as **suspect**, quote tegrastats and logs; do not declare HW encode healthy.

### T5 — Optional full teleop path (only if operator confirms DDS is free)

Same DoD as T3, but RGB comes from teleimager ZMQ (`head.bgr` → RGB → `send_video_frame`). Default process exits 1 on `encode_unavailable`.

1. Confirm nothing else holds debug/DDS on the G1.
2. teleimager server reachable (default `--img-server-ip 127.0.0.1`).
3. LiveKit up.
4. Then:

```bash
docker compose -f docker/compose.yml run --rm --no-deps robot
# CMD: python teleop_robot.py --ee dex3 --arm G1_29
```

**PASS:** logs `HW encode confirmed after N frames` and `Using Jetson MMAPI encoder for H264`; process stays up; tegrastats ENC ≠ 0.  
**FAIL:** `HW encode DoD failed` / `encode_unavailable` / `[OpenH264]` without MMAPI / process exit 1.

Do **not** pass T5 with `--no-img`.

---

## 5. Fallback (only after T3 FAIL with ioctl / permission)

Try **once**, `robot` service only, local compose override — not a committed default:

```yaml
privileged: true
```

On **JetPack 6** only, local device line:

```yaml
- /dev/v4l2-nvenc:/dev/v4l2-nvenc
```

Re-run T2 then T3. Record whether fallback changed the result.

---

## 6. What “done” means

| Layer | Done when |
|---|---|
| Plumbing | nvidia runtime + msenc + Tegra + `video` + `lab-jetson` FFI in `xr-teleop:robot` |
| Fail-closed | Missing msenc or OpenH264-without-MMAPI → no silent software video |
| Encode | After RGB `send_video_frame`: FD on `nvhost-msenc` **and** MMAPI log |
| Not done | FrameBus, depth/RVL, capture in a second container, mock/operator HW encode |

Primary hardware accept is **T3 exit 0 + T4 ENC ≠ 0**. T5 is extra if DDS is free.

---

## 7. Report template (fill this)

```
Date / host hostname:
uname -m:
/etc/nv_tegra_release (first line):
git branch / short SHA:
docker images lab-jetson: yes/no (digest or ID):
nvidia runtime: yes/no
ls -l /dev/nvhost-msenc:

T0: PASS/FAIL — notes:
T1 build: PASS/FAIL — last 30 lines if fail:
T2 env: NVIDIA_DRIVER_CAPABILITIES=  SAG_REQUIRE_HW_ENCODE=  msenc in container: yes/no
T3 check script: exit code=  duration=
    MMAPI log present: yes/no
    OpenH264 log present: yes/no
    FD line (paste):
    EGL / bBlitMode lines (paste or none):
T4 tegrastats ENC/NVENC (paste 3–5 lines while publishing):
T5 teleop_robot (skipped / PASS / FAIL): reason:

Fallback privileged or v4l2-nvenc used: no / yes — result:

Verdict: HW encode proven / not proven
Blocker (if any):
```
