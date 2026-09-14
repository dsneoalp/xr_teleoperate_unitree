# Tests

From the **repo root**. `pytest.ini` collects `teleop/tests`.

## What is covered

| Area | Location | What it checks |
| --- | --- | --- |
| Portal send / sync | `teleop/tests/robot_control/test_tick_slot.py`, `test_obs_record_buffer.py`, `test_portal_operator_send.py`, `test_portal_operator_frames.py` | Same `tick_ts` on state+frame; operator `in_reply_to_ts_us`; recording pairs |
| Video pacing | `teleop/tests/test_video_publish_timing.py` | `encode_ms` / `video_gap_ms` around `send_video_frame` |
| Arm stiffness | `teleop/tests/utils/test_arm_stiffness.py` | fade curve 1→0 |
| HW encode (offline) | `teleop/tests/robot_control/test_portal_hw_encode.py` | probe, OpenH264 fail-closed, `--require-hw-encode` skip, compose PC vs G1 |
| Mock CLI | `teleop/tests/test_mock_cli.py` | mock argparse |
| Episode timestamps | `teleop/tests/utils/test_episode_writer_timestamps.py`, `test_episode_hz.py` | `timestamp_us` / `action_timestamp_us` |

Mocks used in Docker E2E are **only** `teleop/tests/portal_robot_mock.py` and `teleop/tests/portal_operator_mock.py`. Do not add a third mocker.

## Host unit tests

```bash
pytest teleop/tests
```

No LiveKit, no DDS, no Jetson. HW-encode tests fake `/dev/nvhost-msenc` and FFI logs.

## Docker communication (after code changes)

Linux, host network. Copy `teleop/.env.local.example` → `teleop/.env.local` once.

```bash
./docker/run.sh build
./docker/run.sh local
```

Success: mock robot 1 Hz action echo; mock operator `echoed=True` and `obs_frames=True` (matched `tick_ts`).

One-shot:

```bash
docker compose -f docker/compose.yml -f docker/compose.local.yml run --rm operator-mock \
  python tests/portal_operator_mock.py \
  --portal-yaml portal_local.yaml --env-file .env.local \
  --expect-echo --duration 20
```

Metrics in the real loops (not asserted in pytest): `action_gap_ms`, `rtt_ms`, `loop_ms` on operator; `encode_ms` / `video_gap_ms` on robot video thread.

## G1 hardware encode (on the robot)

Not pytest. Follow [`docker/HW_ENCODE_G1_PROTOCOL.md`](../../docker/HW_ENCODE_G1_PROTOCOL.md).

```bash
./docker/run.sh build-g1
./docker/run.sh robot-g1 python tests/portal_hw_encode_check.py
# full teleop with fail-closed encode:
./docker/run.sh robot-g1 --require-hw-encode
```
