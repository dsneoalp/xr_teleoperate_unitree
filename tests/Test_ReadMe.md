# Tests

Alle First-Party-Tests liegen unter `tests/`. Submodule-Beispiele
(`teleop/televuer/example/`) bleiben unangetastet.

## Ausführen

Vom Repo-Root, conda-Umgebung `tv`:

```bash
conda run -n tv python -m pytest tests/
# oder ohne pytest:
conda run -n tv python -m unittest discover -s tests -v
```

LiveKit-Roundtrip (braucht `teleop/.env` mit `LIVEKIT_URL`):

```bash
python -m pytest tests/test_portal_smoke.py -v
```

Ohne LiveKit wird `TestPortalRoundtrip` übersprungen. `TestCliHelp` läuft immer.

## Was getestet wird

| Datei | Inhalt |
|-------|--------|
| `test_arm_stiffness.py` | Fade-Kurve der Arm-Steifigkeit. Kein SDK. |
| `test_dex3_pose_mapping.py` | Dex3-Combo-Mapping, YAML, Ramp, GUI-State. Kein Tk-Fenster. |
| `test_portal_mapping.py` | `portal.yaml` / `portal_mapping.yaml` pack/unpack. |
| `test_portal_unified_timestamp.py` | Ein Capture-Tick für State+Video, 1-Slot drop-oldest. |
| `test_portal_recording_sync.py` | Join `obs.timestamp_us == action.in_reply_to_ts_us`, kein Emit ohne Match, Hz-Guard (Floor 20 Hz, Ziel 30 Hz). |
| `test_portal_smoke.py` | `--help` plus optionaler LiveKit-Roundtrip gegen `teleop/portal_robot_mock.py`. |

Recording-Items tragen `timestamp_us` und `in_reply_to_ts_us`. Der Live-Pfad
zwischen Operator und Robot bleibt drop-oldest (kein zusätzlicher Queue).
