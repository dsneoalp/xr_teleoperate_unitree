"""Load portal_mapping.yaml and pack/unpack Portal action/state dicts.

portal.yaml is the LiveKit contract (name + dtype). This file is the teleop
binding: which of those names form the arm / hand / loco vectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

JOINT_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")


def _as_name_list(value, key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
        return list(value)
    raise ValueError(f"portal mapping '{key}' must be a string or list of strings")


@dataclass(frozen=True)
class UnpackedAction:
    arm_q: np.ndarray
    hand_q: np.ndarray
    vx: float
    vy: float
    vyaw: float
    fsm_id: int


class PortalMapping:
    def __init__(self, mapping_yaml: str, portal_yaml: str):
        with open(mapping_yaml, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        with open(portal_yaml, "r", encoding="utf-8") as f:
            wire = yaml.safe_load(f) or {}

        self.mapping_yaml = mapping_yaml
        self.portal_yaml = portal_yaml
        self.action_fields = [f["name"] for f in (wire.get("action") or [])]
        self.state_fields = [f["name"] for f in (wire.get("state") or [])]
        action_set = set(self.action_fields)
        state_set = set(self.state_fields)

        self.fsm = raw.get("fsm", "fsm_id")
        if not isinstance(self.fsm, str):
            raise ValueError("portal mapping 'fsm' must be a field name string")
        self.loco = _as_name_list(raw.get("loco", ["vx", "vy", "vyaw"]), "loco")
        if len(self.loco) != 3:
            raise ValueError("portal mapping 'loco' must have exactly 3 names (vx, vy, vyaw)")

        self.left_arm = _as_name_list(raw.get("left_arm", []), "left_arm")
        self.right_arm = _as_name_list(raw.get("right_arm", []), "right_arm")
        self.left_hand = _as_name_list(raw.get("left_hand", []), "left_hand")
        self.right_hand = _as_name_list(raw.get("right_hand", []), "right_hand")

        missing_action = []
        for name in [self.fsm, *self.loco, *self.arm_names, *self.hand_names]:
            if name not in action_set:
                missing_action.append(name)
        missing_state = []
        for name in [self.fsm, *self.arm_names, *self.hand_names]:
            if name not in state_set:
                missing_state.append(name)
        if missing_action or missing_state:
            parts = []
            if missing_action:
                parts.append(f"not in portal.yaml action: {missing_action}")
            if missing_state:
                parts.append(f"not in portal.yaml state: {missing_state}")
            raise ValueError("portal mapping vs schema: " + "; ".join(parts))

        self._arm_index = {n: i for i, n in enumerate(self.arm_names)}
        self._hand_index = {n: i for i, n in enumerate(self.hand_names)}

    @property
    def arm_names(self) -> list[str]:
        return [*self.left_arm, *self.right_arm]

    @property
    def hand_names(self) -> list[str]:
        return [*self.left_hand, *self.right_hand]

    @property
    def arm_dof(self) -> int:
        return len(self.arm_names)

    @property
    def hand_dof(self) -> int:
        return len(self.hand_names)

    def pack_action(self, arm_q, hand_q=None, vx=0.0, vy=0.0, vyaw=0.0, fsm_id=0) -> dict:
        arm_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if arm_q.size != self.arm_dof:
            raise ValueError(f"arm_q length {arm_q.size} != mapping arm_dof {self.arm_dof}")
        if hand_q is None:
            hand_q = np.zeros(self.hand_dof, dtype=np.float64)
        else:
            hand_q = np.asarray(hand_q, dtype=np.float64).reshape(-1)
            if hand_q.size != self.hand_dof:
                raise ValueError(f"hand_q length {hand_q.size} != mapping hand_dof {self.hand_dof}")
        loco = (float(vx), float(vy), float(vyaw))
        values = {}
        for name in self.action_fields:
            if name == self.fsm:
                values[name] = int(fsm_id)
            elif name in self._arm_index:
                values[name] = float(arm_q[self._arm_index[name]])
            elif name in self._hand_index:
                values[name] = float(hand_q[self._hand_index[name]])
            elif name in self.loco:
                values[name] = loco[self.loco.index(name)]
            else:
                values[name] = 0.0
        return values

    def unpack_action(self, values: dict) -> UnpackedAction:
        raw = values or {}
        arm_q = np.zeros(self.arm_dof, dtype=np.float64)
        hand_q = np.zeros(self.hand_dof, dtype=np.float64)
        for name, idx in self._arm_index.items():
            if raw.get(name) is not None:
                arm_q[idx] = float(raw[name])
        for name, idx in self._hand_index.items():
            if raw.get(name) is not None:
                hand_q[idx] = float(raw[name])
        loco = [0.0, 0.0, 0.0]
        for i, name in enumerate(self.loco):
            if raw.get(name) is not None:
                loco[i] = float(raw[name])
        fsm_raw = raw.get(self.fsm, 0)
        return UnpackedAction(
            arm_q=arm_q,
            hand_q=hand_q,
            vx=loco[0],
            vy=loco[1],
            vyaw=loco[2],
            fsm_id=int(fsm_raw or 0),
        )

    def unpack_arm_q(self, state: dict):
        """Arm q vector from a state dict, or None if any mapped arm field is missing."""
        if not state:
            return None
        vals = [state.get(n) for n in self.arm_names]
        if any(v is None for v in vals):
            return None
        return np.array([float(v) for v in vals], dtype=np.float64)

    def unpack_hand_q(self, state: dict):
        if not state or not self.hand_names:
            return None
        vals = [state.get(n) for n in self.hand_names]
        if any(v is None for v in vals):
            return None
        return np.array([float(v) for v in vals], dtype=np.float64)

    def pack_state(self, motor_q=None, arm_q=None, hand_q=None, fsm_id=0) -> dict:
        """Fill declared state fields.

        `motor_q` is body q in portal.yaml state order excluding hands and fsm
        (G1-EDU: 29 DoF, JointIndex 0..28). `arm_q` overwrites mapped arm fields.
        Missing names stay 0.
        """
        values = {}
        for name in self.state_fields:
            if name == self.fsm:
                values[name] = int(fsm_id)
            else:
                values[name] = 0.0

        if motor_q is not None:
            body_names = [n for n in self.state_fields if n not in self._hand_index and n != self.fsm]
            motor_q = np.asarray(motor_q, dtype=np.float64).reshape(-1)
            for name, q in zip(body_names, motor_q):
                values[name] = float(q)

        if arm_q is not None:
            arm_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
            for name, idx in self._arm_index.items():
                if idx < arm_q.size:
                    values[name] = float(arm_q[idx])

        if hand_q is not None:
            hand_q = np.asarray(hand_q, dtype=np.float64).reshape(-1)
            for name, idx in self._hand_index.items():
                if idx < hand_q.size:
                    values[name] = float(hand_q[idx])

        return values
