"""Dex3 custom pose bindings: YAML, combos, ramp. No Tk, no Portal."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import yaml

LEFT_JOINTS: Tuple[str, ...] = (
    "left_thumb_mcp",
    "left_thumb_aux",
    "left_thumb_pip",
    "left_index_mcp",
    "left_index_pip",
    "left_middle_mcp",
    "left_middle_pip",
)
RIGHT_JOINTS: Tuple[str, ...] = (
    "right_thumb_mcp",
    "right_thumb_aux",
    "right_thumb_pip",
    "right_index_mcp",
    "right_index_pip",
    "right_middle_mcp",
    "right_middle_pip",
)
JOINT_ORDER: Tuple[str, ...] = LEFT_JOINTS + RIGHT_JOINTS

# Soft URDF limits (rad), keyed by portal_mapping joint names.
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "left_thumb_mcp": (-1.04719755, 1.04719755),
    "left_thumb_aux": (-0.72431163, 0.920),
    "left_thumb_pip": (0.0, 1.74532925),
    "left_index_mcp": (-1.57079632, 0.0),
    "left_index_pip": (-1.74532925, 0.0),
    "left_middle_mcp": (-1.57079632, 0.0),
    "left_middle_pip": (-1.74532925, 0.0),
    "right_thumb_mcp": (-1.04719755, 1.04719755),
    "right_thumb_aux": (-0.920, 0.72431163),
    "right_thumb_pip": (-1.74532925, 0.0),
    "right_index_mcp": (0.0, 1.57079632),
    "right_index_pip": (0.0, 1.74532925),
    "right_middle_mcp": (0.0, 1.57079632),
    "right_middle_pip": (0.0, 1.74532925),
}

DISPLAY_TO_INTERNAL: Dict[str, str] = {
    "L1": "L_trigger",
    "L2": "L_squeeze",
    "X": "L_A",
    "Y": "L_B",
    "R1": "R_trigger",
    "R2": "R_squeeze",
    "B": "R_B",
}
INTERNAL_TO_DISPLAY: Dict[str, str] = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}
COMBO_DISPLAY_ORDER: Tuple[str, ...] = ("L1", "L2", "X", "Y", "R1", "R2", "B")

INTERNAL_TO_TELEDATA: Dict[str, str] = {
    "L_trigger": "left_ctrl_trigger",
    "L_squeeze": "left_ctrl_squeeze",
    "L_A": "left_ctrl_aButton",
    "L_B": "left_ctrl_bButton",
    "R_trigger": "right_ctrl_trigger",
    "R_squeeze": "right_ctrl_squeeze",
    "R_B": "right_ctrl_bButton",
}

MAX_JOINT_DELTA = 1.74532925


def zero_hand_dicts() -> Tuple[Dict[str, float], Dict[str, float]]:
    return {n: 0.0 for n in LEFT_JOINTS}, {n: 0.0 for n in RIGHT_JOINTS}


def hand_dict_to_q(
    left: Optional[Dict[str, Any]] = None,
    right: Optional[Dict[str, Any]] = None,
    default_q: Optional[Sequence[float]] = None,
) -> List[float]:
    q = list(default_q) if default_q is not None else [0.0] * 14
    if len(q) != 14:
        raise ValueError(f"default_q length {len(q)} != 14")
    if left:
        unknown = set(left) - set(LEFT_JOINTS)
        if unknown:
            raise ValueError(f"unknown left_hand joints: {sorted(unknown)}")
        for i, name in enumerate(LEFT_JOINTS):
            if name in left:
                q[i] = float(left[name])
    if right:
        unknown = set(right) - set(RIGHT_JOINTS)
        if unknown:
            raise ValueError(f"unknown right_hand joints: {sorted(unknown)}")
        for i, name in enumerate(RIGHT_JOINTS):
            if name in right:
                q[7 + i] = float(right[name])
    return q


def q_to_hand_dict(q: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if len(q) != 14:
        raise ValueError(f"q length {len(q)} != 14")
    left = {name: float(q[i]) for i, name in enumerate(LEFT_JOINTS)}
    right = {name: float(q[7 + i]) for i, name in enumerate(RIGHT_JOINTS)}
    return left, right


def parse_combo(combo: Any) -> FrozenSet[str]:
    if isinstance(combo, str):
        parts = [p.strip() for p in combo.replace("+", ",").split(",") if p.strip()]
    elif isinstance(combo, Iterable):
        parts = [str(p).strip() for p in combo if str(p).strip()]
    else:
        raise ValueError(f"invalid combo: {combo!r}")
    ids: List[str] = []
    for part in parts:
        if part in DISPLAY_TO_INTERNAL:
            ids.append(DISPLAY_TO_INTERNAL[part])
        elif part in INTERNAL_TO_DISPLAY:
            ids.append(part)
        else:
            raise ValueError(f"unknown button {part!r}")
    if not ids:
        raise ValueError("combo must contain at least one button")
    return frozenset(ids)


def combo_to_display_list(combo: FrozenSet[str]) -> List[str]:
    displays = {INTERNAL_TO_DISPLAY[i] for i in combo}
    return [d for d in COMBO_DISPLAY_ORDER if d in displays]


def pressed_from_tele_data(tele_data: Any) -> FrozenSet[str]:
    pressed = []
    for internal, attr in INTERNAL_TO_TELEDATA.items():
        if bool(getattr(tele_data, attr, False)):
            pressed.append(internal)
    return frozenset(pressed)


def resolve_mapping_mode(custom_mapping: bool, yaml_exists: bool) -> str:
    """Return 'xy', 'load', or 'gui'."""
    if not custom_mapping:
        return "xy"
    if yaml_exists:
        return "load"
    return "gui"


def needs_gui(path: str) -> bool:
    return not os.path.isfile(path)


@dataclass
class Binding:
    combo: FrozenSet[str]
    q: List[float]
    has_left: bool = True
    has_right: bool = True


def binding_side_tag(binding: Binding) -> str:
    if binding.has_left and binding.has_right:
        return "LR"
    if binding.has_left:
        return "L"
    return "R"


def binding_label(binding: Binding) -> str:
    combo = "+".join(combo_to_display_list(binding.combo))
    return f"{combo} ({binding_side_tag(binding)})"


@dataclass
class PoseSpec:
    duration_s: float = 1.5
    default_q: List[float] = field(default_factory=lambda: [0.0] * 14)
    bindings: List[Binding] = field(default_factory=list)


def match_binding(
    pressed: FrozenSet[str],
    bindings: Sequence[Binding],
    side: Optional[str] = None,
) -> Optional[Binding]:
    """Longest combo subset of pressed. side='left'|'right' filters has_left/has_right."""
    if side not in (None, "left", "right"):
        raise ValueError(f"invalid side {side!r}")
    best: Optional[Binding] = None
    best_len = 0
    for binding in bindings:
        if side == "left" and not binding.has_left:
            continue
        if side == "right" and not binding.has_right:
            continue
        if binding.combo <= pressed and len(binding.combo) > best_len:
            best = binding
            best_len = len(binding.combo)
    return best


def target_q(pressed: FrozenSet[str], spec: PoseSpec) -> List[float]:
    q = list(spec.default_q)
    left = match_binding(pressed, spec.bindings, side="left")
    right = match_binding(pressed, spec.bindings, side="right")
    if left is not None:
        q[:7] = left.q[:7]
    if right is not None:
        q[7:] = right.q[7:]
    return q


def ramp_towards(current: Sequence[float], target: Sequence[float], step: float) -> List[float]:
    if len(current) != len(target):
        raise ValueError("current and target length mismatch")
    out: List[float] = []
    for c, t in zip(current, target):
        delta = t - c
        if delta > step:
            out.append(c + step)
        elif delta < -step:
            out.append(c - step)
        else:
            out.append(t)
    return out


def ramp_step(duration_s: float, frequency: float, max_delta: float = MAX_JOINT_DELTA) -> float:
    denom = duration_s * frequency
    if denom <= 0:
        raise ValueError("duration_s * frequency must be > 0")
    return max_delta / denom


def _clip_q(q: Sequence[float]) -> List[float]:
    clipped: List[float] = []
    for i, name in enumerate(JOINT_ORDER):
        lo, hi = JOINT_LIMITS[name]
        clipped.append(min(hi, max(lo, float(q[i]))))
    return clipped


def _hand_block_from_q(q: Sequence[float], side: str, include: bool) -> Optional[Dict[str, float]]:
    if not include:
        return None
    left, right = q_to_hand_dict(q)
    return left if side == "left" else right


def _ordered_hand(d: Dict[str, float], names: Sequence[str]) -> Dict[str, float]:
    return {n: float(d[n]) for n in names}


def spec_to_dict(spec: PoseSpec) -> Dict[str, Any]:
    def_l, def_r = q_to_hand_dict(spec.default_q)
    payload: Dict[str, Any] = {
        "duration_s": float(spec.duration_s),
        "default": {
            "left_hand": _ordered_hand(def_l, LEFT_JOINTS),
            "right_hand": _ordered_hand(def_r, RIGHT_JOINTS),
        },
        "bindings": [],
    }
    for binding in spec.bindings:
        item: Dict[str, Any] = {"combo": combo_to_display_list(binding.combo)}
        left_block = _hand_block_from_q(binding.q, "left", binding.has_left)
        right_block = _hand_block_from_q(binding.q, "right", binding.has_right)
        if left_block is not None:
            item["left_hand"] = _ordered_hand(left_block, LEFT_JOINTS)
        if right_block is not None:
            item["right_hand"] = _ordered_hand(right_block, RIGHT_JOINTS)
        payload["bindings"].append(item)
    return payload


def spec_from_dict(raw: Dict[str, Any]) -> PoseSpec:
    duration_s = float(raw.get("duration_s", 1.5))
    default_raw = raw.get("default") or {}
    default_q = _clip_q(hand_dict_to_q(
        default_raw.get("left_hand"),
        default_raw.get("right_hand"),
    ))
    bindings: List[Binding] = []
    for item in raw.get("bindings") or []:
        combo = parse_combo(item.get("combo"))
        has_left = "left_hand" in item and item["left_hand"] is not None
        has_right = "right_hand" in item and item["right_hand"] is not None
        if not has_left and not has_right:
            raise ValueError("binding must set left_hand and/or right_hand")
        q = _clip_q(hand_dict_to_q(
            item.get("left_hand") if has_left else None,
            item.get("right_hand") if has_right else None,
            default_q=default_q,
        ))
        bindings.append(Binding(combo=combo, q=q, has_left=has_left, has_right=has_right))
    return PoseSpec(duration_s=duration_s, default_q=default_q, bindings=bindings)


def dumps(spec: PoseSpec) -> str:
    return yaml.dump(spec_to_dict(spec), default_flow_style=False, sort_keys=False, allow_unicode=True)


def loads(text: str) -> PoseSpec:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError("pose yaml root must be a mapping")
    return spec_from_dict(raw)


def save(path: str, spec: PoseSpec) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(spec))


def load(path: str) -> PoseSpec:
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read())
