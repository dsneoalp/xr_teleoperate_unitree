"""Thread-safe slider/bindings state for the Dex3 pose GUI."""
from __future__ import annotations

import threading
from typing import Callable, List, Optional

from teleop.utils.dex3_pose_mapping import (
    Binding,
    PoseSpec,
    binding_label,
    parse_combo,
)


class Dex3PoseGUIState:
    def __init__(self, spec: Optional[PoseSpec] = None,
                 on_change: Optional[Callable[[List[float]], None]] = None) -> None:
        self.lock = threading.Lock()
        self.accepted = threading.Event()
        self.cancelled = threading.Event()
        self.on_change = on_change
        if spec is None:
            spec = PoseSpec()
        self.spec = spec
        self.current_q: List[float] = list(spec.default_q)

    def get_q(self) -> List[float]:
        with self.lock:
            return list(self.current_q)

    def set_q(self, q: List[float], notify: bool = True) -> None:
        with self.lock:
            self.current_q = list(q)
            cb = self.on_change
        if notify and cb is not None:
            cb(list(q))

    def set_joint(self, index: int, value: float, notify: bool = True) -> None:
        with self.lock:
            self.current_q[index] = float(value)
            q = list(self.current_q)
            cb = self.on_change
        if notify and cb is not None:
            cb(q)

    def bindings_labels(self) -> List[str]:
        with self.lock:
            return [binding_label(b) for b in self.spec.bindings]

    def add_or_replace_binding(
        self,
        combo_displays: List[str],
        has_left: bool = True,
        has_right: bool = True,
    ) -> str:
        if not has_left and not has_right:
            raise ValueError("binding must include left and/or right")
        combo = parse_combo(combo_displays)
        with self.lock:
            q = list(self.current_q)
            binding = Binding(
                combo=combo,
                q=q,
                has_left=has_left,
                has_right=has_right,
            )
            replaced = False
            for i, existing in enumerate(self.spec.bindings):
                if (existing.combo == combo
                        and existing.has_left == has_left
                        and existing.has_right == has_right):
                    self.spec.bindings[i] = binding
                    replaced = True
                    break
            if not replaced:
                self.spec.bindings.append(binding)
        return binding_label(binding)

    def load_binding(self, index: int) -> Optional[List[float]]:
        with self.lock:
            if index < 0 or index >= len(self.spec.bindings):
                return None
            q = list(self.spec.bindings[index].q)
            self.current_q = q
            cb = self.on_change
        if cb is not None:
            cb(q)
        return q

    def delete_binding(self, index: int) -> None:
        with self.lock:
            if 0 <= index < len(self.spec.bindings):
                del self.spec.bindings[index]

    def snapshot_spec(self) -> PoseSpec:
        with self.lock:
            bindings = [
                Binding(combo=b.combo, q=list(b.q), has_left=b.has_left, has_right=b.has_right)
                for b in self.spec.bindings
            ]
            return PoseSpec(
                duration_s=self.spec.duration_s,
                default_q=list(self.spec.default_q),
                bindings=bindings,
            )

    def accept(self) -> None:
        self.accepted.set()

    def cancel(self) -> None:
        self.cancelled.set()
