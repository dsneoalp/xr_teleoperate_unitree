"""Tkinter GUI to bind Dex3 joint poses to controller combos.

Standalone (no Portal, no robot):

    python -m teleop.utils.dex3_pose_gui --output teleop/my_poses.yaml
    python -m teleop.utils.dex3_pose_gui --output teleop/my_poses.yaml --input teleop/my_poses.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from teleop.utils.dex3_pose_gui_state import Dex3PoseGUIState
from teleop.utils.dex3_pose_mapping import (
    COMBO_DISPLAY_ORDER,
    JOINT_LIMITS,
    LEFT_JOINTS,
    PoseSpec,
    RIGHT_JOINTS,
    load,
    save,
)

JOINT_LABELS = {
    "left_thumb_mcp": "Thumb MCP",
    "left_thumb_aux": "Thumb Aux",
    "left_thumb_pip": "Thumb PIP",
    "left_index_mcp": "Index MCP",
    "left_index_pip": "Index PIP",
    "left_middle_mcp": "Middle MCP",
    "left_middle_pip": "Middle PIP",
    "right_thumb_mcp": "Thumb MCP",
    "right_thumb_aux": "Thumb Aux",
    "right_thumb_pip": "Thumb PIP",
    "right_index_mcp": "Index MCP",
    "right_index_pip": "Index PIP",
    "right_middle_mcp": "Middle MCP",
    "right_middle_pip": "Middle PIP",
}


def run_dex3_pose_gui(
    state: Dex3PoseGUIState,
    output_path: Optional[str] = None,
    stop_event=None,
) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Dex3 Pose Mapping")
    root.geometry("780x820")

    ttk.Label(
        root,
        text="Dex3 poses — bind slider configuration to a button combo",
        font=("Arial", 13, "bold"),
    ).pack(pady=(8, 2))
    ttk.Label(
        root,
        text="A = e-stop (not bindable). Left/Right independently: release one combo → that hand returns to default.",
        font=("Arial", 9),
    ).pack(pady=(0, 6))

    updating = {"value": False}
    sliders = []

    def apply_q_to_sliders(q: List[float]) -> None:
        updating["value"] = True
        try:
            for i, (_name, sl, lbl) in enumerate(sliders):
                sl.set(q[i])
                lbl.config(text=f"{q[i]:.3f}")
        finally:
            updating["value"] = False

    body = ttk.Frame(root)
    body.pack(fill=tk.BOTH, expand=True, padx=10)

    def add_hand_column(parent, title, names, offset):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        for i, name in enumerate(names):
            lo, hi = JOINT_LIMITS[name]
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=JOINT_LABELS[name], width=12, anchor="w").pack(side=tk.LEFT)
            val_lbl = ttk.Label(row, text="0.000", width=7)
            val_lbl.pack(side=tk.RIGHT)
            idx = offset + i

            def on_slide(val, index=idx, label=val_lbl):
                if updating["value"]:
                    return
                v = float(val)
                label.config(text=f"{v:.3f}")
                state.set_joint(index, v)

            sl = ttk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL, command=on_slide)
            sl.set(state.get_q()[idx])
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            sliders.append((name, sl, val_lbl))

    add_hand_column(body, "Left hand (rad)", LEFT_JOINTS, 0)
    add_hand_column(body, "Right hand (rad)", RIGHT_JOINTS, 7)

    combo_frame = ttk.LabelFrame(root, text="Button combo (AND)", padding=8)
    combo_frame.pack(fill=tk.X, padx=10, pady=8)
    combo_vars = {}
    for name in COMBO_DISPLAY_ORDER:
        var = tk.BooleanVar(value=False)
        combo_vars[name] = var
        ttk.Checkbutton(combo_frame, text=name, variable=var).pack(side=tk.LEFT, padx=6)

    side_frame = ttk.Frame(combo_frame)
    side_frame.pack(side=tk.RIGHT, padx=8)
    has_left_var = tk.BooleanVar(value=True)
    has_right_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(side_frame, text="Left", variable=has_left_var).pack(side=tk.LEFT, padx=4)
    ttk.Checkbutton(side_frame, text="Right", variable=has_right_var).pack(side=tk.LEFT, padx=4)

    list_frame = ttk.Frame(root)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
    ttk.Label(list_frame, text="Bindings").pack(anchor="w")
    listbox = tk.Listbox(list_frame, height=6)
    listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
    sb = ttk.Scrollbar(list_frame, command=listbox.yview)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=sb.set)

    def refresh_list():
        listbox.delete(0, tk.END)
        for label in state.bindings_labels():
            listbox.insert(tk.END, label)

    def on_bind():
        selected = [n for n, var in combo_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Combo", "Select at least one button.")
            return
        has_left = bool(has_left_var.get())
        has_right = bool(has_right_var.get())
        if not has_left and not has_right:
            messagebox.showwarning("Hands", "Select Left and/or Right.")
            return
        label = state.add_or_replace_binding(
            selected, has_left=has_left, has_right=has_right)
        refresh_list()
        messagebox.showinfo("Bound", f"Saved pose for {label}")

    def on_load(_evt=None):
        sel = listbox.curselection()
        if not sel:
            return
        q = state.load_binding(int(sel[0]))
        if q is not None:
            apply_q_to_sliders(q)

    def on_delete():
        sel = listbox.curselection()
        if not sel:
            return
        state.delete_binding(int(sel[0]))
        refresh_list()

    btns = ttk.Frame(root)
    btns.pack(fill=tk.X, padx=10, pady=4)
    ttk.Button(btns, text="Bind", command=on_bind).pack(side=tk.LEFT, padx=4)
    ttk.Button(btns, text="Delete selected", command=on_delete).pack(side=tk.LEFT, padx=4)
    listbox.bind("<<ListboxSelect>>", on_load)

    def finish_accept():
        if output_path:
            try:
                save(output_path, state.snapshot_spec())
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))
                return
        state.accept()
        root.quit()
        root.destroy()

    def finish_cancel():
        state.cancel()
        root.quit()
        root.destroy()

    footer = ttk.Frame(root)
    footer.pack(fill=tk.X, padx=10, pady=8)
    ttk.Button(footer, text="Accept", command=finish_accept).pack(side=tk.RIGHT)

    def poll_stop():
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            finish_cancel()
            return
        root.after(200, poll_stop)

    root.protocol("WM_DELETE_WINDOW", finish_cancel)
    refresh_list()
    apply_q_to_sliders(state.get_q())
    root.after(200, poll_stop)
    root.mainloop()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dex3 pose mapping GUI (no Portal)")
    parser.add_argument("--output", required=True, help="YAML path written on Accept")
    parser.add_argument("--input", default=None, help="Optional YAML to preload")
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--on-change", action="store_true",
                        help="Print slider q to stdout on change")
    args = parser.parse_args(argv)

    spec = PoseSpec(duration_s=args.duration)
    if args.input:
        spec = load(args.input)
        spec.duration_s = args.duration if args.duration else spec.duration_s

    def _print_q(q: List[float]) -> None:
        if args.on_change:
            print(" ".join(f"{v:.4f}" for v in q), flush=True)

    state = Dex3PoseGUIState(spec=spec, on_change=_print_q if args.on_change else None)
    if spec.default_q:
        state.set_q(spec.default_q, notify=False)
    run_dex3_pose_gui(state, output_path=args.output)
    if state.cancelled.is_set() and not state.accepted.is_set():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
