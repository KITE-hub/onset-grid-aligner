from __future__ import annotations

from collections.abc import Mapping

from onset_format import signed
from onset_resolution import DelayCandidates


def render_delay_candidates(
    candidates: DelayCandidates, branch_shifts: Mapping[str, float] | None = None
) -> list[str]:
    shifts = branch_shifts or {}
    lines = [
        f"  {name}: {signed(value)} ms" + branch_note(shifts.get(name, 0.0))
        for name, value in candidates
    ]
    return ["[補正候補: 方式別、集約判定なし]", *(lines or ["  なし"])]


def branch_note(shift_ms: float) -> str:
    if abs(shift_ms) < 0.005:
        return ""
    return f" (枝を {signed(shift_ms)} ms 補正)"
