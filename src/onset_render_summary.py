from __future__ import annotations

from onset_format import signed
from onset_resolution import DelayCandidates


def render_delay_candidates(candidates: DelayCandidates) -> list[str]:
    lines = [f"  {name}: {signed(value)} ms" for name, value in candidates]
    return ["[補正候補: 方式別、集約判定なし]", *(lines or ["  なし"])]
