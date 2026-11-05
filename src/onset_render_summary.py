from __future__ import annotations

from onset_format import signed
from onset_pipeline import AnalysisRun
from onset_resolution import DelayResolution
from onset_wavelet import WAVELET_NAMES

PRIMARY_CANDIDATES = frozenset(
    {"fixed_grid", "SuperFlux風", "複素領域", "HFC", *WAVELET_NAMES}
)


def render_delay_candidates(
    resolution: DelayResolution, run: AnalysisRun | None = None
) -> list[str]:
    checks = (
        {}
        if run is None or run.analysis.cross is None
        else {check.name: check for check in run.analysis.cross.checks}
    )
    lines = []
    for name, value in resolution.candidates:
        if name not in PRIMARY_CANDIDATES:
            continue
        check = checks.get(name)
        confidence = (
            ""
            if check is None
            else f" (位相候補の優位差 {check.phase_margin:.1%})"
        )
        lines.append(f"  {name}: {signed(value)} ms{confidence}")
    return ["[補正候補: 方式別、集約判定なし]", *(lines or ["  なし"])]
