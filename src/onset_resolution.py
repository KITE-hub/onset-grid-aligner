from __future__ import annotations

from dataclasses import dataclass

from onset_analysis import Analysis
from onset_band_check import BandCheck
from onset_common import phase_to_delay_ms, wrap_symmetric
from onset_grid import MIN_PHASE_MARGIN, STEPS_PER_BEAT, GridAlignment
from onset_refine_check import RefineCheck
from onset_wavelet import WAVELET_FAMILY, WAVELET_NAMES

MAX_FAMILY_SPREAD_MS = 6.0
SPECTRAL_AGREEMENT_MS = 3.0


@dataclass(frozen=True)
class DelayResolution:
    delay_ms: float | None
    status: str
    spread_ms: float | None
    family_count: int
    candidates: tuple[tuple[str, float], ...]
    excluded_candidates: tuple[str, ...] = ()


def midpoint_phase(
    phase_ms: float,
    analysis: Analysis,
    duration_ms: float,
    period_ms: float,
    use_regression: bool = True,
) -> float:
    drift = analysis.drift
    if use_regression and analysis.uses_regression and drift is not None:
        phase_ms += drift.slope * (duration_ms / 2.0 - drift.reference_ms)
    return float(wrap_symmetric(phase_ms, period_ms))


def family_center(values: list[float], period_ms: float) -> float:
    return min(
        values,
        key=lambda candidate: sum(
            abs(float(wrap_symmetric(candidate - other, period_ms)))
            for other in values
        ),
    )


def grid_branch_phase(
    phase_ms: float,
    alignment: GridAlignment | None,
    lattice_ms: float | None = None,
) -> float:
    if alignment is None:
        return phase_ms
    own_lattice_ms = getattr(alignment, "lattice_ms", None)
    if own_lattice_ms is None or own_lattice_ms <= 0.0:
        return phase_ms
    step_ms = own_lattice_ms if lattice_ms is None else lattice_ms
    return float(
        alignment.phase_ms
        + wrap_symmetric(phase_ms - alignment.phase_ms, step_ms)
    )


def resolve_delay(
    analysis: Analysis,
    alignment: GridAlignment | None,
    bands: BandCheck | None,
    refine: RefineCheck | None,
    duration_ms: float,
    period_ms: float,
) -> DelayResolution:
    candidate_rows: list[tuple[str, str, float]] = []
    excluded_candidates: list[str] = []
    family_values: dict[str, list[float]] = {}

    comb_lattice_ms = period_ms / STEPS_PER_BEAT

    def place(
        phase_ms: float,
        use_regression: bool = True,
        align_grid_branch: bool = True,
        branch_lattice_ms: float | None = None,
    ) -> float:
        if align_grid_branch:
            phase_ms = grid_branch_phase(phase_ms, alignment, branch_lattice_ms)
        midpoint = midpoint_phase(
            phase_ms, analysis, duration_ms, period_ms, use_regression
        )
        return float(wrap_symmetric(phase_to_delay_ms(midpoint), period_ms))

    def add(
        name: str,
        family: str,
        phase_ms: float,
        use_regression: bool = True,
        align_grid_branch: bool = True,
        branch_lattice_ms: float | None = None,
    ) -> None:
        delay = place(phase_ms, use_regression, align_grid_branch, branch_lattice_ms)
        candidate_rows.append((name, family, delay))
        family_values.setdefault(family, []).append(delay)

    spectral_rows: list[tuple[str, float, float]] = []

    if alignment is not None:
        add("fixed_grid", "envelope", alignment.phase_ms, align_grid_branch=False)
    add("cluster", "envelope", analysis.conclusion_ms)
    if refine is not None:
        add("refined", "envelope", refine.offset_ms)
    if analysis.cross is not None:
        for check in analysis.cross.checks:
            if check.name in WAVELET_NAMES:
                add(
                    check.name,
                    WAVELET_FAMILY,
                    check.converted_ms,
                    check.uses_regression,
                    branch_lattice_ms=comb_lattice_ms,
                )
                continue
            delay = place(
                check.converted_ms,
                check.uses_regression,
                branch_lattice_ms=comb_lattice_ms,
            )
            candidate_rows.append((check.name, "spectral", delay))
            spectral_rows.append((check.name, delay, check.phase_margin))
    if bands is not None:
        entries = [*bands.bands]
        if bands.ringing is not None:
            entries.append(bands.ringing)
        for entry in entries:
            add(entry.label, "band", entry.phase_ms, analysis.uses_regression)

    if spectral_rows:
        supporting_indices = max(
            (
                tuple(
                    index
                    for index, (_, value, _) in enumerate(spectral_rows)
                    if abs(float(wrap_symmetric(value - center, period_ms)))
                    <= SPECTRAL_AGREEMENT_MS
                )
                for _, center, _ in spectral_rows
            ),
            key=len,
        )
        has_majority = len(supporting_indices) >= 2
        for index, (name, value, margin) in enumerate(spectral_rows):
            is_low_margin_outlier = (
                margin < MIN_PHASE_MARGIN
                and (not has_majority or index not in supporting_indices)
            )
            if is_low_margin_outlier:
                excluded_candidates.append(name)
            else:
                family_values.setdefault("spectral", []).append(value)

    centers = {
        family: family_center(values, period_ms)
        for family, values in family_values.items()
    }
    candidates = tuple((name, value) for name, _, value in candidate_rows)
    excluded = tuple(excluded_candidates)
    if not centers:
        return DelayResolution(None, "unavailable", None, 0, candidates, excluded)
    if len(centers) == 1:
        center = next(iter(centers.values()))
        return DelayResolution(center, "provisional", None, 1, candidates, excluded)

    values = list(centers.values())
    spread = max(
        abs(float(wrap_symmetric(first - second, period_ms)))
        for index, first in enumerate(values)
        for second in values[index + 1 :]
    )
    center = family_center(values, period_ms)
    if spread > MAX_FAMILY_SPREAD_MS:
        return DelayResolution(
            None, "ambiguous", spread, len(centers), candidates, excluded
        )
    return DelayResolution(center, "consensus", spread, len(centers), candidates, excluded)
