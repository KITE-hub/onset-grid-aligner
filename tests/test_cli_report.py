from types import SimpleNamespace

import pytest

from onset_cli import build_parser
from onset_grid import GridLayout, PhaseSupport
from onset_report import render_compact_report

REMOVED_FRAGMENTS = (
    " 次点: ",
    "アタック検出範囲",
    "アタック間隔",
    "次点のグローバル位相候補",
    "優位差",
    "標準誤差",
    "残差中央値",
    "残差P95",
    "許容幅",
    "位相候補の一意性",
    "適合率",
    "固定12位置候補の診断",
)

LAYOUT = GridLayout(120.0)


def make_check(name, phase_ms, support, stability=None):
    return SimpleNamespace(
        name=name, phase_ms=phase_ms, support=support, stability=stability
    )


def make_run(
    candidates,
    grid_support=None,
    stability=None,
    checks=(),
    drift=None,
    uses_regression=False,
    alignment=None,
    decode=None,
):
    return SimpleNamespace(
        candidates=candidates,
        alignment=alignment,
        config=SimpleNamespace(layout=LAYOUT),
        analysis=SimpleNamespace(
            grid_support=grid_support,
            stability=stability,
            cross=SimpleNamespace(checks=checks) if checks else None,
            drift=drift,
            uses_regression=uses_regression,
        ),
        estimate=None,
        decode=decode,
    )


def test_compact_report_lists_every_candidate_without_notes_when_branches_agree():
    run = make_run((("fixed_grid", 3.1), ("複素領域", -71.3)))

    report = "\n".join(render_compact_report(run))

    assert "  fixed_grid: +3.10 ms\n" in report
    assert "  複素領域: -71.30 ms\n" in report
    assert "枝を" not in report


def test_compact_report_shows_reliability_per_method():
    alignment = SimpleNamespace(phase_ms=0.0, lattice_ms=LAYOUT.comb_lattice_ms)
    run = make_run(
        (("fixed_grid", -1.0), ("HFC", -1.2)),
        grid_support=PhaseSupport(660, 480, 0.12),
        stability=1.0,
        checks=(make_check("HFC", 0.0, PhaseSupport(100, 40, 0.5), 0.5),),
        alignment=alignment,
    )

    report = "\n".join(render_compact_report(run))

    assert (
        "  fixed_grid: 根拠アタック内の残差MAD 0.12 ms / 根拠アタック割合 73%"
        " / 検出アタック数 660件 / 窓単位の安定性 100%"
    ) in report
    assert (
        "  HFC: 根拠アタック内の残差MAD 0.50 ms / 根拠アタック割合 40%"
        " / 検出アタック数 100件 / 窓単位の安定性 50%"
    ) in report
    for fragment in REMOVED_FRAGMENTS:
        assert fragment not in report


def test_compact_report_marks_unavailable_values():
    alignment = SimpleNamespace(phase_ms=0.0, lattice_ms=LAYOUT.comb_lattice_ms)
    run = make_run(
        (("fixed_grid", 0.0),),
        grid_support=PhaseSupport(0, 0, None),
        alignment=alignment,
    )

    report = "\n".join(render_compact_report(run))

    assert (
        "  fixed_grid: 根拠アタック内の残差MAD 算出不可 / 根拠アタック割合 算出不可"
        " / 検出アタック数 0件 / 窓単位の安定性 算出不可"
    ) in report


def test_branch_correction_is_reported_next_to_the_candidate():
    alignment = SimpleNamespace(phase_ms=0.0, lattice_ms=LAYOUT.comb_lattice_ms)
    run = make_run(
        (("fixed_grid", 0.0), ("複素領域", -8.88)),
        grid_support=PhaseSupport(100, 90, 0.1),
        checks=(make_check("複素領域", -20.0, PhaseSupport(100, 10, 1.0)),),
        alignment=alignment,
    )

    report = "\n".join(render_compact_report(run))

    assert "  複素領域: -8.88 ms (枝を -20.83 ms 補正)\n" in report
    assert "  fixed_grid: +0.00 ms\n" in report


def test_drift_note_appears_only_when_regression_is_applied():
    drift = SimpleNamespace(slope=0.001)

    applied = "\n".join(
        render_compact_report(make_run((), drift=drift, uses_regression=True))
    )
    skipped = "\n".join(
        render_compact_report(make_run((), drift=drift, uses_regression=False))
    )

    assert "ドリフト補正: 適用 (傾き +60.00 ms/分" in applied
    assert "ドリフト補正" not in skipped


@pytest.mark.parametrize("option", ["--verbose", "--no-band-check", "--no-refine"])
def test_cli_no_longer_accepts_removed_options(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["audio.wav", option])


def test_decoder_variants_are_listed_only_for_decoded_inputs():
    plain = "\n".join(render_compact_report(make_run((("fixed_grid", 3.1),))))
    decode = SimpleNamespace(suffix=".mp3", sample_rate=44100, applied_skip=1105)
    decoded = "\n".join(
        render_compact_report(make_run((("fixed_grid", 3.1),), decode=decode))
    )

    assert "デコーダ遅延" not in plain
    assert "先頭を削らずに再生する場合: 先頭 0 サンプルを削除" in decoded
    assert "全候補に -25.06 ms を加算 / fixed_grid -21.96 ms" in decoded
    assert "デコーダ遅延(529)だけ削る場合" in decoded
    assert "<- 現在のデコード補正" in decoded
