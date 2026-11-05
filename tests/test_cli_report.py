from types import SimpleNamespace

import numpy as np
import pytest

from onset_cli import build_parser
from onset_grid import GridLayout
from onset_report import render_compact_report
from onset_resolution import DelayResolution

REMOVED_FRAGMENTS = (
    " 次点: ",
    "アタック検出範囲",
    "アタック間隔",
    "次点のグローバル位相候補",
)


def make_run(candidates, heads_ms=None, alignment=None, checks=(), decode=None):
    resolution = DelayResolution(
        delay_ms=-3.1,
        status="consensus",
        spread_ms=0.2,
        family_count=3,
        candidates=candidates,
    )
    heads = np.empty(0) if heads_ms is None else heads_ms
    return SimpleNamespace(
        delay_ms=-3.1,
        resolution=resolution,
        alignment=alignment,
        config=SimpleNamespace(layout=GridLayout(120.0)),
        analysis=SimpleNamespace(
            primary=SimpleNamespace(heads_ms=heads),
            cross=SimpleNamespace(checks=checks) if checks else None,
        ),
        estimate=None,
        decode=decode,
    )


def test_compact_report_lists_only_primary_candidates():
    run = make_run((("fixed_grid", 3.1), ("cluster", 3.0), ("複素領域", -71.3)))

    report = "\n".join(render_compact_report(run))

    assert "  fixed_grid: +3.10 ms" in report
    assert "  複素領域: -71.30 ms" in report
    assert "cluster" not in report


def test_compact_report_omits_runner_up_and_attack_range_lines():
    checks = (
        SimpleNamespace(
            name="複素領域",
            phase_margin=0.02,
            converted_ms=0.0,
            runner_up_converted_ms=10.0,
        ),
    )
    alignment = SimpleNamespace(
        phase_ms=1.0,
        runner_up_phase_ms=126.0,
        phase_margin=0.02,
        comb_fit=0.9,
        comb_mad_ms=0.1,
        comb_se95_ms=0.05,
    )
    run = make_run(
        (("fixed_grid", -1.0), ("複素領域", -1.2)),
        heads_ms=1.0 + 125.0 * np.arange(40),
        alignment=alignment,
        checks=checks,
    )

    report = "\n".join(render_compact_report(run))

    assert "  複素領域: -1.20 ms (位相候補の優位差 2.0%)" in report
    for fragment in REMOVED_FRAGMENTS:
        assert fragment not in report


def test_cli_no_longer_accepts_verbose_option():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["audio.wav", "--verbose"])


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
