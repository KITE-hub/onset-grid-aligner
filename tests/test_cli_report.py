from types import SimpleNamespace

from onset_drift import DriftFit
from onset_grid import PhaseSupport
from onset_method import MethodEstimate
from onset_report import render_compact_report


def make_estimate(name, support=PhaseSupport(10, 5, 0.5), stability=None, slope=None):
    drift = None if slope is None else DriftFit(0.0, slope, slope, slope, 0.0)
    return MethodEstimate(name, 0.0, drift, support, stability)


def make_run(candidates, methods=(), decode=None, dominance=None):
    return SimpleNamespace(
        candidates=candidates,
        methods=methods,
        estimate=None if dominance is None else SimpleNamespace(dominance=dominance),
        decode=decode,
    )


def render(run):
    return "\n".join(render_compact_report(run))


def test_compact_report_lists_every_candidate_as_it_is():
    report = render(make_run((("envelope", 3.1), ("複素領域", -71.3))))

    assert "  envelope: +3.10 ms\n" in report
    assert "  複素領域: -71.30 ms" in report


def test_compact_report_marks_missing_candidates():
    report = render(make_run(()))

    assert "[補正候補: 方式別、集約判定なし]\n  なし" in report


def test_compact_report_shows_the_same_reliability_lines_for_every_method():
    run = make_run(
        (("envelope", -1.0), ("HFC", -1.2)),
        methods=(
            make_estimate("envelope", PhaseSupport(660, 480, 0.12), 1.0),
            make_estimate("HFC", PhaseSupport(100, 40, 0.5), 0.5),
        ),
    )

    report = render(run)

    assert (
        "  envelope: 根拠アタック内の残差MAD 0.12 ms / 根拠アタック割合 73%"
        " / 検出アタック数 660件 / 窓単位の安定性 100%"
    ) in report
    assert (
        "  HFC: 根拠アタック内の残差MAD 0.50 ms / 根拠アタック割合 40%"
        " / 検出アタック数 100件 / 窓単位の安定性 50%"
    ) in report


def test_compact_report_marks_unavailable_values():
    run = make_run(
        (("envelope", 0.0),),
        methods=(make_estimate("envelope", PhaseSupport(0, 0, None), None),),
    )

    report = render(run)

    assert (
        "  envelope: 根拠アタック内の残差MAD 算出不可 / 根拠アタック割合 算出不可"
        " / 検出アタック数 0件 / 窓単位の安定性 算出不可"
    ) in report


def test_drift_section_lists_only_the_methods_that_applied_drift():
    run = make_run(
        (("envelope", 0.0), ("HFC", 0.0)),
        methods=(
            make_estimate("envelope", slope=0.001),
            make_estimate("HFC"),
        ),
    )

    report = render(run)

    assert "[ドリフト補正: 方式ごと、各候補は音源の中央時点に換算]" in report
    assert "  envelope: 適用 (傾き +60.00 ms/分)" in report
    assert "  HFC: 適用" not in report


def test_drift_section_is_omitted_when_no_method_applied_drift():
    run = make_run((("envelope", 0.0),), methods=(make_estimate("envelope"),))

    assert "ドリフト補正" not in render(run)


def test_decoder_variants_are_listed_only_for_decoded_inputs():
    candidates = (("envelope", 3.1), ("HFC", 2.9))
    decode = SimpleNamespace(suffix=".mp3", sample_rate=44100, applied_skip=1105)

    plain = render(make_run(candidates))
    decoded = render(make_run(candidates, decode=decode))

    assert "デコーダ遅延" not in plain
    assert "先頭を削らずに再生する場合: 先頭 0 サンプルを削除" in decoded
    assert (
        "全候補に -25.06 ms を加算 / envelope -21.96 ms / HFC -22.16 ms"
    ) in decoded
    assert "デコーダ遅延(529)だけ削る場合" in decoded
    assert "<- 現在のデコード補正" in decoded


def test_unclear_bpm_estimate_is_flagged():
    candidates = (("envelope", 0.0),)

    assert "BPM自動推定が不明瞭です" in render(make_run(candidates, dominance=1.0))
    assert "BPM自動推定が不明瞭です" not in render(make_run(candidates, dominance=2.0))
