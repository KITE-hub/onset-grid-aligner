import re

import numpy as np
import pytest
import soundfile as sf

from onset_cli import main
from verify_synthetic import expected_methods, synth

BPM = 120.0
OFFSET_MS = 23.71
TOLERANCE_MS = 3.0


def test_cli_reports_candidate_and_reliability_for_every_method(tmp_path, capsys):
    audio = synth.render("rock8", synth.constant_segments(BPM, 24.0), OFFSET_MS)
    path = tmp_path / "rock8.wav"
    sf.write(path, audio, synth.SAMPLE_RATE, subtype="PCM_24")

    exit_code = main([str(path), "--bpm", str(BPM)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "[補正候補: 方式別、集約判定なし]" in output
    assert "[信頼性の目安: 方式ごと]" in output
    for name in expected_methods():
        candidate = re.search(
            rf"^  {re.escape(name)}: ([+-]\d+\.\d+) ms", output, re.MULTILINE
        )
        assert candidate is not None, name
        assert float(candidate.group(1)) == pytest.approx(-OFFSET_MS, abs=TOLERANCE_MS)
        reliability = re.search(
            rf"^  {re.escape(name)}: 根拠アタック内の残差MAD .*"
            "根拠アタック割合 .*検出アタック数 .*窓単位の安定性 ",
            output,
            re.MULTILINE,
        )
        assert reliability is not None, name


def test_cli_reports_an_error_for_audio_without_attacks(tmp_path, capsys):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros((2 * synth.SAMPLE_RATE, 2), dtype=np.float32), synth.SAMPLE_RATE)

    exit_code = main([str(path), "--bpm", str(BPM)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "エラー" in captured.err
    assert captured.out == ""
