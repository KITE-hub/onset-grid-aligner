import pytest

from onset_cli import build_parser, validate_arguments


def make_args(audio, *extra):
    return build_parser().parse_args([str(audio), *extra])


def test_validate_returns_the_resolved_path_of_an_existing_wav(tmp_path):
    audio = tmp_path / "drums.wav"
    audio.write_bytes(b"")

    assert validate_arguments(make_args(audio)) == audio.resolve()


@pytest.mark.parametrize(
    "option", ["--bpm", "--head-level", "--low-hz", "--drift-window-sec"]
)
def test_validate_rejects_nan_values(tmp_path, option):
    audio = tmp_path / "drums.wav"
    audio.write_bytes(b"")

    with pytest.raises(ValueError):
        validate_arguments(make_args(audio, option, "nan"))


def test_validate_rejects_unsupported_suffix(tmp_path):
    audio = tmp_path / "drums.s3v"
    audio.write_bytes(b"")

    with pytest.raises(ValueError):
        validate_arguments(make_args(audio))


def test_validate_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_arguments(make_args(tmp_path / "missing.wav"))
