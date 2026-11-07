import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "batch"))

from batch_discovery import load_song_data, normalize_xml_bytes, parse_xml

SONG_XML = (
    "<mdb><music><info><ascii>Song</ascii><title_name>Title</title_name></info>"
    "<difficulty/></music></mdb>"
)
DOCTYPE_XML = '<!DOCTYPE mdb [<!ENTITY a "x">]><mdb><music/></mdb>'


def test_parse_xml_accepts_plain_documents():
    assert parse_xml(SONG_XML.encode("utf-8")).tag == "mdb"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_parse_xml_rejects_doctype_in_any_encoding(encoding):
    declaration = f'<?xml version="1.0" encoding="{encoding}"?>'
    raw = (declaration + DOCTYPE_XML).encode(encoding)

    with pytest.raises(ValueError, match="DOCTYPE"):
        parse_xml(raw)


def test_parse_xml_reports_unsupported_utf32_documents_as_value_error():
    raw = ('<?xml version="1.0" encoding="utf-32"?>' + DOCTYPE_XML).encode("utf-32")

    with pytest.raises(ValueError, match="XMLの形式が不正"):
        parse_xml(raw)


def test_parse_xml_reports_malformed_documents_as_value_error():
    with pytest.raises(ValueError, match="XMLの形式が不正"):
        parse_xml(b"<mdb><music>")


def test_load_song_data_names_the_offending_file(tmp_path):
    path = tmp_path / "music_db.xml"
    path.write_bytes(b"<mdb><music>")

    with pytest.raises(ValueError, match="music_db.xml"):
        load_song_data(path)


SHIFT_JIS_XML = (
    '<?xml version="1.0" encoding="shift-jis"?>'
    "<mdb><music id=\"1\"><info><ascii>Song</ascii>"
    "<title_name>日本語タイトル①</title_name></info>"
    "<difficulty><novice/><exh/></difficulty></music></mdb>"
)


def test_load_song_data_reads_shift_jis_documents(tmp_path):
    path = tmp_path / "music_db.xml"
    path.write_bytes(SHIFT_JIS_XML.encode("cp932"))

    assert load_song_data(path) == {"song": ("日本語タイトル①", ("NOVICE", "EXH"))}


def test_normalize_xml_bytes_leaves_native_encodings_untouched():
    raw = '<?xml version="1.0" encoding="UTF-8"?><mdb/>'.encode("utf-8")

    assert normalize_xml_bytes(raw) == raw
    assert normalize_xml_bytes(b"<mdb/>") == b"<mdb/>"


def test_normalize_xml_bytes_rejects_unknown_encodings():
    raw = b'<?xml version="1.0" encoding="no-such-codec"?><mdb/>'

    with pytest.raises(ValueError, match="未対応"):
        normalize_xml_bytes(raw)


def test_shift_jis_documents_still_reject_doctype():
    raw = (
        '<?xml version="1.0" encoding="shift-jis"?>' + DOCTYPE_XML
    ).encode("cp932")

    with pytest.raises(ValueError, match="DOCTYPE"):
        parse_xml(raw)
