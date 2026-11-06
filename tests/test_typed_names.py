from onset_odf import OnsetKind


def test_onset_kind_values_are_the_report_labels():
    assert [kind.value for kind in OnsetKind] == ["SuperFlux風", "複素領域", "HFC"]
