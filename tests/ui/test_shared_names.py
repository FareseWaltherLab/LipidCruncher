"""
Samples sharing a name are told apart by their upload label (D4).

A name shared by several samples is shown with a label (``QC (s9)``). The
label used to be the sample's label in the space being shown, so one
sample's text changed with every renumbering (sidebar exclusion, regroup),
and PCA exclusion froze the pre-exclusion text instead. Now the label is the
sample's position in the uploaded file, and a name only carries it while
another sample in the same data shares it — the same rule everywhere.
"""

from tests.ui.test_sample_identity import Scenario

# Generic sample: WT s1-s4, ADGAT-DKO s5-s8, BQC s9-s12.
SHARED = {
    's1': 'X', 's2': 'X',
    **{f's{i}': f'N{i}' for i in range(3, 9)},
    **{f's{i}': 'QC' for i in range(9, 13)},
}


def _expected(upload_label, kept):
    """Text for an uploaded sample when only ``kept`` samples are present."""
    name = SHARED[upload_label]
    shared = sum(SHARED[s] == name for s in kept) > 1
    return f'{name} ({upload_label})' if shared else name


def _scenario(captured):
    sc = Scenario(captured)
    sc.at.session_state['sample_names'] = dict(SHARED)
    sc.at.run()
    sc._ok()
    return sc


def _module1_headers(sc):
    out = sc.csv('download_normalized_data')
    return {
        sc.origin_of(c, out): c[len('concentration['):-1]
        for c in out.columns if c.startswith('concentration[')
    }


class TestSharedNames:

    def test_module1_suffix_survives_sidebar_exclusion(self, captured_downloads):
        """Regression: excluding s3 renumbered the BQC samples s9-s12 to
        s8-s11, and their text became "QC (s8)" .. "QC (s11)"."""
        sc = _scenario(captured_downloads)
        sc.exclude_in_sidebar('s3')
        assert _module1_headers(sc) == {s: _expected(s, sc.kept()) for s in sc.kept()}

    def test_pca_exclusion_matches_sidebar_exclusion(self, captured_downloads):
        """Regression: excluding one "X" in the sidebar left the other as
        "X"; excluding it in PCA left "X (s2)"."""
        sidebar = _scenario(captured_downloads)
        sidebar.exclude_in_sidebar('s1')
        assert _module1_headers(sidebar)['s2'] == 'X'

        captured_downloads.clear()
        pca = _scenario(captured_downloads)
        pca.next()
        pca.exclude_in_pca('s1')
        out = pca.csv('pca_csv_download')
        assert out['Sample'].tolist() == [_expected(s, pca.kept()) for s in pca.kept()]
        assert 'X' in out['Sample'].tolist()

    def test_regroup_keeps_the_upload_label(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.regroup({
            'WT': ['s9', 's2', 's3', 's4'],
            'ADGAT-DKO': ['s5', 's6', 's7', 's8'],
            'BQC': ['s1', 's10', 's11', 's12'],
        })
        assert _module1_headers(sc) == {s: _expected(s, sc.kept()) for s in sc.kept()}
