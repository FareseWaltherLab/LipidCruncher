"""Tests for SampleFrame: data plus the identity of its samples."""

import pandas as pd
import pytest
from pydantic import ValidationError

from app.models.experiment import ExperimentConfig
from app.models.sample_frame import SampleFrame


def _experiment(*counts):
    return ExperimentConfig(
        conditions_list=[f'C{i}' for i in range(len(counts))],
        number_of_samples_list=list(counts),
    )


def _df(n):
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)'],
        **{f'intensity[s{i}]': [float(i)] for i in range(1, n + 1)},
    })


@pytest.fixture
def frame():
    return SampleFrame.from_upload(
        _df(4), _experiment(2, 2), {'s1': 'A', 's2': 'B', 's4': 'D'},
    )


class TestFromUpload:

    def test_every_sample_is_its_own_origin(self, frame):
        assert frame.origin == {'s1': 's1', 's2': 's2', 's3': 's3', 's4': 's4'}

    def test_no_names(self):
        assert SampleFrame.from_upload(_df(2), _experiment(2)).names == {}


class TestRenumbered:

    def test_exclusion_moves_names_and_origin_to_the_new_labels(self, frame):
        """Excluding s1 renumbers s2..s4 to s1..s3."""
        after = frame.renumbered(
            _df(3), _experiment(1, 2),
            {'intensity[s2]': 'intensity[s1]', 'intensity[s3]': 'intensity[s2]',
             'intensity[s4]': 'intensity[s3]'},
        )
        assert after.names == {'s1': 'B', 's3': 'D'}
        assert after.origin == {'s1': 's2', 's2': 's3', 's3': 's4'}

    def test_regroup_permutes(self, frame):
        """Swapping s1 and s3 between the two conditions."""
        after = frame.renumbered(
            _df(4), _experiment(2, 2),
            {'s3': 's1', 's2': 's2', 's1': 's3', 's4': 's4'},
        )
        assert after.names == {'s2': 'B', 's3': 'A', 's4': 'D'}
        assert after.origin == {'s1': 's3', 's2': 's2', 's3': 's1', 's4': 's4'}

    def test_chained_events_keep_the_upload_label(self, frame):
        swapped = frame.renumbered(
            _df(4), _experiment(2, 2),
            {'s3': 's1', 's2': 's2', 's1': 's3', 's4': 's4'},
        )
        excluded = swapped.renumbered(
            _df(3), _experiment(1, 2),
            {'concentration[s2]': 'concentration[s1]',
             'concentration[s3]': 'concentration[s2]',
             'concentration[s4]': 'concentration[s3]'},
        )
        assert excluded.origin == {'s1': 's2', 's2': 's1', 's3': 's4'}
        assert excluded.names == {'s1': 'B', 's2': 'A', 's3': 'D'}

    def test_keeps_the_new_data_and_grouping(self, frame):
        df, experiment = _df(3), _experiment(1, 2)
        after = frame.renumbered(df, experiment, {'s2': 's1', 's3': 's2', 's4': 's3'})
        assert after.df is df
        assert after.experiment == experiment

    def test_leaves_the_original_untouched(self, frame):
        frame.renumbered(_df(3), _experiment(1, 2), {'s2': 's1', 's3': 's2', 's4': 's3'})
        assert frame.origin == {'s1': 's1', 's2': 's2', 's3': 's3', 's4': 's4'}

    def test_rejects_a_key_that_is_not_a_sample(self, frame):
        with pytest.raises(ValueError, match='LipidMolec'):
            frame.renumbered(_df(4), _experiment(2, 2), {'LipidMolec': 's1'})


class TestValidation:

    def test_origin_must_cover_the_experiment(self):
        with pytest.raises(ValidationError, match='origin'):
            SampleFrame(df=_df(2), experiment=_experiment(2), origin={'s1': 's1'})

    def test_frozen(self, frame):
        with pytest.raises(ValidationError):
            frame.names = {}
