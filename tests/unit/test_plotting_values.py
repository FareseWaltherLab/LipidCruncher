"""Value-pinning tests for the numbers behind the analysis figures.

The existing plotter tests use constant replicates or a single fully
saturated species, where a swapped weight, a median instead of a mean or
the wrong ddof gives the same number. These use mixed data and pin values.
"""

import numpy as np
import pandas as pd
import pytest

from app.models.experiment import ExperimentConfig
from app.services.plotting.abundance_bar_chart import (
    BarChartPlotterService,
    _compute_log10_stats,
)
from app.services.plotting.saturation_plot import SaturationPlotterService

CONTROL = ExperimentConfig(n_conditions=1, conditions_list=['Control'], number_of_samples_list=[3])


class TestSaturationValues:

    DF = pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1', 'PC 18:2_20:4', 'PC 18:1_18:1'],  # (.5,.5,0) (0,0,1) (0,1,0)
        'ClassKey': ['PC', 'PC', 'PC'],
        'concentration[s1]': [100.0, 10.0, 20.0],
        'concentration[s2]': [200.0, 40.0, 20.0],
        'concentration[s3]': [300.0, 70.0, 80.0],
    })

    def test_per_sample_sfa_mufa_pufa(self):
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            self.DF, CONTROL, ['Control'], ['PC'],
        )
        fa = result.fa_data['PC']
        np.testing.assert_allclose(fa['SFA']['Control'], [50, 100, 150])
        np.testing.assert_allclose(fa['MUFA']['Control'], [70, 120, 230])
        np.testing.assert_allclose(fa['PUFA']['Control'], [10, 40, 70])

    def test_plot_means_and_sample_sds(self):
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            self.DF, CONTROL, ['Control'], ['PC'],
        )
        row = result.plot_data['PC'].iloc[0]
        assert row['SFA_AUC'] == pytest.approx(100)
        assert row['MUFA_AUC'] == pytest.approx(140)
        assert row['PUFA_AUC'] == pytest.approx(40)
        assert row['SFA_std'] == pytest.approx(50)  # ddof=1
        assert row['MUFA_std'] == pytest.approx(np.std([70, 120, 230], ddof=1))
        assert row['PUFA_std'] == pytest.approx(30)


class TestBarChartValues:

    def test_linear_and_log10_statistics(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC 32:0', 'PC 34:1'],
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [60.0, 40.0],
            'concentration[s2]': [150.0, 50.0],
            'concentration[s3]': [300.0, 100.0],
        })  # class totals per sample: 100, 200, 400
        data = BarChartPlotterService.create_mean_std_data(df, CONTROL, ['Control'], ['PC'])
        row = data.abundance_df.iloc[0]
        assert row['mean_AUC_Control'] == pytest.approx(700 / 3)
        assert row['std_AUC_Control'] == pytest.approx(152.7525, rel=1e-6)
        # Mean of the logs (S5.3), not the log of the mean (2.368).
        assert row['log10_mean_AUC_Control'] == pytest.approx(np.log10(200))
        assert row['log10_std_AUC_Control'] == pytest.approx(np.log10(2))

    def test_log10_zero_floor_is_a_tenth_of_the_smallest_value(self):
        log_mean, log_std = _compute_log10_stats(pd.Series([0.0, 100.0, 1000.0]))
        # zero -> 100 / 10 = 10, so logs are 1, 2, 3
        assert log_mean == pytest.approx(2.0)
        assert log_std == pytest.approx(1.0)
