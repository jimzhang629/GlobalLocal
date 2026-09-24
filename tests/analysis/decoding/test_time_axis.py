"""Decoding windows must be placed on the epochs' own time axis.

process_bootstrap used to pass first_time_point=-1 whatever the epochs were, so
a response-locked run on -1.5 to 1.5 s epochs had every window centre half a
second late.
"""

import numpy as np
import pytest
from ieeg.arrays.label import LabeledArray

from src.analysis.decoding.plots.accuracies import time_axis_label
from src.analysis.decoding.process_bootstrap import first_time_point
from src.analysis.utils.labeled_array_utils import set_axis_labels

SAMPLING_RATE = 256


def _roi_arrays(tmin, tmax):
    """ROI arrays labelled the way make_bootstrapped_labeled_arrays_for_roi does."""
    times = np.arange(round(tmin * SAMPLING_RATE), round(tmax * SAMPLING_RATE)) / SAMPLING_RATE
    data = {cond: np.zeros((3, 2, len(times))) for cond in ('Response_c_MI_MR', 'Response_i_MI_MR')}
    arr = LabeledArray.from_dict(data)
    set_axis_labels(arr, 1, np.array(['e1', 'e2']), 'channel')
    set_axis_labels(arr, -1, np.array([str(t) for t in times]), 'time')
    return {'lpfc': arr}


@pytest.mark.parametrize('tmin,tmax', [(-1.0, 1.5), (-1.5, 1.5)])
def test_first_time_point_is_read_from_the_epochs(tmin, tmax):
    assert first_time_point(_roi_arrays(tmin, tmax)) == tmin


@pytest.mark.parametrize('root,event', [
    ('Stimulus_-1.0to1.5sec_decFactor_8_outliers_10', 'stimulus'),
    ('Response_-1.5to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8', 'response'),
    (None, 'stimulus'),
])
def test_time_axis_label_follows_the_epochs_root(root, event):
    assert time_axis_label(root) == f'Time from {event} onset (s)'
