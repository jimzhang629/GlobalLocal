"""Tests for the electrode QC scoring and overlay plotting."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis.power.electrode_qc import (
    MEAN_LINE_LABEL,
    load_roi_evokeds,
    electrode_outlier_stats,
    annotate_subjects,
    plot_electrode_overlay_for_roi,
)


def data_lines(ax):
    """Lines carrying data, excluding the axhline/axvline guides."""
    return [line for line in ax.lines if len(line.get_xdata()) > 2]


def mean_line(ax):
    """The ROI-average line, found by label rather than draw order."""
    return next(line for line in ax.lines
                if str(line.get_label()).startswith(MEAN_LINE_LABEL))


N_TIMES = 200
TIMES = np.linspace(-1.0, 1.5, N_TIMES)


def make_condition_data(n_elec=20, spike_elec=3, spike_amp=8.0,
                        spike_condition='Stimulus_c75', seed=0):
    """Clean noise everywhere, plus one contact spiking at the epoch start."""
    rng = np.random.default_rng(seed)
    data = {}
    for condition in ('Stimulus_c25', 'Stimulus_c75', 'Stimulus_i25'):
        arr = rng.normal(0, 0.05, size=(n_elec, N_TIMES))
        if condition == spike_condition:
            # A transient in the first three samples only.
            arr[spike_elec, :3] += spike_amp * np.array([1.0, 0.7, 0.4])
        data[condition] = arr
    return data


def write_run_dir(tmp_path, condition_data, roi='lpfc', ch_names=None,
                  conditions_save_name='testset_19_subjects'):
    """Write .npz files in the layout a power-traces run produces."""
    roi_dir = tmp_path / roi
    roi_dir.mkdir(parents=True, exist_ok=True)
    n_elec = next(iter(condition_data.values())).shape[0]
    ch_names = ch_names or [f'E{i}' for i in range(n_elec)]
    for condition, data in condition_data.items():
        np.savez(roi_dir / f'{conditions_save_name}_{condition}_{roi}_evoked.npz',
                 data=data, times=TIMES, ch_names=np.array(ch_names))
    return str(tmp_path), ch_names


class TestLoadRoiEvokeds:
    def test_loads_every_condition_and_recovers_names(self, tmp_path):
        condition_data = make_condition_data()
        save_dir, ch_names = write_run_dir(tmp_path, condition_data)

        loaded, times, loaded_ch = load_roi_evokeds(
            save_dir, 'lpfc', 'testset_19_subjects')

        assert set(loaded) == set(condition_data)
        assert loaded_ch == ch_names
        np.testing.assert_allclose(times, TIMES)
        for condition, data in condition_data.items():
            np.testing.assert_allclose(loaded[condition], data)

    def test_works_when_pointed_at_the_roi_dir_itself(self, tmp_path):
        condition_data = make_condition_data()
        save_dir, _ = write_run_dir(tmp_path, condition_data)

        loaded, _, _ = load_roi_evokeds(os.path.join(save_dir, 'lpfc'), 'lpfc')

        assert set(loaded) == set(condition_data)

    def test_raises_when_nothing_matches(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='No .* files under'):
            load_roi_evokeds(str(tmp_path), 'lpfc')

    def test_rejects_files_from_different_runs(self, tmp_path):
        """Mismatched time axes mean separate runs; pooling them is a bug."""
        condition_data = make_condition_data()
        save_dir, ch_names = write_run_dir(tmp_path, condition_data)
        # A file with a different (longer) time vector.
        np.savez(tmp_path / 'lpfc' / 'testset_19_subjects_Stimulus_x_lpfc_evoked.npz',
                 data=np.zeros((len(ch_names), N_TIMES + 5)),
                 times=np.linspace(-1.0, 1.5, N_TIMES + 5),
                 ch_names=np.array(ch_names))

        with pytest.raises(ValueError, match='different time vector'):
            load_roi_evokeds(save_dir, 'lpfc')

    def test_rejects_files_with_different_electrodes(self, tmp_path):
        condition_data = make_condition_data()
        save_dir, ch_names = write_run_dir(tmp_path, condition_data)
        np.savez(tmp_path / 'lpfc' / 'testset_19_subjects_Stimulus_x_lpfc_evoked.npz',
                 data=np.zeros((len(ch_names), N_TIMES)), times=TIMES,
                 ch_names=np.array([f'OTHER{i}' for i in range(len(ch_names))]))

        with pytest.raises(ValueError, match='different electrode names'):
            load_roi_evokeds(save_dir, 'lpfc')


class TestElectrodeOutlierStats:
    def test_finds_the_planted_edge_outlier(self):
        condition_data = make_condition_data(spike_elec=3)
        ch_names = [f'E{i}' for i in range(20)]

        stats = electrode_outlier_stats(condition_data, TIMES, ch_names,
                                        window='edge', edge_duration=0.05)

        assert stats.iloc[0]['electrode'] == 'E3'
        assert stats.iloc[0]['robust_z'] > 10
        # Everyone else sits near the ROI median.
        assert (stats.iloc[1:]['robust_z'] < 10).all()

    def test_names_the_condition_the_artifact_lives_in(self):
        condition_data = make_condition_data(spike_elec=3,
                                             spike_condition='Stimulus_i25')
        ch_names = [f'E{i}' for i in range(20)]

        stats = electrode_outlier_stats(condition_data, TIMES, ch_names)

        assert stats.iloc[0]['worst_condition'] == 'Stimulus_i25'

    def test_post_onset_max_separates_edge_artifact_from_noisy_contact(self):
        """A pre-onset artifact must not look like a globally bad contact."""
        condition_data = make_condition_data(spike_elec=3)
        ch_names = [f'E{i}' for i in range(20)]

        stats = electrode_outlier_stats(condition_data, TIMES, ch_names,
                                        window='edge')

        row = stats.loc[stats['electrode'] == 'E3'].iloc[0]
        assert row['stat'] > 5              # huge inside the window
        assert row['post_onset_max'] < 1    # ordinary where it matters

    def test_post_onset_max_is_immune_to_a_spike_just_past_the_window(self):
        """A transient peaking past the window edge must not leak into it.

        This is why the column is 'peak after onset' and not 'peak outside the
        scoring window' -- the latter would re-report the same artifact.
        """
        rng = np.random.default_rng(1)
        n_elec = 10
        arr = rng.normal(0, 0.05, size=(n_elec, N_TIMES))
        # Spike straddling the edge window: starts inside, peaks outside it.
        edge_end = int(np.searchsorted(TIMES, TIMES[0] + 0.05))
        arr[2, edge_end - 1:edge_end + 3] += 9.0
        stats = electrode_outlier_stats(
            {'c': arr}, TIMES, [f'E{i}' for i in range(n_elec)],
            window='edge', edge_duration=0.05)

        row = stats.loc[stats['electrode'] == 'E2'].iloc[0]
        assert row['post_onset_max'] < 1

    def test_baseline_window_also_catches_it(self):
        condition_data = make_condition_data(spike_elec=3)
        ch_names = [f'E{i}' for i in range(20)]

        stats = electrode_outlier_stats(condition_data, TIMES, ch_names,
                                        window='baseline')

        assert stats.iloc[0]['electrode'] == 'E3'

    def test_explicit_window_tuple(self):
        condition_data = make_condition_data(spike_elec=3)
        ch_names = [f'E{i}' for i in range(20)]

        stats = electrode_outlier_stats(condition_data, TIMES, ch_names,
                                        window=(-1.0, -0.95))

        assert stats.iloc[0]['electrode'] == 'E3'
        assert stats.attrs['window'] == '-1.0to-0.95s'

    def test_post_stimulus_window_does_not_flag_an_edge_artifact(self):
        """Scoring after onset must not blame a contact for a pre-onset spike."""
        condition_data = make_condition_data(spike_elec=3)
        ch_names = [f'E{i}' for i in range(20)]

        stats = electrode_outlier_stats(condition_data, TIMES, ch_names,
                                        window=(0.0, 1.5))

        assert (stats['robust_z'] < 10).all()

    def test_clean_roi_flags_nobody(self):
        condition_data = make_condition_data(spike_amp=0.0)
        ch_names = [f'E{i}' for i in range(20)]

        stats = electrode_outlier_stats(condition_data, TIMES, ch_names)

        assert (stats['robust_z'] < 10).all()

    def test_identical_electrodes_do_not_all_get_flagged(self):
        """Zero MAD must not divide by zero and flag the entire ROI."""
        arr = np.ones((5, N_TIMES))
        stats = electrode_outlier_stats({'c': arr}, TIMES,
                                        [f'E{i}' for i in range(5)])

        assert (stats['robust_z'] == 0).all()

    def test_marks_ambiguous_disambiguated_names(self):
        condition_data = make_condition_data(n_elec=3, spike_elec=0)
        stats = electrode_outlier_stats(condition_data, TIMES,
                                        ['LFMI5', 'LFMI5-0', 'RAI6'])

        by_name = stats.set_index('electrode')['name_is_ambiguous']
        assert by_name['LFMI5-0']
        assert not by_name['LFMI5']
        assert not by_name['RAI6']

    def test_rejects_empty_window(self):
        condition_data = make_condition_data()
        ch_names = [f'E{i}' for i in range(20)]

        with pytest.raises(ValueError, match='selects no samples'):
            electrode_outlier_stats(condition_data, TIMES, ch_names,
                                    window=(5.0, 6.0))

    def test_rejects_mismatched_shapes(self):
        condition_data = make_condition_data()
        condition_data['Stimulus_c25'] = np.zeros((3, N_TIMES))

        with pytest.raises(ValueError, match='expected'):
            electrode_outlier_stats(condition_data, TIMES,
                                    [f'E{i}' for i in range(20)])

    def test_rejects_unknown_window_name(self):
        condition_data = make_condition_data()
        with pytest.raises(ValueError, match='window must be one of'):
            electrode_outlier_stats(condition_data, TIMES,
                                    [f'E{i}' for i in range(20)],
                                    window='sometime')


class TestAnnotateSubjects:
    def test_lists_every_subject_owning_the_bare_name(self, tmp_path):
        rois_path = tmp_path / 'rois.json'
        rois_path.write_text(
            '{"D0057": {"LFMI5": "lpfc"}, "D0063": {"LFMI5": "lpfc"}, '
            '"D0071": {"RAI6": "lpfc"}}')
        frame = pd.DataFrame({
            'electrode': ['LFMI5-0', 'RAI6', 'NOPE'],
            'stat': [1.0, 1.0, 1.0], 'robust_z': [1.0, 1.0, 1.0],
            'worst_condition': ['c', 'c', 'c'],
            'post_onset_max': [1.0, 1.0, 1.0],
            'name_is_ambiguous': [True, False, False],
        })

        out = annotate_subjects(frame, str(rois_path), roi='lpfc')

        assert out.loc[0, 'candidate_subjects'] == 'D0057|D0063'
        assert out.loc[1, 'candidate_subjects'] == 'D0071'
        assert out.loc[2, 'candidate_subjects'] == 'unknown'

    def test_roi_filter_excludes_other_rois(self, tmp_path):
        rois_path = tmp_path / 'rois.json'
        rois_path.write_text('{"D0057": {"LFMI5": "occ"}}')
        frame = pd.DataFrame({
            'electrode': ['LFMI5'], 'stat': [1.0], 'robust_z': [1.0],
            'worst_condition': ['c'], 'post_onset_max': [1.0],
            'name_is_ambiguous': [False],
        })

        out = annotate_subjects(frame, str(rois_path), roi='lpfc')

        assert out.loc[0, 'candidate_subjects'] == 'unknown'


class TestPlotElectrodeOverlay:
    def test_draws_one_line_per_electrode_plus_the_mean(self):
        condition_data = make_condition_data(n_elec=6, spike_elec=2)
        ch_names = [f'E{i}' for i in range(6)]
        stats = electrode_outlier_stats(condition_data, TIMES, ch_names)

        figures = plot_electrode_overlay_for_roi(
            condition_data, TIMES, ch_names, 'lpfc', stats=stats,
            condition='Stimulus_c75', n_label=2)

        ax = figures['Stimulus_c75'].axes[0]
        assert len(data_lines(ax)) == 6 + 1   # every electrode plus the ROI mean
        plt.close('all')

    def test_labels_the_top_outlier(self):
        condition_data = make_condition_data(n_elec=6, spike_elec=2)
        ch_names = [f'E{i}' for i in range(6)]
        stats = electrode_outlier_stats(condition_data, TIMES, ch_names)

        figures = plot_electrode_overlay_for_roi(
            condition_data, TIMES, ch_names, 'lpfc', stats=stats,
            condition='Stimulus_c75', n_label=1)

        labels = [t.get_text()
                  for t in figures['Stimulus_c75'].axes[0].get_legend().get_texts()]
        assert any(label.startswith('E2') for label in labels)
        plt.close('all')

    def test_mean_line_matches_the_data(self):
        condition_data = make_condition_data(n_elec=6, spike_elec=2)
        ch_names = [f'E{i}' for i in range(6)]

        figures = plot_electrode_overlay_for_roi(
            condition_data, TIMES, ch_names, 'lpfc', condition='Stimulus_c75')

        ax = figures['Stimulus_c75'].axes[0]
        np.testing.assert_allclose(
            mean_line(ax).get_ydata(),
            condition_data['Stimulus_c75'].mean(axis=0))
        plt.close('all')

    def test_writes_files(self, tmp_path):
        condition_data = make_condition_data(n_elec=6)
        ch_names = [f'E{i}' for i in range(6)]

        plot_electrode_overlay_for_roi(
            condition_data, TIMES, ch_names, 'lpfc',
            condition='Stimulus_c75', save_dir=str(tmp_path))

        assert (tmp_path / 'lpfc_Stimulus_c75_electrode_overlay.png').exists()
        assert (tmp_path / 'lpfc_Stimulus_c75_electrode_overlay.pdf').exists()
        plt.close('all')

    def test_all_conditions_by_default(self):
        condition_data = make_condition_data(n_elec=4)
        ch_names = [f'E{i}' for i in range(4)]

        figures = plot_electrode_overlay_for_roi(
            condition_data, TIMES, ch_names, 'lpfc')

        assert set(figures) == set(condition_data)
        plt.close('all')

    def test_unknown_condition_raises(self):
        condition_data = make_condition_data(n_elec=4)
        with pytest.raises(KeyError):
            plot_electrode_overlay_for_roi(
                condition_data, TIMES, [f'E{i}' for i in range(4)], 'lpfc',
                condition='nope')
