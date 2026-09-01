"""Tests for the power_traces -> `anova_labels.csv` converter.

Two contracts matter here and both are asserted end-to-end rather than by
inspecting column names:

1. The written table is read by the SAME loader the A1 conjunction feeds
   (`anova_label_selection.load_anova_label_electrodes`), which is what the
   power-trace and decoding launchers call.
2. Its electrode sets are the ones the brain plots draw --
   `plot_sig_electrodes_dcc.get_condition_electrodes` on the equivalent
   `condition_plot_specs.power_trace_set` spec. Those two now share
   `power_traces.power_trace_electrode_set`, and the test pins the agreement.
"""

import json

import pandas as pd
import pytest

from dcc_scripts.stats import make_power_traces_anova_labels as mk
from src.analysis.power.windowed_anova import power_trace_electrode_set
from src.analysis.utils.anova_label_selection import (
    load_anova_label_electrodes, selected_pairs,
)

CONGRUENCY = 'C(congruency)'
SWITCH_TYPE = 'C(switchType)'
LWPC = 'C(congruency):C(incongruentProportion)'
LWPS = 'C(switchType):C(switchProportion)'


def _write_run(tmp_path, name, effects, roi='lpfc'):
    """`effects` maps effect name -> [(subject, electrode, cluster_p), ...].

    Written the way a real run writes it: one row per electrode per effect,
    `cluster_idx == -1` and `cluster_p_value == 1.0` for the electrodes that had
    no surviving cluster, so the table carries the full tested denominator.
    """
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for effect, entries in effects.items():
        for subject, electrode, p in entries:
            sig = p < 1.0
            rows.append(dict(
                subject=subject, electrode=electrode, roi=roi, effect=effect,
                cluster_idx=0 if sig else -1, sign=1 if sig else 0,
                extent_windows=5 if sig else 0,
                cluster_onset=0.1 if sig else None,
                cluster_offset=0.3 if sig else None,
                cluster_p_value=p, peak_F=5.0, peak_time=0.2,
                best_cluster_p=p, best_cluster_extent=5 if sig else 0,
                best_cluster_sign=1 if sig else 0,
                cluster_p_fdr=p, sig_after_fdr=sig))
    pd.DataFrame(rows).to_csv(run_dir / 'summary.csv', index=False)
    return run_dir


ELECS = [('S1', 'e0'), ('S1', 'e1'), ('S1', 'e2'),
         ('S2', 'e3'), ('S2', 'e4'), ('S2', 'e5')]


def _entries(sig_electrodes):
    return [(s, e, 0.001 if e in sig_electrodes else 1.0) for s, e in ELECS]


@pytest.fixture
def full_run(tmp_path):
    """One full-factorial run with a planted S/F partition over 6 electrodes.

    e0/e3 congruency only, e1/e4 switch type only, e2/e5 both -- so every
    ANOVA_LABEL_EFFECT the launchers submit has a known, non-empty answer.
    """
    return _write_run(tmp_path, 'stimulus_experiment_conditions_24_subjects', {
        CONGRUENCY: _entries({'e0', 'e2', 'e3', 'e5'}),
        SWITCH_TYPE: _entries({'e1', 'e2', 'e4', 'e5'}),
    })


def _run_converter(run_dir, out, *extra):
    mk.main(['--run', str(run_dir), '--roi', 'lpfc', '--out', str(out), *extra])
    return out / 'anova_labels.csv'


def _selected(csv, effect, **kwargs):
    kwargs.setdefault('correction', 'flags')
    return {e for _, e in selected_pairs(
        load_anova_label_electrodes(csv, effect=effect, **kwargs))}


def test_written_table_round_trips_through_the_launcher_loader(full_run, tmp_path):
    csv = _run_converter(full_run, tmp_path / 'labels')

    assert _selected(csv, 'congruency') == {'e0', 'e2', 'e3', 'e5'}
    assert _selected(csv, 'switch_type') == {'e1', 'e2', 'e4', 'e5'}
    assert _selected(csv, 'congruency_only') == {'e0', 'e3'}
    assert _selected(csv, 'switch_type_only') == {'e1', 'e4'}
    assert _selected(csv, 'both') == {'e2', 'e5'}


@pytest.mark.parametrize('effect,include,exclude', [
    ('congruency', (CONGRUENCY,), ()),
    ('switch_type', (SWITCH_TYPE,), ()),
    ('congruency_only', (CONGRUENCY,), (SWITCH_TYPE,)),
    ('switch_type_only', (SWITCH_TYPE,), (CONGRUENCY,)),
    ('both', (CONGRUENCY, SWITCH_TYPE), ()),
])
def test_labels_agree_with_the_brain_plot_electrode_sets(full_run, tmp_path,
                                                         effect, include, exclude):
    """The whole point of the converter: same electrodes as `power_trace_set`."""
    csv = _run_converter(full_run, tmp_path / 'labels')
    plotted = {e for _, e in power_trace_electrode_set(
        full_run, include_effects=include, exclude_effects=exclude, roi='lpfc')}
    assert _selected(csv, effect) == plotted


def test_denominator_is_every_tested_electrode_not_just_the_winners(full_run,
                                                                    tmp_path):
    csv = _run_converter(full_run, tmp_path / 'labels')
    table = pd.read_csv(csv)
    assert len(table) == len(ELECS)
    assert set(zip(table['subject'], table['electrode'])) == set(ELECS)


def test_roi_is_carried_so_roi_filtering_works(full_run, tmp_path):
    csv = _run_converter(full_run, tmp_path / 'labels')
    assert set(load_anova_label_electrodes(
        csv, effect='both', correction='flags', roi='lpfc')) == {'lpfc'}
    # An ROI the table does not cover selects nothing rather than silently
    # falling back to every electrode.
    assert load_anova_label_electrodes(
        csv, effect='both', correction='flags', roi='occ') == {}


def test_p_and_q_columns_reselect_the_same_electrodes(full_run, tmp_path):
    """A downstream ANOVA_LABEL_CORRECTION of none/fdr_bh must not silently
    select nothing, and must not disagree with the saved flags."""
    csv = _run_converter(full_run, tmp_path / 'labels')
    table = pd.read_csv(csv)
    for column in ('p_cong', 'q_cong', 'p_switch', 'q_switch'):
        assert table[column].notna().all(), column

    assert _selected(csv, 'congruency', correction='none', alpha=0.05) == \
        {'e0', 'e2', 'e3', 'e5'}
    assert _selected(csv, 'switch_type', correction='fdr_bh', alpha=0.05) == \
        {'e1', 'e2', 'e4', 'e5'}


def test_two_runs_supply_the_two_sides(tmp_path):
    """LWPC and LWPS live in separate two-factor runs."""
    lwpc = _write_run(tmp_path, 'stimulus_lwpc_conditions_24_subjects',
                      {LWPC: _entries({'e0', 'e2'})})
    lwps = _write_run(tmp_path, 'stimulus_lwps_conditions_24_subjects',
                      {LWPS: _entries({'e1', 'e2'})})
    out = tmp_path / 'labels'
    mk.main(['--s-run', str(lwpc), '--s-effect', LWPC,
             '--f-run', str(lwps), '--f-effect', LWPS,
             '--roi', 'lpfc', '--out', str(out)])

    csv = out / 'anova_labels.csv'
    assert _selected(csv, 'lwpc') == {'e0', 'e2'}
    assert _selected(csv, 'lwps') == {'e1', 'e2'}
    assert _selected(csv, 'lwpc_only') == {'e0'}
    assert _selected(csv, 'both') == {'e2'}


def test_significant_effects_json_is_reported_as_the_actual_threshold(full_run,
                                                                      tmp_path):
    """`load_significant_electrodes` prefers that JSON and thresholds the raw
    cluster p there, ignoring use_fdr -- so the provenance must say so."""
    (full_run / 'significant_effects_structure.json').write_text(json.dumps(
        {'S1': {'e0': {'lpfc': {'0.2': {CONGRUENCY: 0.001}}}}}))
    out = tmp_path / 'labels'
    _run_converter(full_run, out)

    meta = json.loads((out / 'source.json').read_text())
    assert 'significant_effects_structure.json' in meta['thresholded_by']
    # and the flags really did come from the JSON, not from summary.csv
    assert _selected(out / 'anova_labels.csv', 'congruency') == {'e0'}


def test_provenance_and_counts_are_recorded(full_run, tmp_path):
    out = tmp_path / 'labels'
    _run_converter(full_run, out)

    meta = json.loads((out / 'source.json').read_text())
    assert meta['source'] == 'power_traces'
    assert meta['s_effect'] == CONGRUENCY and meta['f_effect'] == SWITCH_TYPE
    assert meta['s_run'] == str(full_run)
    assert meta['counts'] == dict(S=4, F=4, both=2, S_only=2, F_only=2)


def test_default_out_dir_sits_beside_the_runs_and_names_them(tmp_path):
    out = mk.default_out_dir(
        tmp_path / 'stimulus_lwpc_conditions_24_subjects',
        tmp_path / 'stimulus_lwps_conditions_24_subjects',
        'lpfc', LWPC, LWPS, use_fdr=True, p_thresh=0.05)
    assert out.parent == tmp_path / 'power_traces_labels'
    assert out.name == ('lwpc_24_subjects__lwps_24_subjects'
                        '__S-congruency_x_incongruentProportion'
                        '__F-switchType_x_switchProportion__fdr__roi-lpfc')


def test_one_sided_invocation_is_an_error(full_run):
    with pytest.raises(SystemExit):
        mk.main(['--s-run', str(full_run)])


def test_an_effect_absent_from_the_run_names_what_is_available(tmp_path):
    run = _write_run(tmp_path, 'lwpc_only_run', {LWPC: _entries({'e0'})})
    with pytest.raises(ValueError, match=r'C\(congruency\)'):
        mk.main(['--run', str(run), '--out', str(tmp_path / 'labels')])
