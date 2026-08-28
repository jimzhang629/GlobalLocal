"""Re-draw decoding figures from saved MASTER_RESULTS pickles.

Every decoding job writes a ``*_MASTER_RESULTS_*.pkl`` next to its figures,
holding the pooled statistics, the significance masks and the run's arguments.
That is everything a figure needs, so the figures can be regenerated -- with
new labels, colours or layout -- without re-running any decoding.

Typical use, from a notebook::

    from src.analysis.decoding.plots.replot import find_master_results, replot_all

    runs = find_master_results(FIGS_ROOT)          # what is on disk
    runs[runs.condition_label.str.contains('lwpc')]

    replot_all(runs, save_dir=CLEAN_FIGS)          # re-draw the lot

Nothing here is specific to one analysis: every condition set the registry
gives a ``context_comparison`` re-plots the same way. So a sweep that comes
back with only LWPC figures is telling you the *other analyses have no runs on
disk*, not that they cannot be drawn -- run :func:`analysis_coverage` on the
index to see which analysis x electrode-set cells are empty before concluding
anything from the figures that did appear.
"""

import os
import pickle
import re
import traceback
from glob import glob

import numpy as np
import pandas as pd

from src.analysis.config.condition_registry import (
    get_condition_labels,
    get_context_comparison_kwargs,
    get_display_name,
    get_trace_labels,
)
from src.analysis.decoding.anova_electrode_selection import (
    short_decoding_figure_title,
)
from .accuracies import plot_accuracies_with_multiple_sig_clusters

__all__ = [
    'BLOCK_BALANCED_ANALYSES',
    'analysis_coverage',
    'describe_run',
    'find_master_results',
    'find_run_for_figure',
    'load_master_results',
    'replot_master_results',
    'replot_all',
]

# The four block-balanced decoding analyses the paper figures are built from:
# the two list-wide proportion effects and the two cross-proportion controls.
# They are the default expectation for :func:`analysis_coverage`, so a sweep
# missing three of the four says so instead of quietly re-plotting one.
BLOCK_BALANCED_ANALYSES = (
    'stimulus_lwpc_block_balanced_conditions',
    'stimulus_lwps_block_balanced_conditions',
    'stimulus_congruency_by_switch_prop_block_balanced_conditions',
    'stimulus_switch_type_by_inc_prop_block_balanced_conditions',
)

# 20260814_000636_MASTER_RESULTS_job52094028_<condition>_24_subs_<elecs>_LDA_...
_FILENAME_RE = re.compile(
    r'^(?P<timestamp>\d{8}_\d{6})_MASTER_RESULTS_(?P<params>.*)\.pkl$'
)

# Keys of master_results['stats'] that are not comparisons.
_NON_COMPARISON_STATS_KEYS = {'pooled_shuffles'}

# Stand-in electrode-set name for runs that recorded none, so they still get a
# row in the coverage table instead of vanishing from it.
_NO_ELECTRODE_SET = '(no electrode set label)'


def find_master_results(root, pattern='**/*MASTER_RESULTS*.pkl', load_metadata=True):
    """Index every MASTER_RESULTS pickle under ``root``.

    Parameters
    ----------
    root : str
        Directory to search, e.g. the decoding ``figs`` root.
    load_metadata : bool
        Read each pickle's metadata for the authoritative condition label,
        electrode set and ROI list. Slower, but the filename is only a
        convention. With False, fields are parsed from the filename alone.

    Returns
    -------
    pandas.DataFrame
        One row per pickle: ``path``, ``timestamp``, ``condition_label``,
        ``electrode_set_label``, ``rois``, ``n_electrodes``, ``n_subjects``,
        ``params``. Sort or filter it, then hand rows to :func:`replot_all`.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f'no such directory: {root}\n'
            'FIGS_ROOT should be the tree the decoding jobs wrote into -- '
            "run_decoding_dcc.py uses <dcc_scripts/decoding>/figs/<EPOCHS_ROOT_FILE>."
        )

    rows = []
    for path in sorted(glob(os.path.join(root, pattern), recursive=True)):
        match = _FILENAME_RE.match(os.path.basename(path))
        row = {
            'path': path,
            'timestamp': match.group('timestamp') if match else '',
            'params': match.group('params') if match else '',
            'slurm_job_id': '',
            'condition_label': '',
            'electrode_set_label': '',
            'rois': [],
            'n_electrodes': np.nan,
            'n_subjects': np.nan,
            'analysis_params_str': '',
            'error': '',
        }
        if load_metadata:
            try:
                master = load_master_results(path)
                meta = master.get('metadata', {})
                args = meta.get('args', {})
                row['slurm_job_id'] = str(args.get('slurm_job_id', '') or '')
                row['condition_label'] = args.get('condition_label', '')
                row['electrode_set_label'] = (
                    meta.get('electrode_set_label')
                    or meta.get('electrode_set_name')
                    or args.get('electrodes', '')
                )
                row['rois'] = sorted(_rois_in(master))
                row['n_electrodes'] = meta.get('n_electrodes', np.nan)
                row['n_subjects'] = len(args.get('subjects', []) or [])
                row['analysis_params_str'] = meta.get('analysis_params_str', '')
            except Exception as exc:  # a truncated or half-written pickle
                row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)

    return pd.DataFrame(rows, columns=[
        'path', 'timestamp', 'slurm_job_id', 'condition_label',
        'electrode_set_label', 'rois', 'n_electrodes', 'n_subjects',
        'analysis_params_str', 'params', 'error',
    ])


def load_master_results(path):
    """Unpickle one MASTER_RESULTS file."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def analysis_coverage(runs, analyses=BLOCK_BALANCED_ANALYSES,
                      electrode_sets=None, verbose=True):
    """Which analysis x electrode-set cells have runs on disk, and which do not.

    Re-plotting can only draw what a decoding job saved. When a sweep comes
    back with LWPC figures and nothing else, the question is not "why does
    re-plotting skip the other analyses" -- it does not -- but "which jobs
    never wrote a MASTER_RESULTS pickle". This answers that directly: one row
    per (analysis, electrode set), with ``n_runs = 0`` for the cells that are
    empty.

    An empty cell means the decoding job for that analysis and electrode set
    has to be re-run; there is nothing to re-plot from. A condition name that
    has drifted out of the registry is the usual cause -- the job dies at
    ``get_conditions_obj`` before it writes anything.

    Parameters
    ----------
    runs : pandas.DataFrame
        Output of :func:`find_master_results`.
    analyses : sequence of str, optional
        Condition labels that are expected to be present. Defaults to
        :data:`BLOCK_BALANCED_ANALYSES`. Pass
        ``get_condition_labels(with_context_comparison=True)`` for every
        analysis the registry can draw a comparison panel for, or an explicit
        list for a subset.
    electrode_sets : sequence of str, optional
        Electrode-set labels to expect for every analysis. Defaults to every
        electrode set that appears anywhere in ``runs`` -- so an electrode set
        used by one analysis and not the others shows up as a gap.
    verbose : bool
        Print a summary of the missing cells.

    Returns
    -------
    pandas.DataFrame
        ``condition_label``, ``electrode_set_label``, ``n_runs``,
        ``in_registry`` (whether the analysis is one the registry can draw a
        comparison panel for), sorted with the empty cells first.
    """
    if not isinstance(runs, pd.DataFrame):
        runs = find_master_results(runs) if isinstance(runs, str) else pd.DataFrame(runs)

    good = runs[runs['error'] == ''] if 'error' in runs else runs
    if len(good):
        # A run whose electrode set went unrecorded still has to be counted
        # somewhere, or it reads as a missing cell it is not.
        good = good.assign(electrode_set_label=good['electrode_set_label']
                           .fillna('').replace('', _NO_ELECTRODE_SET))
    on_disk = sorted({c for c in good['condition_label'].unique() if c}) if len(good) else []
    counts = (good.groupby(['condition_label', 'electrode_set_label']).size()
              if len(good) else pd.Series(dtype=int))

    analyses = list(analyses) if analyses is not None else list(on_disk)
    # An analysis on disk that was not asked about is still worth a row:
    # dropping it would hide exactly the runs whose condition label has drifted.
    analyses += [a for a in on_disk if a not in analyses]

    if electrode_sets is None:
        electrode_sets = sorted({e for e in good['electrode_set_label'].unique() if e}) \
            if len(good) else []
    electrode_sets = list(electrode_sets)
    if analyses and not electrode_sets:
        # Nothing on disk records an electrode set (or none was passed in).
        # Still report per analysis rather than returning an empty frame.
        electrode_sets = [_NO_ELECTRODE_SET]

    plottable = set(get_condition_labels(with_context_comparison=True))
    rows = [{
        'condition_label': analysis,
        'electrode_set_label': elecs,
        'n_runs': int(counts.get((analysis, elecs), 0)) if len(counts) else 0,
        'in_registry': analysis in plottable,
    } for analysis in analyses for elecs in electrode_sets]

    coverage = pd.DataFrame(rows, columns=[
        'condition_label', 'electrode_set_label', 'n_runs', 'in_registry'])
    if len(coverage):
        coverage = coverage.sort_values(
            ['n_runs', 'condition_label', 'electrode_set_label']
        ).reset_index(drop=True)

    if verbose:
        _print_coverage(coverage, electrode_sets)
    return coverage


def _print_coverage(coverage, electrode_sets):
    if not len(coverage):
        print('No runs on disk, so nothing to report coverage for.')
        return

    missing = coverage[coverage['n_runs'] == 0]
    print(f'{len(coverage) - len(missing)}/{len(coverage)} analysis x '
          f'electrode-set cells have at least one run '
          f'({len(electrode_sets)} electrode sets).')
    if not len(missing):
        print('Every expected analysis is on disk.')
        return

    print(f'\n{len(missing)} cells have NO saved run -- re-plotting cannot draw '
          'these, the decoding jobs have to be re-run:')
    for analysis, group in missing.groupby('condition_label', sort=True):
        note = '' if group['in_registry'].all() else \
            '   [!] not a CONDITION_REGISTRY key with a context comparison'
        print(f'  {analysis}{note}')
        for elecs in group['electrode_set_label']:
            print(f'      · {elecs}')


# Parameters worth seeing first when asking "what produced this figure?".
_KEY_PARAMS = [
    'slurm_job_id', 'timestamp', 'condition_label', 'electrodes',
    'clf_model_str', 'bootstraps', 'n_splits', 'n_repeats',
    'unit_of_analysis', 'explained_variance', 'window_size', 'step_size',
    'sampling_rate', 'percentile', 'cluster_percentile', 'n_cluster_perms',
    'p_thresh_for_time_perm_cluster_stats', 'p_cluster', 'random_state',
    'subjects',
]


def describe_run(master, all_params=False):
    """Every decoding parameter behind one results file.

    Answers "what went into this figure, and which SLURM job was it?" -- the
    figure's timestamp prefix is ``args.timestamp``, which is also the prefix
    of the MASTER_RESULTS filename, so :func:`find_run_for_figure` gets you
    here from a .png.

    Parameters
    ----------
    master : dict or str
        A loaded master_results dict, or a path to one.
    all_params : bool
        False (default) returns the parameters in ``_KEY_PARAMS`` that the run
        actually recorded; True returns everything in ``metadata['args']``.

    Returns
    -------
    pandas.Series
        Parameter name -> value, plus the electrode set and electrode count.
    """
    if isinstance(master, str):
        master = load_master_results(master)

    meta = master.get('metadata', {})
    args = dict(meta.get('args', {}))

    if all_params:
        fields = sorted(args)
    else:
        fields = [k for k in _KEY_PARAMS if k in args]

    described = {}
    for key in fields:
        value = args[key]
        # A 24-subject list is the one field that swamps the printout.
        described[key] = (f'{len(value)} subjects: {", ".join(map(str, value))}'
                          if key == 'subjects' and isinstance(value, (list, tuple))
                          else value)

    described['electrode_set_label'] = (meta.get('electrode_set_label')
                                        or meta.get('electrode_set_name') or '')
    described['n_electrodes'] = meta.get('n_electrodes', np.nan)
    described['analysis_params_str'] = meta.get('analysis_params_str', '')
    return pd.Series(described, dtype=object)


def find_run_for_figure(figure_path, root, **kwargs):
    """The results file whose run produced ``figure_path``.

    Figures are named ``<timestamp>_<comparison>_<roi>_<suffix>``, and the
    timestamp is the run's ``args.timestamp`` -- the same prefix the
    MASTER_RESULTS pickle carries. Two figures that differ only in timestamp
    came from two different runs, and this is how to tell them apart.

    Returns
    -------
    pandas.DataFrame
        Matching runs, from :func:`find_master_results`. More than one row
        means several runs shared a timestamp; compare their
        ``slurm_job_id``.
    """
    name = os.path.basename(figure_path)
    match = re.match(r'^(?P<timestamp>\d{8}_\d{6})_', name)
    if not match:
        raise ValueError(
            f'cannot read a run timestamp off {name!r}; expected a name '
            'beginning YYYYMMDD_HHMMSS_'
        )
    timestamp = match.group('timestamp')
    runs = find_master_results(root, **kwargs)
    hits = runs[runs['timestamp'] == timestamp]
    if not len(hits):
        print(f'No results file under {root} has timestamp {timestamp}. '
              'The figure may predate the pickle, or come from another tree.')
    return hits


def _rois_in(master):
    """ROIs present in a results file's stats."""
    rois = set()
    for key, per_roi in master.get('stats', {}).items():
        if key in _NON_COMPARISON_STATS_KEYS or not isinstance(per_roi, dict):
            continue
        rois.update(per_roi)
    return rois


def _comparison_keys(master):
    return [k for k in master.get('stats', {})
            if k not in _NON_COMPARISON_STATS_KEYS]


def _contrast_entries(master, roi, condition_name, colors, comp1, comp2,
                      significance_label_1, significance_label_2):
    """The two directions of the between-condition test, from stored clusters.

    Older results files key these ``'25_over_75'`` / ``'75_over_25'`` and
    newer ones ``'1_over_2'`` / ``'2_over_1'``, so take whatever two keys are
    there, in order, rather than looking for a fixed pair.
    """
    stored = (master.get('comparison_clusters', {})
              .get(roi, {})
              .get(condition_name.lower(), {}))
    if not stored:
        return {}

    labels = [significance_label_1, significance_label_2]
    bar_colors = [colors.get(comp1), colors.get(comp2)]
    # A solid and a dashed bar: the two directions cannot overlap in time, so
    # they share the top row and the linestyle tells them apart in print.
    linestyles = ['-', '--']

    entries = {}
    for i, (name, info) in enumerate(list(stored.items())[:2]):
        info = info if isinstance(info, dict) else {'clusters': info}
        entries[name] = {
            'clusters': info.get('clusters'),
            'label': info.get('label') or labels[i],
            'color': bar_colors[i] or info.get('color') or 'black',
            'linestyle': linestyles[i],
            'kind': 'contrast',
        }
    return entries


def replot_master_results(
    master,
    save_dir,
    rois=None,
    electrode_set_label=None,
    ylim=(0.3, 0.8),
    include_true_vs_shuffle=True,
    include_difference=False,
    filename_suffix=None,
    **plot_kwargs,
):
    """Re-draw every figure a results file can support.

    Produces, per ROI, the context-comparison panel (two conditions, the
    pooled shuffle, the contrast bars and each condition's own bar against
    chance) when the analysis has one, plus a true-vs-shuffle panel per
    comparison.

    Parameters
    ----------
    master : dict or str
        A loaded master_results dict, or a path to one.
    save_dir : str
        Root for the new figures; a ``<condition>/<roi>`` tree is made inside.
    rois : list, optional
        Defaults to every ROI in the file.
    electrode_set_label : str, optional
        Overrides the electrode-set name used in the title.
    filename_suffix : str, optional
        Tail of every filename. Defaults to the run's own
        ``analysis_params_str`` -- ``job52094028_<condition>_24_subs_<elecs>_
        LDA_20boots_5splits_5reps_repeat_unit_ev_0.9`` -- which is what the
        decoding job itself put in its filenames. The figure title is short
        now, so the filename is the only thing left tying a panel to the
        SLURM job and parameters that produced it: two runs of the same
        analysis otherwise differ by nothing but a timestamp. Pass a short
        string for tidier names, at the cost of that.
    include_difference : bool
        Also draw the (condition 1 - condition 2) difference panel. Off by
        default: it needs the per-sample accuracies of both conditions, which
        the pooled statistics only carry when the units line up.
    **plot_kwargs
        Passed through to the plotting function, e.g. ``base_fontsize=14``,
        ``figsize=(6, 5)``, ``chance_bar_color='black'``.

    Returns
    -------
    list of str
        Paths (without extension) of the figures written.
    """
    if isinstance(master, str):
        master = load_master_results(master)

    meta = master.get('metadata', {})
    args = meta.get('args', {})
    condition_label = args.get('condition_label', '')
    unit = args.get('unit_of_analysis', 'repeat')
    # np.asarray(None) is a size-1 array, not an empty one, so check the key
    # itself or a results file with no time axis fails much later with a
    # confusing message about mismatched significance masks.
    stored_time_points = meta.get('time_window_centers')
    time_points = (np.asarray(stored_time_points, dtype=float)
                   if stored_time_points is not None else np.array([]))
    elec_label = electrode_set_label or meta.get('electrode_set_label') or \
        meta.get('electrode_set_name') or ''
    timestamp = args.get('timestamp', '')

    if time_points.size == 0:
        raise ValueError(
            'results file has no metadata["time_window_centers"]; it was '
            'written by a run too old to re-plot'
        )

    if filename_suffix is None:
        # Keep the job id and parameters in the filename, as the decoding job
        # did: with a short title, that is all that separates two runs of the
        # same analysis on the same ROI. Results files written before
        # analysis_params_str was recorded still get the job id, so a figure
        # is never left attributable by timestamp alone.
        filename_suffix = meta.get('analysis_params_str') or ''
        if not filename_suffix and args.get('slurm_job_id'):
            filename_suffix = f"job{args['slurm_job_id']}"

    rois = list(rois) if rois is not None else sorted(_rois_in(master))
    trace_labels = get_trace_labels(condition_label)
    title_stem = short_decoding_figure_title(
        get_display_name(condition_label), elec_label)
    context = get_context_comparison_kwargs(condition_label) or {}
    # The registry already names the y-axis per analysis; only analyses with
    # no context comparison fall back to the generic label.
    ylabel = context.get('ylabel', 'Decoding accuracy')
    if not context:
        # Every analysis with a 'context_comparison' entry gets a comparison
        # panel; one without gets only the true-vs-shuffle panels. Say which
        # this is, so a thin output tree is not mistaken for a re-plot bug.
        print(f"  · {condition_label or '(no condition_label)'}: no "
              "'context_comparison' in CONDITION_REGISTRY, so no comparison "
              'panel -- true-vs-shuffle panels only')

    written = []
    for roi in rois:
        if context:
            written += _replot_context_comparison(
                master, context, roi, unit, time_points, trace_labels,
                title_stem, save_dir, timestamp, filename_suffix, ylim,
                include_difference, plot_kwargs,
            )
        if include_true_vs_shuffle:
            written += _replot_true_vs_shuffle(
                master, roi, unit, time_points, trace_labels, title_stem,
                ylabel, save_dir, timestamp, filename_suffix, plot_kwargs,
            )
    return written


def _replot_context_comparison(master, context, roi, unit, time_points,
                               trace_labels, title_stem, save_dir, timestamp,
                               filename_suffix, ylim, include_difference,
                               plot_kwargs):
    comp1 = context['condition_comparison_1']
    comp2 = context['condition_comparison_2']
    stats = master.get('stats', {})
    absent = [c for c in (comp1, comp2) if roi not in stats.get(c, {})]
    if absent:
        # The comparison panel needs both sides. Saying which one is missing
        # separates "this run decoded a different pair" from "this ROI had no
        # electrodes", which otherwise both look like a silently thinner tree.
        print(f'  · {roi}: no comparison panel, '
              f"{' and '.join(absent)} absent from stats for this ROI "
              f"(stats has: {sorted(_comparison_keys(master)) or 'nothing'})")
        return []

    colors = context['colors']
    linestyles = context['linestyles']
    condition_name = context['condition_name']
    shuffle_key = f"{context['pooled_shuffle_key']}_across_bootstraps"
    label_1 = trace_labels.get(comp1, comp1.replace('_', ' '))
    label_2 = trace_labels.get(comp2, comp2.replace('_', ' '))
    shuffle_label = 'Shuffle'

    stats_1, stats_2 = stats[comp1][roi], stats[comp2][roi]
    accuracies = {
        label_1: stats_1[f'{unit}_true_accs'],
        label_2: stats_2[f'{unit}_true_accs'],
    }
    pooled_shuffle = (stats.get('pooled_shuffles', {})
                      .get(roi, {})
                      .get(condition_name.lower()))
    if pooled_shuffle is not None:
        accuracies[shuffle_label] = pooled_shuffle

    significance = _contrast_entries(
        master, roi, condition_name, colors, comp1, comp2,
        context.get('significance_label_1', ''),
        context.get('significance_label_2', ''),
    )
    # Each condition's own test against chance, one row below the contrast.
    significance.update({
        f'{comp1}_vs_chance': {
            'clusters': stats_1.get('significant_clusters'),
            'color': colors.get(comp1),
            'kind': 'chance',
        },
        f'{comp2}_vs_chance': {
            'clusters': stats_2.get('significant_clusters'),
            'color': colors.get(comp2),
            'kind': 'chance',
        },
    })

    out_dir = os.path.join(save_dir, f'{condition_name}_comparison', roi)
    kwargs = dict(
        time_points=time_points,
        significance_clusters_dict=significance,
        colors={label_1: colors.get(comp1), label_2: colors.get(comp2),
                shuffle_label: colors.get(shuffle_key, '#949494')},
        linestyles={label_1: linestyles.get(comp1, '-'),
                    label_2: linestyles.get(comp2, '-'),
                    shuffle_label: linestyles.get(shuffle_key, '--')},
        roi=roi,
        save_dir=out_dir,
        timestamp=timestamp,
        ylabel=context.get('ylabel', 'Decoding accuracy'),
        ylim=ylim,
        show_chance_level=False,
        show_sig_legend=True,
        filename_suffix=filename_suffix,
    )
    kwargs.update(plot_kwargs)

    plot_accuracies_with_multiple_sig_clusters(
        accuracies_dict=accuracies,
        comparison_name=f'{condition_name}_comparison',
        title=title_stem,
        **kwargs,
    )
    written = [os.path.join(out_dir, '_'.join(filter(None, [
        timestamp, f'{condition_name}_comparison', roi, filename_suffix])))]

    if include_difference:
        accs_1 = np.asarray(stats_1[f'{unit}_true_accs'], dtype=float)
        accs_2 = np.asarray(stats_2[f'{unit}_true_accs'], dtype=float)
        if accs_1.shape == accs_2.shape:
            differences = accs_1 - accs_2
            spread = np.max(np.abs(np.mean(differences, axis=0))
                            + np.std(differences, axis=0))
            diff_ylim = (-spread * 1.2, spread * 1.2) if spread else (-0.1, 0.1)
            diff_label = f'{label_1} − {label_2}'
            diff_kwargs = dict(kwargs)
            diff_kwargs.update(
                colors={diff_label: '#404040'},
                linestyles={diff_label: '-'},
                ylim=diff_ylim,
                ylabel='Accuracy difference',
                show_chance_level=True,
                chance_level=0,
                filename_suffix=f'{filename_suffix}_ACC_DIFFERENCE',
            )
            # Against-chance bars belong to the individual traces, not to
            # their difference.
            diff_kwargs['significance_clusters_dict'] = {
                k: v for k, v in significance.items()
                if v.get('kind') != 'chance'
            }
            diff_kwargs.update(plot_kwargs)
            plot_accuracies_with_multiple_sig_clusters(
                accuracies_dict={diff_label: differences},
                comparison_name=f'{condition_name}_ACC_DIFFERENCE',
                title=f'{title_stem}: {label_1} − {label_2}',
                **diff_kwargs,
            )
            written.append(os.path.join(out_dir, '_'.join(filter(None, [
                timestamp, f'{condition_name}_ACC_DIFFERENCE', roi,
                f'{filename_suffix}_ACC_DIFFERENCE']))))
        else:
            print(f'  · {roi}: skipping difference panel, '
                  f'{accs_1.shape} vs {accs_2.shape} samples do not pair')

    return written


def _replot_true_vs_shuffle(master, roi, unit, time_points, trace_labels,
                            title_stem, ylabel, save_dir, timestamp,
                            filename_suffix, plot_kwargs):
    written = []
    for comparison in _comparison_keys(master):
        per_roi = master['stats'][comparison]
        if not isinstance(per_roi, dict) or roi not in per_roi:
            continue
        stats = per_roi[roi]
        if f'{unit}_true_accs' not in stats:
            continue

        true_label = trace_labels.get(comparison, comparison.replace('_', ' '))
        shuffle_label = 'Shuffle'
        out_dir = os.path.join(save_dir, comparison, roi)
        kwargs = dict(
            time_points=time_points,
            accuracies_dict={
                true_label: stats[f'{unit}_true_accs'],
                shuffle_label: stats[f'{unit}_shuffle_accs'],
            },
            significance_clusters_dict={
                'vs_shuffle': {
                    'clusters': stats.get('significant_clusters'),
                    'color': '#0173B2',
                    'kind': 'chance',
                },
            },
            colors={true_label: '#0173B2', shuffle_label: '#949494'},
            linestyles={true_label: '-', shuffle_label: '--'},
            comparison_name=f'true_vs_shuffle_{comparison}',
            roi=roi,
            save_dir=out_dir,
            timestamp=timestamp,
            ylabel=ylabel,
            ylim=(0.3, 1.0),
            show_chance_level=False,
            show_sig_legend=True,
            chance_bar_label='> shuffle',
            title=f'{title_stem}: {true_label}',
            filename_suffix=filename_suffix,
        )
        kwargs.update(plot_kwargs)
        plot_accuracies_with_multiple_sig_clusters(**kwargs)
        written.append(os.path.join(out_dir, '_'.join(filter(None, [
            timestamp, f'true_vs_shuffle_{comparison}', roi, filename_suffix]))))
    return written


def replot_all(runs, save_dir, group_by=('condition_label', 'electrode_set_label'),
               **replot_kwargs):
    """Re-plot every run in ``runs``, keeping going past individual failures.

    Parameters
    ----------
    runs : pandas.DataFrame or iterable of str
        Output of :func:`find_master_results` (or a filtered subset), or just
        a list of paths.
    save_dir : str
        Root for the new figures. Each run gets its own subdirectory named
        from ``group_by``, so two electrode sets never overwrite each other.
        Nothing here re-selects electrodes: which electrodes a run used was
        settled when the decoding job ran, and ``electrode_set_label`` just
        reports it. Filtering ``runs`` picks which saved runs to re-draw, not
        which channels go into them.
    **replot_kwargs
        Passed to :func:`replot_master_results`.

    Returns
    -------
    pandas.DataFrame
        One row per run: ``path``, ``condition_label``, ``n_figures``,
        ``error``. Check that ``error`` is empty everywhere before believing
        the output is complete, and read the per-analysis tally printed at the
        end: an analysis absent from it had no run in ``runs`` to re-plot.
    """
    if isinstance(runs, pd.DataFrame):
        records = runs.to_dict('records')
    else:
        records = [{'path': p} for p in runs]

    results = []
    for i, record in enumerate(records, 1):
        path = record['path']
        print(f'[{i}/{len(records)}] {os.path.basename(path)}')
        try:
            master = load_master_results(path)
            meta = master.get('metadata', {})
            args = meta.get('args', {})
            parts = []
            for field in group_by:
                value = record.get(field) or args.get(field) or meta.get(field)
                if value:
                    parts.append(re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)))
            run_dir = os.path.join(save_dir, *parts) if parts else save_dir

            written = replot_master_results(master, run_dir, **replot_kwargs)
            results.append({'path': path,
                            'condition_label': (record.get('condition_label')
                                                or args.get('condition_label') or ''),
                            'out_dir': run_dir,
                            'n_figures': len(written), 'error': ''})
            print(f'    → {len(written)} figures in {run_dir}')
        except Exception as exc:
            traceback.print_exc()
            results.append({'path': path,
                            'condition_label': record.get('condition_label') or '',
                            'out_dir': '', 'n_figures': 0,
                            'error': f'{type(exc).__name__}: {exc}'})

    # Always give the caller the same columns, so indexing the result of an
    # empty sweep says "no rows" rather than raising KeyError on a bare frame.
    frame = pd.DataFrame(results, columns=[
        'path', 'condition_label', 'out_dir', 'n_figures', 'error'])
    if not len(frame):
        print('\nNothing to re-plot: no runs were passed in.')
        return frame
    _print_analysis_tally(frame)
    failed = int((frame['error'] != '').sum())
    print(f'\n{len(frame) - failed}/{len(frame)} runs re-plotted'
          + (f', {failed} failed (see the error column)' if failed else ''))
    return frame


def _print_analysis_tally(frame, expected=BLOCK_BALANCED_ANALYSES):
    """Figures per analysis, and which expected analyses contributed none.

    A sweep is only as complete as the pickles it was handed. Printing the
    tally means "I re-plotted 40 figures" cannot hide the fact that all 40 were
    the same analysis.
    """
    tally = (frame.groupby('condition_label')['n_figures'].sum()
             if 'condition_label' in frame else pd.Series(dtype=int))
    print('\nFigures per analysis:')
    for label, n_figures in tally.sort_index().items():
        print(f'  {label or "(no condition_label)"}: {n_figures}')

    missing = [a for a in expected if int(tally.get(a, 0)) == 0]
    if missing:
        print('\nNo figures for these analyses -- there was no run for them in '
              'the frame passed to replot_all. Either the filter excluded them '
              'or the decoding jobs never wrote a MASTER_RESULTS pickle; '
              'analysis_coverage(runs) tells you which:')
        for label in missing:
            print(f'  · {label}')
