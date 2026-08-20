"""Electrode QC for power traces: find the contacts that distort an ROI mean.

An ROI power trace is an unweighted mean over electrodes, so a single contact
carrying an artifact moves the mean and inflates the SEM ribbon. This module
scores each electrode against its own ROI and draws the individual traces
underneath the mean so a suspect contact can be eyed before it is dropped.

It reads the ``*_evoked.npz`` files the power-traces run already writes
(``{conditions_save_name}_{condition_name}_{roi}_evoked.npz``, holding ``data``,
``times`` and ``ch_names``), so scoring a finished run costs no recomputation.

Scoring is robust-z against the ROI: ``(stat - median) / (1.4826 * MAD)``, where
``stat`` defaults to the largest ``|z|`` in the first 50 ms of the epoch. That
window is where filter/epoch-boundary transients land; pass a different
``window`` to score the pre-stimulus baseline or the whole epoch instead.

Electrode names are **not** unique across subjects -- MNE disambiguates a
collision by appending ``-0``, ``-1`` and so on when the ROI evoked is built, so
a name here identifies a column of that ROI's evoked, not a subject. Use
``subjects_electrodestoROIs_dict.json`` (via ``annotate_subjects``) to recover
the candidate subjects before dropping anything.
"""

import os
import glob
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suffix MNE appends when two subjects contribute the same electrode name.
_DEDUP_SUFFIX = re.compile(r"-\d+$")

# Named scoring windows. Each maps to a callable taking the epoch's time vector
# and returning a boolean sample mask.
WINDOWS = {
    # The first 100 ms of the epoch -- where an epoch-boundary transient sits.
    # Wide enough that a transient peaking a few samples in is fully contained;
    # a window that clips the peak leaks it into the comparison statistic.
    'edge': lambda times, edge_duration=0.1: times < (times[0] + edge_duration),
    # Everything before stimulus onset.
    'baseline': lambda times, edge_duration=None: times < 0,
    # The entire epoch.
    'whole': lambda times, edge_duration=None: np.ones(len(times), dtype=bool),
}


# Label given to the ROI-average line, so callers (and tests) can find it among
# the per-electrode lines without relying on draw order.
MEAN_LINE_LABEL = 'ROI mean'


def _condition_from_filename(path, roi, prefix):
    """Recover the condition name from a saved evoked filename.

    The run writes ``{conditions_save_name}_{condition_name}_{roi}_evoked.npz``.
    When the caller named ``conditions_save_name`` we strip exactly that.
    Otherwise we lean on the save name always ending in ``_{n}_subjects``
    (``get_conditions_save_name``) and take what follows; failing that, the
    whole stem is returned rather than guessing.
    """
    stem = os.path.basename(path)[:-len('.npz')]
    condition = stem[:-len(f'_{roi}_evoked')]
    if prefix and condition.startswith(prefix):
        return condition[len(prefix):]
    marker = '_subjects_'
    if marker in condition:
        return condition.split(marker, 1)[1]
    return condition


def load_roi_evokeds(save_dir, roi, conditions_save_name=None):
    """Load every saved evoked ``.npz`` for one ROI.

    Parameters
    ----------
    save_dir : str
        A power-traces run directory. The ROI's files are looked for in
        ``{save_dir}/{roi}/`` first (where the run writes them) and then in
        ``save_dir`` itself, so either level can be passed.
    roi : str
        ROI name, e.g. ``'lpfc'``.
    conditions_save_name : str, optional
        Restrict to one condition set. Files are named
        ``{conditions_save_name}_{condition_name}_{roi}_evoked.npz``, so a run
        directory holding several sets can be filtered down to one.

    Returns
    -------
    condition_data : dict[str, np.ndarray]
        Condition name -> ``(n_electrodes, n_times)`` array.
    times : np.ndarray
        Shared time vector.
    ch_names : list[str]
        Electrode names, in the column order of every array.

    Raises
    ------
    FileNotFoundError
        If no matching file exists.
    ValueError
        If the files disagree on their time vector or electrode names, which
        means they came from different runs and must not be pooled.
    """
    prefix = f'{conditions_save_name}_' if conditions_save_name else ''
    pattern = f'{prefix}*_{roi}_evoked.npz'
    paths = sorted(glob.glob(os.path.join(save_dir, roi, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(save_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No '{pattern}' files under {save_dir} (looked in {save_dir}/{roi}/ "
            f"and {save_dir}/). Point save_dir at a power-traces run directory."
        )

    condition_data = {}
    times = None
    ch_names = None
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            data = npz['data']
            this_times = npz['times']
            this_ch = [str(c) for c in npz['ch_names']]

        if times is None:
            times, ch_names = this_times, this_ch
        elif len(this_times) != len(times) or not np.allclose(this_times, times):
            raise ValueError(
                f"{os.path.basename(path)} has a different time vector than the "
                f"other files for ROI '{roi}'; these are separate runs and "
                f"cannot be scored together."
            )
        elif this_ch != ch_names:
            raise ValueError(
                f"{os.path.basename(path)} has different electrode names than "
                f"the other files for ROI '{roi}'; these are separate runs and "
                f"cannot be scored together."
            )

        condition_data[_condition_from_filename(path, roi, prefix)] = data

    return condition_data, times, ch_names


def _robust_z(values):
    """Robust z of each value against the vector's own median and MAD.

    Returns zeros when the MAD is zero (every electrode identical), rather than
    dividing by zero and flagging the whole ROI.
    """
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if scale <= 0:
        return np.zeros_like(values)
    return (values - median) / scale


def electrode_outlier_stats(condition_data, times, ch_names, window='edge',
                            edge_duration=0.05):
    """Score every electrode in one ROI against the rest of that ROI.

    For each electrode the statistic is the largest ``|z|`` inside the scoring
    window, maximised over conditions; ``worst_condition`` records which
    condition produced it, which usually identifies the offending trials.

    Parameters
    ----------
    condition_data : dict[str, np.ndarray]
        Condition name -> ``(n_electrodes, n_times)``, as from
        :func:`load_roi_evokeds`.
    times : np.ndarray
        Time vector shared by every array.
    ch_names : list[str]
        Electrode names in column order.
    window : {'edge', 'baseline', 'whole'} or (float, float)
        Which samples to score. A tuple is read as an explicit ``(tmin, tmax)``
        in seconds, inclusive of ``tmin`` and exclusive of ``tmax``.
    edge_duration : float, default 0.1
        Length in seconds of the ``'edge'`` window, measured from the start of
        the epoch. Ignored by the other windows.

    Returns
    -------
    pandas.DataFrame
        One row per electrode, sorted by ``robust_z`` descending, with columns:
        ``electrode``, ``stat`` (max ``|z|`` in the window), ``robust_z``,
        ``worst_condition``, ``post_onset_max`` (max ``|z|`` at or after t=0, to
        separate a pre-onset boundary transient from a contact that is also bad
        where it matters), and ``name_is_ambiguous`` (True when MNE had to
        disambiguate the name, so it maps to more than one subject).

    Raises
    ------
    ValueError
        If the window selects no samples, or the arrays disagree in shape.
    """
    if not condition_data:
        raise ValueError("condition_data is empty; nothing to score.")

    times = np.asarray(times)
    if isinstance(window, (tuple, list)):
        tmin, tmax = window
        mask = (times >= tmin) & (times < tmax)
        window_label = f'{tmin}to{tmax}s'
    else:
        if window not in WINDOWS:
            raise ValueError(
                f"window must be one of {sorted(WINDOWS)} or a (tmin, tmax) "
                f"tuple; got {window!r}"
            )
        mask = WINDOWS[window](times, edge_duration)
        window_label = window

    if not mask.any():
        raise ValueError(
            f"Scoring window '{window_label}' selects no samples from a time "
            f"axis spanning {times[0]:.3f}..{times[-1]:.3f}s."
        )

    n_elec = len(ch_names)
    for condition, data in condition_data.items():
        if data.shape != (n_elec, len(times)):
            raise ValueError(
                f"Condition '{condition}' has shape {data.shape}, expected "
                f"({n_elec}, {len(times)})."
            )

    conditions = list(condition_data)
    # (n_conditions, n_electrodes) peak |z| inside and outside the window.
    in_window = np.stack(
        [np.abs(condition_data[c][:, mask]).max(axis=1) for c in conditions])
    # Peak after stimulus onset, as an unambiguous "is this contact also bad in
    # the part of the epoch I actually analyse?" reading. Deliberately NOT
    # "everything outside the window": a transient peaking a sample or two past
    # the window edge would spill into that and make a boundary artifact look
    # like a globally noisy contact.
    post_mask = times >= 0
    if post_mask.any():
        post_onset_max = np.stack(
            [np.abs(condition_data[c][:, post_mask]).max(axis=1)
             for c in conditions]).max(axis=0)
    else:
        post_onset_max = np.full(n_elec, np.nan)

    stat = in_window.max(axis=0)
    worst = [conditions[i] for i in in_window.argmax(axis=0)]

    frame = pd.DataFrame({
        'electrode': ch_names,
        'stat': stat,
        'robust_z': _robust_z(stat),
        'worst_condition': worst,
        'post_onset_max': post_onset_max,
        'name_is_ambiguous': [bool(_DEDUP_SUFFIX.search(c)) for c in ch_names],
    })
    frame.attrs['window'] = window_label
    return frame.sort_values('robust_z', ascending=False).reset_index(drop=True)


def annotate_subjects(frame, rois_dict_path, roi=None):
    """Add a ``candidate_subjects`` column naming who each electrode could be.

    Electrode names in an ROI evoked are not unique across subjects, so this
    can only narrow the field, never resolve it. The column lists every subject
    whose ROI membership includes that bare name (the MNE ``-0``/``-1``
    disambiguation suffix is stripped before matching), joined by ``'|'``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`electrode_outlier_stats`.
    rois_dict_path : str
        Path to ``subjects_electrodestoROIs_dict.json``.
    roi : str, optional
        If given, only count a subject when that subject maps the electrode to
        this ROI. Without it, any ROI counts.

    Returns
    -------
    pandas.DataFrame
        A copy of ``frame`` with the extra column.
    """
    with open(rois_dict_path) as handle:
        subjects_to_electrodes = json.load(handle)

    by_name = {}
    for subject, electrodes in subjects_to_electrodes.items():
        for electrode, mapped_roi in electrodes.items():
            if roi is not None and str(mapped_roi) != str(roi):
                continue
            by_name.setdefault(electrode, []).append(subject)

    out = frame.copy()
    out['candidate_subjects'] = [
        '|'.join(sorted(by_name.get(_DEDUP_SUFFIX.sub('', name), []))) or 'unknown'
        for name in out['electrode']
    ]
    return out


def plot_electrode_overlay_for_roi(condition_data, times, ch_names, roi,
                                   stats=None, condition=None, n_label=5,
                                   save_dir=None, save_name_suffix=None,
                                   figsize=(12, 8), ylim=None):
    """Draw every electrode's trace under the ROI mean, labelling the outliers.

    Parameters
    ----------
    condition_data : dict[str, np.ndarray]
        Condition name -> ``(n_electrodes, n_times)``.
    times : np.ndarray
        Time vector.
    ch_names : list[str]
        Electrode names in column order.
    roi : str
        ROI name, used in the title and filename.
    stats : pandas.DataFrame, optional
        Output of :func:`electrode_outlier_stats`. When given, the top
        ``n_label`` electrodes by ``robust_z`` are drawn in colour and named in
        the legend. Without it every electrode is drawn the same.
    condition : str, optional
        Plot only this condition. Default draws one figure per condition.
    n_label : int, default 5
        How many outliers to highlight.
    save_dir : str, optional
        Directory to write ``.png``/``.pdf`` into. Created if missing.
    save_name_suffix : str, optional
        Appended to the filename stem.
    figsize : tuple, default (12, 8)
        Figure size.
    ylim : tuple, optional
        Y limits. Default autoscales, which lets an artifact set the scale --
        that is usually what you want when hunting for one.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Condition name -> figure.
    """
    conditions = [condition] if condition is not None else list(condition_data)
    missing = [c for c in conditions if c not in condition_data]
    if missing:
        raise KeyError(f"condition(s) {missing} not in condition_data "
                       f"(have {sorted(condition_data)})")

    highlight = []
    if stats is not None and n_label > 0:
        highlight = list(stats.head(n_label)['electrode'])
    colors = plt.get_cmap('tab10')

    figures = {}
    for cond in conditions:
        data = condition_data[cond]
        fig, ax = plt.subplots(figsize=figsize)

        for idx, name in enumerate(ch_names):
            if name in highlight:
                continue
            ax.plot(times, data[idx], color='0.65', linewidth=0.6,
                    alpha=0.5, zorder=1)

        for rank, name in enumerate(highlight):
            idx = ch_names.index(name)
            row = stats.loc[stats['electrode'] == name].iloc[0]
            ax.plot(times, data[idx], color=colors(rank % 10), linewidth=1.6,
                    zorder=3,
                    label=f"{name} (robust z={row['robust_z']:.0f}, "
                          f"peak={row['stat']:.2f})")

        ax.plot(times, data.mean(axis=0), color='black', linewidth=2.8,
                zorder=4, label=f'{MEAN_LINE_LABEL} (n={len(ch_names)})')

        ax.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax.axvline(0, color='black', linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Time from Stimulus Onset (s)')
        ax.set_ylabel('Power (z)')
        ax.set_title(f'{roi.upper()} — {cond} — individual electrodes')
        if ylim is not None:
            ax.set_ylim(ylim)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f'_{save_name_suffix}' if save_name_suffix else ''
            stem = f'{roi}_{cond}_electrode_overlay{suffix}'
            for ext in ('.png', '.pdf'):
                path = os.path.join(save_dir, stem + ext)
                fig.savefig(path, dpi=200, bbox_inches='tight')
                print(f'Saved electrode overlay to: {path}')

        figures[cond] = fig

    return figures
