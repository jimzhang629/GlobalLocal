#!/usr/bin/env python
"""Rank the electrodes distorting an ROI power trace, and plot them.

Reads the ``*_evoked.npz`` files a finished power-traces run already wrote, so
it needs no MNE, no raw data and no cluster time -- just point it at the run
directory.

Prints a ranked table per ROI, writes ``electrode_outlier_stats.csv``, and (with
``--plot``) writes one overlay figure per condition showing every electrode's
trace in grey under the ROI mean, with the top outliers coloured and named.

Examples
--------
Score the first 100 ms of the epoch (default -- where boundary transients land)::

    python dcc_scripts/power/find_outlier_electrodes.py \\
        --save_dir .../figs/<epochs_root_file>/anova_roi \\
        --rois lpfc occ --plot

Score the whole pre-stimulus baseline instead::

    python dcc_scripts/power/find_outlier_electrodes.py \\
        --save_dir <run_dir> --rois lpfc --window baseline

Score an explicit window::

    python dcc_scripts/power/find_outlier_electrodes.py \\
        --save_dir <run_dir> --rois lpfc --window -1.0 -0.9
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.power.electrode_qc import (  # noqa: E402
    load_roi_evokeds,
    electrode_outlier_stats,
    annotate_subjects,
    plot_electrode_overlay_for_roi,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--save_dir', required=True,
        help='Power-traces run directory holding the saved *_evoked.npz files '
             '(the same path the run used as save_dir).')
    parser.add_argument(
        '--rois', nargs='+', required=True,
        help='ROIs to score, e.g. --rois lpfc occ')
    parser.add_argument(
        '--conditions_save_name', default=None,
        help='Restrict to one condition set when the run directory holds '
             'several (the {conditions_save_name} part of the filenames).')
    parser.add_argument(
        '--window', nargs='+', default=['edge'],
        help="Scoring window: 'edge' (first --edge_duration of the epoch, the "
             "default), 'baseline' (everything before 0), 'whole', or two "
             "numbers giving an explicit tmin tmax in seconds.")
    parser.add_argument(
        '--edge_duration', type=float, default=0.1,
        help="Length in seconds of the 'edge' window. Default 0.1.")
    parser.add_argument(
        '--robust_z_threshold', type=float, default=10.0,
        help='Electrodes at or above this robust z are reported as outliers. '
             'Default 10.')
    parser.add_argument(
        '--top', type=int, default=10,
        help='How many electrodes to print per ROI. Default 10.')
    parser.add_argument(
        '--plot', action='store_true',
        help='Also write per-condition electrode-overlay figures.')
    parser.add_argument(
        '--plot_conditions', nargs='+', default=None,
        help='Restrict --plot to these conditions. Default plots all of them, '
             'which can be a lot of files.')
    parser.add_argument(
        '--n_label', type=int, default=5,
        help='How many outliers to highlight in each overlay. Default 5.')
    parser.add_argument(
        '--rois_dict', default=None,
        help='Path to subjects_electrodestoROIs_dict.json, to add a '
             'candidate_subjects column. Defaults to the copy in '
             'src/analysis/config/ when it exists.')
    parser.add_argument(
        '--out_dir', default=None,
        help='Where to write the CSV and figures. Defaults to '
             '{save_dir}/electrode_qc/.')
    return parser.parse_args(argv)


def resolve_window(tokens):
    """Turn the --window tokens into what electrode_outlier_stats expects."""
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) == 2:
        try:
            return (float(tokens[0]), float(tokens[1]))
        except ValueError:
            raise SystemExit(
                f"--window with two values must be numeric tmin tmax; "
                f"got {tokens!r}")
    raise SystemExit(f"--window takes one name or two numbers; got {tokens!r}")


def main(argv=None):
    args = parse_args(argv)
    window = resolve_window(args.window)
    out_dir = args.out_dir or os.path.join(args.save_dir, 'electrode_qc')
    os.makedirs(out_dir, exist_ok=True)

    rois_dict = args.rois_dict
    if rois_dict is None:
        default_dict = os.path.join(project_root, 'src', 'analysis', 'config',
                                    'subjects_electrodestoROIs_dict.json')
        rois_dict = default_dict if os.path.isfile(default_dict) else None

    all_stats = []
    for roi in args.rois:
        try:
            condition_data, times, ch_names = load_roi_evokeds(
                args.save_dir, roi, args.conditions_save_name)
        except FileNotFoundError as err:
            print(f'[{roi}] {err}')
            continue

        stats = electrode_outlier_stats(
            condition_data, times, ch_names,
            window=window, edge_duration=args.edge_duration)
        if rois_dict:
            stats = annotate_subjects(stats, rois_dict, roi=roi)

        label = stats.attrs.get('window', str(window))
        flagged = stats[stats['robust_z'] >= args.robust_z_threshold]

        print(f"\n=== {roi}: {len(ch_names)} electrodes, "
              f"{len(condition_data)} conditions, window '{label}' ===")
        print(f"  ROI median peak |z| in window: {stats['stat'].median():.3f}")
        print(f"  {len(flagged)} electrode(s) at robust z >= "
              f"{args.robust_z_threshold:g}")
        columns = ['electrode', 'stat', 'robust_z', 'post_onset_max',
                   'worst_condition', 'name_is_ambiguous']
        if 'candidate_subjects' in stats:
            columns.append('candidate_subjects')
        with pd.option_context('display.width', 200,
                               'display.max_colwidth', 40):
            print(stats.head(args.top)[columns].to_string(index=False))

        if flagged['name_is_ambiguous'].any():
            print("  NOTE: a flagged name carries MNE's -0/-1 disambiguation "
                  "suffix, so it maps to more than one subject. Check "
                  "candidate_subjects before dropping it.")

        stats.insert(0, 'roi', roi)
        all_stats.append(stats)

        if args.plot:
            plot_dir = os.path.join(out_dir, roi)
            conditions = args.plot_conditions or [None]
            for cond in conditions:
                figures = plot_electrode_overlay_for_roi(
                    condition_data, times, ch_names, roi,
                    stats=stats, condition=cond, n_label=args.n_label,
                    save_dir=plot_dir)
                for fig in figures.values():
                    plt.close(fig)

    if not all_stats:
        raise SystemExit(
            f'No evoked files found for ROIs {args.rois} under {args.save_dir}.')

    combined = pd.concat(all_stats, ignore_index=True)
    csv_path = os.path.join(out_dir, 'electrode_outlier_stats.csv')
    combined.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}')
    return combined


if __name__ == '__main__':
    main()
