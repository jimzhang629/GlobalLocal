#!/usr/bin/env python
"""Write an A1-format ``anova_labels.csv`` from a within-electrode power-trace run.

This is the bridge that lets the ordinary launchers select on electrodes defined
by the **power traces** rather than by the A1 window-mean ANOVA:

    dcc_scripts/power/submit_specific_conditions_power_traces_dcc.sh
    dcc_scripts/decoding/submit_specific_conditions_decoding_dcc.sh

Both already accept ``ANOVA_LABELS_CSV`` + ``ANOVA_LABEL_EFFECT`` (``both``,
``congruency``, ``switch_type``, ``congruency_only``, ``switch_type_only``, ...),
but the only producer of that table was the A1 conjunction
(``stability_flexibility_anova_conjunction_dcc.py``).

The electrode sets written here are the SAME ones the brain plots draw: the
flags come from ``power_traces.power_trace_electrode_set`` -- the primitive
behind ``plot_sig_electrodes_dcc.get_condition_electrodes``' power-trace source
and behind the ``congruency_only`` / ``switch_type_only`` / ``both`` entries of
``dcc_scripts/vis/condition_plot_specs.py``. So

    ANOVA_LABEL_EFFECT=congruency_only ANOVA_LABEL_CORRECTION=flags

selects exactly the electrodes ``PLOT_SETS=congruency_only`` renders.

It is a pure read of the run's ``summary.csv`` /
``significant_effects_structure.json``: no epoched data, no permutations, no
SLURM. It takes seconds on a login node.

Examples
--------
The congruency / switchType main effects out of the full-factorial run (the
default, and what ``condition_plot_specs`` uses)::

    PT=/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/figs/$EPOCHS/anova_within_electrode
    python make_power_traces_anova_labels.py \
        --run $PT/stimulus_experiment_conditions_24_subjects --roi lpfc

The LWPC / LWPS interactions, which live in two separate two-factor runs::

    python make_power_traces_anova_labels.py \
        --s-run $PT/stimulus_lwpc_conditions_24_subjects \
        --s-effect 'C(congruency):C(incongruentProportion)' \
        --f-run $PT/stimulus_lwps_conditions_24_subjects \
        --f-effect 'C(switchType):C(switchProportion)' --roi lpfc

The printed output path is what you paste into ``ANOVA_LABELS_CSVS`` in either
submit script.

Note on circularity: selecting on the congruency main effect and then plotting or
decoding *that same contrast* is the diagonal cell -- see
``docs/nested_electrode_selection.md``. The off-diagonal cells (select on
congruency, measure switch type or the LWPC/LWPS interaction) are fine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

# Imported from the module rather than the `power_traces` facade: that facade
# pulls in the epoched-data stack (`ieeg`), and this script is a pure read of
# finished result files that must stay runnable without it.
from src.analysis.power.windowed_anova import (
    per_electrode_cluster_stats,
    power_trace_electrode_set,
)

# `load_anova_label_electrodes` reads S (congruency / LWPC) and F (switchType /
# LWPS) plus the A1 p/q spellings; the construct-group aliases keep the table
# interchangeable with the power-traces conjunction's own labels.csv.
S_ALIASES = ('CPC',)
F_ALIASES = ('SPS',)

KEY = ['subject', 'electrode', 'roi']

DEFAULT_S_EFFECT = 'C(congruency)'
DEFAULT_F_EFFECT = 'C(switchType)'


def short_run_tag(run_dir):
    """Compact, still-unique name for a run directory.

    ``stimulus_lwpc_conditions_24_subjects`` -> ``lwpc_24_subjects``. The full
    paths are recorded in ``source.json``, so this only has to stay readable and
    distinguishing -- the output directory name is carried verbatim into the
    downstream ``anova_label_selections/`` slug, which has a length budget.
    """
    name = os.path.basename(os.path.normpath(str(run_dir)))
    return name.replace('stimulus_', '').replace('_conditions', '') or name


def build_labels(s_run, s_effect, f_run, f_effect, roi=None, use_fdr=True,
                 p_thresh=0.05):
    """One row per electrode TESTED, with the S/F flags of the plotted sets.

    The universe is every electrode in the run's ``summary.csv``, not just the
    significant ones: a labels table whose denominator is the winners would make
    every downstream selection trivially complete and would hide how many
    electrodes were considered.
    """
    sides = {
        'S': dict(prefix='cong', run=s_run, effect=s_effect, aliases=S_ALIASES,
                  stats=per_electrode_cluster_stats(s_run, s_effect, roi=roi)),
        'F': dict(prefix='switch', run=f_run, effect=f_effect, aliases=F_ALIASES,
                  stats=per_electrode_cluster_stats(f_run, f_effect, roi=roi)),
    }

    labels = (pd.concat([side['stats'][KEY] for side in sides.values()])
                .drop_duplicates().sort_values(KEY).reset_index(drop=True))

    for flag, side in sides.items():
        selected = set(power_trace_electrode_set(
            side['run'], include_effects=(side['effect'],), roi=roi,
            use_fdr=use_fdr, p_thresh=p_thresh))
        labels[flag] = [int((sub, elec) in selected) for sub, elec
                        in zip(labels['subject'], labels['electrode'])]
        for alias in side['aliases']:
            labels[alias] = labels[flag]

        # The same cluster p/q under the A1 column names, so a downstream
        # ANOVA_LABEL_CORRECTION of `none` or `fdr_bh` re-thresholds these
        # instead of silently selecting nothing.
        prefix = side['prefix']
        graded = side['stats'].rename(columns={
            'p_cluster': f'p_{prefix}', 'q_cluster': f'q_{prefix}',
            'extent': f'{prefix}_extent', 'sign': f'{prefix}_sign',
            'best_cluster_p': f'best_cluster_p_{prefix}'})
        labels = labels.merge(graded, on=KEY, how='left')
    return labels


def default_out_dir(s_run, f_run, roi, s_effect, f_effect, use_fdr, p_thresh):
    """``<anova_within_electrode>/power_traces_labels/<slug>``.

    Kept next to the runs it is derived from rather than under ``stats/results``,
    because it is a pure restatement of those runs and nothing was re-fit.
    """
    def effect_tag(effect):
        return (effect.replace('C(', '').replace(')', '')
                      .replace(':', '_x_'))

    tags = list(dict.fromkeys(short_run_tag(p) for p in (s_run, f_run)))
    threshold = 'fdr' if use_fdr else f"p{p_thresh}"
    slug = (f"{'__'.join(tags)}__S-{effect_tag(s_effect)}"
            f"__F-{effect_tag(f_effect)}__{threshold}__roi-{roi or 'all'}")
    return Path(s_run).parent / 'power_traces_labels' / slug


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', help="one within-electrode run holding both "
                        "effects (e.g. a stimulus_experiment_conditions run). "
                        "Shorthand for --s-run RUN --f-run RUN.")
    parser.add_argument('--s-run', help="run supplying S (the 'congruency' / "
                        "'lwpc' side of every ANOVA_LABEL_EFFECT)")
    parser.add_argument('--f-run', help="run supplying F (the 'switch_type' / "
                        "'lwps' side)")
    parser.add_argument('--s-effect', default=DEFAULT_S_EFFECT,
                        help=f"statsmodels term for S (default: {DEFAULT_S_EFFECT}). "
                             "Use 'C(congruency):C(incongruentProportion)' for LWPC.")
    parser.add_argument('--f-effect', default=DEFAULT_F_EFFECT,
                        help=f"statsmodels term for F (default: {DEFAULT_F_EFFECT}). "
                             "Use 'C(switchType):C(switchProportion)' for LWPS.")
    parser.add_argument('--roi', default='lpfc',
                        help="ROI to restrict to, or 'all' to keep every ROI in "
                             "the run (default: lpfc)")
    parser.add_argument('--p-thresh', type=float, default=0.05,
                        help="raw cluster-p cutoff, used with --no-fdr "
                             "(default: 0.05)")
    parser.add_argument('--no-fdr', action='store_true',
                        help="threshold the raw cluster p instead of the run's "
                             "saved BH flags -- the same use_fdr=False the brain "
                             "plots take")
    parser.add_argument('--out', help="output directory (default: "
                                      "<anova_within_electrode>/power_traces_labels/<slug>)")
    args = parser.parse_args(argv)

    if args.run and (args.s_run or args.f_run):
        parser.error("pass either --run or --s-run/--f-run, not both")
    s_run, f_run = args.s_run or args.run, args.f_run or args.run
    if not (s_run and f_run):
        parser.error("need --run, or both --s-run and --f-run: the S/F table "
                     "has two sides")

    roi = None if args.roi.strip().lower() == 'all' else args.roi.strip()
    use_fdr = not args.no_fdr
    labels = build_labels(s_run, args.s_effect, f_run, args.f_effect, roi=roi,
                          use_fdr=use_fdr, p_thresh=args.p_thresh)

    out_dir = Path(args.out) if args.out else default_out_dir(
        s_run, f_run, roi, args.s_effect, args.f_effect, use_fdr, args.p_thresh)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'anova_labels.csv'
    labels.to_csv(out_csv, index=False)

    n_s, n_f = int(labels['S'].sum()), int(labels['F'].sum())
    n_both = int((labels['S'].eq(1) & labels['F'].eq(1)).sum())

    # `load_significant_electrodes` prefers significant_effects_structure.json
    # when a run has one -- and current runs always write it. That path thresholds
    # the RAW cluster p at --p-thresh and ignores --no-fdr entirely, so record
    # which rule actually produced the flags rather than the flag we passed.
    legacy_json = any((Path(run) / 'significant_effects_structure.json').exists()
                      for run in (s_run, f_run))
    rule = ('saved BH flags' if use_fdr else f'raw cluster p < {args.p_thresh}')
    if legacy_json:
        rule = (f'raw cluster p < {args.p_thresh} (from '
                f'significant_effects_structure.json, which '
                f'load_significant_electrodes prefers over summary.csv)')

    with open(out_dir / 'source.json', 'w') as f:
        json.dump(dict(
            written=datetime.now().isoformat(timespec='seconds'),
            source='power_traces', roi=args.roi,
            s_run=str(s_run), s_effect=args.s_effect,
            f_run=str(f_run), f_effect=args.f_effect,
            use_fdr=use_fdr, p_thresh=args.p_thresh,
            thresholded_by=rule,
            n_electrodes=int(len(labels)),
            n_subjects=int(labels['subject'].nunique()),
            counts=dict(S=n_s, F=n_f, both=n_both,
                        S_only=n_s - n_both, F_only=n_f - n_both),
        ), f, indent=2)

    print(f"{len(labels)} electrodes tested | {labels['subject'].nunique()} subjects "
          f"| threshold: {rule}")
    print(f"  S ({args.s_effect}): {n_s}   F ({args.f_effect}): {n_f}   both: {n_both}")
    print(f"  congruency_only: {n_s - n_both}   switch_type_only: {n_f - n_both}")
    print(f"\nwrote {out_csv}")
    print("\nUse it from either launcher, e.g.:")
    print(f'  ANOVA_LABELS_CSV="{out_csv}" ANOVA_LABEL_CORRECTION=flags \\\n'
          f'      bash submit_specific_conditions_power_traces_dcc.sh')
    return out_csv


if __name__ == '__main__':
    main()
