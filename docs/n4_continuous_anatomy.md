# N4 — continuous electrode scores → anatomy, brain maps, and descriptive centres

**What this document is.** A standalone, end-to-end walkthrough of beats **5–7**
in [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md):

1. the primary coverage-conditioned anatomical test of continuous LWPC and LWPS
   scores;
2. the five continuous-score brain maps; and
3. the per-subject, per-hemisphere weighted centres (descriptive only).

It explains what the analysis asks, where the code lives, which inputs are safe
to use, how data move through the code, exact cluster commands, and how to read
every output. It deliberately does **not** cover the older categorical S/F-group
anatomy arm except where necessary to distinguish it from this analysis.

If you read only three things, read [the estimand](#2-the-estimand-and-sign-convention),
[the primary test](#4-the-primary-anatomical-test), and
[the interpretation checklist](#11-interpretation-checklist).

If you are **writing up the lPFC run** rather than running it, start at
[§15 Findings](#15-findings-from-the-lpfc-run), then
[§16](#16-reading-the-archived-summarytxt) (which blocks of the archived
`summary.txt` are safe to quote — several are not),
[§17](#17-what-the-result-means--discussion) (the Discussion claims and their
limits), and [§18](#18-communicating-a-nested-estimand) (how to present a
four-level estimand without the depth becoming the story).

---

## 0. The short version

N4 asks:

> **Does the balance between an electrode's LWPC and LWPS effects vary with its
> anatomical location?**

For every anatomically eligible electrode, the pipeline estimates an LWPC score
and an LWPS score on disjoint trial halves. Each score is a signed,
equal-cell-weighted, standardized difference-of-differences. After one pooled
scale factor per effect, it forms

```text
delta = lwpc_s - lwps_s
```

and asks whether `delta` differs by ROI/Destrieux parcel. This is an explicit
**effect type × anatomy interaction**. It does not infer a dissociation from
“LWPC significant here, LWPS not significant there.”

The primary null independently swaps the LWPC and LWPS labels within every
electrode. A swap negates `delta`, so the implementation is a sign-flip
permutation. It holds the subject, electrode, anatomy, coverage, responsiveness,
and the two marginal score distributions fixed. The primary result is the
omnibus permutation `F` and `p` in `summary.txt` / `score_anatomy.json`.

The brain maps visualize the two signed scores, their magnitudes, and their
difference. They do not add five new tests. The coordinate regression is a
secondary inferential description of spatial gradients. The weighted medoids
are descriptive and must not replace either anatomical test.

---

## 1. Where the code lives

| Role | File / function |
|---|---|
| Analysis specification | `docs/analysis_plan_concurrent_regulation.md`, §§5–7 |
| Score estimation | `src/analysis/stats/stability_flexibility_segregation.py`: `compute_sensitivities_per_split`, `average_over_splits`, `add_responsiveness` |
| Core N4 statistics | `src/analysis/stats/stability_flexibility_anatomy.py`: `attach_scores`, `relative_score_roi_test`, `relative_score_coordinate_test`, `map_reliability`, `leave_one_subject_out` |
| Brain maps | same file: `SCORE_MAPS`, `plot_score_by_roi`, `plot_scores_on_brain`, `plot_score_maps` |
| Descriptive centres | same file: `score_centers_per_subject` |
| DCC orchestration and output writing | `dcc_scripts/stats/stability_flexibility_anatomy_dcc.py`: `load_scores`, `run_score_anatomy`, `write_score_summary` |
| Environment-variable entry point | `dcc_scripts/stats/run_stability_flexibility_anatomy_dcc.py` |
| Slurm submitter / display wrapper | `dcc_scripts/stats/submit_stability_flexibility_anatomy_dcc.sh`, `sbatch_stability_flexibility_anatomy_dcc.sh` |
| Recommended upstream score-producing run | `dcc_scripts/stats/submit_stability_flexibility_segregation_dcc.sh` |
| Ground-truth regression tests | `tests/analysis/stats/test_stability_flexibility_anatomy.py` |

### Do not choose the wrong arm

`run_stability_flexibility_anatomy_dcc.py` defaults to `ARM=categorical`. That is
the older thresholded S/F-group enrichment analysis. **For this document, always
set `ARM=continuous`**. `ARM=both` runs categorical first and then writes N4 into
a `continuous/` subdirectory, but it costs more and can obscure which result is
primary.

`LABEL_SOURCE` is relevant to the categorical arm. It does not define the N4
continuous electrode set when `SCORES_CSV` is supplied. The continuous arm uses
the electrodes present in that score CSV and then applies the anatomical
`ROI_FILTER`.

---

## 2. The estimand and sign convention

### 2.1 What the two scores are

With the required `CONTRAST_MODE=proportion`, the two scores are:

```text
LWPC = (incongruent - congruent | low incongruent proportion)
     - (incongruent - congruent | high incongruent proportion)

LWPS = (switch - repeat | low switch proportion)
     - (switch - repeat | high switch proportion)
```

The implementation uses four cell means with equal weights and divides their
difference-of-differences by the pooled within-cell SD. It therefore avoids
letting the frequent cells dominate an unbalanced proportion design.

> **Positive LWPC or LWPS means adaptation:** the condition effect is smaller in
> the high-proportion block. Negative means the condition effect grew rather
> than shrank.

For each split, both effects are evaluated on both halves (`xA`, `xB`, `yA`,
`yB`). `average_over_splits` averages the two half-estimates and all requested
splits to give `x` (LWPC) and `y` (LWPS) per electrode. Cross-effect reliability
calculations continue to use the split-resolved values, so same-trial noise
cannot masquerade as co-localization.

### 2.2 Scaling and the relative score

`attach_scores` renames `x/y` to `lwpc_score/lwps_score`, then divides each
entire column by that effect's SD across all pooled electrodes:

```text
lwpc_s = lwpc_score / SD(lwpc_score across all electrodes)
lwps_s = lwps_score / SD(lwps_score across all electrodes)
delta  = lwpc_s - lwps_s
```

**Units.** `lwpc_score`/`lwps_score` are Cohen's *d* — a difference-of-differences
of cell means over the pooled within-cell SD of single-trial high-gamma — so the
raw scores are already unit-free. `lwpc_s`/`lwps_s`/`delta` are that *d* divided
by its own across-electrode SD, so their unit is **"SDs of this effect across
this dataset's electrodes"**: `lwpc_s = 1.5` means an electrode 1.5 cross-electrode
SDs above the mean LWPC *d*, not *d* = 1.5 and not a z-score (the mean is not
removed, and nothing is standardized within subject). They are a *relative*,
sample-dependent scale: the same electrode rescored against a different electrode
set gets a different number, so compare them within a run, never across runs, and
quote `lwpc_score`/`lwps_score` when an absolute effect size is wanted.

There is no subtraction of the pooled mean because the interaction asks about
relative spatial variation; a global offset cannot create between-ROI spread
after the nuisance intercept. There is deliberately no within-subject z-score.
With two electrodes it would force values to ±0.707 regardless of magnitude,
and with one electrode it would silently lose the subject. The scores are
already standardized, so one pooled spread adjustment per effect is sufficient.

Interpret `delta` as:

| Value | Meaning |
|---|---|
| `delta > 0` | relatively more LWPC-dominant |
| `delta < 0` | relatively more LWPS-dominant |
| `delta ≈ 0` | similar scaled LWPC and LWPS scores; it does **not** imply both effects are absent |

Always inspect the signed and magnitude maps alongside `delta`. For example,
`delta > 0` could mean strong positive LWPC, weak LWPS, negative LWPS, or both.

### 2.3 The electrode set is anatomical, not effect-selected

The defensible N4 population is all electrodes in the predeclared anatomical
scope (`ROIS=all` upstream for whole-brain N4, or a predeclared ROI scope), not
electrodes selected for an LWPC or LWPS effect. Selecting significant effects
before asking where those effects occur would make the anatomical answer partly
a consequence of the selection rule and would discard the many weak but
informative continuous observations.

For that reason, use `ELECTRODES=all`, `ROIS=all`, and
`CONTRAST_MODE=proportion` in the upstream score run for the whole-brain primary
analysis. A predeclared `ROI_FILTER=lpfc` is a legitimate separate lPFC analysis,
but is not a substitute for whole-brain N4.

---

## 3. Data flow through the code

The recommended path computes expensive scores once in the segregation job and
reuses its CSVs in the anatomy job:

```text
high-gamma Epochs on disk
  EPOCHS_ROOT_FILE
       │
       │ load_HG_ev1_rescaled_per_subject(..., acc_trials_only=True)
       ▼
per-subject MNE epochs + trial metadata
       │
       │ assemble_long_df(window_tmin, window_tmax,
       │                  effect_measure='cohens_d')
       ▼
long trial table
  subject, electrode, congruency, switchType,
  incongruent_proportion, switch_proportion, hg
       │
       │ compute_sensitivities_per_split(...,
       │     contrast_mode='proportion', n_splits=200)
       │ each electrode's trials are divided into disjoint halves
       ▼
per_split.csv
  one row / electrode / split: xA, xB, yA, yB
       │
       ├─ average_over_splits() + add_responsiveness()
       ▼
electrodes.csv
  subject, electrode, x, y, resp, ...
       │
       │ N4 anatomy job: load_scores()
       │ load atlas maps and optional fsaverage/MNI coordinates
       ▼
attach_scores()
  join ROI + Destrieux + coordinates
  pooled scaling; abs_lwpc, abs_lwps; delta = lwpc_s - lwps_s
       │
       ├─ restrict_to_roi(ROI_FILTER)
       ├─ build_coverage_matrix()
       │
       ├─ PRIMARY: relative_score_roi_test()
       │      delta ~ anatomy + responsiveness + subject dummies
       │      within-electrode score-label-swap null
       │
       ├─ LEVERAGE: leave_one_subject_out()
       │
       ├─ SECONDARY: relative_score_coordinate_test()
       │      delta ~ y + z + x + responsiveness + subject dummies
       │      all electrodes and each hemisphere
       │
       ├─ RELIABILITY: map_reliability()
       │      electrode and parcel levels
       │
       ├─ DESCRIPTIVE: score_centers_per_subject()
       │      subject × hemisphere weighted medoids
       │
       └─ FIGURES + CSV + JSON + summary.txt
```

### 3.1 Anatomy joins

The atlas cache supplies both:

- `roi`: a coarse group defined by `src/analysis/config/rois.py`; and
- `anat`: the raw Destrieux label.

`ANAT_LEVEL=group` tests `roi`; `ANAT_LEVEL=destrieux` tests `anat`.
`ANAT_LEVEL=auto` uses coarse groups for whole brain, but automatically switches
to Destrieux labels when `ROI_FILTER` contains one coarse group—otherwise every
electrode would have the same uninformative coarse label.

Coordinates follow the same reconstruction path used for brain figures:
`jim_mri.subject_to_info` → channel montage → forced MRI frame → subject
Talairach transform → fsaverage/MNI millimetres. Missing reconstruction files
produce warnings and partial coordinates rather than killing the primary ROI
test.

### 3.2 Responsiveness and subject nuisance terms

When the score CSV is produced by the full segregation job, `resp` is carried
into N4. In the current pipeline it defaults to mean absolute HG unless an
external responsiveness dictionary is supplied in code. It controls general
electrode gain so “large response everywhere” is not mistaken for preferential
LWPC or LWPS anatomy.

The implementation uses subject dummy variables as the fixed-effect,
no-shrinkage equivalent of the plan's `(1 | subject)` nuisance intercept. The
permutation supplies inference, so a parametric mixed-model degrees-of-freedom
approximation is not used.

---

## 4. The primary anatomical test

### 4.1 The question

At the selected anatomical level, the conceptual model is:

```text
delta_ij ~ anatomy_j + responsiveness_ij + subject_i
```

The omnibus statistic is a one-way `F`: between-anatomy variation in
nuisance-adjusted `delta` divided by within-anatomy variation. The `F` is only a
scale-free statistic; its reported significance comes from permutations, not a
parametric F distribution.

Because `delta` is formed within each electrode before the model, anatomy is
being tested against the difference between effect types. That is the required
**effect type × anatomy interaction**.

### 4.2 Coverage conditioning

iEEG sampling is clinically determined. `coverage_matrix.csv` has one row per
subject and one column per anatomical unit; a `1` means that subject has at least
one electrode in the unit. Anatomical units covered in fewer than
`MIN_SUBJECTS` are excluded before testing (default `3`).

This does not make implant coverage random. It prevents a label represented by
only one or two people from driving the tested ROI family. Report the retained
units and their subject counts with every anatomical claim.

### 4.3 The null

For each permutation, every electrode independently keeps its subject,
coordinates, anatomy, responsiveness, and pair of observed scores, but LWPC and
LWPS are randomly exchanged. Algebraically this is a sign flip of `delta`.

The Monte Carlo p-value is `(extreme + 1) / (N_PERM + 1)`. With the default
10,000 permutations its smallest possible value is about `0.0001`. Never report
`p = 0`.

### 4.4 Omnibus and per-anatomy rows

The omnibus permutation `p` answers whether relative LWPC/LWPS balance varies
anywhere across the retained anatomical units. `delta_per_roi.csv` then provides
unit-level descriptions:

- `mean_delta`: raw mean `delta`;
- `mean_delta_adj`: mean after removing subject and responsiveness nuisance
  effects—the quantity used by the statistic;
- `p`: two-sided within-electrode-swap permutation p for that adjusted mean;
- `q`: Benjamini–Hochberg value across the retained units;
- `n_electrodes`, `n_subjects`: the sampling behind the row.

Lead with the omnibus test. Use `q`, not an isolated raw row-level `p`, when
identifying the units that explain a significant omnibus result. A positive
adjusted mean is LWPC-dominant; a negative adjusted mean is LWPS-dominant.

### 4.5 Leave-one-subject-out leverage

The same ROI test is rerun after dropping each subject. These folds use fewer
permutations (`max(1000, N_PERM/10)`) because this is an estimate-leverage check,
not a second inferential family.

Read shifts in `observed_stat`, not only shifts in `p`: permutation p-values can
saturate at their resolution floor. A large F collapse when one subject is
removed means the pooled result is fragile even if every fold still prints a
small p-value.

---

## 5. Secondary coordinate test

When coordinates are available, the pipeline also fits, separately by
hemisphere as well as in the pooled coordinate table:

```text
delta ~ mni_y + mni_z + mni_x + responsiveness + subject
```

The block `F/p` tests whether the three-coordinate block predicts relative
effect dominance. Per-axis slopes and permutation p-values describe direction:

| Axis | Positive slope means |
|---|---|
| `mni_y` | LWPC dominance increases anteriorly |
| `mni_z` | LWPC dominance increases superiorly |
| `mni_x` | LWPC dominance increases toward more positive/rightward x; interpret only within hemisphere |

The code reports an `all` fit and, when possible, `lh` and `rh`. Prefer the
hemisphere-specific x slopes because mirrored bilateral x coordinates can
cancel. Treat axis p-values as follow-ups to the coordinate-block test and be
explicit about any multiplicity policy used in the manuscript.

The coordinate model is secondary to the coverage-conditioned categorical
anatomy test, but it is the preferred analysis for a directional claim such as
“LWPC dominance is more anterior.” It uses all positions rather than reducing a
spatial distribution to one point.

---

## 6. Brain maps of continuous scores

The job writes five maps from the same `scores_with_anatomy.csv`:

| Base filename | Values | Colour scale | What it shows |
|---|---|---|---|
| `score_map_lwpc_s.png` | signed pooled-scaled LWPC | symmetric `coolwarm` | direction and location of LWPC adaptation |
| `score_map_lwps_s.png` | signed pooled-scaled LWPS | symmetric `coolwarm` | direction and location of LWPS adaptation |
| `score_map_abs_lwpc.png` | `abs(lwpc_s)` | sequential `viridis` | LWPC magnitude regardless of sign |
| `score_map_abs_lwps.png` | `abs(lwps_s)` | sequential `viridis` | LWPS magnitude regardless of sign |
| `score_map_delta.png` | `lwpc_s - lwps_s` | symmetric `coolwarm` | the relative map tested by N4 |

Each map bins a continuous value into nine colours because the shared brain
renderer accepts one colour per electrode set. Its scale is clipped at the 98th
percentile so a few extremes do not flatten the remaining electrodes. The
adjacent `*_colorbar.png` records the actual range; do not compare hues between
figures without their own colourbars.

Signed and delta scales are centred on zero. Magnitude scales are sequential.
The maps pool electrodes for display, so dense implants are visually prominent.
They are **visualizations, not independent evidence**, and should be read with
`coverage_matrix.csv`, the ROI test, and the leave-one-subject-out table.

If PyVista/MNE/reconstruction/display dependencies are unavailable, the code
writes the colourbar and a `*_by_roi.png` fallback instead. That is a successful
statistical run, but it is not a rendered cortical surface. The Slurm wrapper
uses `xvfb-run` to provide a virtual display; missing recon/template data can
still trigger the fallback.

### `delta_by_roi.png`

This flat companion figure is always attempted. Bars are mean `delta` ± SEM,
dots are electrodes, and labels report electrode and covered-subject counts.
The zero line separates relative LWPC dominance from relative LWPS dominance.
The error bars are descriptive; significance comes from the swap test and the
subject/responsiveness-adjusted statistics, not whether a bar's SEM crosses zero.

### `joint_scatter.png`

Every point is an electrode (`lwpc_s` on x, `lwps_s` on y). It reports both the
pooled and within-subject correlation and, when `per_split.csv` is supplied,
annotates the split-half ceiling. This is useful context for whether the effects
share electrodes, but it is not the primary anatomy interaction.

---

## 7. Reliability and the noise ceiling

A low LWPC–LWPS spatial correlation is uninterpretable unless each map is
reliable enough to correlate with anything. `map_reliability` uses the
split-resolved table:

```text
between = mean over splits of ½[corr(xA, yB) + corr(xB, yA)]
LWPC reliability = mean corr(xA, xB)
LWPS reliability = mean corr(yA, yB)
noise-corrected between = between / sqrt(rel_LWPC × rel_LWPS)
```

It computes these at both electrode and parcel level. The cross-half pairing
prevents shared trial noise from inflating similarity.

Read the result comparatively:

- `between` near the two within-effect reliabilities: maps are about as similar
  as measurement permits;
- `between` well below healthy positive reliabilities: evidence for different
  maps is more credible;
- either reliability near zero or negative: the between-map correlation and
  attenuation correction are not interpretable; collect/aggregate better or
  soften the claim;
- corrected values can exceed ±1 under noisy finite-sample estimates; treat them
  as an instability warning, not a literal correlation.

The anatomy job also reruns the disjoint-half electrode correlation with
`min_elec` equal to 1, 2, and 3. `min_elec` drops whole subjects that have too few
eligible electrodes, so changes across `min_elec_sweep.csv` reveal an important
sample-definition sensitivity.

**Always pass `PER_SPLIT_CSV`.** The job is allowed to run without it, but then
the ceiling and min-electrode sweep are absent and the summary explicitly marks
the spatial correlation as unreportable.

---

## 8. Centres / medoids — descriptive only

The centre panel exists to describe the maps in millimetres, not to establish
anatomical separation. A single pooled centroid is invalid here because subjects
with more electrodes move the location estimate, clinical coverage can mimic a
physiological centre, bilateral distributions can put a centroid in the
midline/white matter, and signed weights can cancel.

The implemented version avoids the worst failures:

- one LWPC centre and one LWPS centre per **subject × hemisphere**;
- the same electrodes for both effects;
- non-negative weights `abs_lwpc` and `abs_lwps`;
- at least three electrodes in a subject × hemisphere group;
- a weighted **medoid** by default—the observed electrode minimizing weighted
  distance to the other electrodes;
- LWPC-minus-LWPS displacement within group; and
- the same within-electrode score-label-swap null.

`score_centers.csv` contains:

| Column | Meaning |
|---|---|
| `subject`, `hemi` | independent descriptive group |
| `n_electrodes` | sites available to locate both medoids |
| `dx`, `dy`, `dz` | LWPC medoid minus LWPS medoid in mm |
| `distance` | Euclidean separation in mm |
| `p_anterior` | within-group swap p for `dy`; descriptive diagnostic |

The summary averages displacement over the eligible subject × hemisphere
groups. Positive `dy` means the LWPC medoid is anterior to the LWPS medoid;
positive `dz` means superior; positive `dx` means more positive x. Lead with the
coordinate regression for an anterior/posterior claim and present the medoids as
an intuitive secondary panel. Do not call their p-values the primary anatomical
test.

---

## 9. How to run it

Run commands from the repository root unless a command explicitly changes
directory.

### 9.1 First validate the entire path with synthetic data

This exercises scores, atlas joins, primary and coordinate tests, reliability,
maps, centres, persistence, and summary writing without accessing real epochs:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=synthetic N_PERM=1000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

The synthetic default plants a spatial gradient. Also run the null:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=synthetic SYNTHETIC_ENRICHMENT=0 N_PERM=1000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Use 1,000 permutations only as a path check. Use the default 10,000 for the
reported result.

### 9.2 Produce reusable real-data scores

Edit the hard-coded `EPOCHS_ROOT_FILE`, `WINDOW_TMIN`, and `WINDOW_TMAX` near the
top of `submit_stability_flexibility_segregation_dcc.sh` so they match the
predeclared analysis. Then run the anatomically defined whole-brain score set:

```bash
cd dcc_scripts/stats
ROIS=all ELECTRODES=all CONTRAST_MODE=proportion EFFECT_MEASURE=cohens_d \
N_SPLITS=200 N_PERM_CORR=10000 N_PERM_LABEL=2000 \
  bash submit_stability_flexibility_segregation_dcc.sh
```

Why these values matter:

- `ROIS=all`, `ELECTRODES=all`: no effect-based or lPFC-only selection before
  the whole-brain anatomy question;
- `CONTRAST_MODE=proportion`: score the LWPC/LWPS interactions, not congruency
  and switch main effects;
- `EFFECT_MEASURE=cohens_d`: the signed, balanced, unit-free measure specified
  for N4;
- `N_SPLITS=200`: stable disjoint-half averages and a split-resolved ceiling.

Do **not** use `SCATTER_ONLY=1` as the N4 input. That path is diagnostic and does
not write the complete full-run product set expected here.

The upstream directory is printed as `Save dir:` in the Slurm log and normally
has this shape:

```text
dcc_scripts/stats/results/<epochs>/segregation_results/
  window_<tmin>to<tmax>s_all_all_rois_proportion_cohens_d_fdr_bh/
```

Verify that it contains at least `electrodes.csv` and `per_split.csv`.

### 9.3 Run N4 from those finished scores — recommended route

Set both paths to the files from the **same** segregation run:

```bash
cd dcc_scripts/stats
SEG_DIR="/absolute/path/to/segregation_results/window_..."

ARM=continuous \
SCORES_CSV="$SEG_DIR/electrodes.csv" \
PER_SPLIT_CSV="$SEG_DIR/per_split.csv" \
ROI_FILTER='' ANAT_LEVEL=group \
MIN_SUBJECTS=3 N_PERM=10000 SEED=0 \
MAKE_BRAIN=1 BRAIN_HEMI=both USE_COORDS=1 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Environment variables not explicitly repeated in the submitter's
`--export=...` list still reach Slurm through `--export=ALL`; the shell-prefix
form above is therefore intentional. Use absolute CSV paths because the batch
job's working directory may differ.

The score-CSV route does not load epochs. It still needs the electrode atlas and,
for coordinate/centre panels, reconstruction files. Set `ROI_DICT_DIR` if the
atlas JSON cache is not in its normal lab location.

### 9.4 Restricted lPFC / Destrieux follow-up

For a predeclared lPFC analysis using raw labels inside lPFC:

```bash
cd dcc_scripts/stats
SEG_DIR="/absolute/path/to/the/same/wholebrain/segregation/run"

ARM=continuous \
SCORES_CSV="$SEG_DIR/electrodes.csv" \
PER_SPLIT_CSV="$SEG_DIR/per_split.csv" \
ROI_FILTER=lpfc ANAT_LEVEL=destrieux HIST_TOP_N=20 \
MIN_SUBJECTS=3 N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

`ANAT_LEVEL=auto` would also select Destrieux here. `HIST_TOP_N` belongs to the
categorical histogram path and does not change the continuous primary test or
`delta_by_roi.png`; setting it here is harmless but not necessary.

### 9.5 Compute scores inside the anatomy job — supported but expensive

If no upstream score run exists, omit `SCORES_CSV` and `PER_SPLIT_CSV` and let
the anatomy job load epochs and score them:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=real ROI_FILTER='' ANAT_LEVEL=group \
CONTRAST_MODE=proportion ELECTRODES=all N_SPLITS=200 N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Before doing this, edit the anatomy submitter's hard-coded `EPOCHS_ROOT_FILE`,
window, and electrode setting. The submitter currently sets `ELECTRODES=sig`, so
the command-line prefix alone cannot override that assignment; change it to
`all` in the file for the anatomically defined N4 population. The DCC core pins
the in-job estimator to `EFFECT_MEASURE='cohens_d'`; it is not an environment
variable exposed by the anatomy runner. Reusing a verified segregation
`electrodes.csv` is nevertheless clearer and safer.

### 9.6 Run without reconstruction/display support

The primary ROI test needs anatomy labels but not coordinates or 3-D rendering:

```bash
cd dcc_scripts/stats
ARM=continuous SCORES_CSV="/abs/path/electrodes.csv" \
PER_SPLIT_CSV="/abs/path/per_split.csv" USE_COORDS=0 MAKE_BRAIN=0 \
ANAT_LEVEL=group N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

This intentionally omits the coordinate test, cortical surfaces, and centres.
It still produces the primary test, coverage table, ROI figure, reliability,
scatter, leverage sweep, CSVs, JSON, and text summary.

### 9.7 Local invocation for debugging

The entry point can be run directly from the repository root, although real
data/recon path discovery is lab-specific:

```bash
ARM=continuous DATA_SOURCE=synthetic N_PERM=200 MAKE_BRAIN=0 \
python dcc_scripts/stats/run_stability_flexibility_anatomy_dcc.py
```

The Slurm wrapper is preferable for real maps because it launches Python under
`xvfb-run`.

---

## 10. Output directory and every output

The runner constructs:

```text
dcc_scripts/stats/results/<tag>/
  anatomy_<label_source>_<scope>_window_<tmin>to<tmax>s_<electrodes>/
    continuous/
```

On the CSV route `<tag>` is not derived from the score filename. With the default
`LABEL_SOURCE=a1` and no epoch directory it currently resolves to the synthetic
fallback tag; with `LABEL_SOURCE=power_traces` it resolves to `power_traces`.
This is only an output-path naming quirk—the loaded CSV remains the data source.
Trust the `Save dir:` line in the job log and archive the exact command/CSV paths
with the result. All files below live in `continuous/`.

### 10.1 Read these first

| File | Contents | How to use it |
|---|---|---|
| `summary.txt` | human-readable primary F/p, per-anatomy rows, LOSO, coordinates, centres, ceiling, correlations | Start here; it states skipped components explicitly. |
| `score_anatomy.json` | machine-readable compact summary of the same analysis | Manuscript/table automation and provenance. It excludes the full permutation arrays. |
| `delta_by_roi.png` | mean ± SEM and electrode dots for the tested relative score | Flat visual companion to the primary model. |
| `coverage_matrix.csv` | subject × tested anatomy presence/absence | Required context for every anatomical conclusion. |

### 10.2 Input/intermediate audit tables

| File | Important columns / interpretation |
|---|---|
| `scores.csv` | loaded or freshly computed per-electrode input (`x`, `y`, `resp`). This is a copy inside the run for provenance. |
| `per_split.csv` | `xA`, `xB`, `yA`, `yB` for every split/electrode; source of reliability and `min_elec` sensitivity. |
| `scores_with_anatomy.csv` | canonical audit table: raw scores, scaled scores, magnitudes, `delta`, ROI, Destrieux label, and coordinates. Use this to reproduce plots or inspect exclusions. |

In `scores_with_anatomy.csv`, check:

- unique `electrode_id` and sensible `subject` counts;
- fraction missing `roi` and `anat`;
- fraction missing MNI coordinates (coordinates may be missing without harming
  the primary ROI test);
- finite `lwpc_s`, `lwps_s`, and `delta`;
- that the requested `ROI_FILTER` was actually applied; and
- extreme values before accepting a percentile-clipped display.

### 10.3 Primary test and robustness tables

| File | Columns | Interpretation |
|---|---|---|
| `delta_per_roi.csv` | anatomy label, electrode/subject N, raw and adjusted means, p, q | Follow-up anatomy rows after the omnibus test. Sign gives relative dominance. |
| `delta_roi_loso.csv` | dropped subject, observed F, p, electrode N | Subject leverage; compare F against `(none)`. |
| `min_elec_sweep.csv` | threshold, cross-effect correlation/p, N, reliabilities, corrected r | Sensitivity to dropping subjects with fewer than 1/2/3 electrodes. Not the primary ROI test. |

### 10.4 Map and scatter figures

| Files | Interpretation |
|---|---|
| `joint_scatter.png` | electrode-level LWPC–LWPS relationship with pooled/within-subject diagnostics and ceiling annotation |
| `score_map_<value>.png` | rendered surface for each of the five values |
| `score_map_<value>_colorbar.png` | required scale for that map |
| `score_map_<value>_by_roi.png` | fallback only when the surface cannot render |

Depending on `BRAIN_HEMI` and the rendering helper, additional view-specific
files may accompany the combined map. `score_anatomy.json` records the combined
path actually returned, or the fallback path.

### 10.5 Centre table

`score_centers.csv` is written only if at least one subject × hemisphere group
has coordinates and at least three eligible electrodes. Its absence can simply
mean insufficient coordinates/coverage; read `summary.txt` and the job log
before treating absence as a pipeline error.

---

## 11. Interpretation checklist

Use this order. It prevents an attractive surface or centroid from outrunning
the actual test.

### A. Validate the population and coverage

- [ ] `ARM=continuous`, `CONTRAST_MODE=proportion`, and the intended time window.
- [ ] Scores came from an anatomically defined set (`all` for whole brain), not
      LWPC/LWPS-significant electrodes.
- [ ] `SCORES_CSV` and `PER_SPLIT_CSV` came from the same upstream run.
- [ ] Mapped electrode and subject counts are plausible.
- [ ] Retained anatomical units meet `MIN_SUBJECTS`; report their coverage.
- [ ] Missing coordinates affect only coordinate/maps/centres, not the primary
      label-based ROI test.

### B. Read the primary interaction

- [ ] Read the omnibus `F` and permutation `p` in `summary.txt`.
- [ ] If significant, use adjusted means and BH `q` in `delta_per_roi.csv` to
      describe where relative dominance lies.
- [ ] Say “relative LWPC/LWPS balance varies by anatomy,” not “LWPC is present
      here and LWPS absent there.”
- [ ] Do not interpret `delta` without checking both signed component maps.

### C. Check robustness

- [ ] Inspect LOSO changes in **F**, especially the largest shift.
- [ ] Compare pooled and within-subject LWPC–LWPS correlations.
- [ ] Inspect `min_elec=1,2,3` rather than reporting only the default survivor
      set.
- [ ] Interpret between-map similarity only relative to both split-half
      reliabilities.
- [ ] Flag undefined/unstable noise correction when reliability is non-positive.

### D. Add secondary spatial descriptions

- [ ] Coordinate block test first, then directional slopes.
- [ ] Prefer hemisphere-specific interpretation of `mni_x`.
- [ ] Use maps to show the data and coverage, not as independent tests.
- [ ] Describe weighted medoids in millimetres as convergent/descriptive only.

### Suggested result language

If the omnibus test is significant and robust:

> The within-electrode balance of pooled-scaled LWPC and LWPS scores varied
> across coverage-eligible anatomical units (swap-permutation omnibus F = …,
> p = …). Adjusted relative scores were LWPC-dominant in … and LWPS-dominant in
> … (BH q = …). The statistic remained similar in leave-one-subject-out fits,
> and the spatial comparison was interpretable given split-half reliabilities of
> … and … .

If it is null with a healthy ceiling:

> Relative LWPC/LWPS balance did not vary detectably across coverage-eligible
> anatomical units (F = …, swap-permutation p = …). Split-half spatial
> reliabilities were … and …, and cross-effect similarity was …; thus the null
> is not readily explained by wholly unreliable maps.

If it is null with a poor ceiling:

> The anatomy interaction was not significant, but the LWPC and/or LWPS map had
> low split-half reliability. The analysis therefore cannot distinguish a
> shared anatomical organization from insufficient precision, and the null is
> inconclusive.

---

## 12. Parameters that change the answer

| Variable | Default in runner | Recommendation / effect |
|---|---:|---|
| `ARM` | `categorical` | **Set `continuous`.** |
| `SCORES_CSV` | unset | Recommended: full segregation run's `electrodes.csv`. |
| `PER_SPLIT_CSV` | unset | Recommended/needed for reportable ceiling and min-electrode sweep. |
| `N_SPLITS` | `200` | Used only if scoring inside anatomy; more splits stabilize estimates but cost time. |
| `CONTRAST_MODE` | `proportion` | Must remain `proportion` for LWPC/LWPS N4. |
| `ROI_FILTER` | whole brain | Changes the tested population; predeclare whole-brain primary vs regional follow-up. |
| `ANAT_LEVEL` | `auto` | Whole brain → coarse groups; one-group restriction → Destrieux. Explicit values improve provenance. |
| `MIN_SUBJECTS` | `3` | Coverage threshold; higher is more conservative but drops anatomical units. |
| `N_PERM` | `10000` | Primary swap-null resolution; lower only for dry runs. |
| `SEED` | `0` | Reproducibility of permutations and plot jitter. |
| `USE_COORDS` | `1` | Controls coordinate test and centres, not the primary ROI test. |
| `MAKE_BRAIN` | `1` | Controls five surface/fallback maps, not statistics. |
| `BRAIN_HEMI` | `both` | Display choice (`both`, `lh`, `rh`, `split`). `split` draws one hemisphere per panel in a window twice as wide. |
| `BRAIN_ZOOM` | renderer default | Per-panel camera zoom (`<1` zooms out). Display choice only; lower it to widen the gap between the `split` hemispheres. |
| `ELECTRODES` | runner `all`, submitter `sig` | CSV route inherits its input population; in-job scoring must be changed to `all` for N4. |

Two non-obvious distinctions:

1. `ALPHA` labels significance in the text and is used by some score estimators;
   the N4 primary decision still comes from the reported permutation `p`.
2. `FDR_CORRECTION` and `LABEL_SOURCE` largely describe the categorical arm;
   do not mistake them for filtering continuous scores loaded from CSV.

---

## 13. Common failure modes

### “My job ran categorical anatomy”

`ARM` was omitted. Rerun with `ARM=continuous`; do not interpret the categorical
S/F table as beats 5–7.

### “The anatomy job says the ceiling was not computed”

`PER_SPLIT_CSV` was omitted or unreadable. Point it to `per_split.csv` from the
same segregation run as `SCORES_CSV`.

### “Only lPFC electrodes are present in my whole-brain run”

The upstream segregation submitter defaults to `ROIS=lpfc`. Recompute scores
with `ROIS=all`; changing only `ROI_FILTER` cannot restore electrodes that never
entered `electrodes.csv`.

### “The in-job run used significant electrodes”

The anatomy submitter assigns `ELECTRODES=sig`. Edit it to `all`, or—preferably—
provide a verified all-electrode score CSV.

### “No cortical surfaces appeared”

Read the Slurm log for the renderer exception and look for `*_by_roi.png` files.
Statistics may be complete. Confirm `xvfb-run`, PyVista/MNE, fsaverage/template
surfaces, and subject reconstructions are available.

### “No coordinate result or `score_centers.csv`”

Either `USE_COORDS=0`, reconstruction lookup failed, coordinates were missing,
or no subject × hemisphere had three eligible electrodes. This does not
invalidate the label-based primary test.

### “The omnibus is significant but no row has q < .05”

The omnibus asks whether any between-unit variation exists collectively;
row-level BH tests ask which adjusted unit means differ from zero. These are not
identical questions. Report the omnibus and avoid naming a specific unit as the
driver unless its corrected follow-up supports that claim.

### “The delta map looks dramatic but the primary test is null”

The display pools electrodes, clips extremes, and does not account visually for
subject leverage or coverage. Trust the prespecified swap test and LOSO result;
describe the map as exploratory/descriptive.

### “A medoid p-value is smaller than the ROI-test p-value”

Do not promote it. The centre statistic answers a narrower, coverage-sensitive
location-summary question. The plan explicitly makes it descriptive; the ROI
interaction and coordinate regression remain the inferential analyses.

---

## 14. Reproducibility and tests

Before a production run:

```bash
pytest tests/analysis/stats/test_stability_flexibility_anatomy.py -v
```

The continuous-arm tests verify pooled scaling without losing tiny subjects,
electrode-ID/anatomy/coordinate joins, recovery of planted and null ROI effects,
coverage filtering, sign invariance of the swap test, coordinate-gradient
recovery, reliability degradation with noise, real-electrode medoids, LOSO
coverage, and surface-render fallback behavior.

For every reported run archive:

- the git commit;
- the exact submission command and Slurm log;
- absolute upstream CSV paths;
- epoch directory, window, electrode and ROI scope;
- `ANAT_LEVEL`, `MIN_SUBJECTS`, `N_SPLITS`, `N_PERM`, and seed;
- `summary.txt`, `score_anatomy.json`, all CSVs, and all figures; and
- warnings about missing atlas labels, coordinates, or render fallback.

Related reading:

- [`n2_direction_tests.md`](n2_direction_tests.md) — the direction test that
  should be settled before N4;
- [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md)
  §§5–7 — the statistical rationale and reporting hierarchy;
- [`stability_flexibility_data_flow.md`](stability_flexibility_data_flow.md) —
  broader A1–A6 data flow; and
- [`stability_flexibility_outputs_guide.md`](stability_flexibility_outputs_guide.md)
  — the wider segregation/anatomy output family.

---

## 15. Findings from the lPFC run

Results from the continuous arm on the lPFC-restricted electrode set
(**398 electrodes, 22 subjects**, 254 lh / 144 rh, `roi == 'lpfc'`), using
`scores_with_anatomy.csv`, `per_split.csv` (200 splits × 4 columns: `xA`/`xB` =
LWPC on disjoint trial halves A and B, `yA`/`yB` = LWPS on the same halves) and
the pipeline's own `summary.txt`. Permutation p-values, 20 000 permutations
unless stated.

### 15.1 Takeaway

**Two results, both now confirmed against the split-half data, and one question
settled in the negative.**

1. **The two effects share their reliable spatial variance.** LWPC and LWPS are
   carried by the same electrodes, the correlation is not an artefact of shared
   trials, and after the correct reliability correction the two maps are
   correlated **at their noise ceiling** — whatever is reliably mapped is
   essentially common to both. Mixed selectivity, not segregated subpopulations.
2. **Within that shared population, the LWPC-versus-LWPS balance shifts along
   the dorsoventral axis**, and the gradient **replicates across disjoint trial
   halves** (98.5 % sign agreement; cross-validated p = 0.005). Roughly 91 % of
   the observed slope is signal.
3. **The magnitude hypothesis is dead, not merely unsupported.** Tested with an
   unbiased cross-validated estimator, there is no dorsoventral gradient in
   either effect's magnitude (p = 0.40). The effect is about which effect
   *dominates in sign*, not about which is *larger*.

Two cautions that shape how this is presented:

- **Per-electrode maps are mostly noise** (split-half reliability ≈ 0.18 at half
  data, ≈ 0.30 at full data). Individual electrodes must not be interpreted; the
  reliable quantities are the low-dimensional summaries — the correlation and
  the gradient.
- **`summary.txt`'s noise-corrected correlations (+1.374, +1.369) are out of
  range and should not be reported.** The cause was a rank-transform artefact,
  now fixed in `map_reliability`; [§15.3](#153-how-reliable-are-the-maps) has
  the corrected values.

The dorsoventral axis was still one of three tested, and the cross-validation
below controls trial noise, not subject sampling — see
[§15.9](#159-what-to-do-next).

### 15.2 Vocabulary used in this section

Each electrode carries two **signed** scores: positive means the effect runs in
the direction behaviour predicts, negative means it runs the other way.

```text
delta = lwpc_s - lwps_s        # positive = LWPC-dominant, negative = LWPS-dominant
```

Electrodes are grouped by whether their two scores point the same way:

- **concordant** — both scores have the **same** sign (`++` or `--`).
- **discordant** — the scores have **opposite** signs (`+-` or `-+`).

This matters because `delta` mixes two things a reader might not want mixed:

| electrode | `lwpc_s` | `lwps_s` | `delta` | group | bigger in magnitude |
|---|---|---|---|---|---|
| A | +1.5 | +0.5 | **+1.0** | concordant `++` | LWPC |
| B | −1.5 | −0.5 | **−1.0** | concordant `--` | LWPC |
| C | +1.0 | −1.0 | **+2.0** | discordant `+-` | neither (tied) |
| D | +1.0 | +1.0 | **0.0** | concordant `++` | neither (tied) |

A and B have identical magnitude relationships but **opposite deltas**; C has the
largest `delta` despite its two effects being equal in size.
[§15.7](#157-why-magnitude-versions-do-not-work) resolves that ambiguity
empirically.

### 15.3 How reliable are the maps?

Reliability must be computed **within a split**, where `xA` and `xB` are
genuinely disjoint trial halves:

| split-half reliability (electrode level) | LWPC | LWPS |
|---|---|---|
| Spearman (the `method` default, used for `between`) | +0.162 | +0.097 |
| Pearson (used for the noise correction) | +0.178 | +0.163 |

⚠️ **Do not compute reliability after averaging over splits.** The A-half of one
split overlaps the B-half of another, so `corr(x̄A, x̄B)` across split-averaged
maps returns **+0.987** — an artefact, not a ceiling.

⚠️ **Do not Spearman-Brown these upward either.** Both sides of
`map_reliability` are already half-length — `between` correlates two half-trial
estimates and so do the reliabilities — so numerator and denominator sit at the
same trial count and the ratio is already the attenuation correction it should
be. Correcting only the denominator to full length would understate it.

⚠️ **The out-of-range correction was a rank-transform artefact.**
`summary.txt` reported `noise-corrected +1.374`, impossible for a correlation.
The cause is that the attenuation formula is classical-test-theory algebra for
**Pearson** correlations, while `method` defaults to Spearman. Ranking deflates
the self-reliabilities much more than the cross term:

| | `between` | ceiling √(rel·rel) | corrected |
|---|---|---|---|
| Spearman (as previously reported) | +0.172 | 0.125 | **+1.374** ❌ |
| Pearson (correct algebra) | +0.174 | 0.170 | **+1.022** ✅ |

Note `between` is nearly identical either way (0.172 vs 0.174) — only the
ceiling moves. This is not sampling noise: 95 % of individual splits exceed 1
under Spearman, and the bootstrap interval on the Pearson value is
**[0.998, 1.046]**. `map_reliability` now always computes
`between_noise_corrected` from Pearson, reports
`between_noise_corrected_ci`, and sets `note` when the value is out of range or
undefined.

A corrected value of ~1.0 is itself the finding: **the two maps are correlated
at their ceiling** — whatever is reliably mapped is essentially common to both.

⚠️ **The parcel-level ceiling is not estimable and should be dropped.**
`summary.txt` reports `noise-corrected +1.369` over 21 Destrieux parcels, but
the Pearson reliability of the LWPC parcel map is **−0.026** — no recoverable
signal — so the ratio is undefined, and `map_reliability` now returns `nan`
with a note rather than a number. Quote `between` (+0.275) and the
reliabilities instead.

**What low reliability does and does not invalidate.** A per-electrode
reliability of ~0.30 means individual electrode scores are mostly noise, so no
single electrode should be interpreted and any dot map is largely noise. It does
**not** invalidate a gradient: the regression compresses 398 noisy electrodes
into 3 slope parameters, and averaging suppresses noise. A noisy
high-dimensional map with a reliable low-dimensional summary is the normal case,
and [§15.5](#155-a-dorsoventral-gradient-in-the-relative-balance) verifies the
summary directly rather than inferring it.

### 15.4 The two effects share their reliable variance

The per-electrode scores are centred near zero, not positive:

| | mean | median | fraction < 0 |
|---|---|---|---|
| `lwpc_s` | +0.034 | −0.024 | **51.0 %** (203/398) |
| `lwps_s` | +0.181 | +0.096 | 43.7 % (174/398) |

Positive-LWPC and positive-LWPS electrodes **co-occur above chance**: 135
positive on both, against a within-subject permutation null of 121.6 ± 4.2
(**p = 0.0016**); independence predicts ~110.

**The correlation is not shared-trial noise.** Because the averaged scores draw
on both halves, the correlation was recomputed across **disjoint** halves:

| | pooled r | within-subject r |
|---|---|---|
| `xA` vs `yA` (shares trials) | +0.169 | +0.116 |
| `xB` vs `yB` (shares trials) | +0.200 | +0.146 |
| **`xA` vs `yB`** (disjoint) | **+0.174** | **+0.126** |
| **`xB` vs `yA`** (disjoint) | **+0.174** | **+0.114** |

Same-half and disjoint-half estimates agree, so no shared-trial inflation. On
the split-averaged maps the disjoint-half within-subject correlation is
**r = +0.243, p = 0.0005** (summary.txt reports +0.224 for the equivalent
quantity).

**Claim supported, and strengthened by §15.3:** the two effects are carried by an
overlapping population, and their reliable spatial variance is essentially
*entirely* shared (noise-corrected r ≈ 1.0).

**Consequence for figures.** Two thresholded maps will *look* disjoint whatever
the truth: at marginal positive rates of 49 % and 56 %, independence alone leaves
only ~28 % of electrodes in both maps. Apparent segregation in a thresholded dot
map is not evidence of segregation.

### 15.5 A dorsoventral gradient in the relative balance

`relative_score_coordinate_test(value_col='delta')`: block **F = 2.87,
p = 0.032**; **z slope = −0.0077 per mm, p = 0.0075** (~0.6 pooled SD across the
dorsoventral extent of the coverage). Negative slope means **LWPS dominance
increases dorsally, LWPC dominance increases ventrally**.

**Cross-validated confirmation.** The slope was refitted independently on each
disjoint trial half of all 200 splits:

| | result |
|---|---|
| mean z slope, half A | **−0.00780** |
| mean z slope, half B | **−0.00759** |
| sign agreement between halves | **98.5 %** (both negative) |
| E[slope_A × slope_B] | **+4.96 × 10⁻⁵** |
| within-subject coordinate-permutation null | p = **0.0050** |

Because A and B are disjoint trials, E[slope_A · slope_B] is unbiased for
slope², so a positive value means a real gradient with no noise floor to
subtract. √(4.96 × 10⁻⁵) = 0.0070 against an observed slope of 0.0077 — about
**91 % of the observed slope is signal**. This is the confirmation the earlier
revision of this section listed as outstanding.

| robustness check | result |
|---|---|
| add hemisphere dummy to nuisance design | z slope −0.0074, p = 0.0094 |
| `z × hemisphere` interaction | coef +0.0031, **p = 0.65** (one shared slope) |
| leave-one-subject-out | z stays p < 0.05 in **21/22** folds (worst: drop D0146 → p = 0.090) |
| drop the `resp` covariate | z p = 0.0070 |

The per-hemisphere fits (lh z p = 0.21, rh z p = 0.98) are **not** a
non-replication: the interaction test finds no heterogeneity to explain and the
subsets are underpowered. Only 6 of 22 subjects are bilateral, so the subject
dummies already absorb most of hemisphere.

**Corroboration at the parcel level.** `relative_score_roi_test(roi_col='anat')`:
omnibus **F = 1.90, p = 0.0099** (summary.txt: p = 0.0104, different seed). The
extremes order dorsoventrally, matching the continuous slope:

| Destrieux label | n | adj. mean `delta` | p | q |
|---|---|---|---|---|
| `lh_S_front_sup` | 33 | −0.54 | 0.009 | 0.126 |
| `lh_G_front_sup` | 47 | −0.30 | 0.013 | 0.126 |
| `rh_G_front_middle` | 34 | +0.41 | 0.020 | 0.129 |
| `lh_G_front_inf-Triangul` | 26 | +0.31 | 0.124 | 0.337 |

No label survives FDR (min q = 0.13), so **the omnibus is the claim**.

**Is this compatible with §15.4?** Yes. "Correlated at the ceiling" is a global
scalar summary with wide error bars at these reliabilities; it leaves room for a
small systematic difference concentrated on one spatial axis. The two results
together say: **a dominant shared component, plus a small but reliable
dorsoventral difference.**

### 15.6 The anterior–posterior hypothesis is null

`delta ~ y`: p = 0.58 pooled, 0.77 lh, 0.22 rh. The §8 centre machinery is built
around this axis (`p_anterior`). Report it as a null, not a pending result.

### 15.7 Why magnitude versions do not work

Every magnitude-based formulation is null:

| value tested | z slope | z p | null used |
|---|---|---|---|
| `delta = lwpc_s − lwps_s` | **−0.0077** | **0.0075** | sign-flip swap |
| `abs_lwpc − abs_lwps` | +0.0023 | 0.32 | sign-flip swap |
| **cross-validated μ²: E[xA·xB] − E[yA·yB]** | **+0.0046** | **0.40** | sign-flip swap |
| `lwpc_s` alone | −0.0037 | 0.12 | within-subject permutation |
| `lwps_s` alone | +0.0040 | 0.062 | within-subject permutation |
| `delta`, both-positive electrodes only (n = 135) | −0.0059 | 0.26 | sign-flip swap |

**The magnitude question is now settled, not merely unsupported.** `|score|` is a
biased magnitude estimator — for a null electrode E`|score|` ≈ 0.8 σ, pure noise
floor — so the null on `abs_lwpc − abs_lwps` was weak evidence. The unbiased
estimator uses the disjoint halves directly: for independent estimates x₁, x₂ of
the same effect, E[x₁·x₂] = μ², with no rectification bias. Computed that way:

- mean μ²: LWPC **+0.300**, LWPS **+0.310** — the two effects are the same size
  overall;
- gradient of the μ² difference along z: **p = 0.40** (block F p = 0.109);
- corr(μ²_LWPC, z) = **+0.028**; corr(μ²_LWPS, z) = **−0.059**.

There is no dorsoventral magnitude gradient to find, with the best available
estimator.

**Is `delta` just picking up sign disagreement?** Tested by splitting on
concordance — these partitions are invariant under the per-electrode label swap,
so the null stays valid inside each (a split on the *sign of `delta`* would not
be, since the swap moves electrodes across it):

| subset | n | `delta` z slope | p |
|---|---|---|---|
| all | 398 | −0.0077 | 0.0075 |
| **concordant (`++` or `--`)** | **249** | **−0.0082** | **0.0032** |
| discordant (`+-` or `-+`) | 149 | −0.0149 | 0.029 |
| `++` only | 135 | −0.0059 | 0.26 |
| `--` only | 114 | −0.0079 | 0.023 |

**No** — the gradient is present in the concordant electrodes alone, slightly
*more* cleanly than in the full sample, with the same sign in every quadrant.

But the magnitude contrast is null even there (+0.0008, **p = 0.76**), and
electrodes A and B in [§15.2](#152-vocabulary-used-in-this-section) show why:
among `++` electrodes `delta = abs_lwpc − abs_lwps`, while among `--` electrodes
`delta = −(abs_lwpc − abs_lwps)`. The two concordant quadrants contribute
**opposite-signed magnitude gradients that cancel when pooled**, while their
`delta` gradients agree.

Finally, **neither score's own sign varies with z** — only their ordering does:

| | r | p |
|---|---|---|
| P(`lwpc_s` > 0) vs z | −0.021 | 0.68 |
| P(`lwps_s` > 0) vs z | +0.043 | 0.42 |
| **P(`delta` > 0) vs z** | **−0.144** | **0.0058** |

A *relative reordering* along the dorsoventral axis, not a sign reversal of
either effect. The last row is the most presentable form of the result — the
fraction of LWPC-dominant electrodes falls dorsally, stated as a proportion,
with no negative values on display.

⚠️ **Null validity.** The sign-flip null in `_swap_null` is valid only for a
paired difference, where negation equals the label swap. It is valid for
`abs_lwpc − abs_lwps` and for the cross-validated μ² difference, and **invalid**
for any single score — passing `value_col='lwpc_s'` or `'abs_lwpc'` to
`relative_score_coordinate_test` tests nothing. The single-score rows use a
within-subject permutation instead.

### 15.8 Why the centres are null, and how to draw them honestly

`score_centers_per_subject` is null on every axis under every weighting:

| weighting / centre | dx (p) | dy (p) | dz (p) |
|---|---|---|---|
| `abs`, `medoid=True` | +0.69 (0.59) | +0.46 (0.80) | −3.26 (0.15) |
| `abs`, `medoid=False` | +0.61 (0.35) | +0.89 (0.33) | +1.42 (0.26) |
| positive-clipped, `medoid=True` | +2.08 (0.29) | −0.86 (0.76) | −1.95 (0.56) |

**Expected, not a contradiction.** The centres weight by `|score|`, and as
`_synthetic_scores` states, they are "blind to a purely signed dissociation" —
which [§15.7](#157-why-magnitude-versions-do-not-work) now establishes with an
unbiased estimator. A null centre does not qualify §15.5.

Two implementation hazards, both visible here:

- `medoid=True` returns **exactly zero** displacement for 6 of 25
  subject × hemisphere groups (including groups of 26, 20, 15 and 15
  electrodes). The medoid takes only *n* discrete values, and on clustered depth
  shafts the weighted-distance argmin is insensitive to the weights, so both
  labels snap to the same contact. `medoid=False` produces none.
- The two centre definitions **disagree in sign on dz** (−3.26 vs +1.42).
- Positive clipping leaves 2 groups with all-zero LWPC weights and 1 with
  all-zero LWPS weights — undefined centres returned as zero.

**A centre figure that depicts the real result.** Show centres of the two
sign-defined *electrode sets* rather than of the two scores:

| | LWPC-dominant (`delta > 0`) | LWPS-dominant (`delta < 0`) | Δz |
|---|---|---|---|
| pooled | n = 186, z̄ = 24.6 | n = 212, z̄ = 29.3 | **−4.6 mm** |
| lh | n = 118, z̄ = 21.0 | n = 136, z̄ = 28.3 | −7.3 mm |
| rh | n = 68, z̄ = 30.9 | n = 76, z̄ = 31.1 | −0.1 mm |

Within subject × hemisphere (both sets ≥ 2 electrodes, 21 groups): mean
Δz = **−1.83 mm**, 14/21 in the expected direction. The pooled numbers are
coverage-inflated — quote the within-subject value. This is a **descriptive
depiction of §15.5, not an independent test**: the sets are defined by the sign
of the quantity the regression models, so testing the separation would be
circular.

### 15.9 What to do next

The three analyses the previous revision listed as blocking are **done**
(§15.3 ceiling, §15.5 cross-validation, §15.7 magnitude). What remains:

1. ~~**Fix the noise correction in the pipeline.**~~ **Done.**
   `map_reliability` now computes `between_noise_corrected` from Pearson
   correlations whatever `method` is, returns a bootstrap
   `between_noise_corrected_ci`, and sets `note` when the ratio is out of range
   or undefined. `split_resolved_corr` gained a `reliability_note` explaining a
   non-positive reliability instead of returning a bare `NaN`. **Re-run the
   pipeline to regenerate `summary.txt`** — the archived one still carries the
   +1.374 / +1.369 values.
2. **The §5.1 `min_elec` sweep's negative `reliability_y` (−0.094) is
   explained, not a bug.** `split_resolved_corr` residualises on responsiveness
   and **within-subject centres**; `map_reliability` does not. At these
   per-subject electrode counts (median 14, min 1, two subjects with ≤ 3)
   centring removes most of the between-electrode variance the reliability is
   computed over, driving it to ~0. Reproduced here: +0.080 / −0.058 after
   centring, against +0.162 / +0.097 without. Read the sweep's `corr` as
   uninterpretable, not as a null — `reliability_note` now says so.

3. **Confirming across held-out subjects is deferred.** The cross-validation in
   §15.5 splits *trials*, so it controls trial noise but not subject sampling;
   leave-one-subject-out (21/22 folds) is reassuring but is not a held-out test.
   Deliberately not run — revisit if a reviewer asks.
4. **Keep the multiplicity caveat.** z was one of three axes. The block F
   (p = 0.032) is the protected headline; Bonferroni over three axes puts the
   z slope at 0.0225.

For reporting: lead with §15.4 (shared population, correlated at ceiling) and
§15.5 (dorsoventral gradient, cross-validated), present the gradient as
P(LWPC-dominant) falling with z or on a diverging **LWPS-dominant ←→
LWPC-dominant** scale, add the §15.8 sign-defined centres as a labelled
descriptive annotation, report §15.6 as an explicit null, and state §15.7 as a
settled negative rather than an absence of evidence. Do not interpret individual
electrodes anywhere ([§15.3](#153-how-reliable-are-the-maps)).

[§16](#16-reading-the-archived-summarytxt) says which blocks of the archived
`summary.txt` are safe to quote, [§17](#17-what-the-result-means--discussion)
turns these findings into Discussion claims, and
[§18](#18-communicating-a-nested-estimand) handles the "this is four levels deep"
objection.

---

## 16. Reading the archived `summary.txt`

The `summary.txt` in the archived lPFC run predates the `map_reliability` fix
(§15.9 item 1). Until the pipeline is re-run, read it block by block:

| Block in `summary.txt` | Verdict | Why |
|---|---|---|
| §5.2 PRIMARY `F = 1.895, p = 0.0104` | ✅ **quote as-is** | The primary result. 396 electrodes / 19 parcels / 22 subjects. |
| per-anatomy table | ⚠️ **ordering only** | Omnibus is the claim; min `q` = 0.126, no row survives FDR. Read `mean_delta_adj`, never `mean_delta` (see below). |
| §9.2 leverage | ✅ **quote as-is** | `F` = 1.59–2.38 across folds; the statistic never collapses. |
| §5.2 SECONDARY coordinates | ✅ **quote, but read the right axis** | Block `F` = 2.872, p = 0.0302; **z** carries it (p = 0.0074), **y is null** (p = 0.5742). |
| the `a POSITIVE mni_y slope` annotation | ⚠️ **generic boilerplate** | Printed every run regardless of result. This run's finding is on z, which the summary prints but does not annotate. |
| §7 DESCRIPTIVE medoids | ❌ **do not report** | Tests `dy`, an axis that is null anyway; `abs` weighting is blind to a signed dissociation; 6/25 groups return exactly-zero displacement (§15.8). Use the §15.8 sign-defined centres instead. |
| §5.4 `noise-corrected +1.374` / `+1.369` | ❌ **invalid** | Pre-fix Spearman artefact. Correct electrode-level value is **+1.022**, CI [0.998, 1.046]; the parcel-level ratio is **undefined** (LWPC parcel reliability −0.026). See §15.3. |
| §5.4 `between` and the reliabilities | ✅ **quote** | +0.172 electrode / +0.275 parcel, against reliabilities of +0.162 / +0.097 (Spearman) or +0.178 / +0.163 (Pearson). |
| §5.1 `min_elec` sweep | ❌ **uninterpretable, not a null** | `split_resolved_corr` residualises *and within-subject centres*; `map_reliability` does not. Centring removes most of the between-electrode variance at these per-subject counts. Also: 1→2 drops one subject and one electrode, 2→3 drops nothing — there is no sensitivity to observe. |
| pooled `r = +0.316`, within-subject `+0.224` | ✅ **quote** | They agree, which licenses pooling (§5.1). |

### 16.1 `mean_delta` versus `mean_delta_adj`

`mean_delta_adj` is the mean of the **residualised** delta — the quantity the `F`
actually uses — and is centred on ~zero across the sample by construction, since
the subject dummies span the intercept. Two consequences:

1. **The adjusted means are relative to the lPFC grand mean, not to absolute
   zero.** The whole-sample mean `delta` is −0.147 (slightly LWPS-dominant
   overall), so "LWPC-dominant parcel" means *LWPC-dominant relative to this
   sample's average balance*.
2. **Where raw and adjusted diverge, the raw mean was a subject artefact.**
   `lh_S_front_inf` goes −0.506 → −0.010; `rh_G_front_inf-Orbital` goes −0.036 →
   **+0.849** on 4 electrodes from 3 subjects. Those raw means were mostly
   reporting *which subjects happened to be wired there*.

### 16.2 Counts that look wrong but are not

- **398 → 396 electrodes, 21 → 19 parcels.** `MIN_SUBJECTS=3` drops
  `ctx_lh_Lat_Fis-ant-Vertical` and `ctx_lh_S_circular_insula_ant`, one subject
  and one electrode each.
- **The coordinate test uses all 398.** It does not coverage-filter, because it
  has no parcel family to condition on.
- **`[lh] F = 4.243, p = 0.0048` is not the dorsoventral result.** It is driven
  by `mni_x` (p = 0.067), a within-left-hemisphere medial–lateral trend. The z
  slope in that fit is p = 0.21. Do not quote the lh block `F` as support for
  §15.5; quote the pooled block `F` and the `z × hemisphere` interaction test
  (p = 0.65) instead.

---

## 17. What the result means — Discussion

The lPFC run supports a **shared-population-with-a-bias** account, not a
dissociation. The ordering below is the order the Discussion should make the
claims in; leading with the gradient inverts the finding.

### 17.1 The three claims, in descending strength

**1. Shared population — the strongest claim, and a positive one.**

This is what the noise ceiling was built to license and what most designs cannot
say. The usual approach — two thresholded maps, compared by eye — cannot tell
"different maps" from "same map, measured noisily". §15.3 measures the ceiling
directly, so the claim is not *"we failed to find segregation"* but:

> **We measured how different these maps could possibly be given our precision,
> and the answer is essentially not at all.**

**2. A dorsoventral bias in relative dominance — solid, but small.**

Block `F` = 2.87 (p = 0.030), z slope −0.0077/mm (p = 0.0074), cross-validated
across disjoint trial halves (98.5 % sign agreement, ~91 % of the slope is
signal), stable in 21/22 LOSO folds, and independently corroborated by the
parcel omnibus whose extremes order dorsoventrally (§17.3).

**3. A reordering, not a magnitude effect — settled, and worth stating as such.**

Neither effect's own sign varies with position; only their ordering does
(§15.7). Combined with the unbiased magnitude test (p = 0.40, μ² = +0.300 LWPC
vs +0.310 LWPS), the picture is: **both regulatory signals are present
throughout lPFC at comparable strength; what shifts along the dorsoventral axis
is which one wins.**

### 17.2 What it argues for and against

**Against modular segregation.** There is no "stability region" and "flexibility
region" here — and §17.4 explains why prior work might have concluded otherwise.

**Against a *competitive* implementation of the tradeoff.** This is the most
constraining result in the run and the easiest to miss. "Stability–flexibility
tradeoff" invites the reading that these are two ends of one seesaw: sites
supporting shielding should be poor at switching. **That predicts a negative
across-site correlation.** The observed correlation is **positive** (pooled
+0.32, within-subject +0.22, above-chance co-occurrence p = 0.0016, §15.4).
Sites that engage in proportion-based regulation at all engage in **both kinds**
of it. Whatever the behavioural tradeoff's status, it is **not implemented as
anticorrelated neural populations in lPFC.**

Two objections to preempt in the text:

- *Does shared electrode SNR manufacture the positive correlation?* No. A shared
  gain factor inflates a correlation of **magnitudes**; to inflate a correlation
  of **signed** values it needs both effects to carry a substantial positive
  mean. `lwpc_s` has a mean of +0.034 and is negative at 51 % of electrodes —
  there is almost no positive mean for gain to multiply.
- *Is responsiveness controlled?* Yes — it is a covariate in every model here and
  is residualised inside `split_resolved_corr`. Say so explicitly.

**For a shared substrate with graded weighting.** One population computing a
regulatory signal, with dorsoventral position biasing how that signal is
apportioned between conflict-shielding and switch-readiness demands. This sits
naturally with mixed-selectivity accounts of PFC — sites carrying combinations
of task variables rather than dedicated functions, with the gradient as a bias
in the mixing weights rather than a boundary between modules. *(Supply the
citations; §15.1's "mixed selectivity, not segregated subpopulations" is the
frame.)*

### 17.3 The direction is the convergent part

Dorsal (superior frontal) → LWPS-dominant; ventral (middle/inferior frontal) →
LWPC-dominant. The parcel adjusted means bear this out independently of the
coordinate fit:

| | value |
|---|---|
| corr(parcel mean MNI-z, parcel `mean_delta_adj`), unweighted | **−0.32** |
| same, weighted by electrode count | **−0.44** |
| mean `mean_delta_adj`, five most dorsal parcels | **−0.20** |
| mean `mean_delta_adj`, five most ventral parcels | **+0.20** |

That direction is broadly consistent with the conventional association of dorsal
lPFC (bordering pre-SMA/FEF territory; task-set updating, action selection) with
switching and of ventral lPFC (IFG; interference resolution, representational
selection) with conflict control.

> **The novelty is not the direction — it is that what prior work would describe
> as two regions is here a graded bias within one shared population.**

Frame it as a refinement of the existing picture rather than a contradiction of
it, and cite the conventional dorsal/ventral distinction so the convergence is
visible.

### 17.4 The methodological contribution

§15.4's arithmetic deserves main-text space, because it explains the prior
literature:

> At marginal positive rates of 49 % and 56 %, independence alone leaves only
> ~28 % of electrodes in both thresholded maps.

**Two thresholded dot maps will look disjoint whether or not the underlying
populations differ.** With per-electrode reliability of ~0.18–0.30, apparent
segregation in a thresholded map is not evidence of segregation — it is the
expected appearance of *any* pair of noisy maps. This motivates the analysis
choices (continuous scores, anatomically-defined electrode set, a within-electrode
difference as the response, an explicit noise ceiling) as necessary rather than
fussy, and it is the honest justification for refusing to interpret individual
electrodes.

### 17.5 The two nulls that are findings, not omissions

- **The rostrocaudal axis is null** (p = 0.58). The most prominent organizational
  account of lPFC is a rostrocaudal hierarchy of abstraction; this says the
  stability–flexibility balance is organized on an **orthogonal** axis. Note that
  the §8 centre machinery was aimed at this axis from the start (`p_anterior`),
  which is part of why it is null.
- **The magnitude hypothesis is settled, not unsupported.** Because it was tested
  with an unbiased estimator (E[xA·xB] = μ²) rather than with `|score|` (noise
  floor ≈ 0.8 σ for a null electrode), the sentence is *"settled negative"*, not
  *"no evidence for"*. Those read very differently to a reviewer, and the
  stronger one is earned.

### 17.6 Limitations the Discussion has to carry

- **lPFC only.** The stability–flexibility literature implicates striatal gating
  heavily. The shared-population claim is about lPFC, not about the control
  system. Say so.
- **The gradient is small.** The z coverage spans 97.6 mm (−21.7 to +75.9;
  73.9 mm between the 2.5th and 97.5th percentiles), so the −0.0077/mm slope is
  **0.57–0.75 `delta` units end to end**. Since `lwpc_s`/`lwps_s` are Cohen's *d*
  divided by that effect's across-electrode SD (0.279 and 0.269 here, §2.2), one
  `delta` unit ≈ 0.28 *d*, putting the gradient at **≈ 0.16–0.21 Cohen's *d***.
  Real, cross-validated, modest. Quote the range with the extent it is computed
  over; do not let the text drift into "lPFC is organized dorsoventrally for
  stability vs flexibility."
- **Keep claims 1 and 2 in proportion.** The shared component is essentially all
  the reliable variance; the gradient is a second-order bias on top of it. They
  are not in tension (§15.5), but if the writing loses the proportion, claim 2
  reads as a retraction of claim 1.
- **One of three axes.** Block `F` (p = 0.032) is the protected headline;
  Bonferroni over three axes puts the z slope at 0.0225 — still significant, so
  state it rather than letting a reviewer raise it.
- **Trial-level CV, not subject-level.** §15.5 splits trials, so it controls
  trial noise but not subject sampling. LOSO (21/22) is reassuring but is not a
  held-out test (§15.9 item 3).
- **Clinical coverage.** 22 subjects, 1–55 electrodes each (median 14), coverage
  determined by surgical need. Every anatomical claim is conditioned on
  `coverage_matrix.csv`.
- **Correlational.** Position predicts the bias; nothing here says position
  causes it.

### 17.7 Discussion skeleton

1. Restate in one sentence — shared population, graded bias, reordering not
   magnitude.
2. **Shared population** — lead here; the ceiling makes it a positive claim.
3. **Against a competitive tradeoff implementation** — the positive correlation
   constrains models; preempt the SNR objection.
4. **The dorsoventral bias** — direction, convergence with conventional
   dorsal/ventral accounts, the reframe from "two regions" to "one population
   with a bias". Keep proportionate to (2).
5. **A reordering, not a magnitude effect** — use the P(`delta` > 0) framing.
6. **The rostrocaudal null** — explicitly not the hierarchy axis.
7. **Why thresholded maps mislead** — §17.4.
8. **Limitations** — §17.6.
9. **Forward** — held-out-subject confirmation; extension beyond lPFC,
   especially striatum.

⚠️ **The single biggest framing risk is leading with the gradient.** If §17.1
claim 2 comes before claim 1, the paper reads as *"we found a dorsoventral
dissociation between stability and flexibility"* — roughly the opposite of what
the data say, and exactly the claim the noise ceiling was built to rule out.
**The shared population is the finding; the gradient is the qualifier.**

---

## 18. Communicating a nested estimand

`delta` is a difference of difference-of-differences, tested for modulation by
location. That is four levels, and readers do notice. This section is how to
present it without the depth becoming the story.

### 18.1 Count the layers honestly, then say which are yours

| | quantity | operation |
|---|---|---|
| 1 | conflict effect | incongruent − congruent |
| 2 | **LWPC** | that effect, low-PC block − high-PC block |
| 3 | **`delta`** | LWPC − LWPS |
| 4 | the test | does `delta` vary with location |

**Layers 1–2 are definitional, not analytic.** LWPC *is* a difference of
differences; there is no such thing as "LWPC without the interaction". A reviewer
objecting to layer 2 is objecting to the existence of the list-wide proportion
congruency effect, not to this analysis. **Layer 4 is the research question.**

**So layer 3 is the only added one — and it is a correction, not an elaboration:**

> The "simpler" alternative is to map LWPC and LWPS separately and compare where
> each is significant. That analysis is shorter to describe and **wrong**, because
> the difference between a significant effect and a non-significant effect is not
> itself significant. Testing the within-electrode difference is the **minimum**
> analysis that licenses a claim about differential anatomical distribution.

Put a sentence to that effect in Methods with the interaction-fallacy citation
(Nieuwenhuis, Forstmann & Wagenmakers, 2011 — verify before submitting).
Reviewers who recognise it will be on your side immediately. This reframes the
exchange: you are not defending added complexity, you are pointing out that the
intuitive alternative is the one that needs defending.

### 18.2 Six moves that make the depth stop reading as depth

**1. Describe the model term, not the arithmetic.**

> ❌ "a difference of difference-of-differences"
> ✅ "an effect-type × location interaction"

Nobody blinks at "we tested a three-way interaction"; everybody blinks at "we
subtracted a subtraction from a subtraction". Same quantity, different reception,
because arithmetic descriptions always sound worse than model descriptions.
**Never describe the arithmetic in the main text** — it belongs in Methods, once.

**2. Call them *scores*, not contrasts.** Encapsulation is what kills the nesting:

> Each electrode receives a **stability score** (how strongly its conflict effect
> adapts to conflict frequency) and a **flexibility score** (how strongly its
> switch effect adapts to switch frequency). We ask whether the **balance**
> between them depends on location.

Three sentences, zero visible nesting. This is how Cohen's *d* works: nobody
objects that a t-test on *d* is "a test on a ratio of a difference to a pooled SD".
*d* is a score with a name. Give these two quantities names and the same thing
happens.

**3. Report the headline as a proportion.** By tercile of MNI z (raw descriptive
binning of the 398-electrode table):

| | n | mean z | **% LWPC-dominant** |
|---|---|---|---|
| ventral | 133 | +2.9 | **54.9 %** |
| mid | 132 | +26.7 | 45.5 % |
| dorsal | 133 | +51.8 | **39.8 %** |

It survives within-subject centring (52 % below vs 41 % above each subject's own
mean z), so it is not a coverage artefact. Raw point-biserial r(`delta` > 0, z) =
−0.108; the nuisance-adjusted value §15.7 tests is −0.144, p = 0.0058.

> **The proportion of lPFC sites where stability regulation dominates falls from
> 55 % ventrally to 40 % dorsally.**

Zero subtractions visible, no negative numbers on display. This is the abstract
sentence and probably the figure.

**4. Convert the noise objection into a measured answer.** The legitimate worry
behind "how many layers deep" is that nested contrasts compound variance. Most
papers can only wave at this; §15.3 and §15.5 answer it — measured split-half
reliability, 98.5 % sign agreement across disjoint trial halves, ~91 % of the
slope as signal. **Lead with this rather than burying it**: it converts the
biggest-looking liability into the strongest methodological claim.

**5. One schematic figure panel.** Four small panels left to right: 2 × 2 cell
means → one bar (stability score) → the other 2 × 2 → one bar (flexibility score)
→ the two bars side by side with the gap labelled. Kills the objection visually in
a way prose cannot. See [`figure_plan.md`](figure_plan.md) F1.

**6. Fix the double negative in the sign convention.** Positive LWPC means the
conflict effect is **smaller** in the high-proportion block — a subtraction where
"more" means "less". A reader holding that inverted *while* tracking `delta` is
lost regardless of how good the rest is. Define **adaptation** once, in words,
with the direction stated, then use "adaptation score" throughout and never
return to raw subtraction language.

### 18.3 The two objections that will actually come

**"Why not just compare |LWPC| vs |LWPS|? That's simpler."** `|score|` is a
**biased** magnitude estimator — E|score| ≈ 0.8 σ for a null electrode, a pure
noise floor. Tested with an unbiased estimator instead (E[xA·xB] = μ²), there is
no gradient at all: p = 0.40, μ² = +0.300 LWPC vs +0.310 LWPS (§15.7). So the
intuitive alternative was run, and it is null for a principled reason. §15.2's
four-row A/B/C/D table is the tool for showing why signed and magnitude `delta`
are not interchangeable — consider promoting it out of this doc.

**"Couldn't this be one model instead of a pipeline of subtractions?"** In
principle yes — a single trial-level hierarchical model with location as a
predictor, where the quantity is one high-order interaction coefficient. Worth
knowing, because it shows the depth lives in the **estimand**, not in this
pipeline: any correct approach has the same depth, it just hides it in a
coefficient name. The pipeline formulation was chosen because it is what makes
the **split-half reliability** and the **within-electrode swap null** possible,
and neither is easy to obtain from a monolithic model. That is a substantive
justification, not a preference — state it.

### 18.4 Where each register goes

| Location | Register |
|---|---|
| Abstract / headline | the proportion sentence (55 % → 40 %); no subtractions |
| Results, first mention | "effect-type × location interaction"; model-term language only |
| Methods | the arithmetic, once, plus the two defensive sentences from §18.1 and §18.3 |
| Figure 1 | the four-panel schematic |
| **Never in main text** | "difference of difference-of-differences" |

The depth is a property of the question — "do two adaptation effects have
different anatomical distributions" cannot be asked with fewer layers. What is
controllable is whether the arithmetic is narrated or the estimand is named.
