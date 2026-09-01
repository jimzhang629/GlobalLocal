#!/bin/bash
#SBATCH --output=out/slurm_%j_%x.out
#SBATCH -e out/slurm_%j_%x.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 5
# 225G is the floor for ONE decoding battery: the parent's copy of
# subjects_mne_objects plus the ~5 loky workers that each get a pickled copy of
# it (decoding_dcc.py, Parallel over process_bootstrap; N_JOBS=-1 clamps to -c
# 5). That floor does not depend on how many electrode SETS the job decodes --
# the epochs are loaded once before selection and the sets are just channel-name
# lists decoded sequentially against them.
#
# What did push coupling runs to 500G was creep, not any one battery: joblib
# reuses its loky workers across Parallel calls, so every battery re-pickles the
# epochs into them and the freed copies do not reliably hand RSS back. A job
# doing the coupled set plus all 20 draws paid that 21 times.
# submit_decoding_with_coupling_electrode_sets_dcc.sh now fans the draws out at
# DRAWS_PER_JOB=4, so a chunk decodes at most 5 sets and 250G covers it.
#
# Raise this if you raise DRAWS_PER_JOB; do NOT lower it for a job that decodes
# fewer draws, since the floor is the same either way.
#SBATCH --mem=250G
#SBATCH --time=48:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg

echo "Running condition: $CONDITION_NAME"
python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/decoding/run_decoding_dcc.py