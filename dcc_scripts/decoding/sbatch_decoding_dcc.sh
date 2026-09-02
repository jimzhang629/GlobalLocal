#!/bin/bash
#SBATCH --output=out/slurm_%j_%x.out
#SBATCH -e out/slurm_%j_%x.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 5
#SBATCH --mem=215G # set this to 225 for non-coupling jobs
#SBATCH --time=48:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg

# Each joblib worker is its own process, and sklearn/numpy would otherwise start a
# BLAS thread pool per worker on top of that. On -c 5 that oversubscribes the
# allocation and the threads spend their time fighting each other rather than
# working. One BLAS thread per worker; the parallelism comes from joblib.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running condition: $CONDITION_NAME"
# -u keeps stdout unbuffered. Without it the .out file lags the job by however
# long it takes to fill a 4KB buffer, which on a long run makes a working job
# look like it stalled at whatever line happened to land last.
python -u /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/decoding/run_decoding_dcc.py