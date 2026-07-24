from setuptools import find_packages, setup

# Runtime dependencies for the `src.analysis` code. Declaring them here means a
# single `pip install -e .` sets up a working environment (project + deps),
# including `ieeg` from PyPI, so nobody needs the old hardcoded
# `sys.path.append(".../IEEG_Pipelines/")` hacks.
#
# Versions are left loose on purpose: in the existing `ieeg` conda env these are
# already satisfied (pip won't touch them), while a fresh env resolves current
# releases. Pin here only if you need reproducibility.
setup(
    name='src',
    packages=find_packages(),
    python_requires='<3.13',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'joblib',
        'mne',
        'mne-bids',
        'umap-learn',
        'ieeg==0.7.0',  # pinned to the version validated in the lab's DCC env
    ],
)
