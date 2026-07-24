"""Low-dimensional (PCA / UMAP) projections and trajectory plots."""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, t
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict, windower
import seaborn as sns

from ..data_prep import concatenate_and_balance_data_for_decoding
from ..decoder import Decoder

def plot_static_pca_projection(
    roi_labeled_arrays: dict,
    roi: str,
    strings_to_find: list,
    cats: dict,
    save_dir: None,
    obs_axs: int = 0,
    random_state: int = 42,
    balance_strata=True
):
    """
    Visualizes the first two principal components of the full, flattened dataset.
    
    This helps to see if the data is linearly separable at a global level.
    """
    print(f"--- Generating Static PCA Plot for ROI: {roi} ---")
    
    # 1. Get and balance data.
    # We MUST use 'subsample' because PCA cannot handle NaNs.
    # This function (with 'subsample') conveniently gives us a clean, NaN-free dataset.
    data, labels, _ = concatenate_and_balance_data_for_decoding(
        roi_labeled_arrays,
        roi,
        strings_to_find,
        obs_axs=obs_axs,
        balance_method='subsample', # Critical: removes NaN trials
        random_state=random_state,
        balance_strata=balance_strata
    )
    
    if data.size == 0:
        print(f"Warning: No valid data for {roi} after balancing. Skipping plot.")
        return

    # 2. Reshape for PCA
    # Input shape is (n_trials, n_channels, n_timepoints)
    # We need (n_trials, n_features), where features = channels * timepoints
    n_trials = data.shape[0]
    data_flat = data.reshape(n_trials, -1)
    
    print(f"Data shape for PCA: {data_flat.shape}")

    # 3. Standardize the data (crucial for PCA)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_flat)

    # 4. Run PCA
    pca = PCA(n_components=2)
    pc_scores = pca.fit_transform(data_scaled)
    
    print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"PCA kept {pca.n_components_} components, total var = " 
          f"{pca.explained_variance_ratio_.sum():.1%}")
    
    # 5. Plot
    
    # Get human-readable labels from 'cats'
    # 'cats' maps {('c25',): 0, ('i25',): 1}
    # We want {0: 'c25', 1: 'i25'}
    label_map = {v: str(k[0]) for k, v in cats.items()}
    display_labels = [label_map[l] for l in labels]
    
    df = pd.DataFrame({
        'PC1': pc_scores[:, 0],
        'PC2': pc_scores[:, 1],
        'Condition': display_labels
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='PC1',
        y='PC2',
        hue='Condition',
        palette='deep',
        alpha=0.7,
        s=50
    )
    
    plt.title(f"Static PCA Projection for {roi}\n(Conditions: {', '.join(label_map.values())})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend(title="Condition")
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_roi = roi.replace(" ", "_").replace("/", "_")
        out = os.path.join(save_dir, f"static_pca_{safe_roi}.pdf")
        plt.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved static PCA plot to {out}")
    else:
        plt.show()


def plot_pca_over_time(
    roi_labeled_arrays: dict,
    roi: str,
    strings_to_find: list,
    cats: dict,
    window_size: int,
    step_size: int,
    sampling_rate: float,
    first_time_point: float,
    save_dir: None,
    obs_axs: int = 0,
    random_state: int = 42
):
    """
    Visualizes the first two principal components in a sliding window over time.
    
    This creates a grid of plots showing how the data's geometry evolves.
    """
    print(f"--- Generating Dynamic PCA Plot for ROI: {roi} ---")

    # 1. Get and balance data (use 'subsample' to remove NaNs)
    data, labels, _ = concatenate_and_balance_data_for_decoding(
        roi_labeled_arrays,
        roi,
        strings_to_find,
        obs_axs=obs_axs,
        balance_method='subsample', # Must use subsample
        balance_strata=True, # just hard code this to balance strata, don't know why i wouldn't want this on
        random_state=random_state
    )
    
    if data.size == 0:
        print(f"Warning: No valid data for {roi} after balancing. Skipping plot.")
        return

    # 2. Window the data
    # Input shape: (n_trials, n_channels, n_timepoints)
    # We want to window along the time axis (-1)
    # Resulting shape (assuming insert_at=0): (n_windows, n_trials, n_channels, window_size)
    windowed_data = windower(
        data,
        window_size=window_size,
        axis=-1,
        step_size=step_size,
        insert_at=0
    )
    
    n_windows, n_trials, _, _ = windowed_data.shape
    
    # 3. Get time points for plot titles
    first_sample = first_time_point * sampling_rate
    start_times = [first_sample + step_size * i for i in range(n_windows)]
    time_window_centers = [
        (start + window_size / 2) / sampling_rate
        for start in start_times
    ]

    # 4. Get display labels
    label_map = {v: str(k[0]) for k, v in cats.items()}
    display_labels = [label_map[l] for l in labels]

    # 5. Create grid of plots
    n_cols = 5  # Adjust as needed
    n_rows = int(np.ceil(n_windows / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
    axes = axes.flatten()

    for win_idx in range(n_windows):
        ax = axes[win_idx]
        
        # 6. Get data for this window and flatten
        # Shape: (n_trials, n_channels, window_size)
        window_data = windowed_data[win_idx]
        # Shape: (n_trials, n_channels * window_size)
        window_flat = window_data.reshape(n_trials, -1)

        # 7. Standardize AND PCA (fit_transform on *this window's data*)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(window_flat)
        
        pca = PCA(n_components=2)
        pc_scores = pca.fit_transform(data_scaled)

        # 8. Plot this window
        df = pd.DataFrame({
            'PC1': pc_scores[:, 0],
            'PC2': pc_scores[:, 1],
            'Condition': display_labels
        })
        
        sns.scatterplot(
            data=df,
            x='PC1',
            y='PC2',
            hue='Condition',
            palette='deep',
            alpha=0.7,
            ax=ax,
            legend= (win_idx == 0) # Only show legend on the first plot
        )
        
        var1 = pca.explained_variance_ratio_[0]
        var2 = pca.explained_variance_ratio_[1]
        
        ax.set_title(f"Time: {time_window_centers[win_idx]:.2f} s\n(Var: {var1+var2:.1%})")
        ax.set_xlabel(f"PC1 ({var1:.1%})")
        ax.set_ylabel(f"PC2 ({var2:.1%})")

    # Clean up empty subplots
    for i in range(n_windows, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"PCA Over Time for {roi} (Conditions: {', '.join(label_map.values())})", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_roi = roi.replace(" ", "_").replace("/", "_")
        out = os.path.join(save_dir, f"pca_over_time_{safe_roi}.pdf")
        plt.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved windowed PCA plot to {out}")
    else:
        plt.show()


def plot_pca_3d_trajectory(
    roi_labeled_arrays,
    roi,
    strings_to_find,
    cats,
    save_dir,
    window_size,
    step_size,
    sampling_rate,
    first_time_point,
    obs_axs=0,
    random_state=42,
    explained_variance=0.8,
    clf=None
):
    """
    Trace each condition's centroid through the Decoder's first 3 PCs over time.
    """
    # first get clean data the same way the rest of the pipeline does
    data, labels, _ = concatenate_and_balance_data_for_decoding(
        roi_labeled_arrays, roi, strings_to_find,
        obs_axs=obs_axs, balance_method='subsample', 
        balance_strata=True, random_state=random_state
    )
    if data.size == 0:
        print(f"No data for{roi}, skipping 3D PCA.")
        return
    # data shape: (n_trials, n_channels, n_timepoints)
    n_trials, n_channels, n_timepoints = data.shape
    
    # fit the Decoder on the full flattened epoch, pull its PCA
    if clf is None:
        clf = LinearDiscriminantAnalysis()
    
    X_flat = data.reshape(n_trials, -1) # (n_trials, n_channels x n_timepoints)
    
    decoder = Decoder(
        cats,
        explained_variance=explained_variance,
        oversample=False,
        n_splits=2,
        n_repeats=1,
        clf=clf,
        random_state=random_state
    )
    
    decoder.fit(X_flat, labels)
    
    pca = decoder.model.named_steps['pca']
    n_kept = pca.n_components_
    print(f"[{roi}] Decoder PCA kept {n_kept} comps, "
          f"total var = {pca.explained_variance_ratio_.sum():.1%}")
    if n_kept < 3:
        print(f"  ⚠ Only {n_kept} PCs available — 3D plot will skip missing axis")
    
    # find the grand mean, which becomes the out-of-window filter for the pca
    grand_mean = data.mean(axis=0) # (n_channels, n_timepoints)
    
    # compute per-window/condition centroids, embed them, and project them
    n_windows = (n_timepoints - window_size) // step_size + 1
    first_sample = first_time_point * sampling_rate
    window_centers_s = [
        (first_sample + step_size * w + window_size / 2) / sampling_rate
        for w in range(n_windows)
    ]
    
    unique_labels = np.unique(labels)
    label_to_name = {v: str(k[0]) for k, v in cats.items()}
    
    # trajectories[label] -> ndarray of shape (n_windows, 3)
    trajectories = {lab: np.zeros((n_windows, 3)) for lab in unique_labels}
    
    for w in range(n_windows):
        t0 = w * step_size
        t1 = t0 + window_size
        
        for lab in unique_labels:
            class_data = data[labels == lab] # (n_class, ch, time)
            window_centroid = class_data[:, :, t0:t1].mean(0) # (ch, window_size)
            
            # embed into the full epoch using the grand mean as filler
            full = grand_mean.copy() # (ch, time)
            full[:, t0:t1] = window_centroid
            
            # project: pca.transform handles the centering itself.
            flat = full.reshape(1, -1) # (1, ch*time)
            pcs = pca.transform(flat)[0] # (n_kept,)
            
            # pad with NaN if the decoder kept fewer than 3 components
            traj_3 = np.full(3, np.nan)
            traj_3[:min(3, n_kept)] = pcs[:min(3, n_kept)]
            trajectories[lab][w] = traj_3
    
    # plot with one set of 3D axes, one line per condition, and color-coded markers for time
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=window_centers_s[0], vmax=window_centers_s[-1])
    
    for lab, traj in trajectories.items():
        # Connecting line (one color per condition)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', linewidth=1.5, alpha=0.6, label=label_to_name[lab])
        
        # markers colored by time
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                   c=[cmap(norm(t)) for t in window_centers_s],
                   s=40, edgecolors='k', linewidths=0.5)
        
        # mark the window closest to t=0 with a star
        zero_idx = int(np.argmin(np.abs(np.array(window_centers_s))))
        ax.scatter(*traj[zero_idx], s=200, marker='*',
                   facecolors='none', edgecolors='red', linewidths=2)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
                  if n_kept >= 2 else "PC2 (n/a)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
                  if n_kept >= 3 else "PC3 (n/a)")   
    ax.set_title(
        f"3D PCA trajectory - {roi}\n"
        f"({n_kept} PCs total, "
        f"{pca.explained_variance_ratio_.sum():.1%} variance; "
        f"red ★ = t≈0)"
    )
    ax.legend()
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("Time (s)")
    
    os.makedirs(save_dir, exist_ok=True)
    safe_roi = roi.replace(" ", "_").replace("/", "_")
    out = os.path.join(save_dir, f"pca_3d_trajectory_{safe_roi}.pdf")
    plt.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D PCA trajectory to {out}")


# need to get time window by time window version next. Also this is untested rn. 10/31/25.
def plot_high_dim_decision_slice(
    fitted_pipeline: Pipeline,
    X_data: np.ndarray,
    y_labels: np.ndarray,
    cats: dict,
    roi: str,
    save_dir: str
):
    """
    Visualizes a 2D slice (PC1 vs PC2) of a high-dimensional classifier's
    decision boundary.

    Parameters:
    - fitted_pipeline: A *pre-fitted* sklearn Pipeline (Scaler -> PCA -> CLF).
    - X_data: The raw, balanced, NaN-free data used for fitting (n_trials, n_features_flat).
    - y_labels: The labels for X_data (n_trials,).
    - cats: The 'cats' dictionary mapping labels to names.
    - roi: The name of the ROI for the plot title.
    """
    
    # 1. Get the fitted components from the pipeline
    scaler = fitted_pipeline.named_steps['scaler']
    pca = fitted_pipeline.named_steps['pca']
    clf = fitted_pipeline.named_steps['clf']
    
    # 2. Get the PC scores for the *actual* data
    X_scaled = scaler.transform(X_data)
    X_pc_all = pca.transform(X_scaled)
    
    X_pc_2D = X_pc_all[:, 0:2] # Get just PC1 and PC2 for the scatter plot
    
    # Get human-readable labels
    label_map = {v: str(k[0]) for k, v in cats.items()}
    display_labels = [label_map[l] for l in y_labels]
    display_labels_str = str(display_labels)
    
    df = pd.DataFrame({
        'PC1': X_pc_2D[:, 0],
        'PC2': X_pc_2D[:, 1],
        'Condition': display_labels
    })

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 3. Plot the scatter of real data points
    sns.scatterplot(
        data=df,
        x='PC1',
        y='PC2',
        hue='Condition',
        palette='deep',
        alpha=0.7,
        s=50,
        ax=ax
    )
    
    # 4. Create the 2D grid
    x_min, x_max = X_pc_2D[:, 0].min() - 1, X_pc_2D[:, 0].max() + 1
    y_min, y_max = X_pc_2D[:, 1].min() - 1, X_pc_2D[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))
    
    # 5. Create the high-dimensional grid (2D slice)
    # We create vectors of zeros for all other PC components
    n_components_total = pca.n_components_
    grid_points_high_dim = np.zeros((xx.ravel().shape[0], n_components_total))
    
    # Set the first two columns to our 2D grid values
    grid_points_high_dim[:, 0] = xx.ravel()
    grid_points_high_dim[:, 1] = yy.ravel()
    
    # 6. Get predictions from the *high-D classifier*
    # The classifier was trained on PC space, so we feed it our high-D PC vectors
    Z = clf.decision_function(grid_points_high_dim)
    Z = Z.reshape(xx.shape)
    
    # 7. Plot the decision boundary and shaded regions
    ax.contourf(xx, yy, Z, cmap='RdBu_r', alpha=0.3, 
                levels=np.linspace(Z.min(), Z.max(), 3))
    ax.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['--'], linewidths=2)
    
    # 8. Add info
    clf_name = clf.__class__.__name__
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]
    var_total = np.sum(pca.explained_variance_ratio_)

    ax.set_title(f"{clf_name} Decision Boundary on 2D Slice for {roi}\n"
                 f"Using {n_components_total} PCs (Total Var: {var_total:.1%})")
    ax.set_xlabel(f"PC1 ({var1:.1%})")
    ax.set_ylabel(f"PC2 ({var2:.1%})")
    
    safe_roi_str = roi.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    filename = f'debug_pca_hyperplane_{safe_roi_str}.pdf'
    filepath = os.path.join(save_dir, filename)
    print(f"Saving debug PCA plot to: {filepath}")
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free memory


def plot_static_umap_projection(
    roi_labeled_arrays: dict,
    roi: str,
    strings_to_find: list,
    cats: dict,
    save_dir: None,
    obs_axs: int = 0,
    random_state: int = 42,
    balance_strata=True,
    explained_variance=0.8,
    n_neighbors=30, min_dist=0.3, metric='euclidean',
    decoder=None, clf=None
):
    """
    Visualizes the first two UMAP dimensions of the full, flattened dataset. Does PCA -> UMAP to further project the PCs onto 2 UMAP dimensions.
    
    This helps to see if the data is nonlinearly separable at a global level.
    """
    print(f"--- Generating Static UMAP Plot for ROI: {roi} ---")
    
    data, labels, _ = concatenate_and_balance_data_for_decoding(
        roi_labeled_arrays, roi, strings_to_find,
        obs_axs=obs_axs, balance_method='subsample',
        balance_strata=True, random_state=random_state,
    )
    if data.size == 0:
        return

    n_trials, n_channels, n_timepoints = data.shape
    X_flat = data.reshape(n_trials, -1)

    if decoder is None:
        if clf is None:
            clf = LinearDiscriminantAnalysis()
        decoder = Decoder(
            cats, explained_variance=explained_variance,
            oversample=False, n_splits=2, n_repeats=1,
            clf=clf, random_state=random_state,
        )
        decoder.fit(X_flat, labels)

    pca = decoder.model.named_steps['pca']
    pc_scores = pca.transform(X_flat)
    
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, metric=metric, random_state=random_state
    )
    emb = reducer.fit_transform(pc_scores)
    
    label_map = {v: str(k[0]) for k, v in cats.items()}
    df = pd.DataFrame({
        'UMAP1': emb[:,0],
        'UMAP2': emb[:,1],
        'Condition': [label_map[l] for l in labels]
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='UMAP1', y='UMAP2',
                    hue='Condition', palette='deep', alpha=0.7, s=50)
    plt.title(f"UMAP of PCA scores for {roi} "
              f"(n_pcs={pca.n_components_}, n_neighbors={n_neighbors})")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Condition")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_roi = roi.replace(" ", "_").replace("/", "_")
        out = os.path.join(save_dir, f"static_umap_{safe_roi}.pdf")
        plt.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(out.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_umap_3d_trajectory(
    roi_labeled_arrays, roi, strings_to_find, cats, save_dir,
    window_size, step_size, sampling_rate, first_time_point,
    obs_axs=0, random_state=42,
    explained_variance=0.8,
    n_neighbors=30, min_dist=0.3, metric='euclidean',
    n_umap_components=3,
    decoder=None, clf=None,
):
    data, labels, _ = concatenate_and_balance_data_for_decoding(
        roi_labeled_arrays, roi, strings_to_find,
        obs_axs=obs_axs, balance_method='subsample',
        balance_strata=True, random_state=random_state,
    )
    if data.size == 0:
        return

    n_trials, n_channels, n_timepoints = data.shape
    X_flat = data.reshape(n_trials, -1)

    if decoder is None:
        if clf is None:
            clf = LinearDiscriminantAnalysis()
        decoder = Decoder(
            cats, explained_variance=explained_variance,
            oversample=False, n_splits=2, n_repeats=1,
            clf=clf, random_state=random_state,
        )
        decoder.fit(X_flat, labels)

    pca = decoder.model.named_steps['pca']
    n_kept = pca.n_components_
    if n_kept < 2:
        print(f"Only {n_kept} PC kept — UMAP needs >=2 input dims. Skipping.")
        return

    # fit UMAP on the same trial-level PCA scores the LDA receives
    train_pcs = pca.transform(X_flat)
    reducer = umap.UMAP(
        n_components=n_umap_components,
        n_neighbors=min(n_neighbors, max(2, n_trials - 1)),
        min_dist=min_dist, metric=metric, random_state=random_state,
    )
    reducer.fit(train_pcs)

    # build per-condition / per-window trajectories using the same padded-centroid
    # trick as plot_pca_3d_trajectory, then push through scaler -> pca -> umap.transform
    grand_mean = data.mean(axis=0)
    n_windows = (n_timepoints - window_size) // step_size + 1
    unique_labels = np.unique(labels)
    label_map = {v: str(k[0]) for k, v in cats.items()}

    trajectories = {lab: np.zeros((n_windows, n_umap_components))
                    for lab in unique_labels}
    window_centers_s = np.zeros(n_windows)

    for w in range(n_windows):
        t0 = w * step_size
        t1 = t0 + window_size
        window_centers_s[w] = first_time_point + (t0 + window_size/2) / sampling_rate

        for lab in unique_labels:
            class_data = data[labels == lab]
            window_centroid = class_data[:, :, t0:t1].mean(axis=0)
            full = grand_mean.copy()
            full[:, t0:t1] = window_centroid
            flat = full.reshape(1, -1)
            pcs = pca.transform(flat)     # (1, n_kept)
            trajectories[lab][w] = reducer.transform(pcs)[0]

    # plot — structure mirrors plot_pca_3d_trajectory: one line per condition,
    # viridis-by-time markers, red star at the window closest to t=0
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    palette = sns.color_palette('deep', n_colors=len(unique_labels))

    for i, lab in enumerate(unique_labels):
        traj = trajectories[lab]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=palette[i], label=label_map[lab], linewidth=2)
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                   c=window_centers_s, cmap='viridis', s=40)

    zero_idx = int(np.argmin(np.abs(window_centers_s)))
    for lab in unique_labels:
        p = trajectories[lab][zero_idx]
        ax.scatter(p[0], p[1], p[2], marker='*', s=200, c='red', zorder=10)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    ax.set_title(f"UMAP trajectory for {roi} "
                 f"(n_pcs={n_kept}, n_neighbors={n_neighbors}, min_dist={min_dist})")
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    safe_roi = roi.replace(" ", "_").replace("/", "_")
    out = os.path.join(save_dir, f"umap_3d_trajectory_{safe_roi}.pdf")
    plt.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D UMAP trajectory to {out}")
