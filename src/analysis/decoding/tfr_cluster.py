"""Decoding restricted to significant time-frequency clusters (TFR masks)."""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ieeg.calc.stats import time_perm_cluster
import gc

from .data_prep import concatenate_and_balance_data_for_decoding, mixup2
from .decoder import Decoder

def decode_on_sig_tfr_clusters(
    X_train_raw, y_train, X_test_raw,
    train_indices, test_indices,
    concatenated_data, labels, cats, 
    obs_axs, chans_axs,
    stat_func, p_thresh, n_perm,
    Decoder, explained_variance, oversample,
    ignore_adjacency=1, seed=42, tails=2, alpha=1.
):
    """
    Balance data and decode with TFR cluster masking. Returns the sig tfr cluster masks for later plotting.
    
    Parameters
    ----------
    X_train_raw : np.ndarray
        Raw training data (trials, channels, freqs, times)
    y_train : np.ndarray
        Training labels
    X_test_raw : np.ndarray
        Raw test data (trials, channels, freqs, times)
    train_indices : np.ndarray
        Indices of training trials in the original concatenated data
    test_indices : np.ndarray
        Indices of test trials in the original concatenated data
    concatenated_data : np.ndarray
        Full concatenated data array (all_trials, channels, freqs, times)
    labels : np.ndarray
        All labels corresponding to concatenated_data
    cats : dict
        Dictionary mapping condition names to label integers
    obs_axs : int
        Axis of the data that contains trial labels
    chans_axs : int
        Axis of the data that contains channel labels
    stat_func : callable
        Statistical function for cluster computation
    p_thresh : float
        P-value threshold for significance
    n_perm : int
        Number of permutations for cluster test
    Decoder : class
        Decoder class to use for decoding
    explained_variance : float
        Proportion of variance to explain with PCA
    oversample : bool
        Whether to oversample the training data
    ignore_adjacency : int
        Whether to ignore adjacency in clustering (1=ignore, 0=use adjacency)
    seed : int
        Random seed for reproducibility
    tails : int
        Number of tails for statistical test (1 or 2)
    alpha : float
        Alpha parameter for mixup augmentation
        
    Returns
    -------
    preds : np.ndarray
        Predicted labels for test data
    channel_masks : dict
        Dictionary of channel masks for significant clusters
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """
    # Get condition names from cats dictionary
    condition_names = [k[0] if isinstance(k, tuple) else k for k in cats.keys()]
    
    # Step 1: Create training-only TFR masks
    channel_masks, channel_t_values = compute_sig_tfr_masks_from_concatenated_data(
        concatenated_data, labels, train_indices, condition_names, cats,
        obs_axs, chans_axs,
        stat_func, p_thresh, n_perm, 
        ignore_adjacency, seed, tails
    )
    
    # Step 2: Apply masks and flatten
    X_train_masked = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
        X_train_raw, obs_axs, chans_axs, channel_masks
    )
    X_test_masked = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
        X_test_raw, obs_axs, chans_axs, channel_masks
    )
    
    # Step 3: Decode
    decoder = Decoder(cats, explained_variance=explained_variance, n_splits=1, n_repeats=1, oversample=oversample, clf_params={}, random_state=seed)
    
    # Handle NaN filling using existing mixup2 function
    mixup2(arr=X_train_masked, labels=y_train, obs_axs=obs_axs, alpha=alpha, seed=seed)
    
    # Fill test NaNs with noise (as done in sample_fold)
    is_nan = np.isnan(X_test_masked)
    X_test_masked[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

    # Fit and predict
    decoder.fit(X_train_masked, y_train)
    preds = decoder.predict(X_test_masked)
    
    # debugging
    print(f"Number of significant clusters found: {sum(mask.any() for mask in channel_masks.values())}")
    print(f"Total significant features: {sum(mask.sum() for mask in channel_masks.values())}")

    return preds, channel_masks, channel_t_values


def compute_sig_tfr_masks_from_roi_labeled_array(
    roi_labeled_array, train_indices, condition_names,
    obs_axs, chans_axs, stat_func, p_thresh, n_perm, 
    ignore_adjacency=1, seed=42, tails=2
):
    """
    Compute significant TFR clusters using only training trials from roi_labeled_array.
    
    Returns:
    --------
    channel_masks : dict
        {channel_label: mask_array} where mask is (n_freqs, n_times)
    """
    # Get channel labels from the labeled array
    channel_labels = roi_labeled_array.labels[chans_axs+1]

    # Validate we have exactly 2 conditions for now
    if len(condition_names) != 2:
        raise ValueError(
            f"For now, just doing perm test instead of ANOVA, "
            f"so this will only work for two conditions. Got {len(condition_names)} conditions."
        )

    # Split training data by condition
    train_data_by_condition = {}
    for cond in condition_names:  # hmm the stats only work for two conditions, so just do two conditions for now. Can expand to >2 conditions in the future, would just need to do ANOVA i think instead of time perm cluster.
        # Extract training trials for this condition
        cond_data = roi_labeled_array[cond]  # Shape: (trials, channels, freqs, times). Test this!
        cond_train_data = np.take(cond_data, train_indices, axis=obs_axs) # TODO: keep going through this code 4:45 on 8/1 - huh? this is a 4d array, check what train_indices is. Should grab along the trials axis. Maybe do np.take(train_indices, axis=obs_axs) to be safe.
        train_data_by_condition[cond] = cond_train_data
    
    # Compute significant clusters for each channel
    channel_masks = compute_sig_tfr_masks_for_specified_channels(
        channel_labels, train_data_by_condition, condition_names, obs_axs, chans_axs
    )
    
    return channel_masks


def compute_sig_tfr_masks_for_specified_channels(
    n_channels, train_data_by_condition, condition_names, 
    obs_axs, chans_axs, stat_func, p_thresh, n_perm,
    ignore_adjacency=1, seed=42, tails=2
):
    """
    Compute significant TFR masks for each channel.
    
    Parameters
    ----------
    n_channels : int
        Number of channels to process
    train_data_by_condition : dict
        Dictionary with condition names as keys and data arrays as values
    condition_names : list
        List of condition names (must be exactly 2 for now)
    obs_axs : int
        Axis containing trials
    chans_axs : int
        Axis containing channels
    stat_func : callable
        Statistical function for cluster computation
    p_thresh : float
        P-value threshold
    n_perm : int
        Number of permutations
    ignore_adjacency : int
        Whether to ignore adjacency in clustering
    seed : int
        Random seed
    tails : int
        Number of tails for test
        
    Returns
    -------
    channel_masks : dict
        Dictionary where keys are channel indices (int) and values are 
        boolean masks of shape (n_freqs, n_times)
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """    
    channel_masks = {}
    channel_t_values = {}
    
    # For each channel, compute significant clusters
    for ch_idx in range(n_channels):
        # Get data for this channel across conditions
        cond0_data = train_data_by_condition[condition_names[0]]
        cond1_data = train_data_by_condition[condition_names[1]]
        
        # Extract channel data
        cond0_chan_data = np.take(cond0_data, ch_idx, axis=chans_axs)
        cond1_chan_data = np.take(cond1_data, ch_idx, axis=chans_axs)
        
        # Run time perm cluster test
        if len(cond0_chan_data) > 0 and len(cond1_chan_data) > 0:
            # let's grab the t values too for debugging and plotting - this will only work if using scipy stats ttest ind, otherwise it will crash!
            t_values = stat_func(cond0_chan_data, cond1_chan_data, axis=0).statistic #.statistic only exists for scipy stats
            channel_t_values[ch_idx] = t_values
            
            # get sig tfr mask for this channel
            mask, _ = time_perm_cluster(
                cond0_chan_data, cond1_chan_data,
                stat_func=stat_func,
                p_thresh=p_thresh,
                n_perm=n_perm,
                axis=0,  # trials are now first axis after taking channel
                ignore_adjacency=ignore_adjacency,
                seed=seed,
                tails=tails
            )
            channel_masks[ch_idx] = mask
        else:
            # No data for comparison - create zero mask
            if len(cond1_chan_data) > 0:
                mask_shape = (cond1_chan_data.shape[1], cond1_chan_data.shape[2])
            else:
                mask_shape = (cond0_chan_data.shape[1], cond0_chan_data.shape[2])
            channel_masks[ch_idx] = np.zeros(mask_shape, dtype=bool)
            print(f"Warning: Channel {ch_idx} has insufficient data for comparison")
    
    return channel_masks, channel_t_values


def compute_sig_tfr_masks_from_concatenated_data(
    concatenated_data, labels, train_indices, condition_names, cats,
    obs_axs, chans_axs, stat_func, p_thresh, n_perm, 
    ignore_adjacency=1, seed=42, tails=2
):
    """
    Compute significant TFR clusters using only training trials from concatenated data.
    
    Parameters
    ----------
    concatenated_data : np.ndarray
        Full data array (all_trials, channels, freqs, times)
    labels : np.ndarray
        Labels for all trials in concatenated_data
    train_indices : np.ndarray
        Indices of training trials to use for computing masks
    condition_names : list
        List of condition names to compare
    cats : dict
        Dictionary mapping condition names to label integers
    obs_axs : int
        Axis containing trials
    chans_axs : int
        Axis containing channels
    stat_func : callable
        Statistical function for cluster computation
    p_thresh : float
        P-value threshold for significance
    n_perm : int
        Number of permutations
    ignore_adjacency : int
        Whether to ignore adjacency in clustering
    seed : int
        Random seed
    tails : int
        Number of tails for statistical test
        
    Returns
    -------
    channel_masks : dict
        Dictionary where keys are channel indices and values are boolean masks
        of shape (n_freqs, n_times) indicating significant clusters
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). 
        THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """
    # Validate we have exactly 2 conditions for now
    if len(condition_names) != 2:
        raise ValueError(
            f"For now, just doing perm test instead of ANOVA, "
            f"so this will only work for two conditions. Got {len(condition_names)} conditions."
        )
    
    # Get training data and labels
    train_data = concatenated_data[train_indices]
    train_labels = labels[train_indices]
    
    # Split training data by condition
    train_data_by_condition = {}
    for cond_name in condition_names:
        # Get the label value for this condition
        cond_label = cats[tuple([cond_name]) if isinstance(cond_name, str) else tuple(cond_name)]
        
        # Get indices for this condition
        cond_mask = train_labels == cond_label
        cond_data = train_data[cond_mask]
        train_data_by_condition[cond_name] = cond_data
    
    # Compute significant clusters for each channel
    n_channels = concatenated_data.shape[chans_axs]
    channel_masks, channel_t_values = compute_sig_tfr_masks_for_specified_channels(
        n_channels, train_data_by_condition, condition_names, 
        obs_axs, chans_axs, stat_func, p_thresh, n_perm,
        ignore_adjacency, seed, tails
    )
    
    return channel_masks, channel_t_values


def apply_tfr_masks_and_flatten_to_make_decoding_matrix(data, obs_axs, chans_axs, channel_masks):
    """
    Apply channel-specific TFR masks and flatten feature matrices.
    
    Parameters
    ----------
    data : np.ndarray
        Shape: (n_trials, n_channels, n_freqs, n_times)
    obs_axs : int
        Axis containing trials
    chans_axs : int
        Axis containing channels
    channel_masks : dict
        Dictionary where keys are channel indices and values are boolean masks
        
    Returns
    -------
    decoding_matrix : np.ndarray
        Shape: (n_trials, n_features) where n_features depends on the masks
    """
    n_trials = data.shape[obs_axs]
    n_channels = data.shape[chans_axs]
    feature_vectors = []
    
    # Move trials to first axis if needed
    if obs_axs != 0:
        data = np.moveaxis(data, obs_axs, 0)
        if chans_axs > obs_axs:
            chans_axs = chans_axs - 1
        else:
            chans_axs = chans_axs + 1

    # Iterate through each channel
    for ch_idx in range(n_channels):
        # Extract this channel's data for all trials
        channel_data = np.take(data, ch_idx, axis=chans_axs)
        
        # Check if we have a mask for this channel
        if ch_idx in channel_masks:
            # Get the boolean mask (n_freqs, n_times)
            mask = channel_masks[ch_idx]
            
            # flatten all dimensions except trials (axis 0)
            n_trials_ch = channel_data.shape[0]
            remaining_shape = channel_data.shape[1:]
            channel_data = channel_data.reshape(n_trials_ch, -1)
            
            # flatten mask
            mask_flat = mask.flatten()

            # make sure the mask size matches the flattened features
            if mask_flat.shape[0] != channel_data.shape[1]:
                raise ValueError("Mask size does not match flattened features size")
            else:
                # apply the mask
                masked_features = channel_data[:, mask_flat]
        
            # Add this channel's features to our list
            feature_vectors.append(masked_features)
    
    # Concatenate all channels' features horizontally
    if feature_vectors:
        decoding_matrix = np.concatenate(feature_vectors, axis=1)
    else:
        # Return empty matrix if no features
        raise ValueError("No features found for any channels.")

    return decoding_matrix


def get_confusion_matrix_for_rois_tfr_cluster(
    roi_labeled_arrays, rois, strings_to_find, stat_func, 
    Decoder, explained_variance=0.95,
    p_thresh=0.05, n_perm=100, 
    n_splits=5, n_repeats=5, obs_axs=0, chans_axs=1,
    balance_method='subsample', oversample=False,
    random_state=42, alpha=0.2, ignore_adjacency=1, seed=42, tails=2, normalize: str = None, clear_memory=True
):
    """
    Compute confusion matrices using TFR cluster masking for multiple ROIs. Also returns the sig tfr cluster masks for later plotting.
    
    Parameters
    ----------
    roi_labeled_arrays : dict
        Dictionary of labeled arrays by ROI
    rois : list
        List of ROIs to process
    strings_to_find : list
        List of condition strings to find
    stat_func : callable
        Statistical function for cluster computation
    Decoder : class
        Decoder class to use
    explained_variance : float
        Variance to explain in PCA
    p_thresh : float
        P-value threshold for clusters
    n_perm : int
        Number of permutations
    n_splits : int
        Number of CV splits
    n_repeats : int
        Number of CV repeats
    obs_axs : int
        Observation axis
    chans_axs : int
        Channel axis
    balance_method : str
        Method for balancing ('subsample' or 'pad_with_nans')
    oversample : bool
        Whether to oversample in decoder
    random_state : int
        Random seed
    alpha : float
        Mixup alpha parameter
    ignore_adjacency : int
        Whether to ignore adjacency in clustering
    seed : int
        Random seed for permutation test
    tails : int
        Number of tails for permutation test
    normalize : str
        Whether to normalize the confusion matrix
    Returns
    -------
    confusion_matrices : dict
        Dictionary of confusion matrices by ROI
    cats_dict : dict
        Dictionary of condition labels by ROI
    channel_masks : dict
        Dictionary of channel masks for significant clusters. Nested dictionary: {roi: {repeat: {fold: channel_masks}}}
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). 
        THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """
    confusion_matrices = {}
    cats_dict = {}
    channel_masks = {}
    channel_t_values = {}
    
    for roi in rois:
        channel_masks[roi] = {}
        channel_t_values[roi] = {}
        print(f"Processing ROI: {roi}")
        
        # Get data and labels
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, random_state
        )
        cats_dict[roi] = cats
        
        # Set up cross-validation
        all_cms = []
        
        for repeat in range(n_repeats):
            channel_masks[roi][repeat] = {}
            channel_t_values[roi][repeat] = {}
            repeat_seed = random_state + repeat * 1000
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            
            fold_cms = []
            
            for fold_idx, (train_indices, test_indices) in enumerate(skf.split(concatenated_data, labels)):
                print(f"  Repeat {repeat+1}/{n_repeats}, Fold {fold_idx+1}/{n_splits}")
                
                # Get train/test data
                X_train_raw = concatenated_data[train_indices]
                X_test_raw = concatenated_data[test_indices]
                y_train = labels[train_indices]
                y_test = labels[test_indices]
                
                # Balance and decode with TFR masking
                preds, fold_channel_masks, fold_channel_t_values = decode_on_sig_tfr_clusters(
                    X_train_raw, y_train, X_test_raw,
                    train_indices, test_indices,
                    concatenated_data, labels, cats,
                    obs_axs, chans_axs,
                    stat_func, p_thresh, n_perm,
                    Decoder, explained_variance, oversample,
                    ignore_adjacency=ignore_adjacency, 
                    seed=repeat_seed + fold_idx, 
                    tails=tails, 
                    alpha=alpha
                )
                
                channel_masks[roi][repeat][fold_idx] = fold_channel_masks
                channel_t_values[roi][repeat][fold_idx] = fold_channel_t_values
                cm = confusion_matrix(y_test, preds)
                fold_cms.append(cm)

                if clear_memory:
                    del X_train_raw, X_test_raw, y_train, y_test, preds, fold_channel_masks, fold_channel_t_values
                    gc.collect()
            
            # Sum across folds
            repeat_cm = np.sum(fold_cms, axis=0)

            # Normalize the confusion matrix for this repeat if requested
            if normalize:
                with np.errstate(divide='ignore', invalid='ignore'):
                    if normalize == 'true':
                        divisor = np.sum(repeat_cm, axis=-1, keepdims=True)
                    elif normalize == 'pred':
                        divisor = np.sum(repeat_cm, axis=-2, keepdims=True)
                    elif normalize == 'all':
                        divisor = np.sum(repeat_cm)
                    else:
                        divisor = 1
                    
                    # Avoid division by zero by setting the divisor to 1 where it is 0
                    if isinstance(divisor, np.ndarray):
                        divisor[divisor == 0] = 1
                    elif divisor == 0:
                        divisor = 1
                    
                    normalized_cm = repeat_cm.astype('float') / divisor
                    all_cms.append(normalized_cm)
            else:
                all_cms.append(repeat_cm) # Append the raw counts if no normalization
        
        # Average across repeats
        final_cm = np.mean(all_cms, axis=0)
        confusion_matrices[roi] = final_cm

        if clear_memory and roi in roi_labeled_arrays:
            del roi_labeled_arrays[roi]
            gc.collect()

        if clear_memory:
            del concatenated_data, labels, cats, all_cms
            gc.collect()
    
    return confusion_matrices, cats_dict, channel_masks, channel_t_values
