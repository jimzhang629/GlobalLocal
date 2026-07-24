"""Confusion-matrix and cm-trace plotting."""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ieeg.viz.parula import parula_map
from typing import Dict, Optional, Union, List, Tuple

from ..plots.style import NATURE_STYLE

def get_display_labels_from_cats(cats):
    """Extracts clean labels for plotting from the 'cats' dictionary."""
    return [key[0] if isinstance(key, tuple) and len(key) == 1 else str(key) for key in cats.keys()]


def plot_and_save_confusion_matrix(cm, display_labels, file_name, save_dir):
    """
    Plots and saves a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name))
    print(f"Saved figure to: {save_dir}")
    plt.close(fig)


def plot_and_save_tfr_masks(masks_dict, mask_type, subjects_or_rois, ch_names, times, freqs, 
                            spec_method, conditions_save_name, save_dir, 
                            channels_per_page=60, grid_shape=(6, 10)):
    """
    Plot and save TFR masks for subjects or ROIs.
    
    Parameters
    ----------
    masks_dict : dict
        Dictionary of masks (subjects or ROIs as keys)
    mask_type : str
        Type of mask ('sig_elecs' or 'all_elecs')
    subjects_or_rois : list
        List of subjects or ROIs to plot
    ch_names : list
        Channel names
    times : array
        Time points
    freqs : array
        Frequencies
    spec_method : str
        Spectral method used
    conditions_save_name : str
        Name for saving
    save_dir : str
        Directory to save figures
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for key in subjects_or_rois:
        if key not in masks_dict:
            continue
            
        mask = masks_dict[key]
        mask_pages = plot_mask_pages(
            mask,
            ch_names,
            times=times,
            freqs=freqs,
            channels_per_page=channels_per_page,
            grid_shape=grid_shape,
            cmap=parula_map,
            title_prefix=f"{key} ",
            log_freq=True,
            show=False
        )
        
        # Save each page
        for i, fig in enumerate(mask_pages):
            fig_name = f"{key}_{mask_type}_{spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
            fig_pathname = os.path.join(save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            print(f"Saved figure: {fig_name}")


def extract_pooled_cm_traces(
    time_window_decoding_results: dict,
    n_bootstraps: int,
    condition_comparisons: dict,
    rois: list,
    unit_of_analysis: str,
    cats_by_roi: dict
) -> dict:
    """
    Pools raw confusion matrix cell counts across bootstraps based on the unit of analysis.

    Parameters
    ----------
    time_window_decoding_results : dict
        The main results dictionary from the parallel bootstrap processing.
    n_bootstraps : int
        The total number of bootstraps run.
    condition_comparisons : dict
        Dictionary of condition comparisons to analyze.
    rois : list
        List of ROIs to process.
    unit_of_analysis : str
        The sampling unit ('bootstrap', 'repeat', or 'fold').
    cats_by_roi : dict
        Dictionary from bootstrap results containing the 'cats' for each ROI.

    Returns
    -------
    dict
        A nested dictionary: 
        {condition_comparison: {roi: {trace_label: pooled_trace_array}}}
        where pooled_trace_array has shape (n_samples, n_windows).
    """
    pooled_traces = {}

    for condition_comparison, strings_to_find in condition_comparisons.items():
        pooled_traces[condition_comparison] = {}
        
        for roi in rois:
            if roi not in cats_by_roi:
                print(f"Warning: 'cats' not found for ROI {roi}, skipping CM trace extraction.")
                continue

            # Get the class labels from the 'cats' dictionary
            cats = cats_by_roi[roi]
            class_labels = [key[0] if isinstance(key, tuple) and len(key) == 1 else str(key) for key in cats.keys()]
            n_classes = len(class_labels)
            
            # Create trace labels, e.g., "True: c25, Pred: i25"
            trace_labels = []
            for true_idx in range(n_classes):
                for pred_idx in range(n_classes):
                    label = f"True: {class_labels[true_idx]}, Pred: {class_labels[pred_idx]}"
                    trace_labels.append(label)

            # This list will hold arrays for each trace, e.g., all_trace_lists[0] is for 'True: c25, Pred: c25'
            all_trace_lists = [[] for _ in trace_labels]

            for b_idx in range(n_bootstraps):
                if (b_idx in time_window_decoding_results and
                    condition_comparison in time_window_decoding_results[b_idx] and
                    roi in time_window_decoding_results[b_idx][condition_comparison]):
                    
                    # Get raw CMs - shape: (n_windows, n_samples_per_boot, n_classes, n_classes)
                    cm_raw = time_window_decoding_results[b_idx][condition_comparison][roi]['cm_true']
                    
                    n_windows, n_samples_per_boot, _, _ = cm_raw.shape

                    # This will hold traces for *this bootstrap*
                    # List of arrays, each shape (n_samples_per_boot, n_windows)
                    boot_traces_T = []
                    for true_idx in range(n_classes):
                        for pred_idx in range(n_classes):
                            # Extract trace: (n_windows, n_samples_per_boot)
                            trace = cm_raw[:, :, true_idx, pred_idx]
                            # Transpose to (n_samples_per_boot, n_windows)
                            boot_traces_T.append(trace.T)
                    
                    if unit_of_analysis == 'bootstrap':
                        # Average across samples for this bootstrap
                        for i in range(len(trace_labels)):
                            mean_trace = np.mean(boot_traces_T[i], axis=0) # Shape: (n_windows,)
                            all_trace_lists[i].append(mean_trace)
                    
                    elif unit_of_analysis in ['repeat', 'fold']:
                        # Append each sample individually
                        for i in range(len(trace_labels)):
                            for sample_idx in range(n_samples_per_boot):
                                all_trace_lists[i].append(boot_traces_T[i][sample_idx, :]) # Shape: (n_windows,)
                    else:
                         raise ValueError(f"Invalid unit_of_analysis: '{unit_of_analysis}'")

            if not all_trace_lists[0]:
                print(f"Warning: No CM traces found for {condition_comparison} - {roi}")
                pooled_traces[condition_comparison][roi] = {}
                continue

            # Stack all collected samples
            roi_traces_dict = {}
            for i, label in enumerate(trace_labels):
                # Stack to get (n_total_samples, n_windows)
                pooled_array = np.vstack(all_trace_lists[i])
                roi_traces_dict[label] = pooled_array
            
            pooled_traces[condition_comparison][roi] = roi_traces_dict

    return pooled_traces


def plot_cm_traces_nature_style(
    time_points: np.ndarray,
    cm_traces_dict: Dict[str, np.ndarray],
    comparison_name: str = "",
    roi: str = "",
    save_dir: str = ".",
    timestamp: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    ylabel: str = "Mean Trial Count",
    ylim: Optional[Tuple[float, float]] = None, # Make ylim optional
    xlim: Tuple[float, float] = (-0.5, 1.5),
    single_column: bool = True,
    show_legend: bool = True,
    filename_suffix: str = "",
    return_fig: bool = False,
    samples_axis=0
):
    """
    Plots raw confusion matrix traces over time in Nature journal style.
    
    This is adapted from plot_accuracies_nature_style to plot
    mean counts (e.g., TP, FN) instead of accuracies.
    """
    
    # Apply Nature style settings
    with plt.rc_context(NATURE_STYLE):
        if single_column:
            fig_width = 89 / 25.4  # 89mm to inches
            fig_height = fig_width * 0.7  # Aspect ratio
        else:
            fig_width = 183 / 25.4 # 183mm to inches
            fig_height = fig_width * 0.4
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Define Nature-appropriate colors if not provided
        if colors is None:
            # Differentiate True vs. Predicted
            # Example: True c25 -> blues, True i25 -> oranges
            colors = {}
            if any('c25' in k for k in cm_traces_dict.keys()):
                colors['True: c25, Pred: c25'] = '#0173B2' # Dark Blue (TP)
                colors['True: c25, Pred: i25'] = '#56B4E9' # Light Blue (FN)
                colors['True: i25, Pred: i25'] = '#DE8F05' # Dark Orange (TP)
                colors['True: i25, Pred: c25'] = '#CC78BC' # Light Purple (FN) # Using purple for i25->c25 FP
            else:
                # Fallback
                nature_colors = ['#0173B2', '#56B4E9', '#DE8F05', '#CC78BC']
                for i, label in enumerate(cm_traces_dict.keys()):
                    colors[label] = nature_colors[i % len(nature_colors)]

        if linestyles is None:
             linestyles = {}
             # Example: Solid for "correct" prediction (TP), dashed for "incorrect" (FN/FP)
             for label in cm_traces_dict.keys():
                 if 'c25, Pred: c25' in label or 'i25, Pred: i25' in label:
                     linestyles[label] = '-' # Solid for TP
                 else:
                     linestyles[label] = '--' # Dashed for FN/FP
        
        max_y_val = 0 # For setting ylim automatically

        # Plot each trace
        for i, (label, traces) in enumerate(cm_traces_dict.items()):
            if traces.ndim == 2:
                n_samples = traces.shape[samples_axis]
                mean_trace = np.mean(traces, axis=samples_axis)
                std_trace = np.std(traces, axis=samples_axis)
            else:
                mean_trace = traces
                std_trace = np.zeros_like(traces)
                print(f"⚠️ Warning: Trace data for '{label}' is 1D. Cannot compute STD.")

            max_y_val = max(max_y_val, np.max(mean_trace + std_trace))

            color = colors.get(label, '#949494') # Default to gray
            linestyle = linestyles.get(label, '-')
            
            ax.plot(time_points, mean_trace,  
                    label=label,  
                    color=color,  
                    linestyle=linestyle,
                    linewidth=1,
                    zorder=3)
            
            ax.fill_between(
                time_points,
                mean_trace - std_trace,
                mean_trace + std_trace,
                alpha=0.25,
                color=color,
                linewidth=0,
                zorder=1
            )
        
        # Add stimulus onset line
        ax.axvline(x=0,  
                    color='#666666',  
                    linestyle='-',  
                    linewidth=0.5,
                    alpha=0.5,
                    zorder=2)
        
        ax.set_xlabel('Time from stimulus onset (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=7)
        
        # Set axis limits
        if ylim is None:
            ax.set_ylim(0, max_y_val * 1.15) # Auto-scale from 0
        else:
            ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        
        ax.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
        
        x_ticks = np.arange(-0.5, 1.6, 0.5)
        ax.set_xticks(x_ticks)
        
        # Auto-set y-ticks
        if ylim is None:
            y_ticks = np.linspace(0, max_y_val * 1.1, num=5)
        else:
            y_ticks = np.linspace(ylim[0], ylim[1], num=5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.1f}' for y in y_ticks]) # Use 1 decimal place for counts

        if show_legend:
            legend = ax.legend(
                loc='upper right',
                fontsize=5, # Smaller fontsize for more labels
                frameon=False,
                handlelength=1.5,
                handletextpad=0.5,
                borderpad=0.2,
                columnspacing=0.5
            )
            for line in legend.get_lines():
                line.set_linewidth(1.5)

        if title:
            ax.set_title(title, fontsize=7, pad=3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        plt.tight_layout(pad=0.5)
        
        if return_fig:
            return fig
        else:
            timestamp_str = f"{timestamp}_" if timestamp else ""
            
            filename_parts = [timestamp_str.rstrip('_')]
            if comparison_name:
                filename_parts.append(comparison_name)
            if roi:
                filename_parts.append(roi)
            if filename_suffix:
                filename_parts.append(filename_suffix)
            
            filename = "_".join(filter(None, filename_parts)) + "_CM_Traces.pdf" # Add suffix
            filepath = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            
            plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(filepath.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
            
            plt.close(fig)
            print(f"Saved CM trace plot to: {filepath}")
