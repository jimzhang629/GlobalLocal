"""Nature-style plotting of decoding accuracy time courses."""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from typing import Union, List, Sequence
from typing import Dict, Optional, Union, List, Tuple

from ..accuracy_stats import find_contiguous_clusters
from ..plots.style import NATURE_STYLE

def plot_accuracies_nature_style(
    time_points: np.ndarray,
    accuracies_dict: Dict[str, np.ndarray],
    significant_clusters: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    sampling_rate: float = 256,
    comparison_name: str = "",
    roi: str = "",
    save_dir: str = ".",
    timestamp: Optional[str] = None,
    p_thresh: float = 0.05,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    ylabel: str = "Accuracy",
    ylim: Tuple[float, float] = (0.0, 1.0),  # CHANGED: Default Y-axis is now 0 to 1
    xlim: Tuple[float, float] = (-1.0, 1.5),
    single_column: bool = True,
    show_legend: bool = True,
    show_significance: bool = True,
    significance_y_position: Optional[float] = None,
    filename_suffix: str = "",
    return_fig: bool = False,
    show_chance_level: bool = True,
    chance_level: float = 0.5,
    samples_axis=0
):
    """
    Plot accuracies in Nature journal style.
    
    Follows Nature's guidelines:
    - Single column width: 89mm (3.5 inches)
    - Double column width: 183mm (7.2 inches)
    - Font: Arial or Helvetica
    - Font sizes: 7pt for labels, 6pt for tick labels
    - Minimal design with no unnecessary elements
    - High contrast colors suitable for print
    """
    
    # Apply Nature style settings
    with plt.rc_context(NATURE_STYLE):
        # Set figure size based on column width
        if single_column:
            fig_width = 89 / 25.4  # 89mm to inches
            fig_height = fig_width * 0.7  # Aspect ratio
        else:
            fig_width = 183 / 25.4 # 183mm to inches
            fig_height = fig_width * 0.4
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Define Nature-appropriate colors if not provided
        if colors is None:
            nature_colors = [
                '#0173B2',  # Blue
                '#DE8F05',  # Orange
                '#029E73',  # Green
                '#CC78BC',  # Light purple
                '#ECE133',  # Yellow
                '#56B4E9',  # Light blue
                '#F0E442',  # Light yellow
                '#949494',  # Gray
            ]
            colors = {}
            for i, label in enumerate(accuracies_dict.keys()):
                colors[label] = nature_colors[i % len(nature_colors)]
        
        # Plot each accuracy time series
        for i, (label, accuracies) in enumerate(accuracies_dict.items()):
            # Compute statistics
            if accuracies.ndim == 2:
                n_samples = accuracies.shape[samples_axis]
                mean_accuracy = np.mean(accuracies, axis=samples_axis)
                std_accuracy = np.std(accuracies, axis=samples_axis)
                sem_accuracy = std_accuracy / np.sqrt(n_samples)
            else:
                mean_accuracy = accuracies
                std_accuracy = np.zeros_like(accuracies)
                sem_accuracy = np.zeros_like(accuracies) # huhh? why is this zero???
                print(f"⚠️ Warning: Accuracy data for '{label}' is 1D. Cannot compute SEM; plotting without error bars.")

            # Get color and linestyle
            color = colors.get(label, '#0173B2') if colors else '#0173B2'
            linestyle = linestyles.get(label, '-') if linestyles else '-'
            
            # Plot mean line with higher contrast
            ax.plot(time_points, mean_accuracy, 
                    label=label, 
                    color=color, 
                    linestyle=linestyle,
                    linewidth=1,
                    zorder=3)
            
            # Plot SEM as shaded area
            ax.fill_between(
                time_points,
                mean_accuracy - std_accuracy,
                mean_accuracy + std_accuracy,
                alpha=0.25,  # Lighter shading for Nature style
                color=color,
                linewidth=0,
                zorder=1
            )
        
        # Add chance level line if requested
        if show_chance_level:
            ax.axhline(y=chance_level, 
                        color='#666666', 
                        linestyle='--', 
                        linewidth=0.5,
                        zorder=2,
                        label='Chance')
        # Add stimulus onset line
        ax.axvline(x=0, 
                    color='#666666', 
                    linestyle='-', 
                    linewidth=0.5,
                    alpha=0.5,
                    zorder=2)
        
        # Add significance markers
        if show_significance and significant_clusters is not None and np.any(significant_clusters):
            clusters = find_contiguous_clusters(significant_clusters)
            
            # CHANGED: Position significance bars at a fixed high point (e.g., 95% of the y-axis)
            if significance_y_position is None:
                y_range = ylim[1] - ylim[0]
                # Place the bar at 95% of the total y-axis height, well above the data
                significance_y_position = ylim[0] + y_range * 0.95
            
            for start_idx, end_idx in clusters:
                
                start_time = time_points[start_idx]
                end_time = time_points[end_idx]
                
                # Draw significance bar
                ax.plot([start_time, end_time], 
                        [significance_y_position, significance_y_position],
                        color='black', 
                        linewidth=1,
                        solid_capstyle='butt')
                
                # Add significance marker
                center_time = (start_time + end_time) / 2
                ax.text(center_time, 
                        significance_y_position + (ylim[1] - ylim[0]) * 0.02, # Position asterisk just above bar
                        '*', 
                        ha='center', 
                        va='bottom', 
                        fontsize=8,
                        fontweight='bold')
        
        # Set labels
        # changed to size 12 font
        # changed to response
        # ax.set_xlabel('Time from response onset (s)', fontsize=12)
        ax.set_xlabel('Time from stimulus onset (s)', fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        
        # Set axis limits
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        
        # Configure ticks
        # changed to size 10 font
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=2)
        
        # Set specific x-ticks for clarity
        # changed to 1.0
        x_ticks = np.arange(-1.0, 1.6, 0.5)
        ax.set_xticks(x_ticks)
        
        # CHANGED: Set specific y-ticks for consistency
        y_ticks = np.linspace(ylim[0], ylim[1], num=5) 
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])

        # Configure legend
        if show_legend:
            # The legend is placed in the upper right. With the new ylim, it should have enough space.
            legend = ax.legend(
                loc='upper right',
                fontsize=6,
                frameon=False,
                handlelength=1,
                handletextpad=0.5,
                borderpad=0.2,
                columnspacing=0.5
            )
            if len(accuracies_dict) > 1: # Only make lines thicker if there's more than just the chance line
                # Make legend lines thicker for visibility
                for line in legend.get_lines():
                    line.set_linewidth(1.5)

        # Add title only if specified
        if title:
            ax.set_title(title, fontsize=7, pad=3)
        
        # Remove top and right spines (Nature style)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make remaining spines thinner
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        if return_fig:
            return fig
        else:
            # Create filename
            timestamp_str = f"{timestamp}_" if timestamp else ""
            
            filename_parts = [timestamp_str.rstrip('_')]
            if comparison_name:
                filename_parts.append(comparison_name)
            if roi:
                filename_parts.append(roi)
            if filename_suffix:
                filename_parts.append(filename_suffix)
            
            filename = "_".join(filter(None, filename_parts)) + ".pdf"  # PDF for publication
            filepath = os.path.join(save_dir, filename)
            
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Save in multiple formats
            plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(filepath.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(filepath.replace('.pdf', '.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)
            
            plt.close(fig)
            print(f"Saved Nature-style plot to: {filepath}")


# Convenience function to create multi-panel Nature figures
def create_multipanel_nature_figure(
    panels_data: List[Dict],
    panel_labels: List[str] = None,
    n_cols: int = 2,
    save_path: str = None,
    fig_title: str = None
):
    """
    Create a multi-panel figure in Nature style.
    
    Parameters
    ----------
    panels_data : List[Dict]
        List of dictionaries, each containing data for one panel.
        Each dict should have keys matching plot_accuracies_nature_style parameters.
    panel_labels : List[str], optional
        Labels for each panel (e.g., ['a', 'b', 'c', 'd']).
    n_cols : int
        Number of columns in the figure grid.
    save_path : str, optional
        Path to save the figure.
    fig_title : str, optional
        Overall figure title.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete multi-panel figure.
    """
    
    n_panels = len(panels_data)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    with plt.rc_context(NATURE_STYLE):
        # Full page width for Nature
        fig_width = 183 / 25.4  # 183mm to inches
        fig_height = fig_width * (n_rows / n_cols) * 0.7
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (panel_data, ax) in enumerate(zip(panels_data, axes)):
            plt.sca(ax)
            
            # Add panel label
            if panel_labels and i < len(panel_labels):
                ax.text(-0.15, 1.05, panel_labels[i], 
                       transform=ax.transAxes,
                       fontsize=8, fontweight='bold',
                       va='top', ha='right')
            
            # Plot the panel (simplified - you'd need to adapt the plotting code)
            # This is a placeholder for the actual plotting logic
            
        # Remove empty subplots
        for i in range(n_panels, len(axes)):
            fig.delaxes(axes[i])
        
        if fig_title:
            fig.suptitle(fig_title, fontsize=8, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
        
        return fig


def plot_true_vs_shuffle_accuracies(time_points, accuracies_true, accuracies_shuffle, significant_clusters,
                    window_size, step_size, sampling_rate, condition_comparison, roi, save_dir, timestamp=None, p_thresh=0.05, other_string_to_add=None):
    """
    Plot mean true and shuffled accuracies over time with significance.

    This function visualizes the average decoding accuracy from true labels and
    shuffled labels across different time windows. It highlights significant
    time clusters (where true accuracy is significantly higher than shuffled)
    with horizontal bars and asterisks. The plot is saved to a file.

    Parameters
    ----------
    time_points : array-like
        The center time points (in seconds) for each window.
    accuracies_true : numpy.ndarray
        Accuracies for true labels. Shape: (n_windows, n_repeats).
    accuracies_shuffle : numpy.ndarray
        Accuracies for shuffled labels. Shape: (n_windows, n_perm).
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    window_size_samples : int
        The size of the decoding window in samples.
    step_size_samples : int
        The step size of the decoding window in samples. (Not directly used in plot rendering logic beyond filename).
    sampling_rate : float
        The sampling rate of the data in Hz.
    condition_comparison : str
        A string describing the condition comparison (e.g., "TaskA_vs_TaskB").
        Used in the plot title and filename.
    roi : str
        The Region of Interest (ROI) being plotted. Used in the plot title
        and filename.
    save_dir : str
        The directory where the plot image will be saved.
    first_time_point_s : float, optional
        The time in seconds of the first sample of the epoch, used for x-axis limits
        if needed, though current xlim are fixed. Default is 0.
    timestamp : str
        Timestamp string for filenaming purposes
    p_thresh : float
        p-value threshold for determining significant clusters
    """
    n_repeats = accuracies_true.shape[1]
    n_perm = accuracies_shuffle.shape[1]

    # Compute mean and standard error
    mean_true_accuracy = np.mean(accuracies_true, axis=1)
    std_true_accuracy = np.std(accuracies_true, axis=1)
    se_true_accuracy = std_true_accuracy / np.sqrt(n_repeats)

    mean_shuffle_accuracy = np.mean(accuracies_shuffle, axis=1)
    std_shuffle_accuracy = np.std(accuracies_shuffle, axis=1)
    se_shuffle_accuracy = std_shuffle_accuracy / np.sqrt(n_perm)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, mean_true_accuracy, label='True Accuracy', color='blue')
    plt.fill_between(
        time_points,
        mean_true_accuracy - se_true_accuracy,
        mean_true_accuracy + se_true_accuracy,
        alpha=0.2,
        color='blue'
    )

    plt.plot(time_points, mean_shuffle_accuracy, label='Shuffled Accuracy', color='red')
    plt.fill_between(
        time_points,
        mean_shuffle_accuracy - se_shuffle_accuracy,
        mean_shuffle_accuracy + se_shuffle_accuracy,
        alpha=0.2,
        color='red'
    )

    # Compute window duration
    window_duration = window_size / sampling_rate

    # Find contiguous significant clusters
    def find_clusters(significant_clusters: Union[np.ndarray, List[bool], Sequence[bool]]):
        """Helper to find start and end indices of contiguous True blocks."""
        clusters = []
        in_cluster = False
        for idx, val in enumerate(list(significant_clusters)):
            if val and not in_cluster:
                # Start of a new cluster
                start_idx = idx
                in_cluster = True
            elif not val and in_cluster:
                # End of the cluster
                end_idx = idx - 1
                clusters.append((start_idx, end_idx))
                in_cluster = False
        # Handle the case where the last value is in a cluster
        if in_cluster:
            end_idx = len(list(significant_clusters)) - 1
            clusters.append((start_idx, end_idx))
        return clusters

    clusters = find_clusters(significant_clusters)

    # # Determine y position for the bars
    # max_y = np.max(mean_true_accuracy + se_true_accuracy)
    # min_y = np.min(mean_shuffle_accuracy - se_shuffle_accuracy)
    # y_bar = max_y + 0.02  # Adjust as needed
    # plt.ylim([min_y, y_bar + 0.05])  # Adjust ylim to accommodate the bars

    # Set y_bar to a fixed value within the y-axis limits
    y_bar = 0.95  # Fixed value near the top of the y-axis

    # Plot horizontal bars and asterisks for significant clusters
    for cluster in clusters:
        start_idx, end_idx = cluster
        start_time = time_points[start_idx] - (window_duration / 2)
        end_time = time_points[end_idx] + (window_duration / 2)
        plt.hlines(y=y_bar, xmin=start_time, xmax=end_time, color='black', linewidth=2)
        # Place an asterisk at the center of the bar
        center_time = (start_time + end_time) / 2
        plt.text(center_time, y_bar + 0.01, '*', ha='center', va='bottom', fontsize=14)

    # Set axis limits
    plt.ylim(0, 1)  # Y-axis limits
    plt.xlim(-1, 1.5)  # X-axis limits

    plt.xlabel('Time from Stim Onset (s)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoding Accuracy over Time for {condition_comparison} in ROI {roi}')
    plt.legend()
    
    # CREATE TIMESTAMP PREFIX
    timestamp_str = f"{timestamp}_" if timestamp else ""

    # CREATE P THRESH PREFIX
    p_thresh_str = str(p_thresh)
    
    # ADD other string if provided
    other_str = f"_{other_string_to_add}" if other_string_to_add else ""
    
    # Construct the filename
    filename = (f"{timestamp_str}{condition_comparison}_ROI_{roi}_window{window_size}_step{step_size}_"
                f"{n_repeats}_repeats_{n_perm}_perm_{p_thresh_str}_p_thresh{other_str}.png") 
    filepath = os.path.join(save_dir, filename)

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Save and close the plot
    plt.savefig(filepath)
    plt.close()


def plot_accuracies_with_multiple_sig_clusters(
    time_points: np.ndarray,
    accuracies_dict: Dict[str, np.ndarray],
    significance_clusters_dict: Dict[str, Dict],
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    sampling_rate: float = 256,
    comparison_name: str = "",
    roi: str = "",
    save_dir: str = ".",
    timestamp: Optional[str] = None,
    p_thresh: float = 0.05,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    ylabel: str = "Accuracy",
    ylim: Tuple[float, float] = (0.0, 1.0),
    xlim: Tuple[float, float] = (-0.5, 1.5),
    single_column: bool = True,
    show_legend: bool = True,
    show_chance_level: bool = True,
    chance_level: float = 0.5,
    filename_suffix: str = "",
    return_fig: bool = False,
    samples_axis: int = 0,
    # New parameters for multiple significance clusters
    sig_bar_height: float = 0.02,  # Height of significance bars as fraction of y-range
    sig_bar_spacing: float = 0.01,  # Spacing between different significance bars as fraction of y-range
    sig_bar_colors: Optional[Dict[str, str]] = None,
    sig_bar_labels: Optional[Dict[str, str]] = None,
    sig_bar_base_position: Optional[float] = None,  # Base y-position for significance bars (default: 0.9 of y-range)
    show_sig_legend: bool = False,  # Whether to show legend for significance bars
    sig_marker_style: str = "*",  # Marker style for significance ('*', '**', '***', or custom)
):
    """
    Enhanced wrapper for plotting accuracies with multiple significance clusters.
    
    This function extends plot_accuracies_nature_style to handle multiple significance
    clusters from different comparisons (e.g., different directions in LWPC).
    
    Parameters
    ----------
    time_points : np.ndarray
        Time points for x-axis.
    accuracies_dict : Dict[str, np.ndarray]
        Dictionary mapping condition names to accuracy arrays.
    significance_clusters_dict : Dict[str, Dict]
        Dictionary mapping cluster names to cluster information.
        Each entry should have:
        - 'clusters': np.ndarray or List[bool] - Boolean mask of significant time points
        - 'label': str (optional) - Label for this cluster in legend
        - 'color': str (optional) - Color for significance bar
        - 'marker': str (optional) - Marker style ('*', '**', etc.)
        - 'position_offset': float (optional) - Additional offset from base position
    sig_bar_height : float
        Height of each significance bar as fraction of y-range.
    sig_bar_spacing : float
        Vertical spacing between different significance bars as fraction of y-range.
    sig_bar_colors : Dict[str, str]
        Dictionary mapping cluster names to colors (overrides individual settings).
    sig_bar_labels : Dict[str, str]
        Dictionary mapping cluster names to labels for legend.
    sig_bar_base_position : float
        Base y-position for significance bars (default: 0.9 of y-range from bottom).
    show_sig_legend : bool
        Whether to show a separate legend for significance bars.
    sig_marker_style : str
        Default marker style for significance indicators.
    
    Other parameters are passed through to the original plot_accuracies_nature_style function.
    
    Returns
    -------
    fig : matplotlib.figure.Figure (if return_fig=True)
        The figure object.
        
    Examples
    --------
    # Example with LWPC having significant clusters for two directions
    significance_clusters_dict = {
        'lwpc_direction1': {
            'clusters': sig_mask_direction1,
            'label': 'LWPC >',
            'color': 'red',
            'marker': '*'
        },
        'lwpc_direction2': {
            'clusters': sig_mask_direction2,
            'label': 'LWPC <',
            'color': 'blue',
            'marker': '**'
        }
    }
    
    plot_accuracies_with_multiple_sig_clusters(
        time_points=time_window_centers,
        accuracies_dict=accuracies_dict,
        significance_clusters_dict=significance_clusters_dict,
        sig_bar_spacing=0.015,
        show_sig_legend=True,
        # ... other parameters
    )
    """
    
    # Import the original NATURE_STYLE (you'll need to adjust this import based on your setup)
    NATURE_STYLE = {
        'figure.figsize': (89/25.4, 89/25.4),
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'lines.linewidth': 1,
        'lines.markersize': 3,
        'legend.frameon': False,
        'legend.columnspacing': 0.5,
        'legend.handlelength': 1,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    }
    
    # Apply Nature style settings
    with plt.rc_context(NATURE_STYLE):
        # Set figure size based on column width
        if single_column:
            fig_width = 89 / 25.4  # 89mm to inches
            fig_height = fig_width * 0.8  # Aspect ratio
        else:
            fig_width = 183 / 25.4  # 183mm to inches
            fig_height = fig_width * 0.4
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Define Nature-appropriate colors if not provided
        if colors is None:
            nature_colors = [
                '#0173B2',  # Blue
                '#DE8F05',  # Orange
                '#029E73',  # Green
                '#CC78BC',  # Light purple
                '#ECE133',  # Yellow
                '#56B4E9',  # Light blue
                '#F0E442',  # Light yellow
                '#949494',  # Gray
            ]
            colors = {}
            for i, label in enumerate(accuracies_dict.keys()):
                colors[label] = nature_colors[i % len(nature_colors)]
        
        # Plot each accuracy time series
        for i, (label, accuracies) in enumerate(accuracies_dict.items()):
            # Compute statistics
            if accuracies.ndim == 2:
                n_samples = accuracies.shape[samples_axis]
                mean_accuracy = np.mean(accuracies, axis=samples_axis)
                std_accuracy = np.std(accuracies, axis=samples_axis)
                sem_accuracy = std_accuracy / np.sqrt(n_samples)
            else:
                mean_accuracy = accuracies
                std_accuracy = np.zeros_like(accuracies)
                sem_accuracy = np.zeros_like(accuracies)
                print(f"⚠️ Warning: Accuracy data for '{label}' is 1D. Cannot compute SEM; plotting without error bars.")

            # Get color and linestyle
            color = colors.get(label, '#0173B2') if colors else '#0173B2'
            linestyle = linestyles.get(label, '-') if linestyles else '-'
            
            # Plot mean line with higher contrast
            ax.plot(time_points, mean_accuracy, 
                    label=label, 
                    color=color, 
                    linestyle=linestyle,
                    linewidth=1,
                    zorder=3)
            
            # Plot SEM as shaded area
            ax.fill_between(
                time_points,
                mean_accuracy - std_accuracy,
                mean_accuracy + std_accuracy,
                alpha=0.25,  # Lighter shading for Nature style
                color=color,
                linewidth=0,
                zorder=1
            )
        
        # Add chance level line if requested
        if show_chance_level:
            ax.axhline(y=chance_level, 
                      color='#666666', 
                      linestyle='--', 
                      linewidth=0.5,
                      zorder=2,
                      label='Chance')
        
        # Add stimulus onset line
        ax.axvline(x=0, 
                  color='#666666', 
                  linestyle='-', 
                  linewidth=0.5,
                  alpha=0.5,
                  zorder=2)
        
        # =================================================================
        # ENHANCED SECTION: Handle multiple significance clusters
        # =================================================================
        if significance_clusters_dict:
            y_range = ylim[1] - ylim[0]
            
            # Calculate base position for significance bars
            if sig_bar_base_position is None:
                sig_bar_base_position = ylim[0] + y_range * 0.9
            
            # Track lines for significance legend
            sig_legend_elements = []
            
            # Process each set of significant clusters
            for cluster_idx, (cluster_name, cluster_info) in enumerate(significance_clusters_dict.items()):
                # Extract cluster information
                if isinstance(cluster_info, dict):
                    clusters_mask = cluster_info.get('clusters')
                    cluster_label = cluster_info.get('label', cluster_name)
                    cluster_color = cluster_info.get('color', 'black')
                    cluster_marker = cluster_info.get('marker', sig_marker_style)
                    position_offset = cluster_info.get('position_offset', 0)
                else:
                    # If just a boolean array is provided
                    clusters_mask = cluster_info
                    cluster_label = cluster_name
                    cluster_color = 'black'
                    cluster_marker = sig_marker_style
                    position_offset = 0
                
                # Override with global settings if provided
                if sig_bar_colors and cluster_name in sig_bar_colors:
                    cluster_color = sig_bar_colors[cluster_name]
                if sig_bar_labels and cluster_name in sig_bar_labels:
                    cluster_label = sig_bar_labels[cluster_name]
                
                # Skip if no significant clusters
                if clusters_mask is None or not np.any(clusters_mask):
                    continue
                
                # Calculate y-position for this set of clusters
                y_position = sig_bar_base_position + (cluster_idx * (sig_bar_height + sig_bar_spacing) * y_range) + position_offset
                
                # Find contiguous clusters
                clusters = find_contiguous_clusters(clusters_mask)
                
                # Draw each cluster
                for start_idx, end_idx in clusters:
                    
                    start_time = time_points[start_idx]
                    end_time = time_points[end_idx]
                    
                    # Draw significance bar
                    line = ax.plot([start_time, end_time], 
                                  [y_position, y_position],
                                  color=cluster_color, 
                                  linewidth=1.5,
                                  solid_capstyle='butt',
                                  alpha=0.8)[0]
                    
                    # Add significance marker
                    center_time = (start_time + end_time) / 2
                    ax.text(center_time, 
                           y_position + y_range * 0.01, 
                           cluster_marker, 
                           ha='center', 
                           va='bottom', 
                           fontsize=8,
                           fontweight='bold',
                           color=cluster_color)
                
                # Add to legend elements (only once per cluster type)
                if show_sig_legend and clusters:
                    from matplotlib.lines import Line2D
                    sig_legend_elements.append(
                        Line2D([0], [0], color=cluster_color, linewidth=1.5, 
                              label=cluster_label, alpha=0.8)
                    )
        
        # Set labels
        ax.set_xlabel('Time from stimulus onset (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=7)
        
        # Set axis limits
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        
        # Configure ticks
        ax.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)
        
        # Set specific x-ticks for clarity
        x_ticks = np.arange(-0.5, 1.6, 0.5)
        ax.set_xticks(x_ticks)
        
        # Set specific y-ticks for consistency
        y_ticks = np.linspace(ylim[0], ylim[1], num=5) 
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])

        # Configure legends
        if show_legend:
            # Main legend for accuracy lines
            main_legend = ax.legend(
                loc='upper right',
                fontsize=6,
                frameon=False,
                handlelength=1,
                handletextpad=0.5,
                borderpad=0.2,
                columnspacing=0.5
            )
            if len(accuracies_dict) > 1:
                for line in main_legend.get_lines():
                    line.set_linewidth(1.5)
            
            # Add significance legend if requested
            if show_sig_legend and sig_legend_elements:
                sig_legend = ax.legend(
                    handles=sig_legend_elements,
                    loc='upper left',
                    fontsize=6,
                    frameon=False,
                    handlelength=1,
                    handletextpad=0.5,
                    borderpad=0.2,
                    title='Significance',
                    title_fontsize=6
                )
                # Add back the main legend (matplotlib removes it when adding second legend)
                ax.add_artist(main_legend)

        # Add title only if specified
        if title:
            ax.set_title(title, fontsize=7, pad=3)
        
        # Remove top and right spines (Nature style)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make remaining spines thinner
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        if return_fig:
            return fig
        else:
            # Create filename
            timestamp_str = f"{timestamp}_" if timestamp else ""
            
            filename_parts = [timestamp_str.rstrip('_')]
            if comparison_name:
                filename_parts.append(comparison_name)
            if roi:
                filename_parts.append(roi)
            if filename_suffix:
                filename_parts.append(filename_suffix)
            
            filename = "_".join(filter(None, filename_parts)) + ".pdf"
            filepath = os.path.join(save_dir, filename)
            
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Save in multiple formats
            plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(filepath.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(filepath.replace('.pdf', '.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)
            
            plt.close(fig)
            print(f"Saved Nature-style plot with multiple significance clusters to: {filepath}")
