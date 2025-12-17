from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FixedLocator
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors


def scatter(single_model_alignment, lens):
    plt.scatter(lens, single_model_alignment)

    # fit a line
    m, b = np.polyfit(lens, single_model_alignment, 1)
    print(m)
    plt.plot(lens, m*np.array(lens) + b, color='red')
    plt.xlabel("Poem Length")
    plt.ylabel("Alignment Score")
    plt.xlim(0, 10 * np.median(lens))
    plt.title("Alignment Score vs Poem Length")
    plt.show()

def quantile(single_model_alignment_float, lens):
    lens = [len(poem['content'].split()) for poem in dataset]
    lens = np.array(lens)
    scores = np.array(single_model_alignment_float)
    len_1500 = [i for i, l in enumerate(lens) if l <= 1500]
    lens = lens[len_1500]
    scores = scores[len_1500]

    num_bins = 20
    bins = np.linspace(lens.min(), lens.max(), num_bins+1)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_quantiles = []

    for i in range(num_bins):
        mask = (lens >= bins[i]) & (lens < bins[i+1])
        if np.any(mask):
            bin_quantiles.append(np.percentile(scores[mask], 10))  # 10% 分位数
        else:
            bin_quantiles.append(np.nan)

    bin_quantiles = np.array(bin_quantiles)


    plt.figure(figsize=(8,5))
    plt.scatter(lens, scores, alpha=0.5, label="Alignment Scores")
    plt.plot(bin_centers, bin_quantiles, color='red', linewidth=2, label='10% Quantile')
    plt.xlabel("Poem Length")
    plt.ylabel("Alignment Score")
    plt.title("Alignment Score vs Poem Length (Quantile)")
    plt.legend()
    plt.show()

def quantile_fig(single_model_alignment_float):
    lens = np.array([len(poem['content'].split()) for poem in dataset])
    scores = np.array(single_model_alignment_float)

    mask_1500 = lens <= 1500
    lens = lens[mask_1500]
    scores = scores[mask_1500]

    num_bins = 20
    bins = np.linspace(lens.min(), lens.max(), num_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_data = []
    for i in range(num_bins):
        mask = (lens >= bins[i]) & (lens < bins[i+1])
        bin_scores = scores[mask]
        if len(bin_scores) == 0:
            bin_scores = [np.nan] 
        bin_data.append(bin_scores)

    plt.figure(figsize=(12,6))
    box = plt.boxplot(
        bin_data, 
        positions=bin_centers, 
        widths=(bins[1]-bins[0])*0.8, 
        patch_artist=True,
        manage_ticks=False
    )

    for patch in box['boxes']:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)
    for whisker in box['whiskers']:
        whisker.set_color('green')
    for cap in box['caps']:
        cap.set_color('green')
    for median in box['medians']:
        median.set_color('darkgreen')

    plt.scatter(lens, scores, alpha=0.3, color='green', s=10)

    plt.xticks(np.linspace(0, 1500, 10))

    plt.xlabel("Poem Length")
    plt.ylabel("Alignment Score")
    plt.title("Alignment Score Distribution per Length Bin")
    plt.show()




def plot_alignment_grid_custom(lens, scores_matrix, model_names, model_params_B):
    num_models = len(model_names)
    fig, axes = plt.subplots(num_models, num_models, figsize=(20, 20), squeeze=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Set universal y-limit for all plots to 0.4-1.1
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylim(0.4, 1.1)
            ax.set_xlim(0, 1500)

    # Scale parameters for radius visualization
    max_param = max(model_params_B)
    radii = [ (p / max_param) * 0.3 + 0.1 for p in model_params_B] 

    for i in range(num_models): # rows (y-axis label index)
        for j in range(num_models): # columns (x-axis label index)
            ax = axes[i, j]
            
            # Get scores for this pair (model i vs model j)
            current_scores = scores_matrix[:, i, j].copy()
            current_scores = np.array([
                float(x) if np.isscalar(x) else float(np.mean(x))
                for x in current_scores
            ])
            noise = np.random.normal(0, 0.05, current_scores.shape)
            current_scores += noise

            # Diagonal: Centered Circle with gradient yellow color
            if i == j:
                # Clear axis settings to allow clean circle drawing
                ax.clear() 
                # Create a radial gradient effect using multiple circles
                # Gradient from bright yellow to darker orange
                n_circles = 50
                for k in range(n_circles):
                    alpha_val = 0.8 - (k / n_circles) * 0.4  # Fade from 0.8 to 0.4
                    radius_val = radii[i] * (1 - k / n_circles)
                    # Color gradient from bright yellow (#FFD700) to white (#FFFFFF)
                    color_val = mcolors.to_rgb('#FFFFFF') if k < n_circles//2 else mcolors.to_rgb("#FFD700")
                    # Blend colors
                    t = k / n_circles
                    color = tuple(
                        (1-t) * mcolors.to_rgb('#FFFFFF')[c] + t * mcolors.to_rgb("#FFD700")[c]
                        for c in range(3)
                    )
                    circle = Circle((0.5, 0.5), radius_val, color=color, alpha=alpha_val, transform=ax.transAxes)
                    ax.add_patch(circle)
                
                # Keep the axis limits but hide ticks and spines
                ax.set_xlim(0, 1500)
                ax.set_ylim(0.4, 1.1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Lower triangle (i > j): Quantile box plot
            elif i > j:
                num_bins = 15
                bins = np.linspace(lens.min(), lens.max(), num_bins + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_data = []
                for k in range(num_bins):
                    mask = (lens >= bins[k]) & (lens < bins[k+1])
                    bin_scores = current_scores[mask]
                    if len(bin_scores) == 0:
                        bin_scores = [np.nan]
                    bin_data.append(bin_scores)
                
                box = ax.boxplot(
                    bin_data,
                    positions=bin_centers,
                    widths=(bins[1]-bins[0])*0.8,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False
                )
                for patch in box['boxes']: patch.set_facecolor('green')
                for item in ['whiskers', 'caps', 'medians']:
                    for line in box[item]: line.set_color('green' if item != 'medians' else 'darkgreen')
                
                ax.scatter(lens, current_scores, alpha=0.1, color='green', s=5)
                ax.set_xticks(np.linspace(0, 1500, 4, dtype=int))
                ax.set_yticks([0.5, 0.7, 0.9, 1.0])
                # Show tick labels
                if j == 0:
                    ax.set_yticklabels(['0.5', '0.7', '0.9', '1.0'], fontsize=9)
                else:
                    ax.set_yticklabels([])
                if i == num_models - 1:
                    ax.set_xticklabels(['0', '500', '1000', '1500'], fontsize=9)
                else:
                    ax.set_xticklabels([])

            # Upper triangle (i < j): Scatter plot with fit line and slope text
            elif i < j:
                ax.scatter(lens, current_scores, alpha=0.3, color='red', s=5)
                m, b = np.polyfit(lens, current_scores, 1)
                ax.plot(lens, m*lens + b, color='black', linestyle='--', linewidth=1)

                # Display the slope 'm' in the lower right corner in 10e-4 format
                slope_text = f"{m * 10000:.2f}"
                ax.text(0.98, 0.02, slope_text, ha='right', va='bottom', fontsize=30, color='black', transform=ax.transAxes)
                ax.set_xticks(np.linspace(0, 1500, 4, dtype=int))
                ax.set_yticks([0.5, 0.7, 0.9, 1.0])
                # Show tick labels
                if j == 0:
                    ax.set_yticklabels(['0.5', '0.7', '0.9', '1.0'], fontsize=9)
                else:
                    ax.set_yticklabels([])
                if i == num_models - 1:
                    ax.set_xticklabels(['0', '500', '1000', '1500'], fontsize=9)
                else:
                    ax.set_xticklabels([])

            # Set Y-labels for the first column
            if j == 0:
                if i != 0:
                    ax.set_ylabel(model_names[i], fontsize=15, rotation=90, va='center', ha='center', labelpad=10)
                else:
                    ax.set_ylabel(model_names[i], fontsize=15, rotation=90, va='center', ha='center', labelpad=32)
            
            # Set X-labels for the bottom row
            if i == num_models - 1 and j != num_models - 1:
                ax.set_xlabel("Poem Length (words)", fontsize=10)

    plt.savefig("alignment_grid_ppl.png", dpi=125)


def plot_alignment_grid_asym(lens, scores_matrix, model_names, model_names2, model_params_B):
    num_models = len(model_names)
    fig, axes = plt.subplots(num_models, num_models, figsize=(20, 20), squeeze=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Calculate appropriate y-limits based on data
    all_scores = []
    for i in range(num_models):
        for j in range(num_models):
            current_scores = scores_matrix[:, i, j].copy()
            current_scores = np.array([
                float(x) if np.isscalar(x) else float(np.mean(x))
                for x in current_scores
            ])
            all_scores.extend(current_scores)
    
    y_min = np.percentile(all_scores, 1)  # 1st percentile
    y_max = np.percentile(all_scores, 99)  # 99th percentile
    y_margin = (y_max - y_min) * 0.1
    y_min -= y_margin
    y_max += y_margin
    
    # Set universal limits for all plots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(1.5, 5.5)

    # Scale parameters for radius visualization
    max_param = max(model_params_B)
    radii = [ (p / max_param) * 0.3 + 0.1 for p in model_params_B] 

    # First pass: calculate all slopes for normalization
    slopes = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            current_scores = scores_matrix[:, i, j].copy()
            current_scores = np.array([
                float(x) if np.isscalar(x) else float(np.mean(x))
                for x in current_scores
            ])
            m, b = np.polyfit(lens, current_scores, 1)
            slopes[i, j] = m
    
    # Normalize slopes: positive slopes -> warm colors (yellow), negative -> cool colors (purple)
    # Split normalization for positive and negative slopes
    slope_min = slopes.min()
    slope_max = slopes.max()
    slope_abs_max = max(abs(slope_min), abs(slope_max))
    slopes_normalized = slopes / slope_abs_max  # Range: [-1, 1]
    
    for i in range(num_models): # rows (y-axis label index)
        for j in range(num_models): # columns (x-axis label index)
            ax = axes[i, j]
            
            # Get scores for this pair (model i vs model j)
            current_scores = scores_matrix[:, i, j].copy()
            current_scores = np.array([
                float(x) if np.isscalar(x) else float(np.mean(x))
                for x in current_scores
            ])
            noise = np.random.normal(0, 0.05, current_scores.shape)
            current_scores += noise
            
            # Get normalized slope for color mapping
            slope_norm = slopes_normalized[i, j]
            
            # Map slope to color: yellow for negative, purple for positive
            # Use a custom colormap: yellow (negative) -> white (0) -> purple (positive)
            if slope_norm < 0:
                # Negative slope: map to yellow/orange shades
                intensity = abs(slope_norm)
                # Background: light yellow
                bg_color = (1.0 - intensity * 0.05, 0.95 - intensity * 0.2, 0.7 - intensity * 0.5)
                # Box: darker yellow/orange
                box_color = (1.0 - intensity * 0.1, 0.8 - intensity * 0.3, 0.3 - intensity * 0.2)
                median_color = (0.9 - intensity * 0.2, 0.6 - intensity * 0.3, 0.1)
            else:
                # Positive slope: map to purple shades
                intensity = slope_norm
                # Background: light purple
                bg_color = (0.9 - intensity * 0.3, 0.85 - intensity * 0.35, 1.0 - intensity * 0.2)
                # Box: darker purple
                box_color = (0.6 - intensity * 0.3, 0.4 - intensity * 0.2, 0.9 - intensity * 0.1)
                median_color = (0.4 - intensity * 0.2, 0.2 - intensity * 0.1, 0.8)
            
            # Set background color
            ax.set_facecolor(bg_color)

            # Lower triangle (i >= j): Quantile box plot for discrete values
            if True:
                unique_lens = np.unique(lens)
                bin_data = []
                for val in unique_lens:
                    mask = lens == val
                    bin_scores = current_scores[mask]
                    if len(bin_scores) == 0:
                        bin_scores = [np.nan]
                    bin_data.append(bin_scores)
                
                box = ax.boxplot(
                    bin_data,
                    positions=unique_lens,
                    widths=0.6,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False
                )
                # Use the calculated box_color for boxes
                for patch in box['boxes']: 
                    patch.set_facecolor(box_color)
                    patch.set_alpha(0.8)
                for item in ['whiskers', 'caps', 'medians']:
                    for line in box[item]: 
                        line.set_color(box_color if item != 'medians' else median_color)
                        line.set_linewidth(1.5 if item == 'medians' else 1.0)
                
                # Add fit line
                m = slopes[i, j]
                b = np.mean(current_scores) - m * np.mean(lens)
                x_range = np.linspace(lens.min(), lens.max(), 100)
                ax.plot(x_range, m * x_range + b, color=median_color, linestyle='--', linewidth=2, alpha=0.7)
                
                ax.scatter(lens, current_scores, alpha=0.15, color=box_color, s=5)
                ax.set_xticks(unique_lens)
                # Auto-generate y-ticks based on data range
                y_ticks = np.linspace(y_min, y_max, 5)
                ax.set_yticks(y_ticks)
                # Show tick labels
                if j == 0:
                    ax.set_yticklabels([f'{y:.2f}' for y in y_ticks], fontsize=9)
                else:
                    ax.set_yticklabels([])
                if i == num_models - 1:
                    ax.set_xticklabels([str(int(v)) for v in unique_lens], fontsize=9)
                else:
                    ax.set_xticklabels([])

            # Upper triangle (i < j): Scatter plot with fit line and slope text
            elif False:
                ax.scatter(lens, current_scores, alpha=0.3, color=box_color, s=5)
                m = slopes[i, j]
                b = np.mean(current_scores) - m * np.mean(lens)
                x_range = np.linspace(lens.min(), lens.max(), 100)
                ax.plot(x_range, m*x_range + b, color=median_color, linestyle='--', linewidth=2, alpha=0.7)

                # Display the slope 'm' in the lower right corner in 10e-4 format
                slope_text = f"{m * 10000:.2f}"
                ax.text(0.98, 0.02, slope_text, ha='right', va='bottom', fontsize=30, color=median_color, transform=ax.transAxes)
                unique_lens = np.unique(lens)
                ax.set_xticks(unique_lens)
                y_ticks = np.linspace(y_min, y_max, 5)
                ax.set_yticks(y_ticks)
                # Show tick labels
                if j == 0:
                    ax.set_yticklabels([f'{y:.2f}' for y in y_ticks], fontsize=9)
                else:
                    ax.set_yticklabels([])
                if i == num_models - 1:
                    ax.set_xticklabels([str(int(v)) for v in unique_lens], fontsize=9)
                else:
                    ax.set_xticklabels([])

            # Set Y-labels for the first column
            if j == 0:
                ax.set_ylabel(model_names[i], fontsize=20, rotation=90, va='center', ha='center', labelpad=10)
            
            # Set X-labels (model_names2) for the top row
            if i == 0:
                ax.set_title(model_names2[j], fontsize=20, pad=10, rotation=45, ha='left')
            
            # Set X-labels for the bottom row
            if i == num_models - 1:
                ax.set_xlabel("Image Aesthetic Score", fontsize=10)

    # Add colorbar legend for slope values
    from matplotlib.patches import Rectangle
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Create a custom colorbar axis on the right side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    # Create custom colormap: yellow -> white -> purple (reversed)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [
        (0.9, 0.6, 0.1),   # Deep orange/yellow (negative)
        (1.0, 0.9, 0.5),   # Light yellow
        (0.95, 0.95, 0.95), # Near white (zero)
        (0.7, 0.6, 0.95),  # Light purple
        (0.4, 0.2, 0.8)    # Deep purple (positive)
    ]
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('yellow_purple', colors_list, N=n_bins)
    
    # Normalize to slope range
    norm = Normalize(vmin=slope_min, vmax=slope_max)
    cb = ColorbarBase(cbar_ax, cmap=cmap_custom, norm=norm, orientation='vertical')
    cb.set_label('Slope (Alignment Score / Image Aesthetic Score)', fontsize=12, labelpad=15)
    
    
    plt.subplots_adjust(bottom=0.08, right=0.90)
    plt.savefig("alignment_grid_score_poems.png", dpi=125, bbox_inches='tight')


if __name__ == "__main__":
    # load dataset and alignment scores
    dataset = load_dataset('SHENJJ1017/poem_aesthetic_eval', split='train', revision='main')
    result_array_path = "results/alignment/raw_full.npy"
    alignment_scores = np.load(result_array_path, allow_pickle=True)
    lens = np.array([len(poem['content'].split()) for poem in dataset])

    # prepare data: filter length <=1500
    mask_1500 = lens <= 1500
    lens_filtered = lens[mask_1500]
    scores_filtered = alignment_scores[mask_1500]

    # perplexity or length
    ppl_path = "results/perplexity_scores.npy" # length: "results/poem_lengths.npy"
    perplexities = np.load(ppl_path)

    valid_mask = perplexities != float('inf')
    valid_ppls = perplexities[valid_mask]
    scores_filtered_ppls = alignment_scores[valid_mask]
    model_params_B = [8.0, 7.0, 7.0, 2.0, 1.7, 1.0, 0.560]
    plot_alignment_grid_custom(lens_filtered, scores_filtered, model_params_B)

    """result_array_path = "../results/alignment/poems_images_raw_mutual_knn_scores.npy"
    dataset = load_dataset('SHENJJ1017/Image-Text', split='train', revision='main')
    scores = np.array([int(d['score']) for d in dataset])
    # print(scores)
    valid_ppls = scores
    alignment_scores = np.load(result_array_path, allow_pickle=True)
    scores_filtered_ppls = alignment_scores

    model_names = ['Meta-Llama-3-8B', 'Mistral-7B-v0.1', 'Gemma-7b', 'Gemma-2b', 'Bloomz-1b7', 'OLMo-1B-hf', 'Bloomz-560m']
    lvm_models = [
            "vit_tiny_patch16\n_224.augreg_in21k",
            "vit_small_patch16\n_224.augreg_in21k",
            "vit_base_patch16\n_224.mae",
            "vit_base_patch14\n_dinov2.lvd142m",
            "vit_large_patch14\n_dinov2.lvd142m",
            "vit_large_patch14\n_clip_224.laion2b",
            "vit_huge_patch14\n_clip_224.laion2b_ft_in12k",
        ]
    lvm_models.reverse()

    num_models = len(model_names)



    plot_alignment_grid_asym(valid_ppls, scores_filtered_ppls, model_names, lvm_models, model_params_B)"""