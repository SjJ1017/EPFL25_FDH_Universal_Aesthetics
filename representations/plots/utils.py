import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def check_symmetric(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is symmetric.

    Args:
        matrix: A numpy array to check.

    Returns:
        True if the matrix is symmetric, False otherwise.
    """
    if_sym =  np.allclose(matrix, matrix.T)
    if not if_sym:
        print(matrix - matrix.T)
    return if_sym



def str_to_matrix(matrix_str: str, symmetric: bool) -> np.ndarray:
    """
    Convert a string representation of a matrix into a NumPy array.

    Args:
        matrix_str: A string representation of a matrix, such as 
    [[1.         0.74595469 0.66893202 0.61941743 0.72912627 0.73915857 0.70550162]
    [0.74595469 1.         0.7200647  0.62556636 0.70097089 0.74854368 0.68155342]
    [0.66893202 0.7200647  1.         0.69190937 0.58122981 0.65436894  0.54983824]
    [0.61941743 0.62556636 0.69190937 1.         0.6029126  0.62006468  0.57249188]
    [0.72912627 0.70097089 0.58122981 0.6029126  1.         0.70194173  0.76893204]
    [0.73915857 0.74854368 0.65436894 0.62006468 0.70194173 1.          0.6877023 ]
    [0.70550162 0.68155342 0.54983824 0.57249188 0.76893204 0.6877023 1.        ]]



    Returns:
        A NumPy array representing the matrix.
    """
    # Remove whitespace and newlines
    matrix_str = matrix_str.strip().replace('\n', ' ')
    # Remove the outer brackets
    matrix_str = matrix_str[1:-1].strip()
    # Split into rows
    rows = matrix_str.split(']')
    matrix = []
    for row in rows:
        row = row.replace('[', '').strip()
        if row:
            values = [float(x) for x in row.split()]
            matrix.append(values)
    result = np.array(matrix)
    if symmetric:
        if not check_symmetric(result):
            raise ValueError("The resulting matrix is not symmetric.")
    return result


def read_records(file_path: str) -> list[str]:
    """
    Read lines from a text file.

    Args:
        file_path: Path to the text file.

    Returns:
        A list of strings, each representing a line in the file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    validatelines = []
    for line in lines:
        if not line.startswith('--- IGNORE ---'):
            validatelines.append(line)
        else:
            break
    matrices_str = {}
    current_matrix_name = None
    current_matrix_lines = []
    reading_matrix = False
    for line in validatelines:
        if line.endswith('!'):
            current_matrix_name = line[:-1]
            reading_matrix = True
        elif reading_matrix and '\n' not in line and line != '':
            current_matrix_lines.append(line)
        else:
            if reading_matrix and current_matrix_name is not None:
                matrices_str[current_matrix_name] = '\n'.join(current_matrix_lines)
                current_matrix_name = None
                current_matrix_lines = []
                reading_matrix = False
            elif not reading_matrix:
                continue
            else:
                print(line)
                raise ValueError("Unexpected line format while reading matrices.")
    if reading_matrix and current_matrix_name is not None:
        matrices_str[current_matrix_name] = '\n'.join(current_matrix_lines)

    matrices = {}
    for name in matrices_str:
        matrices[name] = str_to_matrix(matrices_str[name], symmetric=True)

    return matrices


def plot_heatmap(matrix1, matrix2, left_name = "Left", right_name = "Right", center_name = "Center", save_path=None):
    if isinstance(matrix1, str):
        matrix1 = str_to_matrix(matrix1, symmetric=True)
    if isinstance(matrix2, str):
        matrix2 = str_to_matrix(matrix2, symmetric=True)

    diff = matrix1 - matrix2

    vmin_val = min(matrix1.min(), matrix2.min())
    vmax_val = max(matrix1.max(), matrix2.max())


    abs_max = np.max(np.abs(diff))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))


    sns.heatmap(
        matrix1, annot=False, cmap="Grays",
        vmin=vmin_val, vmax=vmax_val, square=True, ax=axes[0]
    )
    axes[0].set_title(left_name)
    axes[0].set_yticklabels(['Meta-Llama-3-8B','Mistral-7B-v0.1','Gemma-7b','Gemma-2b','Bloomz-1b7','OLMo-1B-hf','Bloomz-560m'], rotation=0)


    sns.heatmap(
        matrix2, annot=False, cmap="Grays",
        vmin=vmin_val, vmax=vmax_val, square=True, ax=axes[2]
    )
    axes[2].set_title(right_name)
    axes[2].set_yticklabels([])


    sns.heatmap(
        diff, annot=False, cmap="RdYlGn",
        vmin=-abs_max, vmax=abs_max, center=0,
        square=True, ax=axes[1]
    )
    axes[1].set_title(center_name)
    axes[1].set_yticklabels([])


    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    print(f"{save_path} saved")


def plot_alignments(alignment_scores_1, alignment_scores_2, label_1="Poems", label_2="Text", xlim=(0, 15), ylim=(0.45, 0.75), save_path=None):
    # Use a clean, academic style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    #alignment_scores = [0.70551378, 0.62969923, 0.58245617, 0.62080199, 0.68859649, 0.59761906]
    #alignment_scores_text = [0.67744356, 0.56929827, 0.57280701, 0.50952387, 0.50789469, 0.47105262]
    capability_data = {
        'Model': ['Mistral-7B-v0.1', 'Gemma-7b', 'Gemma-2b', 'Bloomz-1b7', 'OLMo-1B-hf', 'Bloomz-560m'],
        # 'Capability Score': [44.87, 26.59, 20.18, 10.44, 21.82, 6.20]
        'Capability Score': [12.77, 13.07, 7.32, 4.05, 6.63, 3.51]
    }
    parameter_sizes = [7, 7, 2, 1.7, 1, 0.56]
    df = pd.DataFrame(capability_data)
    df[f'Alignment Score ({label_1})'] = alignment_scores_1
    df[f'Alignment Score ({label_2})'] = alignment_scores_2
    df['Parameter Size (B)'] = parameter_sizes

    plt.figure(figsize=(10, 7))  # High DPI for publication

    size_scale = 400

    scatter_poems = plt.scatter(
        df['Capability Score'],
        df[f'Alignment Score ({label_1})'],
        s=df['Parameter Size (B)'] * size_scale,
        c="#03b347",
        alpha=0.5,
        edgecolors='white',
        linewidth=1.5,
        zorder=3
    )

    scatter_text = plt.scatter(
        df['Capability Score'],
        df[f'Alignment Score ({label_2})'],
        s=df['Parameter Size (B)'] * size_scale,
        c="#114df0",
        alpha=0.5,
        edgecolors='white',
        linewidth=1.5,
        zorder=3
    )

    coeff_poems = np.polyfit(df["Capability Score"], df[f"Alignment Score ({label_1})"], 1)
    poly_poems = np.poly1d(coeff_poems)
    x_line = np.linspace(0, 15, 200)
    plt.plot(x_line, poly_poems(x_line), color="#03b347", linewidth=2.5, alpha=0.8, zorder=2)

    coeff_text = np.polyfit(df["Capability Score"], df[f"Alignment Score ({label_2})"], 1)
    poly_text = np.poly1d(coeff_text)
    plt.plot(x_line, poly_text(x_line), color="#114df0", linewidth=2.5, alpha=0.8, zorder=2)

    plt.xlabel('Open LLM Leaderboard Average Score', fontweight='bold', labelpad=10)
    plt.ylabel('Alignment Score', fontweight='bold', labelpad=10)
    plt.ylim(ylim)
    plt.xlim(xlim)

    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    sns.despine(trim=True)

    for i, row in df.iterrows():
        plt.annotate(
            row['Model'],
            (row['Capability Score'], row['Alignment Score (Poems)']),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
            fontsize=11,
            fontweight='medium',
            color='#333333'
        )

    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Poems', markerfacecolor="#03b347", markersize=12),
            plt.Line2D([0], [0], marker='o', color='w', label='Text',  markerfacecolor="#114df0", markersize=12)
        ],
        title="Type",
        title_fontsize=11,
        fontsize=10,
        loc='lower right',
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc'
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    file_path = "results/alignment/alignment.txt"
    matrices = read_records(file_path)
    print(matrices["NUNFORMATED_POEMS"])