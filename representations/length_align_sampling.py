from datasets import load_dataset, Dataset
from utils import load_multiline_jsonl
import json
import numpy as np

def sample_matching_distribution_with_replacement(
    lengths_A,
    lengths_B,
    indices_B,
    n_bins=20,
    bin_edges=None,
    total_samples=None,
    random_state=None
):
    """
    Args:
        lengths_A: 1D array of lengths (target distribution)
        lengths_B: 1D array of lengths (source pool)
        indices_B: 1D array of same length as lengths_B (original indices)
        n_bins: number of quantile bins (ignored if bin_edges is provided)
        bin_edges: optional explicit bin boundaries
        total_samples: number of samples to draw; if None, use len(A)
        random_state: optional RNG seed

    Returns:
        selected_indices_B: indices from indices_B (original indices)
        selected_lengths: values sampled from lengths_B
        diagnostics: dict
    """
    rs = np.random.RandomState(random_state)

    A = np.asarray(lengths_A)
    B = np.asarray(lengths_B)
    idxB = np.asarray(indices_B)
    assert len(B) == len(idxB), "indices_B length must match lengths_B length."

    if total_samples is None:
        total_samples = len(A)

    if bin_edges is None:
        qs = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(A, qs))
    if len(bin_edges) < 2:
        raise ValueError("Bin edges must contain at least two unique values.")

    inds_A = np.digitize(A, bin_edges, right=False) - 1
    inds_B = np.digitize(B, bin_edges, right=False) - 1
    nbins = len(bin_edges) - 1

    p = np.zeros(nbins, float)
    for i in range(nbins):
        p[i] = np.mean(inds_A == i)
    if np.sum(p) == 0:
        raise ValueError("Target distribution empty")
    p /= p.sum()  # normalize

    bin_indices_B = []
    for i in range(nbins):
        bin_indices_B.append(np.where(inds_B == i)[0])

    selected_indices = []
    for _ in range(total_samples):
        # first pick a bin according to p
        chosen_bin = rs.choice(nbins, p=p)
        if len(bin_indices_B[chosen_bin]) == 0:
            # fallback: pick another bin randomly according to p
            nonempty_bins = [i for i in range(nbins) if len(bin_indices_B[i]) > 0]
            chosen_bin = rs.choice(nonempty_bins)
        # pick one index from B in that bin, with replacement
        chosen_idx_in_B = rs.choice(bin_indices_B[chosen_bin])
        selected_indices.append(idxB[chosen_idx_in_B])

    selected_indices = np.array(selected_indices)
    selected_lengths = B[np.isin(idxB, selected_indices)]

    diagnostics = {
        "bin_edges": bin_edges,
        "p": p,
        "total_samples": total_samples,
    }

    return selected_indices, selected_lengths, diagnostics

def sample_matching_distribution(
    lengths_A,
    lengths_B,
    indices_B,
    n_bins=20,
    bin_edges=None,
    random_state=None
):
    """

    Args:
        lengths_A: 1D array of lengths (target distribution)
        lengths_B: 1D array of lengths (source pool)
        indices_B: 1D array of same length as lengths_B (original indices)
        n_bins: number of quantile bins (ignored if bin_edges is provided)
        bin_edges: optional explicit bin boundaries
        random_state: optional RNG seed

    Returns:
        selected_indices_B: indices from indices_B (original indices)
        selected_lengths: values sampled from lengths_B
        diagnostics: dict
    """
    rs = np.random.RandomState(random_state)

    A = np.asarray(lengths_A)
    B = np.asarray(lengths_B)
    idxB = np.asarray(indices_B)
    assert len(B) == len(idxB), "indices_B length must match lengths_B length."

    # Build bins
    if bin_edges is None:
        qs = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(A, qs))
    if len(bin_edges) < 2:
        raise ValueError("Bin edges must contain at least two unique values.")

    inds_A = np.digitize(A, bin_edges, right=False) - 1
    inds_B = np.digitize(B, bin_edges, right=False) - 1
    nbins = len(bin_edges) - 1

    p = np.zeros(nbins, float)
    c = np.zeros(nbins, int)
    for i in range(nbins):
        p[i] = np.mean(inds_A == i)
        c[i] = np.sum(inds_B == i)

    positive = p > 0
    if np.any((p > 0) & (c == 0)):
        missing = np.where((p > 0) & (c == 0))[0].tolist()
        raise ValueError(
            f"Bins {missing} needed by target distribution but B has zero samples. "
            "Reduce n_bins or merge bins."
        )

    ratios = c[positive] / p[positive]
    S_max = int(np.floor(ratios.min()))
    if S_max <= 0:
        raise ValueError("Cannot sample any points while preserving A's proportions.")

    s = np.floor(S_max * p).astype(int)
    s = np.minimum(s, c)

    leftover = S_max - s.sum()
    if leftover > 0:
        fractional = (S_max * p) - np.floor(S_max * p)
        avail = (c - s) > 0
        order = np.argsort(-fractional)
        for i in order:
            if leftover == 0:
                break
            if p[i] > 0 and avail[i]:
                s[i] += 1
                leftover -= 1

    s = np.minimum(s, c)  # safety
    S_final = s.sum()

    selected_indices = []
    for i in range(nbins):
        if s[i] > 0:
            bin_mask = np.where(inds_B == i)[0]
            chosen_pos = rs.choice(bin_mask, size=s[i], replace=False)
            selected_indices.append(idxB[chosen_pos])

    if len(selected_indices) == 0:
        selected_indices = np.array([], dtype=int)
    else:
        selected_indices = np.concatenate(selected_indices)

    selected_lengths = B[np.isin(idxB, selected_indices)]

    diagnostics = {
        "bin_edges": bin_edges,
        "p": p,
        "c": c,
        "s": s,
        "S_max": S_max,
        "S_final": S_final,
    }

    return selected_indices, selected_lengths, diagnostics


if __name__ == "__main__":
    poems_path = '../poems/rated_poems_results_sampled_full.jsonl'
    poems = load_multiline_jsonl(poems_path)

    poems_lens = [len(poem['content'].split()) for poem in poems]
    poems_indices = np.arange(len(poems))

    # filter out poems with 0 score (error in rating)
    indices_without_0_score = [i for i, poem in enumerate(poems) if poem["evaluation"]["score"] > 0]
    non_zero_poems = [poems[i] for i in indices_without_0_score]
    non_zero_poems_lens = [len(poem['content'].split()) for poem in non_zero_poems]

    dataset =  load_dataset('minhuh/prh', revision="wit_1024")

    lens = [] # standard distribution
    for i in range(len(dataset['train'])):
        lens.append(len(dataset['train'][i]['text'][0].split()))


    selected_indices, selected_lengths, diagnostics = sample_matching_distribution(
        lengths_A = lens,
        lengths_B = poems_lens,
        indices_B = poems_indices,
        n_bins = 20,
        random_state = 42,
    )
    
    new_lens = [len(poems[idx]['content'].split()) for idx in selected_indices]
    print(f"Selected lengths stats: mean={np.mean(new_lens):.2f}, median={np.median(new_lens):.2f}, std={np.std(new_lens):.2f}")
    print(f"Standard: mean={np.mean(lens):.2f}, median={np.median(lens):.2f}, std={np.std(lens):.2f}")


    