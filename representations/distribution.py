from representations.main import FeaturesLoader, get_filter_pair_function
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.manifold import TSNE

def gaussian_fit(X):
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    return mean, cov


def draw_gaussian_ellipse(mean2d, cov2d, ax, color, nstd=2, alpha=0.4):
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nstd * np.sqrt(vals)

    ell = Ellipse(xy=mean2d, width=width, height=height,
                  angle=theta, edgecolor=color, facecolor='none', lw=2)
    ax.add_patch(ell)

    for t in np.linspace(0.1, 1, 10):
        ell2 = Ellipse(xy=mean2d,
                       width=width * t, height=height * t,
                       angle=theta,
                       edgecolor='none',
                       facecolor=color,
                       alpha=alpha * (1 - t))
        ax.add_patch(ell2)


def plot_groups_with_pca(
    feature_groups,
    colors=None,
    show_scatter=True,
    show_ellipse=True,
    n_components=2,
    figsize=(7,7),
    nstd=2,
    save_path=None
):
    """
    feature_groups: list of np.array, each array has shape (N_i, D)
    colors: list of matplotlib colors, same length as feature_groups
    """

    # --- Assign default colors ---
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_groups)))

    # --- Build PCA on concatenated features ---
    all_features = np.vstack(feature_groups)
    pca = PCA(n_components=n_components)
    pca.fit(all_features)
    W = pca.components_.T  # (D,2)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    for X, color in zip(feature_groups, colors):
        # Project points
        X2 = X @ W

        # Scatter
        if show_scatter:
            ax.scatter(X2[:,0], X2[:,1], s=8, color=color, alpha=0.4)

        # Gaussian fit + ellipse
        if show_ellipse:
            mean, cov = gaussian_fit(X)
            mean2 = W.T @ mean
            cov2  = W.T @ cov @ W
            draw_gaussian_ellipse(mean2, cov2, ax, color=color, nstd=nstd)

            # plot mean
            ax.scatter(mean2[0], mean2[1], s=80, color=color, marker='x')

    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.axis("equal")
    plt.savefig(save_path, dpi=300)
    plt.show()

    return W 


def tsne_feature_groups(
    feature_groups,
    labels=None,
    colors=None,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    figsize=(7,7),
    alpha=0.45,
    marker='o'
):
    """
    feature_groups: list of arrays, each (N_i, D)
    labels: list of str, optional
    colors: list of matplotlib colors, optional
    """

    # --- Default labels ---
    n_groups = len(feature_groups)
    if labels is None:
        labels = [f"Group {i}" for i in range(n_groups)]

    # --- Default colors ---
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))

    # --- Combine all features for t-SNE ---
    all_feats = np.vstack(feature_groups)
    group_sizes = [g.shape[0] for g in feature_groups]

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        verbose=1
    )
    all_2d = tsne.fit_transform(all_feats)

    # --- Split the projected 2D back into groups ---
    projections = []
    idx = 0
    for size in group_sizes:
        projections.append(all_2d[idx:idx+size])
        idx += size

    # --- Plot ---
    plt.figure(figsize=figsize)
    for pts, col, lab in zip(projections, colors, labels):
        plt.scatter(pts[:,0], pts[:,1], s=10, alpha=alpha,
                    color=col, label=lab, marker=marker)

    plt.legend()
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.show()

    return projections

def draw_range_ellipse(points2d, ax, color, nstd=2, alpha=0.12, lw=2):
    """
    points2d: (N,2)
    color: matplotlib color
    
    This fits a 2D Gaussian inside t-SNE space and draws an ellipse.
    """
    mean = points2d.mean(axis=0)
    cov = np.cov(points2d, rowvar=False)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width, height = 2 * nstd * np.sqrt(vals)

    # fill ellipse (light alpha)
    ell = Ellipse(xy=mean, width=width, height=height,
                  angle=theta, facecolor=color, edgecolor='none',
                  alpha=alpha)
    ax.add_patch(ell)

    # boundary line
    ell2 = Ellipse(xy=mean, width=width, height=height,
                   angle=theta, facecolor='none', edgecolor=color,
                   linewidth=lw)
    ax.add_patch(ell2)


# t-SNE + ellipse 
def tsne_feature_groups_with_ellipses(
    feature_groups,
    labels=None,
    colors=None,
    show_scatter=True,
    show_ellipse=True,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    figsize=(7,7),
    scatter_alpha=0.45,
    marker='o',
    ellipse_nstd=2,
    save_path=None
):
    n_groups = len(feature_groups)

    if labels is None:
        labels = [f"Group {i}" for i in range(n_groups)]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))

    # Concatenate
    all_feats = np.vstack(feature_groups)
    group_sizes = [g.shape[0] for g in feature_groups]

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        verbose=1
    )
    all_2d = tsne.fit_transform(all_feats)

    # Split back
    projections = []
    idx = 0
    for size in group_sizes:
        projections.append(all_2d[idx:idx+size])
        idx += size

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    for proj, col, lab in zip(projections, colors, labels):

        if show_scatter:
            ax.scatter(proj[:,0], proj[:,1], s=10, color=col,
                       alpha=scatter_alpha, marker=marker, label=lab)

        if show_ellipse:
            draw_range_ellipse(proj, ax=ax, color=col, nstd=ellipse_nstd)

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend()
    ax.axis('equal')
    plt.savefig(save_path, dpi=300)
    plt.show()

    return projections

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def sample_matching_small_from_large(
    X_large,           # (N_large, D)
    X_small,           # (N_small, D)
    n_components=50,   # PCA dims for density-estimation space
    sample_size=None,  # default = len(X_small)
    clf_penalty='l2',
    clf_C=1.0,
    random_state=0
):
    rng = np.random.default_rng(random_state)
    if sample_size is None:
        sample_size = X_small.shape[0]

    # PCA（fit）
    X_all = np.vstack([X_large, X_small])
    scaler = StandardScaler()
    X_all_std = scaler.fit_transform(X_all)

    pca = PCA(n_components=min(n_components, X_all_std.shape[1]))
    X_all_low = pca.fit_transform(X_all_std)

    N_large = X_large.shape[0]
    X_large_low = X_all_low[:N_large]
    X_small_low = X_all_low[N_large:]

    X_train = np.vstack([X_large_low, X_small_low])
    y_train = np.hstack([np.zeros(X_large_low.shape[0]), np.ones(X_small_low.shape[0])])

    clf = LogisticRegression(
        penalty=clf_penalty, C=clf_C, solver='lbfgs', max_iter=2000, random_state=random_state
    )
    clf.fit(X_train, y_train)

    s_large = clf.predict_proba(X_large_low)[:, 1]  
    s_large = np.clip(s_large, 1e-12, 1 - 1e-12)

    weights = s_large / s_large.sum()


    replace = False
    if sample_size > N_large * 0.9:
        replace = True

    chosen_idx = rng.choice(np.arange(N_large), size=sample_size, replace=replace, p=weights)

    return chosen_idx, weights, clf, scaler, pca

def main(save_path=None):
    filters = [lambda dp: dp['score'] == i for i in [1,2,3,4,5]]

    loader = FeaturesLoader(f"../platonic/features/poems/NousResearch_Meta-Llama-3-8B_pool-avg.pt")
    loader_2 = FeaturesLoader(f"../platonic/features/text/wit_1024/NousResearch_Meta-Llama-3-8B_pool-avg.pt")
    filter_1, filter_2 = get_filter_pair_function("length")

    features_1 = loader.load_features(layer=-1, dataset=True, numpy=True) 
    #features_2 = loader.load_features(layer=-1, dataset=True, numpy=True, filter_function=filter_1)
    #features_3 = loader.load_features(layer=-1, dataset=True, numpy=True, filter_function=filter_2)
    features_4 = loader_2.load_features(layer=-1, dataset=False, numpy=True)

    parts_num = 2
    colors = plt.cm.viridis(np.linspace(0,1,parts_num))
    tsne_feature_groups_with_ellipses(
        feature_groups=[features_1, features_4],
        labels=['Poems', 'Texts'],
        colors=colors,
        show_scatter=True,
        show_ellipse=True,
        perplexity=35,
        max_iter=1200,
        ellipse_nstd=2,
        save_path=save_path
    )

if __name__ == "__main__":  
    main(save_path="tsne_plot.png")