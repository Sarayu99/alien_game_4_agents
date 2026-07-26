"""
visualize_landscape_paths.py
==============================
Adds two more figures on top of analyze_results.py's four, for ONE
landscape (one K value):

  Graph 5 (fig5_landscape_paths.png)
      A 2D map of the entire NK landscape (all 2^N configurations),
      built with multidimensional scaling (MDS) so that configurations
      that are close in Hamming distance stay close on the map. Each
      point is colored by its payoff (like an elevation map). All four
      agents (A: myopic local search, B: free replication, C:
      epsilon-greedy at EPSILON_FOR_COMPARISON, D: streetlight) each get
      their own panel showing their actual trial-by-trial path drawn
      over this map -- start (black square), intermediate trials (small
      red dots), and final configuration (gold star).

  Graph 6 (fig6_attribute_flip_heatmaps.png)
      For each of the four agents, a heatmap with the 10 attributes on
      the y-axis and trial number on the x-axis, shaded black/white for
      ON/OFF.

Can be run standalone for a single K value:
    python visualize_landscape_paths.py

Or called programmatically (used by run_experiment.py /
run_multi_k_experiment.py):
    import visualize_landscape_paths
    visualize_landscape_paths.main(K=5, N=10, seed=42, output_dir="outputs/K5", figure_dir="figures/K5")
"""

import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

import config
from nk_landscape import NKLandscape, ATTRIBUTE_NAMES


# ---------------------------------------------------------------------------
# Data loading (mirrors analyze_results.py)
# ---------------------------------------------------------------------------
def load_csv(filename, output_dir):
    """Load one of run_experiment.py's output CSVs as a DataFrame, with the
    'config' column converted back from text into a real Python tuple."""
    path = os.path.join(output_dir, filename)
    df = pd.read_csv(path)
    df["config"] = df["config"].apply(ast.literal_eval)
    return df


def build_landscape(N, K, seed):
    """Rebuild the exact same landscape used by run_experiment.py (same
    N, K, and seed), so the MDS map lines up with the agents' CSVs."""
    return NKLandscape(N=N, K=K, seed=seed)


# ---------------------------------------------------------------------------
# Graph 5: MDS landscape map + agent paths
# ---------------------------------------------------------------------------
def compute_mds_embedding(landscape, output_dir):
    """
    Enumerate every possible configuration in the landscape, compute all
    pairwise Hamming distances, and project them into 2D with MDS so that
    configurations that are close in Hamming distance end up close on the
    map. Results are cached to disk (keyed by N, K, seed) since this is
    the slow step and N_TRIALS/epsilon changes don't require recomputing it.
    """
    cache_path = os.path.join(
        output_dir,
        f"mds_embedding_N{landscape.N}_K{landscape.K}_seed{landscape.seed}.npz",
    )
    if os.path.exists(cache_path):
        print(f"Loading cached MDS embedding from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        # Defensive tuple coercion: even though config_list is saved as an
        # object array of tuples (see below), np.array(list_of_equal_length_
        # tuples, dtype=object) is a classic NumPy gotcha -- NumPy infers a
        # rectangular (n_configs, N) shape and stores plain ints instead of
        # one tuple per row, EVEN with dtype=object explicitly requested.
        # After a save/load round trip through .npz, .tolist() then returns
        # a list of LISTS instead of tuples, which breaks config_to_index
        # (dict keys must be hashable). Wrapping each element in tuple(...)
        # here recovers the correct tuples regardless of which shape ended
        # up on disk (old caches included).
        config_list = [tuple(c) for c in cached["config_list"].tolist()]
        return cached["coords"], cached["fitness"], config_list

    all_configs = landscape.enumerate_all_configs()  # sorted low -> high payoff
    config_list = [c for c, _ in all_configs]
    fitness = np.array([p for _, p in all_configs])
    configs_array = np.array(config_list, dtype=int)

    print(f"Computing pairwise Hamming distances for {len(config_list)} configurations...")
    dist_matrix = squareform(pdist(configs_array, metric="hamming")) * landscape.N

    print("Running MDS (roughly 30-90 seconds for 1,024 points)...")
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=landscape.seed,
        n_init=1,
        max_iter=300,
        normalized_stress=False,
    )
    coords = mds.fit_transform(dist_matrix)

    os.makedirs(output_dir, exist_ok=True)
    # Build a genuine 1D object array (one tuple per element) instead of
    # np.array(config_list, dtype=object), which NumPy silently turns into
    # a 2D int array here since every config is the same length -- see the
    # loading branch above for why that matters.
    config_array = np.empty(len(config_list), dtype=object)
    for i, c in enumerate(config_list):
        config_array[i] = tuple(c)
    np.savez(cache_path, coords=coords, fitness=fitness, config_list=config_array)
    print(f"Cached MDS embedding to {cache_path}")

    return coords, fitness, config_list


def plot_landscape_paths(coords, fitness, config_list, agent_dfs, output_filename, figure_dir):
    """One panel per agent: the full MDS landscape map (colored by payoff)
    with that agent's actual visited-configuration path drawn on top."""
    config_to_index = {c: i for i, c in enumerate(config_list)}

    os.makedirs(figure_dir, exist_ok=True)
    n_agents = len(agent_dfs)
    fig, axes = plt.subplots(1, n_agents, figsize=(6 * n_agents, 5.5), sharex=True, sharey=True)
    if n_agents == 1:
        axes = [axes]

    scatter = None
    for ax, (label, df) in zip(axes, agent_dfs.items()):
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], c=fitness, cmap="viridis",
            s=8, alpha=0.35, linewidths=0,
        )

        path_idx = [config_to_index[tuple(c)] for c in df["config"]]
        path_coords = coords[path_idx]

        ax.plot(path_coords[:, 0], path_coords[:, 1], color="crimson",
                linewidth=1.2, alpha=0.85, zorder=3)
        if len(path_coords) > 2:
            ax.scatter(path_coords[1:-1, 0], path_coords[1:-1, 1], color="crimson",
                       s=25, zorder=4, edgecolors="white", linewidths=0.5)
        ax.scatter(*path_coords[0], color="black", marker="s", s=70, zorder=5,
                   label="Start", edgecolors="white")
        ax.scatter(*path_coords[-1], color="gold", marker="*", s=180, zorder=5,
                   label="Final", edgecolors="black", linewidths=0.5)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("MDS dimension 1")
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("MDS dimension 2")
    fig.colorbar(scatter, ax=axes, label="Payoff (fitness)", shrink=0.8)
    fig.suptitle("Graph 5: Agent Search Paths Over the NK Landscape (MDS Projection)")

    path = os.path.join(figure_dir, output_filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Graph 6: attribute flip heatmaps
# ---------------------------------------------------------------------------
def plot_attribute_flip_heatmaps(agent_dfs, output_filename, figure_dir):
    """For each agent, a black/white heatmap of ON/OFF state per attribute
    (rows) across trials (columns)."""
    os.makedirs(figure_dir, exist_ok=True)
    n_agents = len(agent_dfs)
    fig, axes = plt.subplots(n_agents, 1, figsize=(10, 2.3 * n_agents), sharex=False)
    if n_agents == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, agent_dfs.items()):
        matrix = np.array([list(c) for c in df["config"]]).T  # (N_attributes, n_trials_shown)
        ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax.set_yticks(range(len(ATTRIBUTE_NAMES)))
        ax.set_yticklabels(ATTRIBUTE_NAMES, fontsize=8)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["trial"].tolist(), fontsize=7)
        ax.set_title(label, fontsize=10, loc="left")

    axes[-1].set_xlabel("Trial number")
    fig.suptitle("Graph 6: Attribute ON/OFF State Across Trials (black = ON) per Agent")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(figure_dir, output_filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(K=None, N=None, seed=None, output_dir=None, figure_dir=None):
    K = config.K_COMPLEXITY if K is None else K
    N = config.N_ATTRIBUTES if N is None else N
    seed = config.RANDOM_SEED if seed is None else seed
    output_dir = config.OUTPUT_DIR if output_dir is None else output_dir
    figure_dir = config.FIGURE_DIR if figure_dir is None else figure_dir

    landscape = build_landscape(N, K, seed)
    coords, fitness, config_list = compute_mds_embedding(landscape, output_dir)

    df_a = load_csv("agent_a.csv", output_dir)
    df_b = load_csv("agent_b.csv", output_dir)
    df_c = load_csv(f"agent_c_eps{config.EPSILON_FOR_COMPARISON}.csv", output_dir)
    df_d = load_csv("agent_d.csv", output_dir)

    agent_dfs = {
        "Agent A (myopic local search)": df_a,
        "Agent B (free replication)": df_b,
        f"Agent C (epsilon={config.EPSILON_FOR_COMPARISON})": df_c,
        "Agent D (streetlight)": df_d,
    }

    plot_landscape_paths(coords, fitness, config_list, agent_dfs, "fig5_landscape_paths.png", figure_dir)
    plot_attribute_flip_heatmaps(agent_dfs, "fig6_attribute_flip_heatmaps.png", figure_dir)

    print(f"\nGraphs 5 and 6 generated successfully for K={K}.")


if __name__ == "__main__":
    main()
