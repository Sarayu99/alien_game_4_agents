"""
visualize_landscape_paths.py
==============================
For ONE landscape (one K value):

  OPTIONAL: Graph 6 (fig6_attribute_flip_heatmaps.png), a per-K heatmap
     of each agent's attribute ON/OFF state across trials for one episode.
     This is the only per-K figure that is NOT repeated in common_figures,
     so it is OFF by default. Set SAVE_ATTRIBUTE_HEATMAPS = True below to
     get it (saved in figures_N.._E../K{K}/).

Can be run standalone for a single K value:
    python visualize_landscape_paths.py
"""

import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from nk_landscape import ATTRIBUTE_NAMES


SAVE_ATTRIBUTE_HEATMAPS = False   # True -> also write the per-K fig6 heatmaps

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
    fig.suptitle("Graph 6: Attribute ON/OFF State Across Trials (black = ON) per Agent; one episode")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(figure_dir, output_filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(K=None, N=None, seed=None, output_dir=None, figure_dir=None, episode=None):
    if not SAVE_ATTRIBUTE_HEATMAPS:
        print("Per-K figures are off (SAVE_ATTRIBUTE_HEATMAPS=False); nothing to do.")
        return

    K = config.K_COMPLEXITY if K is None else K
    output_dir = config.output_dir_for_k(K) if output_dir is None else output_dir
    figure_dir = config.figure_dir_for_k(K) if figure_dir is None else figure_dir
    episode = config.PATH_EPISODE if episode is None else episode
    if episode is None:   # default: last episode actually run
        episode = int(load_csv("agent_a.csv", output_dir)["episode"].max())

    def one_episode(filename):
        df = load_csv(filename, output_dir)
        return df[df["episode"] == episode].sort_values("trial").reset_index(drop=True)

    agent_dfs = {
        "Agent A (myopic local search)": one_episode("agent_a.csv"),
        "Agent B (free replication)": one_episode("agent_b.csv"),
        f"Agent C (epsilon={config.EPSILON_FOR_COMPARISON})": one_episode(
            f"agent_c_eps{config.EPSILON_FOR_COMPARISON}.csv"),
        f"Agent D (Q-learning, eps={config.RL_EPSILON})": one_episode("agent_d.csv"),
    }
    plot_attribute_flip_heatmaps(agent_dfs, "fig6_attribute_flip_heatmaps.png", figure_dir)
    print(f"\nGraph 6 generated for K={K} (episode {episode}).")


if __name__ == "__main__":
    main()
