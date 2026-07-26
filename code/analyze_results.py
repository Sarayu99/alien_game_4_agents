"""
analyze_results.py
====================
Reads the CSV files produced by run_experiment.py for ONE landscape (one
K value) and produces four comparison graphs, saved as PNG files inside
the given figure_dir:

  Graph 1 (fig1_performance_comparison.png)
      Best-known payoff vs. trial number (trial 0 included), comparing
      all four agents: A (myopic local search), B (free replication),
      C (epsilon-greedy, epsilon = EPSILON_FOR_COMPARISON), and D
      (streetlight) -- all on one chart. The x-axis shows only whole
      trial-number ticks.

  Graph 2 (fig2_epsilon_comparison_performance.png)
      Best-known payoff vs. trial number (trial 0 included), comparing
      Agent C at the three epsilon values in config.EPSILON_VALUES --
      all on one chart.

  Graph 3 (fig3_search_distance_comparison.png)
      Hamming search distance vs. trial number, comparing all four
      agents. Search distance is undefined at trial 0 (there is no prior
      best-known configuration yet), so trial 0 instead shows each
      agent's Hamming distance from the TRUE global optimum of the
      landscape -- i.e. how far each agent's true starting point was
      from the best possible solution.

  Graph 4 (fig4_epsilon_comparison_search_distance.png)
      Same as Graph 3, but comparing Agent C at the three epsilon values.

These four charts mirror the two main output figures from Billinger et al.
(2014): Figure 1 (average performance over trials) and Figure 2 (average
search distance over trials), extended to include a hardcoded myopic
local-search baseline (Agent A) alongside the three Claude agents.

To keep overlapping series visible even when two agents make the exact
same choices (and their lines would otherwise sit exactly on top of one
another), every series is drawn with its own color, linestyle, and marker
shape.

Can be run standalone for a single K value:
    python analyze_results.py
(uses config.K_COMPLEXITY / config.OUTPUT_DIR / config.FIGURE_DIR)

Or called programmatically with explicit parameters (this is how
run_experiment.py and run_multi_k_experiment.py use it):
    import analyze_results
    analyze_results.main(K=5, N=10, seed=42, output_dir="outputs/K5", figure_dir="figures/K5")
"""

import ast
import os

import pandas as pd
import matplotlib.pyplot as plt

import config
from nk_landscape import NKLandscape


# Distinct linestyle/marker combinations cycled across series, so that even
# if two agents' values are identical at every trial, both remain visually
# distinguishable on the chart.
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

AGENT_A_LABEL = "Agent A (myopic local search)"
AGENT_B_LABEL = "Agent B (free replication)"
AGENT_D_LABEL = "Agent D (streetlight)"


def load_csv(filename, output_dir):
    """Load one of run_experiment.py's output CSVs as a DataFrame.
    The 'config' column is stored as text in the CSV, so we convert it
    back into a real Python tuple here."""
    path = os.path.join(output_dir, filename)
    df = pd.read_csv(path)
    df["config"] = df["config"].apply(ast.literal_eval)
    return df


def get_global_optimum_config(N, K, seed):
    """Rebuild the exact same landscape used by run_experiment.py (same
    N, K, and seed) purely to recover the TRUE global-optimum configuration.
    Used to fill in trial 0's undefined search_distance (see
    add_display_distance below)."""
    landscape = NKLandscape(N=N, K=K, seed=seed)
    optimum_config, _ = landscape.get_global_optimum()
    return optimum_config


def add_display_distance(df, global_optimum_config):
    """Add 'search_distance_display': identical to 'search_distance' for
    every trial > 0, but for trial 0 (where search_distance is undefined,
    since there is no prior best-known configuration yet) it is instead the
    Hamming distance from the agent's starting configuration to the TRUE
    global optimum of the landscape. This lets Graphs 3-4 include trial 0,
    showing each agent's true starting point."""
    df = df.copy()
    df["dist_to_global_optimum"] = [
        NKLandscape.hamming_distance(cfg, global_optimum_config) for cfg in df["config"]
    ]
    df["search_distance_display"] = df["search_distance"]
    trial0_mask = df["trial"] == 0
    df.loc[trial0_mask, "search_distance_display"] = df.loc[trial0_mask, "dist_to_global_optimum"]
    return df


def plot_lines(series_dict, ylabel, title, output_filename, figure_dir, integer_xticks=False):
    """
    Plot several trial-indexed series on one chart.
    """
    os.makedirs(figure_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    all_trials = set()
    for i, (label, (trial_numbers, values)) in enumerate(series_dict.items()):
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        marker = MARKERS[i % len(MARKERS)]
        all_trials.update(list(trial_numbers))
        plt.plot(
            trial_numbers, values, marker=marker, markersize=5, linestyle=linestyle,
            linewidth=1.6, alpha=0.85, label=label,
        )

    plt.xlabel("Trial number")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if integer_xticks:
        plt.xticks(sorted(all_trials))

    plt.tight_layout()

    path = os.path.join(figure_dir, output_filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def load_all_agents(output_dir):
    """Load all four agents' CSVs for one K run, returning
    (df_a, df_b, eps_dfs, df_c_main, df_d)."""
    df_a = load_csv("agent_a.csv", output_dir)
    df_b = load_csv("agent_b.csv", output_dir)
    eps_dfs = {eps: load_csv(f"agent_c_eps{eps}.csv", output_dir) for eps in config.EPSILON_VALUES}
    df_c_main = eps_dfs[config.EPSILON_FOR_COMPARISON]
    df_d = load_csv("agent_d.csv", output_dir)
    return df_a, df_b, eps_dfs, df_c_main, df_d


def main(K=None, N=None, seed=None, output_dir=None, figure_dir=None):
    K = config.K_COMPLEXITY if K is None else K
    N = config.N_ATTRIBUTES if N is None else N
    seed = config.RANDOM_SEED if seed is None else seed
    output_dir = config.OUTPUT_DIR if output_dir is None else output_dir
    figure_dir = config.FIGURE_DIR if figure_dir is None else figure_dir

    df_a, df_b, eps_dfs, df_c_main, df_d = load_all_agents(output_dir)

    global_optimum_config = get_global_optimum_config(N, K, seed)
    df_a = add_display_distance(df_a, global_optimum_config)
    df_b = add_display_distance(df_b, global_optimum_config)
    df_d = add_display_distance(df_d, global_optimum_config)
    eps_dfs = {eps: add_display_distance(df, global_optimum_config) for eps, df in eps_dfs.items()}
    df_c_main = eps_dfs[config.EPSILON_FOR_COMPARISON]

    agent_c_label = f"Agent C (epsilon-greedy, eps={config.EPSILON_FOR_COMPARISON})"

    # -------------------------------------------------------------------
    # Graph 1: performance comparison (all four agents), trial 0 included.
    # -------------------------------------------------------------------
    plot_lines(
        {
            AGENT_A_LABEL: (df_a["trial"], df_a["best_payoff_so_far"]),
            AGENT_B_LABEL: (df_b["trial"], df_b["best_payoff_so_far"]),
            agent_c_label: (df_c_main["trial"], df_c_main["best_payoff_so_far"]),
            AGENT_D_LABEL: (df_d["trial"], df_d["best_payoff_so_far"]),
        },
        ylabel="Best-known payoff so far",
        title=f"Graph 1: Performance Comparison Across Agents A, B, C, and D (K={K})",
        output_filename="fig1_performance_comparison.png",
        figure_dir=figure_dir,
        integer_xticks=True,
    )

    # -------------------------------------------------------------------
    # Graph 2: epsilon comparison for Agent C (performance), trial 0 included.
    # -------------------------------------------------------------------
    plot_lines(
        {
            f"Agent C (epsilon={eps})": (df["trial"], df["best_payoff_so_far"])
            for eps, df in eps_dfs.items()
        },
        ylabel="Best-known payoff so far",
        title=f"Graph 2: Agent C Performance Across Different Epsilon Values (K={K})",
        output_filename="fig2_epsilon_comparison_performance.png",
        figure_dir=figure_dir,
    )

    # -------------------------------------------------------------------
    # Graph 3: search distance comparison (all four agents). Trial 0 is
    # shown using each agent's distance from the TRUE global optimum
    # (search_distance is undefined there); trial 1+ uses the normal
    # search-distance metric.
    # -------------------------------------------------------------------
    plot_lines(
        {
            AGENT_A_LABEL: (df_a["trial"], df_a["search_distance_display"]),
            AGENT_B_LABEL: (df_b["trial"], df_b["search_distance_display"]),
            agent_c_label: (df_c_main["trial"], df_c_main["search_distance_display"]),
            AGENT_D_LABEL: (df_d["trial"], df_d["search_distance_display"]),
        },
        ylabel="Search distance (Hamming distance)",
        title=(
            f"Graph 3: Search Distance Comparison Across Agents A, B, C, and D (K={K})\n"
            "(trial 0 = distance from true global optimum)"
        ),
        output_filename="fig3_search_distance_comparison.png",
        figure_dir=figure_dir,
    )

    # -------------------------------------------------------------------
    # Graph 4: epsilon comparison for Agent C (search distance), same
    # trial-0 treatment as Graph 3.
    # -------------------------------------------------------------------
    plot_lines(
        {
            f"Agent C (epsilon={eps})": (df["trial"], df["search_distance_display"])
            for eps, df in eps_dfs.items()
        },
        ylabel="Search distance (Hamming distance)",
        title=(
            f"Graph 4: Agent C Search Distance Across Different Epsilon Values (K={K})\n"
            "(trial 0 = distance from true global optimum)"
        ),
        output_filename="fig4_epsilon_comparison_search_distance.png",
        figure_dir=figure_dir,
    )

    print(f"\nAll four graphs generated successfully for K={K}.")


if __name__ == "__main__":
    main()
