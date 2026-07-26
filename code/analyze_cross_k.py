"""
analyze_cross_k.py
====================
Reads the per-K output folders produced by run_multi_k_experiment.py
(outputs/K0, outputs/K5, outputs/K9 by default) and produces TWO
additional graphs comparing all four agents across landscape complexity
levels:

  Graph 7 (figures/fig7_cross_k_comparison.png)
      Grouped bar chart: x-axis = K value (0, 5, 9), one bar per agent
      (A: myopic local search, B: free replication, C: epsilon-greedy at
      EPSILON_FOR_COMPARISON, D: streetlight) within each K group, y-axis
      = final best-known payoff (the value at the last completed trial).
      This mirrors the structure of Billinger et al. (2014)'s Table 1
      (human vs. computational-agent performance by complexity), applied
      to your four agents instead.

  Graph 8 (figures/fig8_search_distance_by_agent.png)
      A 2x2 panel of subplots, one per agent (A, B, C, D). Each subplot
      plots search distance vs. trial number, with one line per K value
      (0, 5, 9) so you can see how search-distance behavior shifts with
      landscape ruggedness, agent by agent. This follows the visual
      style of Billinger et al. (2014)'s Figure 4 (line + markers, trial
      on the x-axis, search distance on the y-axis) -- but where Figure 4
      compares actual vs. simulated agents at one fixed K, Graph 8
      compares landscape complexity levels at one fixed agent, repeated
      across all four agents. Trial 0 uses each agent's Hamming distance
      from the TRUE global optimum (search_distance is undefined at
      trial 0), same convention as Graphs 3-4 in analyze_results.py.

Run this AFTER run_multi_k_experiment.py has completed all three K runs
(it is called automatically at the end of that script). Can also be run
standalone if the per-K output folders already exist:

    python analyze_cross_k.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

import config
from analyze_results import load_csv, add_display_distance, get_global_optimum_config

AGENT_LABELS = {
    "A_myopic_local_search": "Agent A (myopic local search)",
    "B_free_replication": "Agent B (free replication)",
    "C_epsilon_greedy": f"Agent C (epsilon-greedy, eps={config.EPSILON_FOR_COMPARISON})",
    "D_streetlight": "Agent D (streetlight)",
}
# Fixed order/colors so the same agent always gets the same bar color
# across every K group.
AGENT_ORDER = ["A_myopic_local_search", "B_free_replication", "C_epsilon_greedy", "D_streetlight"]
BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _final_payoff(csv_path):
    """Return the best_payoff_so_far value at the LAST trial in a CSV."""
    df = pd.read_csv(csv_path)
    return df.sort_values("trial").iloc[-1]["best_payoff_so_far"]


def collect_final_payoffs(k_values=None):
    """
    Returns a dict: {agent_key: {K: final_payoff, ...}, ...}
    for all four agents across all given K values, reading from each K's
    output folder (config.output_dir_for_k(K)).
    """
    k_values = config.K_VALUES if k_values is None else k_values

    results = {agent_key: {} for agent_key in AGENT_ORDER}
    for K in k_values:
        output_dir = config.output_dir_for_k(K)

        agent_files = {
            "A_myopic_local_search": "agent_a.csv",
            "B_free_replication": "agent_b.csv",
            "C_epsilon_greedy": f"agent_c_eps{config.EPSILON_FOR_COMPARISON}.csv",
            "D_streetlight": "agent_d.csv",
        }

        for agent_key, filename in agent_files.items():
            path = os.path.join(output_dir, filename)
            if not os.path.exists(path):
                print(f"Warning: missing {path} -- skipping Agent {agent_key} at K={K}. "
                      "Did run_multi_k_experiment.py finish for this K value?")
                continue
            results[agent_key][K] = _final_payoff(path)

    return results


def plot_cross_k_comparison(results, k_values=None, figure_dir=None, output_filename="fig7_cross_k_comparison.png"):
    k_values = config.K_VALUES if k_values is None else k_values
    figure_dir = config.FIGURE_DIR if figure_dir is None else figure_dir
    os.makedirs(figure_dir, exist_ok=True)

    n_agents = len(AGENT_ORDER)
    bar_width = 0.8 / n_agents
    x_positions = range(len(k_values))

    plt.figure(figsize=(9, 5.5))
    for i, agent_key in enumerate(AGENT_ORDER):
        offsets = [x + (i - (n_agents - 1) / 2) * bar_width for x in x_positions]
        values = [results[agent_key].get(K, float("nan")) for K in k_values]
        plt.bar(
            offsets, values, width=bar_width, label=AGENT_LABELS[agent_key],
            color=BAR_COLORS[i % len(BAR_COLORS)], edgecolor="white", linewidth=0.5,
        )

    plt.xticks(list(x_positions), [f"K={K}" for K in k_values])
    plt.xlabel("Landscape complexity")
    plt.ylabel("Final best-known payoff")
    plt.title("Graph 7: Cross-Complexity Comparison of Agents A, B, C, and D")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(figure_dir, output_filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# Fixed color/marker per K value so the same K always looks the same in
# every one of the four subplots.
K_COLORS = {0: "#4C72B0", 5: "#DD8452", 9: "#55A868"}
K_MARKERS = {0: "o", 5: "s", 9: "^"}

AGENT_FILES = {
    "A_myopic_local_search": "agent_a.csv",
    "B_free_replication": "agent_b.csv",
    "C_epsilon_greedy": f"agent_c_eps{config.EPSILON_FOR_COMPARISON}.csv",
    "D_streetlight": "agent_d.csv",
}


def collect_search_distance_by_agent(k_values=None, N=None, seed=None):
    """
    Returns a nested dict: {agent_key: {K: (trial_numbers, search_distances)}}
    for all four agents across all given K values. Reuses load_csv() and
    add_display_distance() from analyze_results.py so trial 0's undefined
    search_distance is filled in with distance-to-global-optimum exactly the
    same way Graphs 3-4 do it, for a single K.
    """
    k_values = config.K_VALUES if k_values is None else k_values
    N = config.N_ATTRIBUTES if N is None else N
    seed = config.RANDOM_SEED if seed is None else seed

    data = {agent_key: {} for agent_key in AGENT_ORDER}
    for K in k_values:
        output_dir = config.output_dir_for_k(K)
        global_optimum_config = get_global_optimum_config(N, K, seed)

        for agent_key, filename in AGENT_FILES.items():
            path = os.path.join(output_dir, filename)
            if not os.path.exists(path):
                print(f"Warning: missing {path} -- skipping Agent {agent_key} at K={K} "
                      "in the Graph 8 search-distance panel.")
                continue
            df = load_csv(filename, output_dir)
            df = add_display_distance(df, global_optimum_config)
            data[agent_key][K] = (df["trial"].tolist(), df["search_distance_display"].tolist())

    return data


def plot_search_distance_by_agent(
    data, k_values=None, figure_dir=None, output_filename="fig8_search_distance_by_agent.png",
):
    """
    Graph 8: a 2x2 panel of subplots (one per agent), each showing search
    distance vs. trial with one line per K value. Mirrors the visual style
    of Billinger et al. (2014) Figure 4 (line + markers over trial number).
    """
    k_values = config.K_VALUES if k_values is None else k_values
    figure_dir = config.FIGURE_DIR if figure_dir is None else figure_dir
    os.makedirs(figure_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for ax, agent_key in zip(axes_flat, AGENT_ORDER):
        agent_data = data.get(agent_key, {})
        all_trials = set()

        for K in k_values:
            if K not in agent_data:
                continue
            trials, values = agent_data[K]
            all_trials.update(trials)
            ax.plot(
                trials, values,
                color=K_COLORS.get(K, "gray"),
                marker=K_MARKERS.get(K, "o"),
                markersize=5, linewidth=1.6, alpha=0.85,
                label=f"K={K}",
            )

        ax.set_title(AGENT_LABELS[agent_key], fontsize=10)
        ax.set_xlabel("Trial number")
        ax.set_ylabel("Search distance (Hamming)")
        ax.grid(True, alpha=0.3)
        if all_trials:
            ax.set_xticks(sorted(all_trials))
        ax.legend(fontsize=8)

    fig.suptitle(
        "Graph 8: Search Distance vs. Trial by Agent, Across Landscape Complexity\n"
        "(trial 0 = distance from true global optimum; style follows "
        "Billinger et al. 2014, Figure 4)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(figure_dir, output_filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main(k_values=None, figure_dir=None):
    k_values = config.K_VALUES if k_values is None else k_values
    figure_dir = config.FIGURE_DIR if figure_dir is None else figure_dir

    results = collect_final_payoffs(k_values)
    plot_cross_k_comparison(results, k_values=k_values, figure_dir=figure_dir)
    print("Graph 7 (cross-K comparison) generated successfully.")

    search_distance_data = collect_search_distance_by_agent(k_values)
    plot_search_distance_by_agent(search_distance_data, k_values=k_values, figure_dir=figure_dir)
    print("Graph 8 (search distance by agent across K) generated successfully.")


if __name__ == "__main__":
    main()
