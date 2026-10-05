"""
analyze_cross_k.py
====================
Cross-landscape comparison graphs (K = 0, 5, 9), built from the per-K
folders written by run_multi_k_experiment.py. All curves are averaged
across episodes.

  Graph 1 (fig1_cumulative_payoff_across_landscapes.png)
      X: trial number
      Y: average cumulative payoff up to that trial, across individuals
         (agents A-D) AND episodes
      Lines: one per K (0, 5, 9)
      Higher = more accumulated earnings; steeper = faster accumulation.

  Graph 2, across landscapes (fig2_cumulative_payoff_across_landscapes.png)
      Three side-by-side panels (K=0, 5, 9, shared y-axis), each one the
      per-K Graph 2: x = trial number, y = cumulative payoff as a % of the
      maximum possible (global optimum played every trial), averaged across
      episodes, one line per agent.

  Graph 2b, across landscapes (fig2b_cumulative_payoff_raw_across_landscapes.png)
      Same three panels as Graph 2 but in actual cumulative payoff units.

  Graph 3, across landscapes (fig3_search_distance_across_landscapes.png)
      Same three-panel layout for average search distance vs. trial
      (trial 0 = distance from true global optimum).

  Graph 4 (fig4_final_cumulative_payoff_bars.png)
      Grouped bars: final cumulative payoff (after the last trial), mean
      across episodes with +/-1 SE error bars, per agent within each K.

  Graph 7, across landscapes (fig7_learning_curve_across_landscapes.png)
      Three panels (K=0, 5, 9): episode number vs the episode's total payoff
      (% of maximum possible), one line per agent plus the grey dashed
      no-memory control copy of Agent D.

  Graph 8 (fig8_per_agent_cumulative_payoff_across_landscapes.png)
      2x2 panels, ONE PANEL PER AGENT, lines = K (0, 5, 9): trial number vs
      cumulative payoff (% of maximum possible, mean across episodes).
      Shows how each agent's accumulation changes with ruggedness.

  Graph 8b (fig8b_per_agent_cumulative_payoff_raw_across_landscapes.png)
      Same as Graph 8 in actual cumulative payoff units.

  Graph 9 (fig9_per_agent_search_distance_across_landscapes.png)
      Same 2x2 layout for average search distance vs trial
      (trial 0 = distance from true global optimum).

All of these are saved in figures_N.._E../common_figures/ (they compare all
K values, so they belong to no single K folder). Every panel shows its own
y tick labels. The learning-curve x axis is exactly episodes 1..N_EPISODES.

Standalone (after all K runs finished):  python analyze_cross_k.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config
from analyze_results import (
    AGENT_KEYS, AGENT_LABELS, AGENT_STYLE, load_all_agents, draw_agent_lines,
    mean_by_trial, load_control, draw_learning_curves,
)

K_COLORS = {0: "#4C72B0", 5: "#DD8452", 9: "#55A868"}
K_MARKERS = {0: "o", 5: "s", 9: "^"}


def load_all_k(k_values, N, seed):
    return {K: load_all_agents(config.output_dir_for_k(K), N, K, seed) for K in k_values}


def plot_graph1(data, k_values, figure_dir):
    """Cumulative payoff vs trial, pooled over agents and episodes, per K."""
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for K in k_values:
        pooled = pd.concat(data[K].values(), ignore_index=True)
        trials, vals = mean_by_trial(pooled, "cumulative_payoff")
        ax.plot(trials, vals, color=K_COLORS.get(K), marker=K_MARKERS.get(K, "o"),
                markersize=4.5, linewidth=1.8, label=f"K={K}")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Average cumulative payoff\n(across agents A-D and episodes)")
    ax.set_title("Graph 1: Average Cumulative Payoff Across All Agents (A-D), by Landscape")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(figure_dir, "fig1_cumulative_payoff_across_landscapes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _three_panel(data, k_values, column, x_mode, title, xlabel, ylabel, filename, figure_dir, min_trial=0):
    fig, axes = plt.subplots(1, len(k_values), figsize=(5.6 * len(k_values), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, K in zip(axes, k_values):
        draw_agent_lines(ax, data[K], column, x_mode, min_trial=min_trial)
        ax.set_title(f"K={K}")
        ax.set_xlabel(xlabel)
        ax.tick_params(labelleft=True)   # show y tick labels on every panel
        if x_mode == "pct":
            ax.set_xlim(0, 100)
        ax.legend(fontsize=7.5)          # legend on every panel
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(figure_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_learning_three_panel(data, controls, k_values, figure_dir):
    fig, axes = plt.subplots(1, len(k_values), figsize=(5.6 * len(k_values), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, K in zip(axes, k_values):
        draw_learning_curves(ax, data[K], controls.get(K))
        ax.set_title(f"K={K}")
        ax.set_xlabel("Episode number")
        ax.tick_params(labelleft=True)   # show y tick labels on every panel
        ax.legend(fontsize=7.5)          # legend on every panel
    axes[0].set_ylabel("Episode's total payoff (% of maximum possible)")
    fig.suptitle("Graph 7 across landscapes: Learning Across Episodes")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(figure_dir, "fig7_learning_curve_across_landscapes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_agent_across_k(data, k_values, column, x_mode, min_trial, title, ylabel,
                            filename, figure_dir):
    """2x2 panels, one per agent, one line per K."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, key in zip(axes.flatten(), AGENT_KEYS):
        for K in k_values:
            trials, vals = mean_by_trial(data[K][key], column)
            keep = trials >= min_trial
            ax.plot(trials[keep], vals[keep], color=K_COLORS.get(K), marker=K_MARKERS.get(K, "o"),
                    markersize=4.5, linewidth=1.7, alpha=0.9, label=f"K={K}")
        ax.set_title(AGENT_LABELS[key], fontsize=10)
        ax.set_xlabel("Trial number")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(figure_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_graph4(data, k_values, figure_dir):
    n_agents = len(AGENT_KEYS)
    width = 0.8 / n_agents
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, key in enumerate(AGENT_KEYS):
        means, ses = [], []
        for K in k_values:
            df = data[K][key]
            last = df[df["trial"] == df["trial"].max()]["cumulative_payoff"]
            means.append(last.mean())
            ses.append(last.std(ddof=1) / np.sqrt(len(last)) if len(last) > 1 else 0.0)
        xs = [j + (i - (n_agents - 1) / 2) * width for j in range(len(k_values))]
        ax.bar(xs, means, width=width, yerr=ses, capsize=3, label=AGENT_LABELS[key],
               color=AGENT_STYLE[key][0], edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"K={K}" for K in k_values])
    ax.set_ylabel("Final cumulative payoff (mean across episodes, +/-1 SE)")
    ax.set_title("Graph 4: Final Cumulative Payoff by Agent and Landscape")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(figure_dir, "fig4_final_cumulative_payoff_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main(k_values=None, figure_dir=None, N=None, seed=None):
    k_values = config.K_VALUES if k_values is None else k_values
    figure_dir = config.common_figure_dir() if figure_dir is None else figure_dir
    N = config.N_ATTRIBUTES if N is None else N
    seed = config.RANDOM_SEED if seed is None else seed
    os.makedirs(figure_dir, exist_ok=True)

    data = load_all_k(k_values, N, seed)
    n_ep = data[k_values[0]][AGENT_KEYS[0]]["episode"].nunique()

    plot_graph1(data, k_values, figure_dir)
    _three_panel(
        data, k_values, "cumulative_pct", "trial",
        "Graph 2 across landscapes: Cumulative Payoff by Agent (% of maximum possible; "
        "trial 0 = start payoff as % of optimum)",
        "Trial number",
        f"Cumulative payoff (% of maximum possible;\nmean across {n_ep} episode(s))",
        "fig2_cumulative_payoff_across_landscapes.png", figure_dir,
    )
    _three_panel(
        data, k_values, "cumulative_payoff", "trial",
        "Graph 2b across landscapes: Cumulative Payoff by Agent, Actual Values",
        "Trial number",
        f"Cumulative payoff (actual units;\nmean across {n_ep} episode(s))",
        "fig2b_cumulative_payoff_raw_across_landscapes.png", figure_dir,
    )
    _three_panel(
        data, k_values, "search_distance_display", "trial",
        "Graph 3 across landscapes: Average Search Distance per Trial "
        "(trial 0 = distance from true global optimum)",
        "Trial number",
        f"Search distance (Hamming; mean across {n_ep} episode(s))",
        "fig3_search_distance_across_landscapes.png", figure_dir,
    )
    plot_graph4(data, k_values, figure_dir)

    controls = {K: load_control(config.output_dir_for_k(K), N, K, seed) for K in k_values}
    plot_learning_three_panel(data, controls, k_values, figure_dir)
    plot_per_agent_across_k(
        data, k_values, "cumulative_pct", "trial", 0,
        f"Graph 8: Cumulative Payoff (% of maximum possible) by Agent, Across Landscapes "
        f"(mean across {n_ep} episode(s); trial 0 = start payoff as % of optimum)",
        "Cumulative payoff (% of max)",
        "fig8_per_agent_cumulative_payoff_across_landscapes.png", figure_dir,
    )
    plot_per_agent_across_k(
        data, k_values, "cumulative_payoff", "trial", 0,
        f"Graph 8b: Cumulative Payoff by Agent, Across Landscapes, Actual Values "
        f"(mean across {n_ep} episode(s))",
        "Cumulative payoff (actual units)",
        "fig8b_per_agent_cumulative_payoff_raw_across_landscapes.png", figure_dir,
    )
    plot_per_agent_across_k(
        data, k_values, "search_distance_display", "trial", 0,
        f"Graph 9: Average Search Distance by Agent, Across Landscapes "
        f"(mean across {n_ep} episode(s); trial 0 = distance from true global optimum)",
        "Search distance (Hamming)",
        "fig9_per_agent_search_distance_across_landscapes.png", figure_dir,
    )
    print("Cross-landscape graphs generated.")


if __name__ == "__main__":
    main()
