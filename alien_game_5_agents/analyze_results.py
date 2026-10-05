"""
analyze_results.py
====================
Shared helpers (no figures are written from this file any more).

Loads the per-agent CSVs written by run_experiment.py, adds the derived
columns (search_distance_display, cumulative_pct), and provides the drawing
helpers used by analyze_cross_k.py, which produces ALL the figures in
figures_N.._E../common_figures/.
"""

import ast
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

import config
from nk_landscape import NKLandscape

AGENT_KEYS = ["A_myopic_local_search", "B_free_replication", "C_epsilon_greedy", "D_q_learning"]
AGENT_LABELS = {
    "A_myopic_local_search": "Agent A (myopic local search)",
    "B_free_replication": "Agent B (free replication)",
    "C_epsilon_greedy": f"Agent C (epsilon-greedy, eps={config.EPSILON_FOR_COMPARISON})",
    "D_q_learning": f"Agent D (Q-learning, eps={config.RL_EPSILON})",
}
AGENT_FILES = {
    "A_myopic_local_search": "agent_a.csv",
    "B_free_replication": "agent_b.csv",
    "C_epsilon_greedy": f"agent_c_eps{config.EPSILON_FOR_COMPARISON}.csv",
    "D_q_learning": "agent_d.csv",
}
AGENT_STYLE = {  # fixed color / linestyle / marker per agent
    "A_myopic_local_search": ("#4C72B0", "-", "o"),
    "B_free_replication": ("#DD8452", "--", "s"),
    "C_epsilon_greedy": ("#55A868", "-.", "^"),
    "D_q_learning": ("#C44E52", ":", "D"),
}


# ---------------------------------------------------------------------------
# Loading + aggregation helpers (also used by analyze_cross_k.py)
# ---------------------------------------------------------------------------
def load_csv(filename, output_dir):
    df = pd.read_csv(os.path.join(output_dir, filename))
    df["config"] = df["config"].apply(ast.literal_eval)
    return df


def get_global_optimum_config(N, K, seed):
    return NKLandscape(N=N, K=K, seed=seed).get_global_optimum()[0]


def add_display_distance(df, global_optimum_config):
    """search_distance_display = search_distance for trial > 0; at trial 0
    it is the Hamming distance to the true global optimum."""
    df = df.copy()
    dist_opt = [NKLandscape.hamming_distance(c, global_optimum_config) for c in df["config"]]
    df["search_distance_display"] = df["search_distance"]
    mask = df["trial"] == 0
    df.loc[mask, "search_distance_display"] = pd.Series(dist_opt, index=df.index)[mask]
    return df


def add_cumulative_pct(df, optimum_payoff, start_payoff):
    """Add 'cumulative_pct': cumulative payoff as a % of the maximum
    achievable cumulative payoff up to that trial, i.e. optimum_payoff * t
    (plus the fixed trial-0 payoff if config.INCLUDE_TRIAL0_IN_CUMULATIVE).
    At trial 0 (nothing earned yet, baseline 0) the value is set to 0."""
    df = df.copy()
    offset = start_payoff if config.INCLUDE_TRIAL0_IN_CUMULATIVE else 0.0
    baseline = optimum_payoff * df["trial"] + offset
    pct = 100.0 * df["cumulative_payoff"] / baseline.where(baseline > 0)
    df["cumulative_pct"] = pct.fillna(0.0)
    return df


CONTROL_KEY = "D_q_learning_reset_control"
CONTROL_FILE = "agent_d_reset_control.csv"
CONTROL_LABEL = "Agent D control (Q reset every episode)"


def load_control(output_dir, N, K, seed):
    """DataFrame for the no-memory Agent D control, or None if not run."""
    if not os.path.exists(os.path.join(output_dir, CONTROL_FILE)):
        return None
    landscape = NKLandscape(N=N, K=K, seed=seed)
    opt, opt_payoff = landscape.get_global_optimum()
    _, start_payoff = landscape.get_lowest_performing_configuration()
    df = add_display_distance(load_csv(CONTROL_FILE, output_dir), opt)
    return add_cumulative_pct(df, opt_payoff, start_payoff)


def episode_totals(df):
    """(episode numbers, final cumulative payoff as % of max) per episode."""
    last = df[df["trial"] == df["trial"].max()].sort_values("episode")
    return last["episode"].to_numpy(), last["cumulative_pct"].to_numpy()


def draw_learning_curves(ax, agent_dfs, control_df):
    n_ep = 1
    for key in AGENT_KEYS:
        eps, vals = episode_totals(agent_dfs[key])
        n_ep = max(n_ep, int(eps.max()))
        color, ls, marker = AGENT_STYLE[key]
        ax.plot(eps, vals, color=color, linestyle=ls, marker=marker, markersize=5,
                linewidth=1.7, alpha=0.9, label=AGENT_LABELS[key])
    if control_df is not None:
        eps, vals = episode_totals(control_df)
        ax.plot(eps, vals, color="#7f7f7f", linestyle="--", marker="x", markersize=6,
                linewidth=1.5, alpha=0.9, label=CONTROL_LABEL)
    ax.grid(True, alpha=0.3)
    # x axis: exactly the episode numbers 1..n_ep (a single tick if n_ep == 1)
    ax.set_xticks(range(1, n_ep + 1))
    ax.set_xlim(0.5, n_ep + 0.5)


def load_all_agents(output_dir, N, K, seed):
    """Returns {agent_key: DataFrame} with search_distance_display and
    cumulative_pct added."""
    landscape = NKLandscape(N=N, K=K, seed=seed)
    opt, opt_payoff = landscape.get_global_optimum()
    _, start_payoff = landscape.get_lowest_performing_configuration()
    out = {}
    for key in AGENT_KEYS:
        df = add_display_distance(load_csv(AGENT_FILES[key], output_dir), opt)
        out[key] = add_cumulative_pct(df, opt_payoff, start_payoff)
    return out


def mean_by_trial(df, column):
    """Average `column` across episodes (and any other rows) at each trial."""
    g = df.groupby("trial")[column].mean().sort_index()
    return g.index.to_numpy(), g.to_numpy()


def pct_rounds(trials, total_trials):
    return np.asarray(trials) / total_trials * 100.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def draw_agent_lines(ax, agent_dfs, column, x_mode, min_trial=0):
    """x_mode: 'pct' (percentage of rounds completed) or 'trial'."""
    for key in AGENT_KEYS:
        trials, vals = mean_by_trial(agent_dfs[key], column)
        keep = trials >= min_trial
        trials, vals = trials[keep], vals[keep]
        total = trials.max()
        x = pct_rounds(trials, total) if x_mode == "pct" else trials
        color, ls, marker = AGENT_STYLE[key]
        ax.plot(x, vals, color=color, linestyle=ls, marker=marker, markersize=4.5,
                linewidth=1.7, alpha=0.9, label=AGENT_LABELS[key])
    ax.grid(True, alpha=0.3)
    if x_mode == "trial":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
