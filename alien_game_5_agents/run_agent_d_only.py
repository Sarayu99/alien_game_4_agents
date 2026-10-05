"""
run_agent_d_only.py
===================
Runs ONLY Agent D (tabular Q-learning, no API calls) on the NK landscapes
K = config.K_VALUES (default 0, 5, 9) for N = 24 trials per episode and
E = 100 episodes, for EVERY epsilon in EPSILON_VALUES (edit the single line
below), and draws TWO figures, each with one subplot per epsilon:

    Figure 1  agent_d_mean_payoff_curve.png
        y-axis : MEAN payoff over the episode's trials 1..N
    Figure 2  agent_d_best_payoff_curve.png
        y-axis : BEST (highest) payoff reached within the episode's trials 1..N

    x-axis : episode number (1..E)
    lines  : one per K value (within each epsilon subplot)
    values : averaged over independent replications (random seeds) of Agent D;
             shaded band = +/-1 standard error. Dotted horizontal line = the
             global optimum of that K's landscape (reference only).

A rising line = Agent D earns more in later episodes = it is learning
(its Q-table persists across episodes).

SAME IMPLEMENTATION AS THE MAIN EXPERIMENT
------------------------------------------
Nothing about Agent D is re-implemented here. This file imports and calls:
    config.py        -> N_ATTRIBUTES, RANDOM_SEED, K_VALUES, RL_* parameters
    nk_landscape.py  -> NKLandscape (same landscape as the main runs: same N, K, seed)
    agents.py        -> AgentD  (AgentD.run_episode, unchanged)
Replication 0 uses the exact Agent D seed of run_experiment.py (seed + 7),
so it reproduces that run's Agent D; further replications only change
Agent D's own RNG seed (landscape stays fixed).

Trial 0 (the free starting configuration, identical every episode) is
excluded from both metrics.

Usage:
    python run_agent_d_only.py                       # uses EPSILON_VALUES below
    python run_agent_d_only.py --eps 0.2 0.3 0.9     # one-off override
    python run_agent_d_only.py --trials 24 --episodes 100 --reps 50

Outputs (folder agent_d_only_N24_E100_eps<list>/):
    agent_d_mean_payoff_curve.png
    agent_d_best_payoff_curve.png
    payoff_by_episode.csv      (mean, SE per epsilon x K x episode, both metrics)
    payoff_raw.csv             (every epsilon x K x replication x episode value)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # no display on the cluster
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config
from nk_landscape import NKLandscape
from agents import AgentD

# ============================================================================
# >>> EDIT THIS ONE LINE to choose which epsilon values to run <<<
# One subplot per value, e.g. [0.3] or [0.1, 0.3, 0.5] or [0.4, 0.5, 0.6, 0.7]
EPSILON_VALUES = [0.4, 0.5, 0.6, 0.7]
# ============================================================================


def episode_payoffs(rows):
    """(mean payoff, best payoff) over trials 1..N of one episode.
    Trial 0 (free starting configuration) is excluded."""
    played = [r["payoff"] for r in rows if r["trial"] > 0]
    return float(np.mean(played)), float(np.max(played))


def run_one_replication(landscape, trials, episodes, agent_seed, eps):
    """One Agent D (Q-table persists across episodes) for `episodes` episodes.
    Returns a list of (mean_payoff, best_payoff), one per episode."""
    agent_d = AgentD(landscape, n_trials=trials, rng_seed=agent_seed, eps=eps)
    return [episode_payoffs(agent_d.run_episode(ep)) for ep in range(1, episodes + 1)]


def plot_metric(summary, metric, ylabel, title, path, eps_values, global_opt, args):
    """One figure: one subplot per epsilon, one line per K."""
    n = len(eps_values)
    ncols = n if n <= 3 else 2
    nrows = int(np.ceil(n / ncols))
    colors = {0: "#1b9e77", 5: "#d95f02", 9: "#7570b3"}
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()

    for ax, eps in zip(flat, eps_values):
        for K in config.K_VALUES:
            s_ = summary[(summary["epsilon"] == eps) & (summary["K"] == K)]
            c = colors.get(K)
            ax.plot(s_["episode"], s_[f"{metric}_mean"], lw=1.8, color=c, label=f"K = {K}")
            ax.fill_between(s_["episode"],
                            s_[f"{metric}_mean"] - s_[f"{metric}_se"],
                            s_[f"{metric}_mean"] + s_[f"{metric}_se"],
                            color=c, alpha=0.15, linewidth=0)
            ax.axhline(global_opt[K], color=c, ls=":", lw=1, alpha=0.8)   # global optimum (reference)
        ax.set_xlim(1, args.episodes)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"ε = {eps:g}", fontsize=13)
        ax.grid(alpha=0.3)
    for ax in flat[n:]:
        ax.set_visible(False)                      # unused grid cells

    for col in range(ncols):                       # x label on the bottom-most visible axes
        visible = [axes[r, col] for r in range(nrows) if axes[r, col].get_visible()]
        if visible:
            visible[-1].set_xlabel("Episode number", fontsize=11)
            visible[-1].tick_params(labelbottom=True)
    for r in range(nrows):
        axes[r, 0].set_ylabel(ylabel, fontsize=11)

    flat[0].legend(title="Landscape complexity", fontsize=10, loc="center right")
    fig.suptitle(f"{title}\n"
                 f"N = {args.trials} trials per episode, {args.episodes} episodes, "
                 f"α = {config.RL_ALPHA}, γ = {config.RL_GAMMA}; "
                 f"mean over {args.reps} replications (shaded = ±1 SE)",
                 fontsize=13)
    fig.text(0.01, 0.005,
             f"Higher = better. Trials 1-{args.trials} (trial 0 excluded). Dotted lines = global optimum "
             "of each K landscape (reference only). Landscape fixed; only Agent D's random seed varies.",
             fontsize=8, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Agent D only: mean / best payoff vs episode.")
    parser.add_argument("--N", type=int, default=config.N_ATTRIBUTES)
    parser.add_argument("--trials", type=int, default=24, help="trials per episode")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--reps", type=int, default=50,
                        help="independent Agent D replications (seeds) to average over")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="landscape seed")
    parser.add_argument("--eps", type=float, nargs="+", default=None,
                        help="epsilon values (default: EPSILON_VALUES at the top of this file)")
    args = parser.parse_args()
    eps_values = args.eps if args.eps is not None else EPSILON_VALUES

    eps_tag = "-".join(f"{e:g}" for e in eps_values)
    # Anchor the output folder to this script's own directory (not the current
    # working directory), so it lands in alien_game_5_agents/ however it is launched.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, f"agent_d_only_N{args.trials}_E{args.episodes}_eps{eps_tag}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"Epsilon values: {eps_values} | K values: {config.K_VALUES} | "
          f"{args.reps} replications x {args.episodes} episodes", flush=True)

    raw = []          # epsilon, K, replication, episode, mean_payoff, best_payoff
    global_opt = {}   # K -> global optimum payoff (reference line)
    for eps in eps_values:
        for K in config.K_VALUES:
            landscape = NKLandscape(N=args.N, K=K, seed=args.seed)   # same landscape as main runs
            global_opt[K] = landscape.get_global_optimum()[1]
            for rep in range(args.reps):
                agent_seed = args.seed + 7 + 1000 * rep              # rep 0 == run_experiment.py's Agent D seed
                results = run_one_replication(landscape, args.trials, args.episodes, agent_seed, eps)
                for ep, (mean_p, best_p) in enumerate(results, start=1):
                    raw.append({"epsilon": eps, "K": K, "replication": rep, "episode": ep,
                                "mean_payoff": mean_p, "best_payoff": best_p})
            print(f"epsilon={eps:g}, K={K}: done ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)   # re-create if the folder was removed while the run was going
    raw_df = pd.DataFrame(raw)
    raw_df.to_csv(os.path.join(out_dir, "payoff_raw.csv"), index=False)

    g = raw_df.groupby(["epsilon", "K", "episode"])
    summary = g.agg(mean_payoff_mean=("mean_payoff", "mean"), mean_payoff_sd=("mean_payoff", "std"),
                    best_payoff_mean=("best_payoff", "mean"), best_payoff_sd=("best_payoff", "std"),
                    n=("mean_payoff", "count")).reset_index()
    for m in ("mean_payoff", "best_payoff"):
        summary[f"{m}_se"] = summary[f"{m}_sd"] / np.sqrt(summary["n"])
    summary.to_csv(os.path.join(out_dir, "payoff_by_episode.csv"), index=False)

    os.makedirs(out_dir, exist_ok=True)
    title_head = "Agent D (Q-learning, Q-table persists across episodes)"
    plot_metric(summary, "mean_payoff",
                "Mean payoff per episode\n(average over the episode's trials)",
                f"{title_head}: mean payoff per episode",
                os.path.join(out_dir, "agent_d_mean_payoff_curve.png"),
                eps_values, global_opt, args)
    plot_metric(summary, "best_payoff",
                "Best payoff per episode\n(highest payoff reached in the episode)",
                f"{title_head}: best payoff per episode",
                os.path.join(out_dir, "agent_d_best_payoff_curve.png"),
                eps_values, global_opt, args)
    print(f"Done in {time.time() - t0:.1f}s. Outputs in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
