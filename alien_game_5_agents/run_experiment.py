"""
run_experiment.py
==================
Runs the 4-agent, multi-episode experiment on ONE landscape (one K value).

    Agent A -- Myopic Local Search   (no memory across episodes)
    Agent B -- Free Replication      (no memory across episodes)
    Agent C -- Epsilon-Greedy        (no memory across episodes)
    Agent D -- Q-learning, eps = 0.3 (Q-table persists across episodes)
    (+ a no-memory control copy of D, agent_d_reset_control.csv, no API calls)

Each agent plays config.N_EPISODES episodes of config.N_TRIALS trials on the
same landscape. A/B/C are rebuilt from scratch every episode; Agent D is
built once and its Q-table carries over.

CSV outputs (one row per agent x episode x trial) in output_dir:
    agent_a.csv, agent_b.csv, agent_c_eps0.3.csv, agent_d.csv,
    agent_d_qtable.npy, best_and_worst_outcome_summary.csv

Usage:
    python run_experiment.py
    python run_experiment.py --K 5 --trials 24 --episodes 10 --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd

import config
import progress
from nk_landscape import NKLandscape
from claude_client import ClaudeClient
from agents import AgentA, AgentB, AgentC, AgentD


def save_dataframe(df, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def run_single_experiment(N, K, trials, episodes, seed, model, output_dir, figure_dir):
    print(f"Building landscape: N={N}, K={K}, seed={seed}")
    landscape = NKLandscape(N=N, K=K, seed=seed)

    global_config, global_payoff = landscape.get_global_optimum()
    worst_config, worst_payoff = landscape.get_lowest_performing_configuration()
    print(f"Global optimum (reference only): {global_config} -> {global_payoff:.4f}")
    print(f"Global worst (reference only):   {worst_config} -> {worst_payoff:.4f}")

    claude = ClaudeClient(model=model)

    # Single-K run: start the tracker here. (run_multi_k_experiment.py starts
    # one tracker for the whole sweep, so this is skipped there.)
    if not progress.is_active():
        progress.init(episodes * 2 * trials)   # agents B and C only
    eps_c = config.EPSILON_FOR_COMPARISON

    rows_a, rows_b, rows_c, rows_d, rows_ctrl = [], [], [], [], []

    # Agent D is created ONCE: its q table is its memory across episodes.
    agent_d = AgentD(landscape, n_trials=trials, rng_seed=seed + 7)

    # No-memory CONTROL copy of Agent D (same seed, Q re-randomised every
    # episode). Only used for the learning-curve graph.
    run_control = config.RL_RUN_RESET_CONTROL and not config.RL_RESET_Q_EACH_EPISODE
    agent_d_ctrl = (AgentD(landscape, n_trials=trials, rng_seed=seed + 7,
                           reset_q_each_episode=True,
                           agent_name="D_q_learning_reset_control")
                    if run_control else None)

    for ep in range(1, episodes + 1):
        print(f"\n--- K={K} | Episode {ep}/{episodes} ---", flush=True)
        progress.set_context(K=K, episode=ep, n_episodes=episodes)

        # Fresh instances every episode -> no memory for A, B, C.
        # Per-episode RNG seeds so A and C don't replay identical coin flips.
        agent_a = AgentA(landscape, claude=None, n_trials=trials,
                         rng_seed=seed * 1000 + ep, episode=ep)
        rows_a.extend(agent_a.run())

        agent_b = AgentB(landscape, claude, n_trials=trials, episode=ep)
        rows_b.extend(agent_b.run())

        agent_c = AgentC(landscape, claude, epsilon=eps_c, n_trials=trials,
                         rng_seed=seed * 1000 + 500 + ep, episode=ep)
        rows_c.extend(agent_c.run())

        rows_d.extend(agent_d.run_episode(ep))
        if agent_d_ctrl is not None:
            rows_ctrl.extend(agent_d_ctrl.run_episode(ep))

        total = lambda rows: sum(r["payoff"] for r in rows if r["episode"] == ep and r["trial"] > 0)
        print(f"  Total payoff (trials 1-{trials}):  "
              f"A={total(rows_a):.3f}  B={total(rows_b):.3f}  "
              f"C={total(rows_c):.3f}  D={total(rows_d):.3f}   "
              f"(D state-actions updated so far: {agent_d.n_visited_state_actions()})",
              flush=True)
        progress.episode_done()

    save_dataframe(pd.DataFrame(rows_a), "agent_a.csv", output_dir)
    save_dataframe(pd.DataFrame(rows_b), "agent_b.csv", output_dir)
    save_dataframe(pd.DataFrame(rows_c), f"agent_c_eps{eps_c}.csv", output_dir)
    save_dataframe(pd.DataFrame(rows_d), "agent_d.csv", output_dir)
    np.save(os.path.join(output_dir, "agent_d_qtable.npy"), agent_d.q)
    if rows_ctrl:
        save_dataframe(pd.DataFrame(rows_ctrl), "agent_d_reset_control.csv", output_dir)

    all_rows = rows_a + rows_b + rows_c + rows_d
    best_row = max(all_rows, key=lambda r: r["payoff"])
    summary_df = pd.DataFrame([{
        "agent": best_row["agent"],
        "episode": best_row["episode"],
        "trial": best_row["trial"],
        "config": NKLandscape.config_to_dict(best_row["config"]),
        "payoff": best_row["payoff"],
        "global_optimum_config": NKLandscape.config_to_dict(global_config),
        "global_optimum_payoff": global_payoff,
        "global_worst_config": NKLandscape.config_to_dict(worst_config),
        "global_worst_payoff": worst_payoff,
    }])
    save_dataframe(summary_df, "best_and_worst_outcome_summary.csv", output_dir)
    print(f"\nBest single outcome: {best_row['agent']} (episode {best_row['episode']}, "
          f"trial {best_row['trial']}) payoff {best_row['payoff']:.4f} "
          f"vs global optimum {global_payoff:.4f}")


def generate_all_analysis(K, N, seed, output_dir, figure_dir):
    """Per-K post-processing: optional fig6 heatmaps and reasoning analysis.
    All figures come from analyze_cross_k.py (common_figures)."""
    print(f"\nGenerating analysis for K={K}...")

    try:
        import visualize_landscape_paths
        visualize_landscape_paths.main(K=K, N=N, seed=seed, output_dir=output_dir, figure_dir=figure_dir)
    except ImportError as e:
        print(f"\nSkipped per-K landscape figures: missing dependency ({e}). See requirements.txt.")

    try:
        import analyze_reasoning
        analyze_reasoning.main(output_dir=output_dir)
    except Exception as e:
        print(f"\nSkipped reasoning analysis: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run the alien-game experiment on one K.")
    parser.add_argument("--N", type=int, default=config.N_ATTRIBUTES)
    parser.add_argument("--K", type=int, default=config.K_COMPLEXITY)
    parser.add_argument("--trials", type=int, default=config.N_TRIALS, help="trials per episode")
    parser.add_argument("--episodes", type=int, default=config.N_EPISODES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--model", type=str, default=config.MODEL_NAME)
    args = parser.parse_args()

    # CLI overrides go into config FIRST so the output folder names match.
    config.N_TRIALS = args.trials
    config.N_EPISODES = args.episodes
    output_dir = config.output_dir_for_k(args.K)
    figure_dir = config.figure_dir_for_k(args.K)

    run_single_experiment(
        N=args.N, K=args.K, trials=args.trials, episodes=args.episodes, seed=args.seed,
        model=args.model, output_dir=output_dir, figure_dir=figure_dir,
    )
    generate_all_analysis(K=args.K, N=args.N, seed=args.seed,
                          output_dir=output_dir, figure_dir=figure_dir)
    print(f"\nAll done -- check {output_dir}/. (Figures come from analyze_cross_k.py.)")


if __name__ == "__main__":
    main()
