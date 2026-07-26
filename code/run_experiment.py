"""
run_experiment.py
==================
Main script: creates one NK landscape and runs all four agents on it, in
isolation from one another (each agent's search decisions have no
influence on or visibility into the others):

    Agent A -- Myopic Local Search (hardcoded baseline, no Claude calls)
    Agent B -- Free Replication    (full history, Claude decides freely)
    Agent C -- Epsilon-Greedy      (code-triggered explore/exploit)
    Agent D -- Streetlight         (anchored, code-restricted flips)

Every trial's results are saved to CSV files inside OUTPUT_DIR (see
config.py). After the experiment finishes, this script automatically
calls analyze_results.py, visualize_landscape_paths.py, and
analyze_reasoning.py so that every output graph/analysis is generated in
one command -- you do not need to run them separately.

Usage (single K value, from the command line):
    python run_experiment.py
    python run_experiment.py --K 5 --N 10 --trials 3 --seed 42

For a sweep across multiple K values (0, 5, 9) with a combined
cross-complexity comparison graph, use run_multi_k_experiment.py instead,
which calls run_single_experiment() below once per K value.
"""

import argparse
import os

import pandas as pd

import config
from nk_landscape import NKLandscape
from claude_client import ClaudeClient
from agents import AgentA, AgentB, AgentC, AgentD


def history_to_dataframe(history):
    """Convert an agent's list-of-dict history into a pandas DataFrame."""
    return pd.DataFrame(history)


def save_dataframe(df, filename, output_dir):
    """Save a DataFrame as a CSV file inside the given output folder."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def run_single_experiment(N, K, trials, seed, model, output_dir, figure_dir):
    """
    Build one NK landscape (N, K, seed) and run all four agents on it,
    saving their trial-by-trial CSVs (and the best/worst summary CSV)
    into output_dir. Does NOT generate graphs -- call
    generate_all_analysis() afterward for that (kept separate so
    run_multi_k_experiment.py can run all three K values first, then
    generate every graph, including the cross-K comparison, at the end).

    Returns nothing; everything is written to disk.
    """
    print(f"Building landscape: N={N}, K={K}, seed={seed}")
    landscape = NKLandscape(N=N, K=K, seed=seed)

    global_config, global_payoff = landscape.get_global_optimum()
    worst_config, worst_payoff = landscape.get_lowest_performing_configuration()
    print(
        f"Global optimum (for reference only -- never shown to agents): "
        f"{global_config} -> payoff {global_payoff:.4f}"
    )
    print(
        f"Global worst configuration (for reference only): "
        f"{worst_config} -> payoff {worst_payoff:.4f}"
    )

    claude = ClaudeClient(model=model)

    all_rows = []  # collects every trial from every agent/run, for the final summary

    # -------------------------------------------------------------------
    # Agent A: myopic local search (pure code, no Claude calls)
    # -------------------------------------------------------------------
    print("\nRunning Agent A (myopic local search)...")
    agent_a = AgentA(landscape, claude=None, n_trials=trials, rng_seed=seed)
    history_a = agent_a.run()
    save_dataframe(history_to_dataframe(history_a), "agent_a.csv", output_dir)
    all_rows.extend(history_a)

    # -------------------------------------------------------------------
    # Agent B: free replication (run once)
    # -------------------------------------------------------------------
    print("\nRunning Agent B (free replication)...")
    agent_b = AgentB(landscape, claude, n_trials=trials)
    history_b = agent_b.run()
    save_dataframe(history_to_dataframe(history_b), "agent_b.csv", output_dir)
    all_rows.extend(history_b)

    # -------------------------------------------------------------------
    # Agent C: epsilon-greedy (run once per epsilon value in config.py)
    # -------------------------------------------------------------------
    for eps in config.EPSILON_VALUES:
        print(f"\nRunning Agent C (epsilon-greedy, epsilon={eps})...")
        # Fresh agent instance each time -> fully isolated runs, no shared state.
        agent_c = AgentC(landscape, claude, epsilon=eps, n_trials=trials, rng_seed=seed)
        history_c = agent_c.run()
        save_dataframe(history_to_dataframe(history_c), f"agent_c_eps{eps}.csv", output_dir)
        all_rows.extend(history_c)

    # -------------------------------------------------------------------
    # Agent D: streetlight (run once)
    # -------------------------------------------------------------------
    print("\nRunning Agent D (streetlight)...")
    agent_d = AgentD(landscape, claude, n_trials=trials, rng_seed=seed)
    history_d = agent_d.run()
    save_dataframe(history_to_dataframe(history_d), "agent_d.csv", output_dir)
    all_rows.extend(history_d)

    # -------------------------------------------------------------------
    # Identify the single highest-payoff outcome across every agent/run,
    # and pair it with the true global-best and global-worst payoffs of
    # this landscape, all in one reference row.
    # -------------------------------------------------------------------
    best_row = max(all_rows, key=lambda row: row["payoff"])
    best_config_dict = NKLandscape.config_to_dict(best_row["config"])
    global_config_dict = NKLandscape.config_to_dict(global_config)
    worst_config_dict = NKLandscape.config_to_dict(worst_config)

    print("\n=== BEST OUTCOME ACROSS ALL AGENTS AND RUNS ===")
    print(f"Agent:         {best_row['agent']}  (epsilon={best_row['epsilon']})")
    print(f"Trial number:  {best_row['trial']}")
    print(f"Configuration: {best_config_dict}")
    print(f"Payoff:        {best_row['payoff']:.4f}")
    print(f"(For reference, the true global optimum payoff was {global_payoff:.4f} "
          f"at config {global_config_dict}, and the true global worst payoff was "
          f"{worst_payoff:.4f} at config {worst_config_dict} -- this worst config is "
          f"also the free starting configuration given to Agents A, B, C, and D at trial 0 -- "
          f"Agent D additionally sees a separate 'anchor' reference configuration, but does not "
          f"start there)")

    summary_df = pd.DataFrame([{
        "agent": best_row["agent"],
        "epsilon": best_row["epsilon"],
        "trial": best_row["trial"],
        "config": best_config_dict,
        "payoff": best_row["payoff"],
        "global_optimum_config": global_config_dict,
        "global_optimum_payoff": global_payoff,
        "global_worst_config": worst_config_dict,
        "global_worst_payoff": worst_payoff,
    }])
    save_dataframe(summary_df, "best_and_worst_outcome_summary.csv", output_dir)


def generate_all_analysis(K, N, seed, output_dir, figure_dir):
    """Runs analyze_results.py, visualize_landscape_paths.py, and
    analyze_reasoning.py against one already-completed experiment run
    (output_dir), saving graphs into figure_dir. Kept as a separate
    function (rather than inline in run_single_experiment) so
    run_multi_k_experiment.py can call it once per K value."""
    print(f"\nGenerating analysis for K={K} (output_dir={output_dir}, figure_dir={figure_dir})...")

    import analyze_results
    analyze_results.main(K=K, N=N, seed=seed, output_dir=output_dir, figure_dir=figure_dir)

    try:
        import visualize_landscape_paths
        visualize_landscape_paths.main(K=K, N=N, seed=seed, output_dir=output_dir, figure_dir=figure_dir)
    except ImportError as e:
        print(
            f"\nSkipped Graphs 5-6 (visualize_landscape_paths.py): missing dependency ({e}). "
            "Install scipy and scikit-learn (see requirements.txt) to enable them."
        )

    try:
        import analyze_reasoning
        analyze_reasoning.main(output_dir=output_dir)
    except Exception as e:
        print(f"\nSkipped reasoning analysis (analyze_reasoning.py): {e}")


def main():
    parser = argparse.ArgumentParser(description="Run the alien-game NK-landscape experiment.")
    parser.add_argument("--N", type=int, default=config.N_ATTRIBUTES, help="number of attributes")
    parser.add_argument("--K", type=int, default=config.K_COMPLEXITY, help="landscape ruggedness (0-9)")
    parser.add_argument("--trials", type=int, default=config.N_TRIALS, help="number of trials per agent")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="landscape random seed")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Claude model name")
    args = parser.parse_args()

    output_dir = config.OUTPUT_DIR
    figure_dir = config.FIGURE_DIR

    run_single_experiment(
        N=args.N, K=args.K, trials=args.trials, seed=args.seed, model=args.model,
        output_dir=output_dir, figure_dir=figure_dir,
    )

    print("\nExperiment finished. Generating the comparison graphs now...")
    generate_all_analysis(K=args.K, N=args.N, seed=args.seed, output_dir=output_dir, figure_dir=figure_dir)

    print("\nAll done -- check the outputs/ and figures/ folders.")


if __name__ == "__main__":
    main()
