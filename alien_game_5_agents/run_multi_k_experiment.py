"""
run_multi_k_experiment.py
===========================
Runs the full 4-agent, multi-episode experiment once per K in
config.K_VALUES (default 0, 5, 9), then produces the cross-landscape
comparison graphs.

    outputs_N.._E../K0, K5, K9              (CSVs, Q-table)
    figures_N.._E../common_figures/*.png    (all graphs; cross-K comparisons)

Usage:
    python run_multi_k_experiment.py
"""

import config
import progress
import run_experiment
import analyze_cross_k


def main():
    # One tracker for the whole sweep: K values x episodes x {B, C} x trials.
    progress.init(len(config.K_VALUES) * config.N_EPISODES * 2 * config.N_TRIALS)

    for K in config.K_VALUES:
        output_dir = config.output_dir_for_k(K)
        figure_dir = config.figure_dir_for_k(K)

        print(f"\n{'=' * 70}\nRunning full experiment for K={K}\n{'=' * 70}")
        run_experiment.run_single_experiment(
            N=config.N_ATTRIBUTES, K=K, trials=config.N_TRIALS, episodes=config.N_EPISODES,
            seed=config.RANDOM_SEED, model=config.MODEL_NAME,
            output_dir=output_dir, figure_dir=figure_dir,
        )
        run_experiment.generate_all_analysis(
            K=K, N=config.N_ATTRIBUTES, seed=config.RANDOM_SEED,
            output_dir=output_dir, figure_dir=figure_dir,
        )

    print(f"\n{'=' * 70}\nGenerating cross-landscape comparison graphs\n{'=' * 70}")
    analyze_cross_k.main()
    print("\nAll K values complete.")


if __name__ == "__main__":
    main()
