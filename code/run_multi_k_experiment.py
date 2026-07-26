"""
run_multi_k_experiment.py
===========================
Runs the full four-agent experiment (Agents A, B, C, D) once for EACH K
value in config.K_VALUES (by default [0, 5, 9] -- Billinger et al.
(2014)'s low/intermediate/high complexity levels), each with its own
freshly-built landscape and its own output folder:

    outputs/K0/, outputs/K5/, outputs/K9/
    figures/K0/, figures/K5/, figures/K9/

For each K, this generates the same 6 graphs (+ reasoning analysis) that
run_experiment.py produces for a single K. Once all three K values are
done, it generates ONE additional cross-complexity graph:

    Graph 7 (figures/fig7_cross_k_comparison.png)
        A grouped bar chart comparing all four agents' final best-known
        payoff across the three K values, in the style of Billinger et
        al. (2014)'s Table 1 (human vs. computational-agent performance
        by complexity level) -- but for your four agents instead.

Usage:
    python run_multi_k_experiment.py

Each K value's landscape uses the SAME seed (config.RANDOM_SEED), so
differences across K are attributable to landscape ruggedness, not to a
different random landscape draw.

Note: this runs 4 agents x 3 K values = 12 agent runs total (Agent C
counted once per epsilon value), each making real Claude API calls
(except Agent A). At N_TRIALS=24 this is a meaningfully larger amount of
API usage than a single run_experiment.py call -- consider testing with a
small config.N_TRIALS first.
"""

import config
import run_experiment
import analyze_cross_k


def main():
    for K in config.K_VALUES:
        output_dir = config.output_dir_for_k(K)
        figure_dir = config.figure_dir_for_k(K)

        print(f"\n{'=' * 70}\nRunning full experiment for K={K}\n{'=' * 70}")
        run_experiment.run_single_experiment(
            N=config.N_ATTRIBUTES, K=K, trials=config.N_TRIALS, seed=config.RANDOM_SEED,
            model=config.MODEL_NAME, output_dir=output_dir, figure_dir=figure_dir,
        )
        run_experiment.generate_all_analysis(
            K=K, N=config.N_ATTRIBUTES, seed=config.RANDOM_SEED,
            output_dir=output_dir, figure_dir=figure_dir,
        )

    print(f"\n{'=' * 70}\nGenerating Graph 7 (cross-K comparison)\n{'=' * 70}")
    analyze_cross_k.main()

    print("\nAll K values complete -- check outputs/K0, outputs/K5, outputs/K9, "
          "figures/K0, figures/K5, figures/K9, and figures/fig7_cross_k_comparison.png.")


if __name__ == "__main__":
    main()
