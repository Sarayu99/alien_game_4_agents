# Alien Game: 4 Agents Searching a Tunable Rugged (NK) Landscape

This project replicates the "alien game" (Billinger et al. 2014, 2021;
Albert & Billinger 2024/2025) and runs four differently-strategized
agents across search trials, on the same landscape, in complete
isolation from one another.

## Files

| File | Purpose |
|---|---|
| `config.py` | **Edit this file first.** All settings live here: your API key, model name, N/K, number of trials, epsilon values, streetlight anchor-selection settings, K sweep values, output folders. |
| `nk_landscape.py` | Builds the tunable NK-model rugged landscape (the "alien game" engine). Can be run directly (`python nk_landscape.py`) as a quick sanity check. |
| `claude_client.py` | Thin wrapper around the Anthropic API (text + JSON + JSON-with-reasoning responses). You shouldn't need to edit this. |
| `agents.py` | Defines the four agents: `AgentA` (myopic local search), `AgentB` (free replication), `AgentC` (epsilon-greedy), `AgentD` (streetlight). |
| `run_experiment.py` | Runs all four agents on ONE landscape (one K value), saves trial-by-trial CSVs to `outputs/`, then automatically generates Graphs 1-6 and the reasoning analysis. |
| `run_multi_k_experiment.py` | Runs `run_experiment.py`'s full pipeline once for EACH K value in `config.K_VALUES` (default: 0, 5, 9), each into its own `outputs/K{n}/` and `figures/K{n}/` subfolder, then generates Graph 7 (cross-complexity comparison). |
| `analyze_results.py` | Reads one K value's CSVs and produces Graphs 1-4 (PNG). |
| `visualize_landscape_paths.py` | Reads one K value's CSVs and produces Graphs 5-6 (PNG): an MDS landscape map with each agent's search path, and an attribute ON/OFF heatmap per agent. |
| `analyze_reasoning.py` | Reads the `reasoning` column captured for each trial and computes attention-breadth / forward-vs-backward-looking measures. |
| `analyze_cross_k.py` | Reads every K value's output folder and produces Graph 7. |
| `requirements.txt` | Python packages needed (`anthropic`, `pandas`, `matplotlib`, `numpy`, `scipy`, `scikit-learn`). |
| `run_alien_game.lsf` | LSF batch script for Berkeley Haas HPC (`bsub < run_alien_game.lsf`); runs the full K=0/5/9 sweep by default. |

## The four agents

- **Agent A -- Myopic Local Search (the baseline)**: pure code, ZERO Claude API calls, ZERO memory beyond the current best-known configuration. This is Billinger et al. (2014)'s own p=0 computational baseline: one fully random "long jump" on trial 1, then a single random attribute flip per trial afterward, kept only if it improves on the current best. The only agent that never talks to Claude and never sees any history at all -- deliberately, so it can serve as a clean, dependency-free floor to measure the other three agents against.
- **Agent B -- Free Replication**: Claude sees the full trial history and freely proposes its next FULL configuration every round. No framing, no extra reference information, no search-distance rule -- the plain baseline for "what does Claude do with no strategy imposed at all."
- **Agent C -- Epsilon-Greedy**: each round, a fixed probability (`epsilon`), decided by CODE (not by Claude), picks whether this is an "explore" round or an "exploit" round -- that mode-selection being external and random is part of the actual definition of epsilon-greedy in the RL/bandit literature. **Claude is not restricted in either round**: it always proposes a full, unconstrained configuration (identical JSON schema to Agent B), with the full trial history. The only difference between rounds is the *framing text* it receives ("make a small refinement" vs. "try something substantially different") -- whether its resulting search distance actually comes out small or large is something you measure in the CSV, not something the code guarantees.
- **Agent D -- Streetlight**: Claude is given, as extra reference information alongside its own full trial history, the value of ONE other already-explored configuration (the "anchor") -- mirroring exactly how Hoelzemann, Manso, Nagaraj & Tranchero (2024)'s laboratory experiment reveals one project's value to participants who otherwise choose completely freely among all options. **Claude is not restricted to flipping a capped number of attributes near that anchor** -- it proposes a full, unconstrained configuration every round, identical in shape to Agent B. Every row also logs `distance_from_anchor` (the Hamming distance from that trial's choice to the anchor), so you can directly measure whether Claude's search clusters near the anchor anyway, rather than assuming it does.

**Why B, C, and D are now structurally identical in what Claude is *allowed* to do:** all three always propose a full, free 10-attribute configuration with full history access. They differ only in the *framing or reference information* included in the prompt -- never in a code-enforced cap on flip count or an artificially restricted candidate list. This means any difference you see in their resulting search-distance or payoff patterns reflects something real about how that framing/information changes Claude's behavior, rather than being guaranteed in advance by the code. (Agent A remains the one exception, by design -- see above.)

## Reasoning capture ("think aloud")

Every Claude decision (Agent B every trial; Agent C on both explore and exploit rounds, since both now call Claude; Agent D every trial) asks Claude to briefly explain its reasoning before giving its final JSON answer, and captures that explanation in a `reasoning` column -- inspired by the think-aloud extension in Albert & Billinger's LLM replication. Trials with no real Claude choice (the free starting configuration, and all of Agent A's moves) get an explicit `"N/A -- ..."` placeholder instead.

`analyze_reasoning.py` reads this column and computes two lightweight, keyword-based measures per trial: **attention breadth** (how many distinct symbols are mentioned) and a **forward-looking ratio** (predictive vs. backward-referencing language). These are heuristic proxies for the source paper's own measures, not a validated replication -- treat them as a starting point.

## The output graphs

- **Graph 1 / 2 (performance)**: best-known payoff vs. trial number, trial 0 included. Graph 1 compares all four agents; Graph 2 compares Agent C across its three epsilon values.
- **Graph 3 / 4 (search distance)**: Hamming search distance vs. trial number. Trial 0 shows each agent's Hamming distance from the TRUE global optimum (search distance is otherwise undefined at trial 0).
- **Graph 5 (landscape paths)** and **Graph 6 (attribute heatmaps)**: an MDS map of the whole landscape with each of the four agents' actual search paths drawn on it, and a per-agent ON/OFF attribute heatmap across trials.
- **Graph 7 (cross-complexity comparison)**: only produced by `run_multi_k_experiment.py` / `analyze_cross_k.py`. A grouped bar chart comparing all four agents' final best-known payoff across K=0, K=5, and K=9 -- mirroring the structure of Billinger et al. (2014)'s Table 1.

On every line chart (Graphs 1-4), each series uses its own color, linestyle, and marker shape, so overlapping series stay visually distinguishable even when two agents make identical choices.

## How to run it (step by step)

1. **Install packages** (one time only, inside your conda environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your API key.** Open `config.py` and paste your key into the `API_KEY` variable near the top, or leave it blank and set the `ANTHROPIC_API_KEY` environment variable instead (e.g. in your `.lsf` script).

3. **Adjust settings if you want** (also in `config.py`): landscape ruggedness `K_COMPLEXITY` (used only for a single `run_experiment.py` run), number of trials (`N_TRIALS`), epsilon values, streetlight anchor percentiles, and `K_VALUES` (the K sweep used by `run_multi_k_experiment.py`, default `[0, 5, 9]`).

4a. **Run a single landscape (one K value):**
   ```bash
   python run_experiment.py
   python run_experiment.py --K 5 --N 10 --trials 3 --seed 42
   ```

4b. **Or run the full K=0/5/9 sweep with the cross-complexity graph:**
   ```bash
   python run_multi_k_experiment.py
   ```
   Note: this runs 4 agents x 3 K values (Agent C counted once per epsilon value) -- meaningfully more API usage than a single run. Consider testing with a small `N_TRIALS` first.

5. **On the HPC cluster**, submit either script via LSF:
   ```bash
   bsub < run_alien_game.lsf
   ```
   (edit the script to switch between the single-K and multi-K commands -- see the comments inside it).

6. **(Optional) Regenerate just the graphs later**, without rerunning the experiment:
   ```bash
   python analyze_results.py                 # Graphs 1-4, single K (config.K_COMPLEXITY)
   python visualize_landscape_paths.py        # Graphs 5-6, single K
   python analyze_reasoning.py                # reasoning_analysis.csv, single K
   python analyze_cross_k.py                  # Graph 7, reads all of config.K_VALUES
   ```

## Notes

- Each K value's landscape is fixed by `RANDOM_SEED` in `config.py`, so re-running with the same seed and K reproduces the exact same landscape.
- Agents A, B, C, and D all start from the same free, low-performing configuration at trial 0; Agent D additionally sees its anchor as reference information from trial 1 onward, but never starts there.
- `best_and_worst_outcome_summary.csv` reports, per K value: the single best outcome found by any agent, the true global-optimum payoff of the landscape, and the true global-worst payoff of the landscape.
