# Alien Game v4 — Four Agents (A, B, C, D) Searching an NK Landscape Over Multiple Episodes

This project simulates the "alien game" (Billinger, Stieglitz & Schumacher 2014; Billinger et al. 2021; Albert & Billinger 2024/2025) with four artificial agents that search the same tunable rugged (NK) landscape in complete isolation from one another. Three agents are classic or LLM-driven searchers with **no memory across episodes**; the fourth is a **Q-learning agent whose Q-table persists across episodes**. This README is meant to be self-sufficient: it explains the task, every agent, every file, every column in every CSV, every graph and how to read it, every setting, and how to run and interpret the experiment.

---

## Table of contents
1. [The task in one page](#1-the-task-in-one-page)
2. [Experiment structure: landscapes, episodes, trials](#2-experiment-structure-landscapes-episodes-trials)
3. [The four agents](#3-the-four-agents)
4. [Reasoning capture ("think aloud")](#4-reasoning-capture-think-aloud)
5. [Key definitions (payoff, search distance, cumulative payoff, success)](#5-key-definitions)
6. [Files in this folder](#6-files-in-this-folder)
7. [Settings (config.py)](#7-settings-configpy)
8. [How to run](#8-how-to-run)
9. [Outputs: folder layout and CSV data dictionary](#9-outputs-folder-layout-and-csv-data-dictionary)
10. [The graphs and how to read them](#10-the-graphs-and-how-to-read-them)
11. [Reasoning analysis](#11-reasoning-analysis)
12. [API usage, runtime and reproducibility](#12-api-usage-runtime-and-reproducibility)
13. [Interpretation cautions](#13-interpretation-cautions)
14. [What changed relative to the previous (3-agent) version](#14-what-changed-relative-to-the-previous-3-agent-version)
15. [Troubleshooting](#15-troubleshooting)
16. [References](#16-references)

---

## 1. The task in one page

An agent is a seller who has made contact with an alien that buys "art pictures". A picture is made of **N = 10 geometric shapes**, each switched ON (1) or OFF (0). The shapes are named with Greek letters: alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa. A **configuration** is therefore a 10-bit string (2^10 = 1,024 possible pictures).

The agent does not know which configurations the alien prefers. Each **trial** it submits one configuration and learns its **payoff** (what the alien pays, a number between 0 and 1). The goal is to earn as much as possible over the trials.

The hidden payoff function is an **NK landscape** (Kauffman). Two parameters define it:
- **N = 10**: number of attributes.
- **K = 0 … 9**: how many *other* attributes each attribute's contribution depends on. K controls **ruggedness**:
  - **K = 0**: smooth, a single peak; each attribute contributes independently, so local improvements lead to the global optimum.
  - **K = 5**: intermediate ruggedness (many local peaks).
  - **K = 9**: maximally rugged; every attribute's contribution depends on all others, so the landscape is close to random and local moves give little guidance.

How the payoff is built (see `nk_landscape.py`): for each attribute *i*, K partner attributes are drawn at random from the other 9; attribute *i* gets a random lookup table with 2^(K+1) entries (each a uniform(0,1) draw), indexed by the on/off states of *i* and its K partners. The payoff of a configuration is the **mean of the 10 contributions**. The landscape is generated from `random.Random(seed)`, so the same `RANDOM_SEED` and K always reproduce the same landscape.

The code enumerates all 1,024 configurations to know the **true global optimum** and the **true global worst** configuration. These are used only for reference and for the trial-0 search-distance convention; **they are never shown to any agent**.

---

## 2. Experiment structure: landscapes, episodes, trials

- **Landscapes.** One fixed landscape per K value in `K_VALUES` (default 0, 5, 9). All three use the same seed, so differences across K reflect ruggedness, not a different random draw. (Note the landscapes are *not* nested versions of one another: a different K means different partner structures and lookup-table sizes, so raw payoff levels are not strictly comparable across K — see §13.)
- **Trial.** One submission of one configuration and the resulting payoff. `N_TRIALS` trials make one episode (default target: 24).
- **Trial 0.** The starting configuration is **given for free** and is the **global-worst configuration of that landscape**. It is identical for all four agents and every episode. It is logged as `trial = 0` with `round_type = given_start`. It is *not* a choice by the agent.
- **Episode.** One complete play of `N_TRIALS` trials from trial 0. Every agent plays `N_EPISODES` episodes on the same landscape (target: 10).
- **Memory across episodes.**
  - **Agents A, B, C: none.** A fresh agent object is created every episode. It sees only the current episode's trials. (Agents A and C also get a different random seed per episode so that their random choices differ between episodes.)
  - **Agent D: yes.** One agent object is created per landscape and `run_episode()` is called once per episode. Its **Q-table is never reset** between episodes (unless you set `RL_RESET_Q_EACH_EPISODE = True`), so payoffs it experienced in earlier episodes influence its later choices.
- **Isolation.** Agents never see one another's choices or payoffs.
- **With `N_EPISODES = 1`** Agent D has nothing to remember across episodes. It is then simply an epsilon-greedy Q-learner within one 24-step run. Meaningful differences from the other agents' structure only appear for `N_EPISODES > 1`.

Recommended run sequence: (1) `N_TRIALS = 3, N_EPISODES = 1` as a smoke test; (2) `N_TRIALS = 24, N_EPISODES = 1`; (3) `N_TRIALS = 24, N_EPISODES = 10` (or more) to study Agent D's learning.

---

## 3. The four agents

All four start every episode from the same free trial-0 configuration. The three Claude-facing agents (B, C) share one system-prompt story (the alien game with Greek-letter shapes) and one answer format, differing only in the extra framing text. All code is in `agents.py`.

### Agent A — Myopic Local Search (baseline; pure code)
- **Who decides:** code only. **No Claude API calls.**
- **Memory:** none across episodes; within an episode it remembers only its current best-known configuration and payoff.
- **Behaviour:** this is Billinger et al. (2014)'s *p = 0* computational baseline, i.e. random-mutation hill-climbing.
  - Trial 1: a fully random "long jump" (each of the 10 attributes randomly ON/OFF).
  - Trials 2…N: copy the **current best-known configuration**, flip **one randomly chosen attribute**, submit it. The best-known configuration is replaced only if the new payoff is strictly higher.
- **Consequence for the data:** from trial 2 onward Agent A's search distance is **exactly 1 by construction** (it always flips one attribute of its best-known configuration). Its trial-1 distance depends on how far the random jump lands from the trial-0 configuration. It keeps testing new single flips even after it has found the global optimum, so its later configurations need not equal the optimum.
- **Why it exists:** a clean, dependency-free floor against which to measure the other agents; Billinger et al. found that human participants do not systematically outperform pure local search.

### Agent B — Free Replication (Claude)
- **Who decides:** Claude, every trial.
- **Memory:** none across episodes; within an episode it sees the **full history** of its own trials (including trial 0): which symbols were ON and the payoff, plus its **total accumulated payoff so far**.
- **Behaviour:** Claude is asked to submit its next full 10-attribute configuration as JSON (`{"config": {"alpha": 0/1, …}}`) after first explaining its reasoning in 1–3 sentences. No framing, no extra information, no restriction on how many attributes it may change. This is the plain baseline for "what does Claude do when no strategy is imposed", and mirrors Albert & Billinger's LLM version of the game.
- **Note:** the "total accumulated payoff" quoted to Claude in the prompt includes the trial-0 payoff. The `cumulative_payoff` column in the CSVs/graphs excludes trial 0 by default (§5). This only affects the number Claude is told, not the analysis.

### Agent C — Epsilon-Greedy (Claude, with code-chosen explore/exploit framing)
- **Who decides:** a **code coin flip** decides the *mode* each trial; Claude decides the *configuration*.
- **Epsilon:** `EPSILON_FOR_COMPARISON = 0.3`. With probability 0.3 the round is an **explore** round, otherwise an **exploit** round. The choice of mode is external and random — that is what makes it epsilon-greedy in the bandit/RL sense.
- **Memory:** none across episodes; within an episode it sees the full trial history, and in every round it is also told its **current best-known configuration and payoff**.
- **Framing text:**
  - *Exploit:* "try to make only a small, incremental refinement to your current best-known configuration".
  - *Explore:* "try submitting a configuration that is substantially different from your current best-known one".
- **Claude is not restricted in either mode.** It may submit any full configuration; the framing is guidance only. Whether exploit rounds actually produce small search distances and explore rounds large ones is something you **measure** in the `search_distance` and `round_type` columns, not something the code guarantees.

### Agent D — Q-learning (reinforcement-learning agent with memory; pure code)
- **Who decides:** a tabular Q-learning algorithm. **No Claude API calls.**
- **Source:** the Python code in `Q-learning-reference.pdf` (Vyakaranam, *Reinforcement Q-Learning Implementation using Python*, Section V-B), adapted line by line. The mapping from the reference's grid world to this game:

| Reference (grid world) | This project |
|---|---|
| State = grid cell | State = **(trial number, current 10-bit configuration)** = 24 × 1,024 states |
| 4 actions (down/left/up/right) | 10 actions: flip attribute *a* (a = 0…9) |
| `q = np.random.rand(n, n, 4)` | `Q = rng.rand(N_TRIALS, 1024, 10)` — random initial Q-table, indexed `Q[t, s, a]` |
| `if uniform(0,1) < eps: random action` | same, `eps = RL_EPSILON = 0.3` (reference uses 0.2) |
| exploit: first action whose q equals the max | same (first index holding the max, no random tie-break) |
| reward r = 1 at goal, 0 elsewhere, −1 for illegal moves | reward r = **payoff** of the configuration reached |
| `Del = r + gamma*np.amax(q[snew]) - q[s,ind]` | `Del = r + gamma*max(Q[snew,:]) - Q[s,a]` on trials 1…N−1 (with `Q[t+1, snew, :]`); **`Del = r - Q[t,s,a]` on the last trial (terminal)** |
| `q[s,ind] += alpha*Del` | `Q[s,a] += alpha*Del` |
| `gamma = 0.9`, `alpha = 0.9` | `alpha = 0.9` (`RL_ALPHA`); **`gamma = 1.0`** (`RL_GAMMA`) — see *Objective* below |
| `s = snew` | the agent's current configuration becomes the new configuration |
| no special terminal case in the code | **trial N (the last trial of the episode) is terminal**: no bootstrap, target = r |

- **Objective (differs from the reference):** the reference has no terminal case and uses gamma = 0.9, so it maximises an endless, discounted sum of rewards. Here the reward is the payoff reached, gamma = 1.0, and the last trial of each episode is terminal, so the quantity D learns to maximise is the **plain cumulative payoff over trials 1…N** (every payoff weighted equally; trial 0, the free start, is not a reward). This is still tabular Q-learning (same update rule, epsilon-greedy, model-free, off-policy); it is the standard episodic form with the next-state value set to 0 on the terminal step. The **trial number is part of the state** (`Q[t, s, a]`, t = trial − 1), so `Q[t,s,a]` is the exact expected sum of the payoffs still to come until the last trial: the finite-horizon cumulative-payoff objective is solved exactly, and flip/flip-back loops cannot inflate values at gamma = 1 (the same configuration at a later trial is a different table entry). The cost is a table N_TRIALS times larger, so **learning needs many more episodes** (see below). Because returns are sums of up to N payoffs, Q-values of visited entries grow well above the 0–1 random initial values of unvisited ones, which makes exploitation favour actions already tried.
- **Per step (= per trial):** from its current configuration *s*, with probability 0.3 pick a random attribute to flip (`round_type = explore`), otherwise flip the attribute with the highest Q-value in row *s* (`round_type = exploit`). Observe the payoff, update Q, move to the new configuration. **Exploration is 30% on every round, constant, with no decay.**
- **Where D differs from A, B, C in what it does:** D is a *walker*: each step it flips one attribute of its **current** configuration (not of its best-known one) and may therefore wander into worse regions after an exploratory flip. Its search distance from its best-known configuration can exceed 1.
- **Memory (the point of Agent D):** the Q-table is created once and persists across episodes, so what D learned about payoffs in earlier episodes shapes later episodes. Every episode still starts from the same free trial-0 configuration. The reference code re-randomises Q at the start of every episode; set `RL_RESET_Q_EACH_EPISODE = True` to reproduce that and obtain a **no-memory control** for D.
- **Learning is slow by nature:** the table has 24 × 1,024 × 10 = 245,760 entries and each episode makes only 24 updates, so most of the table is never visited, and values must propagate backwards from trial 24 to trial 1. Expect **no visible learning at 100 episodes**; in a quick check (10 independent agents, eps = 0.3) the mean payoff per episode was still flat at episode 100 and only started to rise by episodes ~2,000–3,000. More **episodes** (not more replications) are what reveal learning; replications only reduce noise in the average.
- **Saved artefacts:** the final Q-table is saved as `agent_d_qtable.npy` (shape 24 × 1024 × 10; first axis = trial index t = trial − 1; second axis = the configuration read as a binary number, alpha = most significant bit; last axis *a* = attribute *a* in the order alpha…kappa).

---

## 4. Reasoning capture ("think aloud")
Inspired by the think-aloud extension in Albert & Billinger's LLM study, every real Claude decision (Agent B every trial; Agent C every trial, in both explore and exploit modes) first asks Claude to explain its reasoning in 1–3 sentences before giving its final JSON. The explanation is saved in the `reasoning` column. The client retries up to 3 times if the reply cannot be parsed as JSON and raises an error if all attempts fail. Rows with no real Claude choice carry a placeholder starting with `N/A --`: the free trial-0 row for every agent, all of Agent A's moves ("hardcoded myopic local search") and all of Agent D's moves ("tabular Q-learning agent"). The sampling temperature is not set, so the API default applies.

---

## 5. Key definitions

- **Payoff** of a configuration: mean of the ten NK contributions, in [0, 1].
- **Best-known payoff / configuration (within an episode):** the highest payoff the agent has seen *so far in that episode* and the configuration that produced it (updated only on a strict improvement). Column `best_payoff_so_far`.
- **Success:** `True` when the trial's payoff is **greater than or equal to** the best-known payoff at that moment (ties count). Trial 0 is trivially `True`.
- **Search distance (Hamming):** the number of attributes (0–10) that differ between the configuration submitted at this trial and the agent's **best-known configuration immediately before this trial** (*not* the previous trial's configuration). It is undefined at trial 0 (no prior best), so the CSV stores it empty there.
  - **Display convention for graphs:** at trial 0 the graphs show instead the Hamming distance from the agent's starting configuration to the **true global optimum** (column computed in the analysis step as `search_distance_display`). This is identical for all agents and episodes (it is a property of the landscape), so at trial 0 all lines coincide. The y-value at trial 0 is a different quantity from trials ≥ 1 and should be read as "how far the start is from the best possible solution".
- **Cumulative payoff (`cumulative_payoff`):** running sum of payoffs from trial 1 up to the current trial within an episode. Trial 0 (the free start, identical for everyone) is excluded by default, so every curve starts at 0 at trial 0; set `INCLUDE_TRIAL0_IN_CUMULATIVE = True` to add its payoff instead (a constant offset for all agents).
- **Cumulative payoff as % of maximum (`cumulative_pct`, computed in the analysis step, not stored in the CSVs):** `100 × cumulative_payoff ÷ (global_optimum_payoff × trial)`. Used on the y-axis of Graph 2.
- **"Average across episodes":** for each trial number, the mean of that quantity over all episodes of that agent.

---

## 6. Files in this folder

| File | Purpose |
|---|---|
| `config.py` | **Edit this first.** All settings: API key (or environment variable), model name, N/K, number of trials and episodes, Agent C epsilon, Agent D RL parameters, cumulative-payoff convention, output folders. |
| `nk_landscape.py` | Builds the NK landscape; payoff lookup; enumeration of all 1,024 configurations; global best/worst; Hamming distance; config↔dict helpers. Can be run directly as a sanity check. |
| `claude_client.py` | Thin wrapper around the Anthropic API (text, JSON, and JSON-with-reasoning replies, with retries). Normally not edited. |
| `agents.py` | The four agents (`AgentA`, `AgentB`, `AgentC`, `AgentD`) and the shared trial-logging logic. |
| `run_experiment.py` | Runs all four agents for all episodes on **one** landscape (one K), writes the CSVs and Q-table, then triggers the per-K analysis. |
| `run_multi_k_experiment.py` | Runs `run_experiment` for every K in `K_VALUES`, then produces the cross-landscape graphs. This is the usual entry point. |
| `analyze_results.py` | Per-K graphs: Graph 2 (cumulative payoff as % of maximum, by agent), Graph 3 (search distance by agent) and Graph 7 (learning curve across episodes). Also holds the shared loading/aggregation helpers. |
| `analyze_cross_k.py` | Cross-landscape graphs: Graph 1, three-panel versions of Graphs 2, 3 and 7, the final-payoff bar chart (Graph 4), and the per-agent-across-K panels (Graphs 8 and 9). |
| `visualize_landscape_paths.py` | Optional per-K Graph 6 (attribute ON/OFF heatmaps) for one chosen episode; off by default (`SAVE_ATTRIBUTE_HEATMAPS`). |
| `analyze_reasoning.py` | Keyword-based measures on the captured reasoning text (Agents B and C). |
| `requirements.txt` | Python packages: anthropic, pandas, matplotlib, numpy. |
| `run_alien_game.lsf` | LSF batch script for the Berkeley Haas HPC (`bsub < run_alien_game.lsf`). |
| `README.md` | This file. |

---

## 7. Settings (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `API_KEY` | `""` | Anthropic API key. Leave empty and set the `ANTHROPIC_API_KEY` environment variable instead (recommended; never commit a real key). |
| `MODEL_NAME` | `claude-sonnet-4-6` | Claude model used by Agents B and C. |
| `N_ATTRIBUTES` | 10 | Number of shapes (N). Fixed at 10 in the alien game. |
| `K_COMPLEXITY` | 5 | K used only by a single `run_experiment.py` run. |
| `K_VALUES` | `[0, 5, 9]` | K values swept by `run_multi_k_experiment.py` (low / intermediate / high complexity). |
| `N_TRIALS` | 3 (target 24) | Trials per episode. |
| `N_EPISODES` | 1 (target ≥ 10) | Episodes per agent per landscape. **This is the setting to raise to study Agent D's memory.** |
| `RANDOM_SEED` | 42 | Fixes the landscapes (and seeds Agent A/C/D's random streams). |
| `EPSILON_FOR_COMPARISON` | 0.3 | Agent C's explore probability. |
| `RL_EPSILON` | 0.3 | Agent D's exploration probability, every round. |
| `RL_ALPHA` | 0.9 | Agent D learning rate (reference value). |
| `RL_GAMMA` | 1.0 | Agent D discount factor. 1.0 = all payoffs count equally; together with the terminal update on the last trial, D maximises cumulative payoff over the episode (reference value was 0.9). |
| `RL_RUN_RESET_CONTROL` | `True` | Also run a no-memory control copy of Agent D (same seed, Q re-randomised every episode; no API calls), saved as `agent_d_reset_control.csv` and drawn only in Graph 7. Skipped if `RL_RESET_Q_EACH_EPISODE` is `True`. With one episode it is identical to Agent D. |
| `RL_RESET_Q_EACH_EPISODE` | `False` | `False`: Q persists across episodes (memory). `True`: Q re-randomised each episode (reference behaviour; no-memory control). |
| `INCLUDE_TRIAL0_IN_CUMULATIVE` | `False` | Whether the free trial-0 payoff is part of the cumulative sum. |
| `PATH_EPISODE` | `None` | Which episode (1-based) Graph 6 draws; `None` = the last episode run. |
| `OUTPUT_DIR_TEMPLATE` / `FIGURE_DIR_TEMPLATE` | `outputs_N{n_trials}_E{n_episodes}` / `figures_N{n_trials}_E{n_episodes}` | Output/figure folder names, filled in automatically from the current `N_TRIALS` and `N_EPISODES` (e.g. `outputs_N3_E1`, `figures_N24_E10`), so runs with different settings never overwrite each other. Per-K subfolders (`K0`, `K5`, `K9`) are created inside. |

The command-line flags `--trials` and `--episodes` of `run_experiment.py` override `N_TRIALS` and `N_EPISODES` for a single-K run. `run_multi_k_experiment.py` always uses `config.py`.

---

## 8. How to run

1. **Install packages** (once, inside your conda environment): `pip install -r requirements.txt`
2. **Provide the API key.** Either `export ANTHROPIC_API_KEY="..."` (e.g. in `~/.bashrc` or the `.lsf` script) or paste it into `API_KEY` in `config.py`. Prefer the environment variable.
3. **Set trials and episodes** in `config.py` (see the sequence in §2).
4. **Run.**
   - Full sweep over K = 0, 5, 9, all graphs: `python run_multi_k_experiment.py`
   - Single landscape: `python run_experiment.py` or e.g. `python run_experiment.py --K 5 --trials 24 --episodes 10 --seed 42` (produces the per-K graphs only; the cross-landscape graphs need all K values).
   - HPC: `bsub < run_alien_game.lsf` (the script `cd`s into the project folder, activates the conda environment, and runs the full sweep; walltime is 16 h).
5. **Regenerate graphs without re-running the experiment** (reads the existing CSVs; first set `N_TRIALS` and `N_EPISODES` in `config.py` to the values used for that run, because they determine which folder is read):
   ```bash
   python analyze_results.py            # Graphs 2-3 for config.K_COMPLEXITY (default output folders)
   python visualize_landscape_paths.py  # Graph 6 (only if SAVE_ATTRIBUTE_HEATMAPS=True)
   python analyze_reasoning.py          # reasoning_analysis.csv
   python analyze_cross_k.py            # Graph 1, cross-landscape Graphs 2-3, bar chart
   ```
6. **Important:** CSVs for a given K are written only **after all episodes of that K have finished**. If a job dies mid-K (e.g. an API failure after retries, or walltime), that K's results are lost and must be re-run. Test with small `N_TRIALS`/`N_EPISODES` first.

---

## 9. Outputs: folder layout and CSV data dictionary

Folder names depend on the run settings; below, `N24_E10` stands for `N_TRIALS=24, N_EPISODES=10` (a 3-trial, 1-episode test gives `outputs_N3_E1/` and `figures_N3_E1/`).
```
outputs_N24_E10/
  K0/  K5/  K9/
    agent_a.csv                       Agent A, all episodes
    agent_b.csv                       Agent B, all episodes
    agent_c_eps0.3.csv                Agent C, all episodes
    agent_d.csv                       Agent D, all episodes
    agent_d_reset_control.csv         no-memory control copy of Agent D (Q reset each episode)
    agent_d_qtable.npy                Agent D's final Q-table (1024 x 10)
    best_and_worst_outcome_summary.csv
    reasoning_analysis.csv            (Agents B and C)
figures_N24_E10/
  K0/  K5/  K9/                       per-landscape graphs (same files in each)
    fig2_cumulative_payoff_by_agent.png
    fig2b_cumulative_payoff_raw_by_agent.png
    fig3_search_distance_by_agent.png
    fig5_landscape_paths.png
    fig6_attribute_flip_heatmaps.png
    fig7_learning_curve_across_episodes.png
  common_figures/                     graphs comparing all K values (belong to no single K)
    fig1_cumulative_payoff_across_landscapes.png
    fig2_cumulative_payoff_across_landscapes.png
    fig2b_cumulative_payoff_raw_across_landscapes.png
    fig3_search_distance_across_landscapes.png
    fig4_final_cumulative_payoff_bars.png
    fig7_learning_curve_across_landscapes.png
    fig8_per_agent_cumulative_payoff_across_landscapes.png
    fig8b_per_agent_cumulative_payoff_raw_across_landscapes.png
    fig9_per_agent_search_distance_across_landscapes.png
```

### Agent CSVs (`agent_a.csv`, `agent_b.csv`, `agent_c_eps0.3.csv`, `agent_d.csv`)
One row per **episode × trial** (including trial 0 of each episode). Columns:

| Column | Meaning |
|---|---|
| `agent` | Agent identifier: `A_myopic_local_search`, `B_free_replication`, `C_epsilon_greedy`, `D_q_learning`. |
| `epsilon` | Agent C's / D's epsilon for that row; empty for A and B. |
| `episode` | Episode number, starting at 1. |
| `trial` | Trial number within the episode; 0 = free starting configuration. |
| `config` | The submitted configuration as a tuple of ten 0/1 values, in the order alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa. |
| `payoff` | Payoff received (0–1). |
| `best_payoff_so_far` | Best payoff seen so far in this episode (including this trial). |
| `cumulative_payoff` | Running sum of payoffs within the episode (convention in §5). |
| `search_distance` | Hamming distance to the best-known configuration *before* this trial; empty at trial 0. |
| `round_type` | `given_start` (trial 0); A: `long_jump` (trial 1) / `myopic_local_move`; B: `own_choice`; C: `explore` / `exploit`; D: `explore` / `exploit`. |
| `success` | `True` if payoff ≥ best-known payoff at that moment (ties count). |
| `reasoning` | Claude's brief explanation (B, C), or an `N/A -- …` placeholder (trial 0, A, D). |

Agent D's CSV has three extra columns (empty at trial 0):

| Column | Meaning |
|---|---|
| `flipped_attribute` | Name of the attribute D flipped this step. |
| `td_error` | The update term `Del = r + gamma·max Q[t+1,snew,:] − Q[t,s,a]` computed this step (on the last trial of an episode, `Del = r − Q[t,s,a]`: terminal, no bootstrap). Large positive values early on mean the payoff exceeded D's expectation. |
| `q_value_after` | `Q[s,a]` after the update. |

### `best_and_worst_outcome_summary.csv`
One reference row per K: the single highest-payoff outcome found by any agent in any episode (`agent`, `episode`, `trial`, `config`, `payoff`), alongside the true `global_optimum_config` / `global_optimum_payoff` and `global_worst_config` / `global_worst_payoff` of the landscape. The global worst configuration is also the free trial-0 start for every agent.

### `reasoning_analysis.csv`
One row per trial with real Claude reasoning: `agent, epsilon, episode, trial, round_type, attention_breadth, forward_chars, backward_chars, forward_looking_ratio, reasoning` (see §11).

---

## 10. The graphs and how to read them

All curves are **means across episodes** (number of episodes shown in the axis label). With `N_EPISODES = 1` each curve is a single run and carries no information about variability; the bar chart's error bars are then zero.

### Graph 1 — Average cumulative payoff across all agents (A–D), by landscape (`figures_N.._E../common_figures/fig1_cumulative_payoff_across_landscapes.png`)
- **X-axis:** trial number. **Y-axis:** average cumulative payoff up to that trial, averaged **across the four agents and all episodes**. **Lines:** one per K (0, 5, 9).
- **What it is:** for each landscape (K), take every agent's cumulative payoff at each trial in each episode, then average over all four agents and all episodes. It starts at 0 at trial 0 because the free start is excluded. The value at trial *t* is therefore the average total earned over the first *t* trials by a typical agent on that landscape.
- **Reading it:** higher lines = more accumulated earnings; steeper lines = faster accumulation per trial. The slope at a given trial is approximately the average payoff earned per trial around that point (e.g. a value of 0.48 at trial 1 means the average first-trial payoff was 0.48).
- **Caution:** it pools four very different agents into one line per K, so it describes "what the average of these agents earns on each landscape", not any single agent. Use Graph 2 for agent comparisons. Raw payoff levels are also not strictly comparable across K (§13).

### Graph 2 — Compare performance across individuals (`figures_N.._E../K{k}/fig2_cumulative_payoff_by_agent.png`; three-panel version `figures_N.._E../common_figures/fig2_cumulative_payoff_across_landscapes.png`)
- **X-axis:** trial number (integers). **Y-axis:** each agent's cumulative payoff up to that trial, expressed as a **percentage of the maximum achievable cumulative payoff** and averaged across its episodes. The maximum achievable cumulative payoff at trial *t* is `global_optimum_payoff × t` (what an agent would have earned by submitting the true global optimum on every trial so far; plus the fixed trial-0 payoff if `INCLUDE_TRIAL0_IN_CUMULATIVE = True`). So 100% = playing the best possible configuration every trial. **Lines:** one per agent (A, B, C, D) within the same landscape.
- **Trial 0 is plotted:** the ratio is undefined at trial 0 (nothing earned yet), so there the graph shows the free starting configuration's payoff as a % of the optimum payoff, i.e. where every agent starts (identical for all agents; it is the global-worst configuration, so it is low). This is a different quantity from trials ≥ 1, the same kind of convention as Graph 3's trial 0. The jump from trial 0 to trial 1 shows the effect of the first real choice.
- **Reading it:** the highest line shows who has earned the largest share of the maximum so far. Mathematically this equals the agent's *average payoff per trial so far* divided by the optimum payoff, so it tends to flatten over time rather than keep rising (the raw `cumulative_payoff` column in the CSVs still holds the un-normalised running sum). An agent reaching a given percentage at an earlier trial is accumulating faster in rounds, not in wall-clock time. The three-panel version puts K = 0, 5, 9 side by side with a shared y-axis.

### Graph 2b — Cumulative payoff in actual units (`figures_N.._E../K{k}/fig2b_cumulative_payoff_raw_by_agent.png`; three-panel version `figures_N.._E../common_figures/fig2b_cumulative_payoff_raw_across_landscapes.png`)
The same layout as Graph 2 but the y-axis is the **actual cumulative payoff** (sum of payoffs from trial 1 up to that trial, mean across episodes), not a percentage. It is 0 at trial 0 (no choice made yet; the free start is excluded by default, see `INCLUDE_TRIAL0_IN_CUMULATIVE`). Lines rise over time; the slope at a trial is roughly the average payoff earned per trial around that point. Because raw payoffs depend on each landscape's draw, compare agents within a panel rather than levels across K.

### Graph 3 — Average search distance per trial (`figures_N.._E../K{k}/fig3_search_distance_by_agent.png`; three-panel version `figures_N.._E../common_figures/fig3_search_distance_across_landscapes.png`)
- **X-axis:** trial number. **Y-axis:** mean Hamming distance from the agent's previous best-known configuration (0–10), averaged across episodes. **Lines:** one per agent. **Trial 0** shows the distance from the start to the true global optimum (§5), common to all agents.
- **Reading it:** high values = long jumps (exploration/"wide" search); values near 1 = single-attribute local moves. Agent A should sit at exactly 1 from trial 2 on by construction. For B, C and D the pattern is measured, not imposed. Billinger et al. (2014) use the same quantity (their Figures 2 and 4).

### Graph 4 — Final cumulative payoff (`figures_N.._E../common_figures/fig4_final_cumulative_payoff_bars.png`)
Grouped bars: for each K, one bar per agent giving the cumulative payoff after the last trial, mean across episodes with ±1 standard error (sample SD ÷ √episodes) as error bars. This replaces the previous version's "final best-known payoff" bar chart (Graph 7) and mirrors the layout of Billinger et al. (2014, Table 1).

### Graph 7 — Learning across episodes (`figures_N.._E../K{k}/fig7_learning_curve_across_episodes.png`; three-panel version `figures_N.._E../common_figures/fig7_learning_curve_across_landscapes.png`)
- **X-axis:** episode number. **Y-axis:** that episode's total payoff, i.e. its final cumulative payoff as a % of the maximum possible (same unit as Graph 2). **Lines:** agents A, B, C, D, plus a grey dashed **control copy of Agent D** whose Q-table is re-randomised every episode (same seed; no memory).
- **Reading it:** Agents A, B and C have no memory, so their lines should fluctuate around a flat level. If memory helps Agent D, its line should trend upward over episodes and sit above its own reset control. At episode 1 Agent D and its control are identical by construction. With `N_EPISODES = 1` each line is a single point; the graph needs several episodes to be informative. Each point is a single run, so individual episodes are noisy; look at the trend over many episodes rather than any one point.

### Graph 8 — Cumulative payoff by agent across landscapes (`figures_N.._E../common_figures/fig8_per_agent_cumulative_payoff_across_landscapes.png`)
2×2 panels, **one panel per agent**, with one line per K (0, 5, 9). X = trial number; Y = cumulative payoff as % of the maximum possible (mean across episodes, from trial 1). It shows how each agent's performance changes with landscape ruggedness. Note the y-axis scale differs by panel. Percentages are relative to each landscape's own optimum, which makes them more comparable across K than raw payoffs (but see §13).

### Graph 8b — Per-agent cumulative payoff across landscapes, actual units (`figures_N.._E../common_figures/fig8b_per_agent_cumulative_payoff_raw_across_landscapes.png`)
Same 2×2 layout as Graph 8 (one panel per agent, lines = K) with the y-axis in actual cumulative payoff units. Trial 0 is plotted at 0.

### Graph 9 — Search distance by agent across landscapes (`figures_N.._E../common_figures/fig9_per_agent_search_distance_across_landscapes.png`)
Same 2×2 layout as Graph 8 for mean search distance vs trial (trial 0 = distance from the start to the true global optimum). Follows the visual style of Billinger et al. (2014, Figure 4).

### Graph 6 — Attribute heatmaps (`figures_N.._E../K{k}/fig6_attribute_flip_heatmaps.png`)
For each agent (episode set by `PATH_EPISODE`): rows = the ten attributes, columns = trials, black = ON, white = OFF. Shows which attributes were switched when, and how stable or erratic each agent's configuration is.

---

## 11. Reasoning analysis
`analyze_reasoning.py` computes two lightweight, **keyword-based** measures per trial from Claude's reasoning text (Agents B and C only; A and D are skipped automatically because they have placeholder text):
- **`attention_breadth`:** how many distinct attribute names (alpha … kappa) are mentioned in the reasoning.
- **`forward_looking_ratio`:** each sentence is labelled forward-looking, backward-looking or neither from keyword lists; the ratio is (characters in forward sentences) ÷ (characters in backward sentences), empty when there are no backward sentences.

These are heuristic proxies inspired by the measures in Albert & Billinger's preprint, **not** a validated replication of their classification. Treat them as exploratory. The script prints and saves a per-agent summary (mean attention breadth, mean forward-looking ratio).

---

## 12. API usage, runtime and reproducibility
- **Claude calls:** only Agents B and C call the API: `N_TRIALS × N_EPISODES` calls each per K, i.e. `2 × N_TRIALS × N_EPISODES × len(K_VALUES)` in total. At 24 × 10 × 3 K values that is 1,440 calls. Agents A and D make no calls. Each call may be retried up to 3 times on unparseable replies.
- **Runtime:** dominated by API latency (the calls are sequential). Estimate a single call's time from your 3-trial test and scale; the LSF walltime is set to 16 h to be safe.
- **Reproducibility:** the landscapes, Agent A's and C's random streams and Agent D's initial Q-table and random choices are seeded (`RANDOM_SEED`, with per-episode offsets for A and C), so re-running with the same settings reproduces them. **Agents B and C's Claude responses are not guaranteed to repeat** (sampling temperature is the API default), so their results will vary from run to run.

---

## 13. Interpretation cautions
- **Few episodes ≠ distribution.** `N_TRIALS` controls how long one simulated session is; it does not give you replication. A single episode (or a handful) is a pilot, not a basis for claims about behavioural patterns or about which strategy is better. Distributional claims need many independent episodes with averages and error bands (Albert & Billinger use on the order of 69 per condition).
- **Not directly comparable to the source papers.** The landscapes here come from a different random generator and seed than the papers' original landscapes, and episode counts differ, so numerical results cannot be compared one-to-one with Billinger et al. (2014, 2021) or Albert & Billinger (2024). Be precise about which paper's task you are comparing to: the 2014 paper is pure payoff-maximisation search, the 2021 paper adds a wealth-maximisation incentive and a stop/continue decision.
- **Agents differ in information and mechanics, not only in strategy.** A and D are pure code; B and C are language models with full within-episode history. D moves from its *current* configuration, A from its *best-known* one, and B/C choose freely. Differences in outcomes reflect all of this together.
- **Agent D with 1 episode** has no cross-episode memory; compare D against the `RL_RESET_Q_EACH_EPISODE = True` control to isolate the effect of memory once you run multiple episodes.
- **Raw payoffs across K.** Landscapes with different K are different random functions (different table sizes and partner structures); higher or lower payoffs at one K versus another partly reflect the landscape draw. Within-K comparisons across agents are the safest.
- **Search distance is measured from the best-known configuration**, not from the previous trial's configuration; analyses must use the same definition.
- **Contamination.** The alien-game framing may be familiar to language models; the Greek-letter labels for the shapes serve as a robustness check, not a guarantee.
- **Reasoning measures are heuristics** (§11).

---

## 14. What changed relative to the previous (3-agent) version
- **Episodes added:** every agent now plays `N_EPISODES` episodes of `N_TRIALS` trials per landscape; CSVs gained `episode` and `cumulative_payoff` columns.
- **Agent D replaced:** the streetlight agent (Claude shown one reference configuration) is gone; Agent D is now the Q-learning agent described above. (The streetlight design was dropped because an isolated single-agent setup cannot capture the multi-agent free-riding mechanism that effect depends on.)
- **Agent C** now runs at a single epsilon (0.3); the three-epsilon sweep and its comparison graphs were removed.
- **Added later:** Graph 7 (learning across episodes, with a no-memory Agent D control), Graphs 8 and 9 (per-agent panels with lines for K).
- **Graphs:** "best-known payoff vs. trial" replaced by cumulative payoff (new Graphs 1–2); search distance is now averaged across episodes (Graph 3); the old Graph 7 bar chart became the final-cumulative-payoff bar chart (Graph 4); the old Graph 8 is replaced by the three-panel cross-landscape versions; Graphs 5–6 now draw a single chosen episode.
- **Agents A–C logic is unchanged**, apart from fresh per-episode instances and per-episode random seeds for A and C.

---

## 15. Troubleshooting
- **"No API key found"** — set `ANTHROPIC_API_KEY` or `API_KEY` in `config.py`.
- **"Claude did not return valid JSON … after 3 attempts"** — transient model-output problem; rerun (consider re-running only the affected K with `run_experiment.py --K …`).
- **Missing-file warnings / empty graphs** — a K's CSVs are only written when that K finishes all episodes; check `outputs/K{k}/`.
- **Graph 6 skipped** — a dependency is missing (`pip install -r requirements.txt`).
- **Lines overlap** — with identical choices two agents' lines coincide; each agent has its own colour, linestyle and marker so overlaps remain visible.
- **Everything is flat or identical with `N_EPISODES = 1`** — expected for error bars (zero SE) and for Agent D's memory (nothing to remember).

---

## 16. References
- Billinger, S., Stieglitz, N., & Schumacher, T. R. (2014). Search on rugged landscapes: An experimental study. *Organization Science*.
- Billinger, S., et al. (2021). Exploration and exploitation in complex search tasks: How feedback influences whether and where human agents search. *Strategic Management Journal*.
- Albert, M., & Billinger, S. (2024/2025). LLM replication of the alien game (preprint).
- Hoelzemann, J., Manso, G., Nagaraj, A., & Tranchero, M. (2024). Streetlight effect experiment (basis of the earlier Agent D design, now removed).
- Vyakaranam, S. A. *Reinforcement Q-Learning Implementation using Python* (`Q-learning-reference.pdf`), the template for Agent D.
- Kauffman, S. A. — NK fitness landscape model.
