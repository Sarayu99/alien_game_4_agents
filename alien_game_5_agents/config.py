"""
config.py
==========
Central configuration for the 4-agent, multi-episode Alien Game experiment.

Design (new in this version)
----------------------------
- Each K value has ONE fixed NK landscape (fixed by RANDOM_SEED).
- Every agent plays N_EPISODES episodes on that landscape; each episode
  is N_TRIALS trials long.
- Agents A, B, C are rebuilt from scratch every episode: NO memory of
  previous episodes.
- Agent D is a tabular Q-learning agent. Its Q-table PERSISTS across
  episodes (that is its memory of past payoffs).
"""

import os

# ---------------------------------------------------------------------------
# 1. ANTHROPIC API KEY
# ---------------------------------------------------------------------------
# Paste your key here, or leave "" and export ANTHROPIC_API_KEY instead.
API_KEY = ""


def get_api_key():
    if API_KEY:
        return API_KEY
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not env_key:
        raise ValueError(
            "No API key found. Paste it into config.py (API_KEY) or set the "
            "ANTHROPIC_API_KEY environment variable."
        )
    return env_key


# ---------------------------------------------------------------------------
# 2. CLAUDE MODEL
# ---------------------------------------------------------------------------
MODEL_NAME = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# 3. NK LANDSCAPE + EPISODE STRUCTURE
# ---------------------------------------------------------------------------
N_ATTRIBUTES = 10
K_COMPLEXITY = 5          # used only by a single run_experiment.py call
N_TRIALS = 3              # trials per episode (planned: run 3 first, then 24)
N_EPISODES = 1            # episodes per agent per landscape (planned: 1 first, then raise it;
                          # Agent D's memory only matters when this is > 1)
RANDOM_SEED = 42
K_VALUES = [0, 5, 9]

# ---------------------------------------------------------------------------
# 4. AGENT C (EPSILON-GREEDY, unchanged logic)
# ---------------------------------------------------------------------------
EPSILON_FOR_COMPARISON = 0.3   # Agent C's epsilon (single value now)

# ---------------------------------------------------------------------------
# 5. AGENT D (Q-LEARNING, adapted from Q-learning-reference.pdf)
# ---------------------------------------------------------------------------
# State  = current 10-bit configuration (2^N states)
# Action = flip one attribute (N actions)
# Reward = payoff of the configuration reached
# Q-table starts as np.random.rand(...) exactly like the reference.
RL_EPSILON = 0.3     # 30% exploration EVERY round (reference uses 0.2)
RL_ALPHA = 0.9       # learning rate   (reference value)
RL_GAMMA = 1.0       # discount factor. 1.0 = every payoff counts equally, so with the
                     # terminal update on the last trial (agents.py) Agent D maximises the
                     # CUMULATIVE payoff over the episode (reference code used 0.9)
# False = Q persists across episodes (Agent D's memory; what you asked for).
# True  = Q re-randomised every episode, as the reference code literally does
#         (useful as a no-memory control for Agent D).
RL_RESET_Q_EACH_EPISODE = False
# Also run a CONTROL copy of Agent D whose Q-table is re-randomised every
# episode (reference behaviour, no cross-episode memory), using the same seed.
# It is drawn only in the learning-curve graph (Graph 7) and costs no API
# calls. Skipped automatically if RL_RESET_Q_EACH_EPISODE is already True.
RL_RUN_RESET_CONTROL = True

# ---------------------------------------------------------------------------
# 6. CUMULATIVE PAYOFF DEFINITION
# ---------------------------------------------------------------------------
# The trial-0 starting configuration is given for free and is identical for
# all agents. False = cumulative payoff sums trials 1..N_TRIALS only
# (starts at 0 at trial 0). True = also adds the trial-0 payoff.
INCLUDE_TRIAL0_IN_CUMULATIVE = False

# ---------------------------------------------------------------------------
# 7. PATH / HEATMAP FIGURES (Graphs 5-6)
# ---------------------------------------------------------------------------
PATH_EPISODE = None   # episode (1-based) drawn in Graphs 5-6; None = the last episode run

# ---------------------------------------------------------------------------
# 8. OUTPUT LOCATIONS
# ---------------------------------------------------------------------------
# Folder names are built automatically from the CURRENT N_TRIALS / N_EPISODES,
# e.g. N_TRIALS=3, N_EPISODES=1 -> outputs_N3_E1/ and figures_N3_E1/
# so runs with different settings never overwrite each other.
OUTPUT_DIR_TEMPLATE = "outputs_N{n_trials}_E{n_episodes}"
FIGURE_DIR_TEMPLATE = "figures_N{n_trials}_E{n_episodes}"


def output_root():
    return OUTPUT_DIR_TEMPLATE.format(n_trials=N_TRIALS, n_episodes=N_EPISODES)


def figure_root():
    return FIGURE_DIR_TEMPLATE.format(n_trials=N_TRIALS, n_episodes=N_EPISODES)


def common_figure_dir():
    """Graphs that compare all K values live here, e.g. figures_N24_E10/common_figures"""
    return os.path.join(figure_root(), "common_figures")


def output_dir_for_k(k):
    """e.g. outputs_N24_E10/K5"""
    return os.path.join(output_root(), f"K{k}")


def figure_dir_for_k(k):
    """e.g. figures_N24_E10/K5"""
    return os.path.join(figure_root(), f"K{k}")
