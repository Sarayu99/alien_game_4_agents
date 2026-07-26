"""
config.py
==========
Central configuration file for the Alien Game NK-landscape experiment.

Everything you might want to change before running an experiment lives here:
- Your Anthropic API key
- Which Claude model to call
- The size and ruggedness of the landscape (N, K)
- How many trials each agent gets
- The epsilon values used by Agent B (epsilon-greedy)
- Output folder locations

You should NOT need to edit any other file to change these settings.
"""

import os

# ---------------------------------------------------------------------------
# 1. ANTHROPIC API KEY
# ---------------------------------------------------------------------------
# For security, this file does NOT contain a hardcoded API key -- it's safe
# to commit and push to a public GitHub repo as-is.
#
# Instead, set your key as an environment variable before running any script,
# e.g. in your shell or in your .lsf script:
#     export ANTHROPIC_API_KEY="sk-ant-..."
# or via a separate untracked file such as ~/.anthropic_env that you source
# before running (make sure that file is listed in .gitignore).
#
# If you really want to hardcode a key locally for convenience, you can paste
# it into the string below -- but if you do, make sure this file is added to
# .gitignore BEFORE committing, so the key never ends up in git history.
API_KEY = ""  # <-- optional: paste your API key here for local use only (do not commit if filled in)


def get_api_key():
    """Return the API key to use: the hardcoded value above takes priority;
    otherwise falls back to the ANTHROPIC_API_KEY environment variable."""
    if API_KEY:
        return API_KEY
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not env_key:
        raise ValueError(
            "No API key found. Either paste your key into config.py "
            "(the API_KEY variable, for local use only -- do not commit it) "
            "or set the ANTHROPIC_API_KEY environment variable before "
            "running the script."
        )
    return env_key


# ---------------------------------------------------------------------------
# 2. CLAUDE MODEL
# ---------------------------------------------------------------------------
MODEL_NAME = "claude-sonnet-4-6"  # change this string to use a different model

# ---------------------------------------------------------------------------
# 3. NK LANDSCAPE PARAMETERS (the "alien game" rules)
# ---------------------------------------------------------------------------
N_ATTRIBUTES = 10       # number of geometric shapes / Greek-letter symbols (fixed at 10 in the alien game)
K_COMPLEXITY = 5        # ruggedness parameter used for a SINGLE run_experiment.py call: 0 = smooth, 5 = intermediate, 9 = maximally rugged
N_TRIALS = 3            # number of search trials per agent (TEMPORARILY lowered to 3 for testing; normally 24, matching Billinger et al. 2014)
RANDOM_SEED = 42        # fixes the landscape so results are reproducible across runs

# K values swept by run_multi_k_experiment.py, matching Billinger et al.
# (2014)'s three complexity levels exactly: low (0), intermediate (5), high (9).
K_VALUES = [0, 5, 9]

# ---------------------------------------------------------------------------
# 4. AGENT C (EPSILON-GREEDY) SETTINGS
# ---------------------------------------------------------------------------
# Each round, Agent C's code flips a coin: with probability EPSILON, Claude
# is told this is an "explore" round (framed as: try something substantially
# different); otherwise it's an "exploit" round (framed as: make a small
# refinement). In both cases Claude proposes a full, unconstrained
# configuration -- the framing is guidance only, not a code-enforced rule,
# so whether exploit rounds actually come out smaller than explore rounds is
# something you measure in the resulting search_distance column, not
# something guaranteed by this file.
EPSILON_VALUES = [0.1, 0.3, 0.5]   # the three epsilon values used for Graph 2 and Graph 4
EPSILON_FOR_COMPARISON = 0.3       # the single epsilon value used for Graph 1 and Graph 3 (the 4-agent comparison)

# ---------------------------------------------------------------------------
# 5. AGENT D (STREETLIGHT) SETTINGS
# ---------------------------------------------------------------------------
# Agent D is given, as reference information alongside its own full trial
# history, the value of one "attractive but not optimal" configuration (the
# anchor) -- but is NOT restricted to moving near it; it proposes a full,
# unconstrained configuration every round, same as Agents B and C. Only the
# anchor SELECTION uses these percentile bounds; nothing here caps how far
# Claude's own choices are allowed to be from that anchor.
STREETLIGHT_PERCENTILE_LOW = 0.60    # anchor point must be at/above this fitness percentile...
STREETLIGHT_PERCENTILE_HIGH = 0.70   # ...and below this percentile (attractive, but not the global optimum)

# ---------------------------------------------------------------------------
# 6. OUTPUT LOCATIONS
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"   # folder where CSV trial logs are saved (single-K run default)
FIGURE_DIR = "figures"   # folder where the output graphs (PNG) are saved (single-K run default)


def output_dir_for_k(k):
    """Per-K subfolder used by run_multi_k_experiment.py, e.g. 'outputs/K5'.
    Keeps each complexity level's CSVs fully separate."""
    return os.path.join(OUTPUT_DIR, f"K{k}")


def figure_dir_for_k(k):
    """Per-K subfolder for figures, e.g. 'figures/K5'."""
    return os.path.join(FIGURE_DIR, f"K{k}")
