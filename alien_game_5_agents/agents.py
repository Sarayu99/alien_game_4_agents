"""
agents.py
=========
Four search agents, all playing the SAME NK landscape in isolation, over
multiple EPISODES (each episode = config.N_TRIALS trials):

  Agent A -- Myopic Local Search      (pure code, NO memory across episodes)
  Agent B -- Free Replication         (Claude, NO memory across episodes)
  Agent C -- Epsilon-Greedy           (Claude + code-chosen explore/exploit
                                       framing, NO memory across episodes)
  Agent D -- Q-Learning (RL)          (pure code, tabular Q-learning with
                                       epsilon = 0.3 every round; its
                                       Q-table PERSISTS across episodes)

A, B and C are re-instantiated at the start of every episode, so they have
no recollection of earlier episodes' payoffs. Agent D is instantiated ONCE
per landscape and run_episode() is called repeatedly, so what it learned
about payoffs in earlier episodes is carried forward in its Q-table.
"""

import random

import numpy as np

import config
import progress
from nk_landscape import NKLandscape, ATTRIBUTE_NAMES
from claude_client import ClaudeClient


# Placeholder reasoning text used whenever no real Claude choice was made
# on a given trial (the free starting configuration, or any of Agent A's
# fully code-driven moves). analyze_reasoning.py skips rows whose
# reasoning text starts with "N/A --".
NO_CHOICE_MADE = "N/A -- starting configuration given for free (no choice made)."
NO_API_CALL_MYOPIC = (
    "N/A -- hardcoded myopic local search (Billinger et al. 2014's p=0 "
    "baseline): no Claude API call, no memory beyond the current "
    "best-known configuration."
)

NO_API_CALL_QLEARNING = (
    "N/A -- tabular Q-learning agent (epsilon-greedy over a Q-table): no "
    "Claude API call."
)

CONFIG_JSON_INSTRUCTIONS = (
    "Reply with a JSON object with a single key 'config', whose value is "
    "an object mapping each of the 10 symbol names (alpha, beta, gamma, "
    "delta, epsilon, zeta, eta, theta, iota, kappa) to either 0 (OFF) or "
    "1 (ON). Example: "
    '{"config": {"alpha": 1, "beta": 0, "gamma": 1, "delta": 0, '
    '"epsilon": 1, "zeta": 0, "eta": 1, "theta": 0, "iota": 1, "kappa": 0}}'
)


class SearchAgent:
    agent_name = "base"

    def __init__(self, landscape: NKLandscape, claude: ClaudeClient, n_trials=None, episode=1):
        self.landscape = landscape
        self.claude = claude
        self.n_trials = n_trials or config.N_TRIALS
        self.start_episode(episode)

    def start_episode(self, episode):
        """Reset all WITHIN-episode state (history, best-known, cumulative
        payoff). Anything an agent wants to remember across episodes (only
        Agent D's Q-table) lives outside these fields and is untouched."""
        self.episode = episode
        self.history = []
        self.best_config = None
        self.best_payoff = None
        self._cumulative = 0.0

    def _update_best(self, config_tuple, payoff):
        is_success = self.best_payoff is None or payoff >= self.best_payoff
        if self.best_payoff is None or payoff > self.best_payoff:
            self.best_config = config_tuple
            self.best_payoff = payoff
        return is_success

    def _process_trial(self, trial_number, config_tuple, round_type, epsilon=None,
                        reasoning=None, extra_fields=None):
        payoff = self.landscape.get_payoff(config_tuple)

        if self.best_config is not None:
            search_distance = NKLandscape.hamming_distance(config_tuple, self.best_config)
        else:
            search_distance = None

        is_success = self._update_best(config_tuple, payoff)

        if trial_number > 0 or config.INCLUDE_TRIAL0_IN_CUMULATIVE:
            self._cumulative += payoff

        row = {
            "agent": self.agent_name,
            "epsilon": epsilon,
            "episode": self.episode,
            "trial": trial_number,
            "config": config_tuple,
            "payoff": payoff,
            "best_payoff_so_far": self.best_payoff,
            "cumulative_payoff": self._cumulative,
            "search_distance": search_distance,
            "round_type": round_type,
            "success": is_success,
            "reasoning": reasoning,
        }
        if extra_fields:
            row.update(extra_fields)
        self.history.append(row)
        return payoff

    def _history_text(self):
        """Full trial-by-trial history as plain text, used by every
        Claude-driven agent (B, C, D), so that all three reason over the
        same kind of information -- differences in behavior reflect the
        framing/context each one is given, not unequal information."""
        if not self.history:
            return "No trials have been played yet."

        lines = []
        for row in self.history:
            config_dict = NKLandscape.config_to_dict(row["config"])
            symbols_on = [name for name, state in config_dict.items() if state == 1]
            lines.append(
                f"Trial {row['trial']}: symbols ON = {symbols_on}, "
                f"payoff = {row['payoff']:.4f}"
            )
        return "\n".join(lines)

    def total_wealth(self):
        return sum(row["payoff"] for row in self.history)

    def _free_choice_move(self, system_prompt, extra_context=""):
        """
        Shared 'propose a full, unconstrained configuration' move used by
        Agents B, C, and D alike. extra_context is an optional block of
        additional text inserted into the prompt (e.g. Agent C's
        explore/exploit framing, or Agent D's anchor reference) -- it
        never restricts what Claude is allowed to submit, only what
        information/instruction it's given going in.

        Returns (new_config, reasoning_text).
        """
        history_text = self._history_text()
        context_block = f"{extra_context}\n\n" if extra_context else ""
        user_prompt = (
            f"Here is the full history of your trials so far:\n\n{history_text}\n\n"
            f"Your total accumulated payoff so far is {self.total_wealth():.4f}.\n\n"
            f"{context_block}"
            "Considering what you know so far, please submit your next trial "
            f"combination. {CONFIG_JSON_INSTRUCTIONS}"
        )
        reply, reasoning_text = self.claude.ask_json_with_reasoning(system_prompt, user_prompt)
        config_dict = reply.get("config", {})
        new_config = NKLandscape.dict_to_config(config_dict)
        return new_config, reasoning_text


# =============================================================================
# Agent A -- Myopic Local Search (the baseline)
# =============================================================================
class AgentA(SearchAgent):
    """Billinger et al. (2014)'s p=0 computational baseline: a single
    random attribute flip per trial, kept only if it improves on the
    current best-known configuration, preceded by one fully random "long
    jump" on trial 1. Pure code -- no Claude API calls anywhere, and the
    ONLY agent in this project with no history access, by design."""

    agent_name = "A_myopic_local_search"

    def __init__(self, landscape, claude=None, n_trials=None, rng_seed=None, episode=1):
        super().__init__(landscape, claude, n_trials, episode)
        self._rng = random.Random(rng_seed)

    def run(self):
        start_config, _ = self.landscape.get_lowest_performing_configuration()
        self._process_trial(0, start_config, round_type="given_start", reasoning=NO_CHOICE_MADE)

        for trial_number in range(1, self.n_trials + 1):
            if trial_number == 1:
                new_config = tuple(self._rng.randint(0, 1) for _ in range(self.landscape.N))
                round_type = "long_jump"
            else:
                new_config = list(self.best_config)
                idx = self._rng.randrange(self.landscape.N)
                new_config[idx] = 1 - new_config[idx]
                new_config = tuple(new_config)
                round_type = "myopic_local_move"

            self._process_trial(trial_number, new_config, round_type=round_type,
                                 reasoning=NO_API_CALL_MYOPIC)

        return self.history


# =============================================================================
# Agent B -- Free Replication
# =============================================================================
class AgentB(SearchAgent):
    """Direct replication of Albert & Billinger's LLM alien-game study:
    Claude sees the full trial history and freely decides its next
    configuration every round. No framing, no extra reference info,
    no search-distance rule enforced."""

    agent_name = "B_free_replication"

    SYSTEM_PROMPT = (
        "You are taking part in a game. You have made contact with an alien "
        "from a distant planet who is interested in buying art pictures. "
        "An art picture is made up of 10 distinct geometric shapes, each of "
        "which you can switch ON or OFF. We refer to the 10 shapes using "
        "the Greek letters alpha, beta, gamma, delta, epsilon, zeta, eta, "
        "theta, iota, and kappa. You do not know in advance which "
        "combination of shapes the alien prefers -- you only find out the "
        "payoff (how much the alien pays) after you submit a combination. "
        "Your goal is to maximize your total accumulated payoff across all "
        "of your trials. You will play a fixed number of trials in total. "
        "You will be shown the full history of your own past trials and "
        "their payoffs before each choice."
    )

    def __init__(self, landscape, claude, n_trials=None, episode=1):
        super().__init__(landscape, claude, n_trials, episode)

    def run(self):
        start_config, _ = self.landscape.get_lowest_performing_configuration()
        self._process_trial(0, start_config, round_type="given_start", reasoning=NO_CHOICE_MADE)

        progress.agent_start()
        for trial_number in range(1, self.n_trials + 1):
            new_config, reasoning_text = self._free_choice_move(self.SYSTEM_PROMPT)
            payoff = self._process_trial(trial_number, new_config, round_type="own_choice",
                                          reasoning=reasoning_text)
            progress.trial_done("B", trial_number, self.n_trials, payoff,
                                self.best_payoff, "own_choice")

        return self.history


# =============================================================================
# Agent C -- Epsilon-Greedy
# =============================================================================
class AgentC(SearchAgent):
    """Each round, a fixed probability (epsilon) -- decided by CODE, not
    by Claude -- picks whether this is an 'explore' or 'exploit' round.
    In BOTH cases Claude proposes a full, unconstrained configuration
    (same JSON schema as Agent B); the only difference is the framing
    text it's given for that round. Whether exploit rounds actually come
    out as smaller moves than explore rounds is measured, not enforced."""

    agent_name = "C_epsilon_greedy"

    SYSTEM_PROMPT = (
        "You are taking part in a game. You have made contact with an alien "
        "from a distant planet who is interested in buying art pictures. "
        "An art picture is made up of 10 distinct geometric shapes, each of "
        "which you can switch ON or OFF. We refer to the 10 shapes using "
        "the Greek letters alpha, beta, gamma, delta, epsilon, zeta, eta, "
        "theta, iota, and kappa. You do not know in advance which "
        "combination of shapes the alien prefers -- you only find out the "
        "payoff after you submit a combination. Before each trial you will "
        "be told whether this is an 'explore' round or an 'exploit' round. "
        "Which type of round it is is decided randomly, before you are "
        "asked to choose, and is not up to you -- but you are always free "
        "to submit any full configuration you like; nothing restricts "
        "which or how many symbols you can change. You will be shown the "
        "full history of your own past trials and their payoffs before "
        "each choice."
    )

    EXPLOIT_FRAMING = (
        "This is an EXPLOIT round: try to make only a small, incremental "
        "refinement to your current best-known configuration, in order to "
        "improve your payoff through local search. (You are still free to "
        "submit any configuration you like -- this is guidance, not a rule "
        "enforced by the game.)"
    )
    EXPLORE_FRAMING = (
        "This is an EXPLORE round: try submitting a configuration that is "
        "substantially different from your current best-known one, in "
        "order to search a different part of the landscape. (You are "
        "still free to submit any configuration you like -- this is "
        "guidance, not a rule enforced by the game.)"
    )

    def __init__(self, landscape, claude, epsilon, n_trials=None, rng_seed=None, episode=1):
        super().__init__(landscape, claude, n_trials, episode)
        self.epsilon = epsilon
        self._rng = random.Random(rng_seed)

    def run(self):
        start_config, _ = self.landscape.get_lowest_performing_configuration()
        self._process_trial(
            0, start_config, round_type="given_start", epsilon=self.epsilon,
            reasoning=NO_CHOICE_MADE,
        )

        progress.agent_start()
        for trial_number in range(1, self.n_trials + 1):
            if self._rng.random() < self.epsilon:
                round_type = "explore"
                framing = self.EXPLORE_FRAMING
            else:
                round_type = "exploit"
                framing = self.EXPLOIT_FRAMING

            best_dict = NKLandscape.config_to_dict(self.best_config)
            extra_context = (
                f"{framing}\n\nYour current best-known configuration is: "
                f"{best_dict}, which earned a payoff of {self.best_payoff:.4f}."
            )
            new_config, reasoning_text = self._free_choice_move(self.SYSTEM_PROMPT, extra_context)

            payoff = self._process_trial(trial_number, new_config, round_type=round_type,
                                          epsilon=self.epsilon, reasoning=reasoning_text)
            progress.trial_done("C", trial_number, self.n_trials, payoff,
                                self.best_payoff, round_type)

        return self.history


# =============================================================================
# Agent D -- Q-learning (RL agent with memory across episodes)
# =============================================================================
class AgentD(SearchAgent):
    """Tabular Q-learning agent, a line-by-line adaptation of the Python
    code in Q-learning-reference.pdf (Vyakaranam, "Reinforcement Q-Learning
    Implementation using Python"), Section V-B.

    The variable names below are the reference's own: q, s, snew, ind, r,
    v, v1, Del, k, eps, alpha, gamma. Nothing is renamed.

    Mapping from the reference's grid world to the alien game
    ----------------------------------------------------------
    reference                                   -> here
    s = (row, col)  (cell on an n x n grid)     -> (t, s): t = trial index (0..N_TRIALS-1),
                                                   s = integer index of the current 10-bit
                                                   configuration, 2^N states
    ind in 0..3 (down/left/up/right)            -> ind in 0..N-1  (flip attribute ind)
    q = np.random.rand(number, number, 4)       -> q = np.random.rand(N_TRIALS, 2**N, N)  (random init)
    if random.uniform(0,1) < eps:               -> same
        ind = np.random.randint(0,4)            ->     ind = np.random.randint(0,N)
    else: v = np.amax(q[s[0],s[1],:])           -> else: v = np.amax(q[t,s,:])
          first k with q[s[0],s[1],k] == v      ->        first k with q[t,s,k] == v
    r = reward of the cell reached              -> r = payoff of the configuration reached
    v1 = np.amax(q[snew[0],snew[1],:])          -> v1 = np.amax(q[t+1,snew,:])
    Del = r + (gamma*v1) - q[s[0],s[1],ind]     -> Del = r + (gamma*v1) - q[t,s,ind]
                                                   (on the LAST trial only: Del = r - q[t,s,ind],
                                                   see "Objective" below)
    q[s[0],s[1],ind] = q[...] + alpha*Del       -> q[t,s,ind] = q[t,s,ind] + alpha*Del
    s = snew                                    -> s = snew
    (eqs. (1)-(2) of the paper)

    Parameters: alpha=0.9 (reference); eps = 0.3 here (reference uses 0.2)
    -- 30% random exploration on every round, no decay; gamma = 1.0
    (reference uses 0.9, see "Objective" below).

    Objective: maximise CUMULATIVE payoff over the episode
    -------------------------------------------------------
    The reference has no terminal case and gamma = 0.9, which makes the
    agent maximise an endless, discounted sum of rewards. Here the reward
    is the payoff of the configuration reached, the episode has a known end
    (trial N_TRIALS is TERMINAL) and gamma = 1.0 (config.RL_GAMMA), so
    every payoff counts equally:
        return from trial 1 = r_1 + r_2 + ... + r_N_TRIALS = cumulative payoff
    * Trials 1 .. N_TRIALS-1 : Del = r + gamma*v1 - q[t,s,ind]   (bootstrap,
                               v1 = best value at the NEXT trial's row)
    * Trial  N_TRIALS        : Del = r - q[t,s,ind]              (terminal: the
                               value of the next state is 0, no bootstrap)
    The STATE IS (trial number, configuration): q has shape
    (N_TRIALS, 2^N, N), indexed q[t, s, a] with t = trial_number - 1. Q is
    then the exact expected sum of the payoffs still to come until the last
    trial, so the finite-horizon cumulative-payoff objective is solved
    exactly (and flip/flip-back loops cannot inflate values at gamma = 1,
    because the same configuration at a later trial is a different cell).
    The price is a table N_TRIALS times larger, so learning needs many more
    episodes. Trial 0 (free start) is not a reward.

    Differences forced by the alien-game setting
    --------------------------------------------
    * One alien-game TRIAL = one step (one action, one payoff). The
      reference's "trial" (a walk until the goal) has no analogue here; the
      reference's EPISODE maps to an episode of N_TRIALS steps.
    * Every episode starts from the same given start configuration, like
      the other agents (the reference always restarts at s=(n-1, 0)).
    * MEMORY: the reference re-initialises q at the start of every
      episode (so episodes are independent learning runs). Here q is
      created ONCE and persists across episodes, so payoffs from earlier
      episodes shape later behaviour. Set config.RL_RESET_Q_EACH_EPISODE
      = True to reproduce the reference's per-episode reset instead.
    * The reference calls the global `random` and `np.random`. Each AgentD
      owns its own generators (self._rng, self._np_rng) with the same
      seed, so the memory agent and its control stay reproducible and do
      not disturb each other. Inside run_episode they are bound to the
      local names `random` and `np_random`, so those lines read exactly
      like the reference.

    Logged per step (CSV columns): ind, flipped_attribute (its name),
    Del, and q_s_ind (= q[t,s,ind] after the update).

    No Claude API calls anywhere."""

    agent_name = "D_q_learning"

    def __init__(self, landscape, claude=None, n_trials=None, rng_seed=None,
                 eps=None, alpha=None, gamma=None,
                 reset_q_each_episode=None, agent_name=None):
        super().__init__(landscape, claude, n_trials, episode=1)
        if agent_name is not None:          # e.g. the no-memory control copy
            self.agent_name = agent_name
        self.reset_q_each_episode = (config.RL_RESET_Q_EACH_EPISODE
                                     if reset_q_each_episode is None else reset_q_each_episode)
        self._episodes_run = 0
        self.eps = config.RL_EPSILON if eps is None else eps
        self.alpha = config.RL_ALPHA if alpha is None else alpha
        self.gamma = config.RL_GAMMA if gamma is None else gamma
        self.n_actions = landscape.N        # replaces the literal 4 (directions) of the reference
        self._rng = random.Random(rng_seed)                 # stands in for the reference's `random`
        self._np_rng = np.random.RandomState(rng_seed)      # stands in for the reference's `np.random`
        self._init_q_table()

    def _init_q_table(self):
        # reference: q=np.random.rand(number,number,4)
        # here: one (2^N x N) table per trial number -> shape (N_TRIALS, 2^N, N)
        self.q = self._np_rng.rand(self.n_trials, 2 ** self.landscape.N, self.n_actions)
        self.update_counts = np.zeros_like(self.q, dtype=int)   # bookkeeping only, not in the reference

    @staticmethod
    def _state_index(config_tuple):
        return int("".join(str(int(b)) for b in config_tuple), 2)

    def run_episode(self, episode):
        """Play ONE episode (N_TRIALS steps) and return its rows. q carries
        over from previous calls unless RL_RESET_Q_EACH_EPISODE is True."""
        self.start_episode(episode)
        # Re-randomise q before every episode AFTER the first (the first
        # table was already drawn in __init__, so a control with the same
        # seed reproduces the memory agent's episode 1 exactly).
        if self.reset_q_each_episode and self._episodes_run > 0:
            self._init_q_table()
        self._episodes_run += 1

        # Names exactly as in the reference code
        random = self._rng          # reference: random.uniform(0,1)
        np_random = self._np_rng    # reference: np.random.randint(0,4)
        q = self.q
        eps, alpha, gamma = self.eps, self.alpha, self.gamma

        state_cfg, _ = self.landscape.get_lowest_performing_configuration()
        self._process_trial(0, state_cfg, round_type="given_start", epsilon=eps,
                            reasoning=NO_CHOICE_MADE)
        s = self._state_index(state_cfg)

        for trial_number in range(1, self.n_trials + 1):
            t = trial_number - 1        # trial index = part of the state (row block of q)
            #moving by exploration
            if random.uniform(0, 1) < eps:
                ind = np_random.randint(0, self.n_actions)
                round_type = "explore"
            #moving by exploitation, where the agent
            #flips the attribute corresponding to the max value of the row of q for its current (trial, configuration)
            else:
                v = np.amax(q[t, s, :])
                for k in range(self.n_actions):
                    if q[t, s, k] == v:
                        ind = k
                        break
                round_type = "exploit"

            #changing the configuration to the new one (s->snew)
            next_cfg = list(state_cfg)
            next_cfg[ind] = 1 - next_cfg[ind]
            next_cfg = tuple(next_cfg)
            snew = self._state_index(next_cfg)

            #reward: payoff of the configuration reached
            r = self.landscape.get_payoff(next_cfg)

            #calculation of improvement (Del)
            if trial_number == self.n_trials:
                #TERMINAL step (last trial of the episode): there is no future
                #payoff to add, so the target is just r (value of next state = 0)
                Del = r - q[t, s, ind]
            else:
                #"v1" represents the max value at the new position, at the NEXT trial
                v1 = np.amax(q[t + 1, snew, :])
                #gamma is the discount factor (1.0 here: all payoffs count equally)
                Del = r + (gamma * v1) - q[t, s, ind]
            #updating the entry of q at the current (trial, configuration)
            q[t, s, ind] = q[t, s, ind] + alpha * Del
            self.update_counts[t, s, ind] += 1

            self._process_trial(
                trial_number, next_cfg, round_type=round_type, epsilon=eps,
                reasoning=NO_API_CALL_QLEARNING,
                extra_fields={
                    "ind": int(ind),
                    "flipped_attribute": ATTRIBUTE_NAMES[ind],
                    "Del": float(Del),
                    "q_s_ind": float(q[t, s, ind]),
                },
            )
            #finally making the move
            s = snew
            state_cfg = next_cfg

        return self.history

    def n_visited_state_actions(self):
        """Number of distinct (trial, configuration, action) entries updated so far."""
        return int((self.update_counts > 0).sum())
