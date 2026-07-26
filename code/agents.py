"""
agents.py
=========
Defines the four search agents used in this project, all searching the
SAME NK landscape in complete isolation from one another:

  Agent A -- Myopic Local Search (the baseline)
      Pure code, ZERO Claude API calls, ZERO memory beyond the current
      best-known configuration. Exactly Billinger et al. (2014)'s p=0
      computational baseline: a random full "long jump" on trial 1, then
      a single random attribute flip per trial afterward, kept only if
      it improves on the current best. This is the one agent that never
      talks to Claude and never sees any trial history at all.

  Agent B -- Free Replication
      The direct replication of Albert & Billinger's LLM alien-game
      study. Claude sees the full trial history and freely decides its
      next FULL configuration every round. No search-distance rule, no
      framing, no extra reference information -- a plain baseline for
      "what does Claude do with no strategy imposed at all."

  Agent C -- Epsilon-Greedy
      Each round, a fixed probability (epsilon), decided by CODE (not by
      Claude), decides whether this is an "explore" round or an
      "exploit" round -- that mode-selection being external/random is
      part of the actual definition of epsilon-greedy. BUT unlike a
      code-enforced version, Claude is NOT restricted in what it's
      allowed to submit in either round: it always proposes a full,
      free configuration (identical JSON schema to Agent B), with full
      history. The only difference between rounds is the FRAMING text
      it's given ("make a small refinement" vs. "try something
      substantially different") -- whether Claude's resulting search
      distance actually comes out small or large is something we
      MEASURE, not something the code guarantees.

  Agent D -- Streetlight
      Claude is given, as extra reference information alongside its own
      full trial history, the value of ONE other already-explored
      configuration (the "anchor") -- exactly mirroring how Hoelzemann,
      Manso, Nagaraj & Tranchero (2024)'s experiment reveals one
      project's value to participants who otherwise choose completely
      freely among all options. Claude is NOT restricted to flipping a
      capped number of attributes near that anchor; it proposes a full,
      free configuration every round, identical in shape to Agent B.
      Whether its search ends up clustering near the anchor anyway is
      the actual empirical question this agent is meant to let you test,
      rather than something the code forces to be true.

Agents B, C, and D are now structurally identical in what Claude is
ALLOWED to do (propose any full 10-attribute configuration, with full
history of its own past trials) -- they differ only in the framing or
reference information included in the prompt. This means any difference
in their resulting search-distance or payoff patterns reflects something
about how that framing/information actually changes Claude's behavior,
not a difference in the choice space each agent was given.
"""

import random

import config
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

    def __init__(self, landscape: NKLandscape, claude: ClaudeClient, n_trials=None):
        self.landscape = landscape
        self.claude = claude
        self.n_trials = n_trials or config.N_TRIALS

        self.history = []
        self.best_config = None
        self.best_payoff = None

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

        row = {
            "agent": self.agent_name,
            "epsilon": epsilon,
            "trial": trial_number,
            "config": config_tuple,
            "payoff": payoff,
            "best_payoff_so_far": self.best_payoff,
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

    def __init__(self, landscape, claude=None, n_trials=None, rng_seed=None):
        super().__init__(landscape, claude, n_trials)
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

    def run(self):
        start_config, _ = self.landscape.get_lowest_performing_configuration()
        self._process_trial(0, start_config, round_type="given_start", reasoning=NO_CHOICE_MADE)

        for trial_number in range(1, self.n_trials + 1):
            new_config, reasoning_text = self._free_choice_move(self.SYSTEM_PROMPT)
            self._process_trial(trial_number, new_config, round_type="own_choice",
                                 reasoning=reasoning_text)

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

    def __init__(self, landscape, claude, epsilon, n_trials=None, rng_seed=None):
        super().__init__(landscape, claude, n_trials)
        self.epsilon = epsilon
        self._rng = random.Random(rng_seed)

    def run(self):
        start_config, _ = self.landscape.get_lowest_performing_configuration()
        self._process_trial(
            0, start_config, round_type="given_start", epsilon=self.epsilon,
            reasoning=NO_CHOICE_MADE,
        )

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

            self._process_trial(trial_number, new_config, round_type=round_type,
                                 epsilon=self.epsilon, reasoning=reasoning_text)

        return self.history


# =============================================================================
# Agent D -- Streetlight
# =============================================================================
class AgentD(SearchAgent):
    """Claude is given the value of ONE other already-explored
    configuration (the 'anchor') as extra reference information,
    alongside its own full trial history -- but is NOT restricted to
    flipping a capped number of attributes near it. It proposes a full,
    unconstrained configuration every round (same JSON schema as Agent
    B). Whether its search clusters near the anchor anyway is the
    empirical question this agent lets you test."""

    agent_name = "D_streetlight"

    SYSTEM_PROMPT = (
        "You are taking part in a game. You have made contact with an alien "
        "from a distant planet who is interested in buying art pictures. "
        "An art picture is made up of 10 distinct geometric shapes, each of "
        "which you can switch ON or OFF. We refer to the 10 shapes using "
        "the Greek letters alpha, beta, gamma, delta, epsilon, zeta, eta, "
        "theta, iota, and kappa. You do not know in advance which "
        "combination of shapes the alien prefers -- you only find out the "
        "payoff after you submit a combination. In addition to the full "
        "history of your own past trials, you will also be given, for "
        "reference, the value of ONE other configuration that has already "
        "been explored by someone else. This is extra information only -- "
        "you are not required to move toward it, stay near it, or use it "
        "at all; you are free to submit any configuration you like."
    )

    def __init__(self, landscape, claude, n_trials=None, rng_seed=None):
        super().__init__(landscape, claude, n_trials)
        self.anchor_config = None
        self.anchor_payoff = None

    def run(self):
        # The anchor is revealed information, not a trial the agent
        # played itself -- it is never logged as a row in this agent's
        # own history, only referenced in the prompt (see below).
        self.anchor_config, self.anchor_payoff = self.landscape.get_streetlight_seed(
            config.STREETLIGHT_PERCENTILE_LOW,
            config.STREETLIGHT_PERCENTILE_HIGH,
        )
        anchor_dict = NKLandscape.config_to_dict(self.anchor_config)

        start_config, _ = self.landscape.get_lowest_performing_configuration()
        self._process_trial(
            0, start_config, round_type="given_start", reasoning=NO_CHOICE_MADE,
            extra_fields={"distance_from_anchor": NKLandscape.hamming_distance(start_config, self.anchor_config)},
        )

        for trial_number in range(1, self.n_trials + 1):
            extra_context = (
                f"For reference, here is one other configuration that has "
                f"already been explored: {anchor_dict}, which earned a "
                f"payoff of {self.anchor_payoff:.4f}."
            )
            new_config, reasoning_text = self._free_choice_move(self.SYSTEM_PROMPT, extra_context)

            self._process_trial(
                trial_number, new_config, round_type="own_choice_anchor_shown",
                reasoning=reasoning_text,
                extra_fields={"distance_from_anchor": NKLandscape.hamming_distance(new_config, self.anchor_config)},
            )

        return self.history
