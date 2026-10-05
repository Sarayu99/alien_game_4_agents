"""
progress.py
===========
Lightweight per-trial progress + ETA reporting for the Claude-driven agents
(B and C). Agents A and D are pure code and take ~0 s, so the ETA is based
only on the Claude API trials (the only part that takes real time).

Usage (already wired into run_experiment.py / run_multi_k_experiment.py / agents.py):
    progress.init(total_api_trials)       # once, before the loops
    progress.set_context(K=5, episode=3, n_episodes=10)
    progress.agent_start()                # at the start of B's / C's run()
    progress.trial_done("B", trial, n_trials, payoff, best_payoff, round_type)

All functions are no-ops if init() was never called, so agents.py still works
standalone. Every print is flushed immediately (safe under LSF even without
`python -u`).
"""

import time
from collections import deque
from datetime import datetime, timedelta

_tracker = None


def _fmt(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class _Tracker:
    def __init__(self, total, window):
        self.total = total
        self.done = 0
        self.t0 = time.time()
        self.last = self.t0
        # Rolling window = one full episode of B+C trials, so the average
        # covers every trial index (later trials have longer prompts and
        # are slower; averaging a full cycle avoids a biased ETA).
        self.durations = deque(maxlen=window)
        self.K = None
        self.episode = None
        self.n_episodes = None
        self.ep_t0 = self.t0


def init(total_api_trials, window=48):
    global _tracker
    _tracker = _Tracker(total_api_trials, window)
    print(f"[progress] tracking {total_api_trials} Claude API trials "
          f"(agents B + C only; A and D are instant).", flush=True)


def is_active():
    return _tracker is not None


def set_context(K=None, episode=None, n_episodes=None):
    if _tracker is None:
        return
    _tracker.K, _tracker.episode, _tracker.n_episodes = K, episode, n_episodes
    _tracker.ep_t0 = time.time()


def agent_start():
    if _tracker is not None:
        _tracker.last = time.time()


def trial_done(agent, trial, n_trials, payoff, best_payoff, round_type=None):
    t = _tracker
    if t is None:
        return
    now = time.time()
    t.durations.append(now - t.last)
    t.last = now
    t.done += 1

    elapsed = now - t.t0
    avg = sum(t.durations) / len(t.durations)
    remaining = (t.total - t.done) * avg
    finish = datetime.now() + timedelta(seconds=remaining)
    pct = 100.0 * t.done / t.total

    ctx = ""
    if t.K is not None:
        ctx += f"K={t.K} "
    if t.episode is not None:
        ctx += f"ep {t.episode}/{t.n_episodes} "
    rt = f" ({round_type})" if round_type else ""
    print(f"[{ctx}| {agent} trial {trial}/{n_trials}{rt}] "
          f"payoff={payoff:.3f} best={best_payoff:.3f} | "
          f"overall {t.done}/{t.total} ({pct:.1f}%) | "
          f"{avg:.1f}s/trial | elapsed {_fmt(elapsed)} | "
          f"ETA {_fmt(remaining)} (~{finish:%a %H:%M})", flush=True)


def episode_done():
    t = _tracker
    if t is None:
        return
    remaining = (t.total - t.done) * (sum(t.durations) / len(t.durations)) if t.durations else 0
    print(f"[progress] episode {t.episode}/{t.n_episodes} (K={t.K}) finished in "
          f"{_fmt(time.time() - t.ep_t0)} | elapsed {_fmt(time.time() - t.t0)} | "
          f"ETA for everything {_fmt(remaining)}", flush=True)
