"""
analyze_reasoning.py
=====================
Reads the "reasoning" column captured by run_experiment.py (Claude's
"think aloud" explanation for each trial's choice, via
ClaudeClient.ask_json_with_reasoning) and computes two simple text-based
measures inspired by the think-aloud extension in Albert & Billinger's
LLM replication of the alien game:

  - attention_breadth
      The number of distinct Greek-letter attribute names mentioned in a
      trial's reasoning text.

  - forward_looking_ratio
      A rough proxy for "ratio of character count of output classified
      into forward looking text to backward looking text." Each sentence
      is classified as forward-looking, backward-looking, or neither via
      simple keyword lists; forward_looking_ratio = forward_chars /
      backward_chars.

IMPORTANT CAVEAT: this is a lightweight keyword-based heuristic, not a
validated replication of the paper's own classification method. Treat it
as a starting point for exploration.

Agent A (myopic local search) never involves a real Claude choice, so
every row in agent_a.csv carries the same placeholder reasoning text and
is automatically excluded here -- no special-casing needed.

Can be run standalone for a single K value's output folder:
    python analyze_reasoning.py

Or called programmatically:
    import analyze_reasoning
    analyze_reasoning.main(output_dir="outputs/K5")
"""

import glob
import os
import re

import pandas as pd

import config
from nk_landscape import ATTRIBUTE_NAMES


BACKWARD_LOOKING_CUES = [
    "previously", "previous trial", "previous round", "last trial", "last round",
    "so far", "earlier", "before", "already tried", "history", "learned",
    "based on what", "in trial", "in round", "prior", "past trial", "past round",
    "we saw", "i saw", "resulted in", "yielded", "gave a payoff", "gave us",
]

FORWARD_LOOKING_CUES = [
    "i will", "i'll", "let's try", "let's test", "going to try", "next",
    "expect", "hope", "hoping", "predict", "hypothesize", "hypothesis",
    "should improve", "might improve", "could improve", "aim to", "plan to",
    "try flipping", "want to see", "want to test", "to see if", "to test if",
    "in order to", "this should", "this might", "this could",
]


def _placeholder_row(reasoning_text):
    if not isinstance(reasoning_text, str) or reasoning_text.strip() == "":
        return True
    return reasoning_text.startswith("N/A --")


def _split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def compute_attention_breadth(text):
    text_lower = text.lower()
    mentioned = set()
    for name in ATTRIBUTE_NAMES:
        if re.search(rf"\b{name}\b", text_lower):
            mentioned.add(name)
    return len(mentioned)


def classify_sentence(sentence):
    lower = sentence.lower()
    backward_hits = sum(1 for cue in BACKWARD_LOOKING_CUES if cue in lower)
    forward_hits = sum(1 for cue in FORWARD_LOOKING_CUES if cue in lower)

    if forward_hits == 0 and backward_hits == 0:
        return "neither"
    if forward_hits > backward_hits:
        return "forward"
    if backward_hits > forward_hits:
        return "backward"
    return "neither"


def compute_forward_backward_chars(text):
    forward_chars = 0
    backward_chars = 0
    for sentence in _split_sentences(text):
        label = classify_sentence(sentence)
        if label == "forward":
            forward_chars += len(sentence)
        elif label == "backward":
            backward_chars += len(sentence)
    return forward_chars, backward_chars


def analyze_dataframe(df):
    rows = []
    for _, row in df.iterrows():
        reasoning_text = row.get("reasoning")
        if _placeholder_row(reasoning_text):
            continue

        attention_breadth = compute_attention_breadth(reasoning_text)
        forward_chars, backward_chars = compute_forward_backward_chars(reasoning_text)
        forward_looking_ratio = (forward_chars / backward_chars) if backward_chars > 0 else None

        rows.append({
            "agent": row["agent"],
            "epsilon": row.get("epsilon"),
            "trial": row["trial"],
            "round_type": row["round_type"],
            "attention_breadth": attention_breadth,
            "forward_chars": forward_chars,
            "backward_chars": backward_chars,
            "forward_looking_ratio": forward_looking_ratio,
            "reasoning": reasoning_text,
        })

    return pd.DataFrame(rows)


def find_agent_csvs(output_dir):
    """Locate every agent_*.csv file produced by run_experiment.py inside
    output_dir (agent_a.csv, agent_b.csv, agent_c_eps*.csv, agent_d.csv)."""
    pattern = os.path.join(output_dir, "agent_*.csv")
    return sorted(glob.glob(pattern))


def main(output_dir=None):
    output_dir = config.OUTPUT_DIR if output_dir is None else output_dir

    csv_paths = find_agent_csvs(output_dir)
    if not csv_paths:
        print(f"No agent_*.csv files found in '{output_dir}/'. Run run_experiment.py first.")
        return

    all_results = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "reasoning" not in df.columns:
            print(f"Skipping {path}: no 'reasoning' column found.")
            continue
        result_df = analyze_dataframe(df)
        if not result_df.empty:
            all_results.append(result_df)
        print(f"Processed {os.path.basename(path)}: "
              f"{len(result_df)} trials with real Claude reasoning.")

    if not all_results:
        print("No trials with real Claude reasoning were found across any "
              "agent CSV -- nothing to save.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reasoning_analysis.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    print("\n=== Summary (mean per agent/epsilon) ===")
    group_cols = ["agent", "epsilon"]
    summary = combined.groupby(group_cols, dropna=False).agg(
        n_trials=("trial", "count"),
        mean_attention_breadth=("attention_breadth", "mean"),
        mean_forward_looking_ratio=("forward_looking_ratio", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
