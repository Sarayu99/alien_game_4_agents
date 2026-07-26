"""
claude_client.py
=================
A thin wrapper around the Anthropic API so the rest of the code base never
has to deal with API details directly. All Claude API calls in this
project go through this file.

You should not need to edit this file -- put your API key in config.py instead.
"""

import json
import re

from anthropic import Anthropic

import config


class ClaudeClient:
    """
    Wraps a single Anthropic API client and provides three methods:
    - ask_text(): get a free-form text answer from Claude
    - ask_json(): get a strict JSON answer from Claude (used whenever we
      need Claude's answer to be machine-readable, e.g. a 10-symbol
      configuration or a single attribute name to flip)
    - ask_json_with_reasoning(): same as ask_json(), but also captures
      Claude's "think aloud" explanation for its choice as a separate
      string, instead of discarding it (see docstring below)
    """

    def __init__(self, model=None):
        api_key = config.get_api_key()
        self.client = Anthropic(api_key=api_key)
        self.model = model or config.MODEL_NAME

    def ask_text(self, system_prompt, user_prompt, max_tokens=1000):
        """Send a system + user prompt to Claude and return the plain text
        of its reply (concatenating all text blocks in the response)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "".join(text_parts).strip()

    @staticmethod
    def _extract_json(raw_text):
        """
        Try progressively looser strategies to pull a JSON object out of
        Claude's raw reply. Claude will sometimes "think out loud" before
        giving the JSON answer (e.g. walking through prior trials' payoffs
        before stating the config), so we can't assume the whole reply is
        clean JSON -- we have to go find the object.

        Returns the parsed dict, or raises json.JSONDecodeError if none of
        the strategies work.

        Strategy 1: try parsing the whole reply as-is.
        Strategy 2: strip markdown code fences (```json / ```) and retry.
        Strategy 3: regex-search for the LAST {...} block in the reply and
                    parse just that (handles reasoning-before-JSON replies,
                    where the actual answer is the final JSON block).
        """
        # Strategy 1: whole reply
        try:
            return json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: scan for the LAST valid top-level JSON object in the
        # reply, using a right-to-left brace scan with json.JSONDecoder's
        # raw_decode instead of a single greedy regex.
        #
        # A naive greedy regex like r"\{.*\}" with re.DOTALL is NOT safe here:
        # re.findall returns exactly ONE match spanning from the very FIRST
        # "{" in the whole reply to the very LAST "}" -- not "the last
        # JSON-looking block". If Claude's reasoning text happens to contain
        # its own balanced brace pair (e.g. writing a set of attribute names
        # as "{delta, epsilon, zeta}"), the greedy match swallows that prose
        # too and json.loads fails on it.
        #
        # Instead: try every "{" position from rightmost to leftmost, and at
        # each one attempt raw_decode(). Keep the first one (scanning from
        # the end) whose parse consumes the rest of the string (ignoring
        # trailing whitespace) -- that is guaranteed to be the final,
        # complete JSON object, regardless of any stray brace-like text
        # earlier in the reasoning.
        decoder = json.JSONDecoder()
        brace_positions = [i for i, ch in enumerate(cleaned) if ch == "{"]
        for start in reversed(brace_positions):
            try:
                obj, end = decoder.raw_decode(cleaned, start)
            except json.JSONDecodeError:
                continue
            if cleaned[end:].strip() == "":
                return obj

        # Nothing worked -- raise using the cleaned text so the error message
        # in ask_json() is informative.
        return json.loads(cleaned)

    def ask_json(self, system_prompt, user_prompt, max_tokens=1000, max_retries=3):
        """
        Same as ask_text(), but expects Claude's reply to be a JSON object,
        and returns it already parsed into a Python dict.

        If Claude's reply isn't valid JSON on the first pass (e.g. it added
        reasoning or extra commentary), this tries the fallback extraction
        strategies in _extract_json() before giving up and retrying with a
        stricter reminder prompt.
        """
        strict_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Reply with ONLY a valid JSON object and "
              "nothing else -- no explanation, no markdown formatting, "
              "no code fences."
        )

        raw_text = ""
        last_error = None
        for _ in range(max_retries):
            raw_text = self.ask_text(system_prompt, strict_prompt, max_tokens=max_tokens)
            try:
                return self._extract_json(raw_text)
            except json.JSONDecodeError as e:
                last_error = e
                strict_prompt = (
                    user_prompt
                    + "\n\nYour previous reply could not be parsed as JSON. "
                      "Reply with ONLY a valid JSON object, nothing else."
                )

        raise ValueError(
            f"Claude did not return valid JSON after {max_retries} attempts. "
            f"Last error: {last_error}. Last raw reply: {raw_text}"
        )

    @staticmethod
    def _split_reasoning_and_json(raw_text):
        """
        Like _extract_json(), but ALSO returns whatever text Claude wrote
        BEFORE the final JSON object -- its "think aloud" explanation for
        the choice it's about to make.

        Returns (parsed_dict, reasoning_text). reasoning_text is "" if
        Claude's reply was pure JSON with no explanation attached.
        """
        stripped = raw_text.strip()

        try:
            return json.loads(stripped), ""
        except json.JSONDecodeError:
            pass

        cleaned = stripped.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned), ""
        except json.JSONDecodeError:
            pass

        # Same right-to-left brace scan as _extract_json() -- see that
        # method's docstring for why a single greedy regex is unsafe here
        # (it breaks when the reasoning prose itself contains a balanced
        # "{...}", e.g. "{delta, epsilon, zeta}" used as set notation).
        decoder = json.JSONDecoder()
        brace_positions = [i for i, ch in enumerate(cleaned) if ch == "{"]
        for start in reversed(brace_positions):
            try:
                obj, end = decoder.raw_decode(cleaned, start)
            except json.JSONDecodeError:
                continue
            if cleaned[end:].strip() == "":
                reasoning_text = cleaned[:start].strip()
                return obj, reasoning_text

        return json.loads(cleaned), ""

    def ask_json_with_reasoning(self, system_prompt, user_prompt, max_tokens=1000, max_retries=3):
        """
        Same purpose as ask_json(), but explicitly invites Claude to
        "think aloud" -- to briefly explain its reasoning in plain text --
        before giving its final JSON answer.

        Returns a tuple: (parsed_dict, reasoning_text)
        """
        reasoning_prompt = (
            user_prompt
            + "\n\nFirst, briefly explain your reasoning in 1-3 sentences: "
              "what are you considering, and why are you making this "
              "choice? Then, on a new line, give your final answer as a "
              "single JSON object, with nothing written after it."
        )

        raw_text = ""
        last_error = None
        for _ in range(max_retries):
            raw_text = self.ask_text(system_prompt, reasoning_prompt, max_tokens=max_tokens)
            try:
                return self._split_reasoning_and_json(raw_text)
            except json.JSONDecodeError as e:
                last_error = e
                reasoning_prompt = (
                    user_prompt
                    + "\n\nYour previous reply could not be parsed. Please "
                      "reply again: first briefly explain your reasoning "
                      "in 1-3 sentences, then on a new line give a single "
                      "valid JSON object as your final answer, with "
                      "nothing written after it."
                )

        raise ValueError(
            f"Claude did not return valid JSON (with reasoning) after "
            f"{max_retries} attempts. Last error: {last_error}. "
            f"Last raw reply: {raw_text}"
        )
