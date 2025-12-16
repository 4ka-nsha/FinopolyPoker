# player_template.py
"""
Simplified PokerBot – Player Template

You ONLY need to modify the `decide_action` function below.

The tournament engine (master.py) will:
  - Call this script once per round.
  - Send you a single JSON object on stdin describing the game state.
  - Expect a JSON object on stdout: {"action": "FOLD" or "CALL" or "RAISE"}

Your job:
  - Read the state.
  - Decide whether to FOLD / CALL / RAISE using a *quantitative* strategy.
  - Output the action as JSON.

You are free to:
  - Add helper functions.
  - Use probability / EV calculations.
  - Use opponent statistics for adaptive strategies.
  - As long as you keep the I/O format the same.
"""

import json
import sys
from typing import List, Tuple

# -------------------------
# 1. Basic card utilities
# -------------------------

# Ranks from lowest to highest. T = 10, J = Jack, Q = Queen, K = King, A = Ace.
RANKS = "23456789TJQKA"
# Map rank character -> numeric value (2..14)
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}  # 2..14 (A=14)


def parse_card(card_str: str) -> Tuple[int, str]:
    """
    Convert a string like "AH" or "7D" into (rank_value, suit).

    Example:
        "AH" -> (14, 'H')
        "7D" -> (7, 'D')

    card_str[0]: rank in "23456789TJQKA"
    card_str[1]: suit in "CDHS"  (Clubs, Diamonds, Hearts, Spades)
    """
    return RANK_VALUE[card_str[0]], card_str[1]


def is_straight_3(rank_values: List[int]) -> Tuple[bool, int]:
    """
    Check if 3 cards form a straight under our custom rules.

    Rules:
      - 3 cards are a straight if they are in sequence.
      - A can be:
          * LOW in A-2-3  (treated as the LOWEST straight)
          * HIGH in Q-K-A (treated as the highest normal case)
      - Return:
          (is_straight: bool, high_card_value_for_straight: int)

    Examples:
      [2, 3, 4]    -> (True, 4)
      [12, 13, 14] -> Q-K-A -> (True, 14)
      [14, 2, 3]   -> A-2-3 -> (True, 3)  (lowest straight)
    """
    r = sorted(rank_values)

    # Normal consecutive: x, x+1, x+2
    if r[0] + 1 == r[1] and r[1] + 1 == r[2]:
        return True, r[2]

    # A-2-3 special: {14,2,3} -> treat as straight with high=3
    if set(r) == {14, 2, 3}:
        return True, 3

    return False, 0


# --------------------------------------
# 2. Hand category evaluation (3 cards)
# --------------------------------------

"""
Hand is always: your 2 hole cards + 1 community card = 3 cards.

We classify them into 6 categories (from weakest to strongest):

  0: HIGH CARD
  1: PAIR
  2: FLUSH
  3: STRAIGHT
  4: TRIPS  (Three of a kind)
  5: STRAIGHT FLUSH

And the *global ranking* is:

  STRAIGHT_FLUSH (5) > TRIPS (4) > STRAIGHT (3) > FLUSH (2) > PAIR (1) > HIGH_CARD (0)

This function only returns the category index 0..5,
not the tie-break details (you don't strictly need tie-breaks inside your bot).
"""


def hand_category(hole: List[str], table: str) -> int:
    """
    Compute the hand category for your 3-card hand.

    Input:
        hole  = ["AS", "TD"], etc. (your two private cards)
        table = "7H"           (community card)

    Returns:
        0..5 as defined above.
    """
    cards = hole + [table]
    rank_values, suits = zip(*[parse_card(c) for c in cards])
    flush = len(set(suits)) == 1  # True if all 3 suits are the same

    # Count how many times each rank appears
    counts = {}
    for v in rank_values:
        counts[v] = counts.get(v, 0) + 1

    straight, _ = is_straight_3(list(rank_values))

    if straight and flush:
        return 5  # Straight Flush
    if 3 in counts.values():
        return 4  # Trips
    if straight:
        return 3  # Straight
    if flush:
        return 2  # Flush
    if 2 in counts.values():
        return 1  # Pair
    return 0      # High Card


# ----------------------------------------
# 3. Scoring summary (for your reference)
# ----------------------------------------
"""
IMPORTANT: these are the points awarded PER ROUND
depending on the actions and showdown result.

Notation:
  - "Showdown" means nobody folded: cards are compared.
  - result = which player wins the hand, not part of your code directly.

Fold scenarios (no showdown):
  P1: FOLD, P2: FOLD  ->  (0, 0)
  P1: FOLD, P2: CALL  ->  (-1, +2)
  P1: FOLD, P2: RAISE ->  (-1, +3)
  P1: CALL, P2: FOLD  ->  (+2, -1)
  P1: RAISE, P2: FOLD ->  (+3, -1)

Showdown scenarios (someone has better hand):
  Both CALL:
    - P1 wins: (+2, -2)
    - P2 wins: (-2, +2)

  P1 RAISE, P2 CALL:
    - P1 wins: (+3, -2)
    - P2 wins: (-3, +2)

  P1 CALL, P2 RAISE:
    - P1 wins: (+2, -3)
    - P2 wins: (-2, +3)

  P1 RAISE, P2 RAISE (High-Risk round):
    - P1 wins: (+3, -3)
    - P2 wins: (-3, +3)

Any showdown where hands are *exactly* identical:
  -> (0, 0)

Your bot does NOT see the opponent's current action,
but it CAN see opponent action frequencies over previous rounds
(via `opponent_stats`).
Use this to think in terms of EXPECTED VALUE (EV), not just raw hand strength.
"""


# -----------------------------------
# 4. Main strategy function to edit
# -----------------------------------

def decide_action(state: dict) -> str:
    """
    This is the ONLY function you need to modify.

    Input: `state` is a dictionary with at least:
      - state["your_hole"]      -> list of 2 strings, e.g. ["AS", "TD"]
      - state["table_card"]     -> string, e.g. "7H"
      - state["opponent_stats"] -> dict like {"fold": x, "call": y, "raise": z}
      - state["round"]          -> current round number (1..500+)
      - state["rules"]          -> descriptive info about rules (optional to use)

    Your task:
      - Use the information above.
      - Estimate the EV of FOLD / CALL / RAISE.
      - Return one of the strings: "FOLD", "CALL", or "RAISE".

    You may:
      - Add more helper functions above.
      - Do simulation / Monte Carlo within time limits.
      - Use simple heuristics + probabilities.
    """

    # 1) Extract basic information
    hole = state["your_hole"]                 # e.g. ["AS", "TD"]
    table = state["table_card"]               # e.g. "7H"
    opp = state.get("opponent_stats") or {"fold": 0, "call": 0, "raise": 0}
    round_number = state.get("round", 1)

    # 2) Basic hand strength (0..5)
    #    0: high, 1: pair, 2: flush, 3: straight, 4: trips, 5: straight flush
    category = hand_category(hole, table)

    # 3) Basic opponent tendencies
    total_opp_actions = opp["fold"] + opp["call"] + opp["raise"]
    if total_opp_actions == 0:
        # Early rounds: no information yet
        opp_fold_rate = 0.0
        opp_raise_rate = 0.0
    else:
        opp_fold_rate = opp["fold"] / total_opp_actions
        opp_raise_rate = opp["raise"] / total_opp_actions

    # ------------------------------------------------------------------
    # 4) PLACE TO MAKE THIS "QUANTITATIVE"
    #
    # Ideas:
    #   - Approximate win probability given (hole, table).
    #     For example, simulate random opponent hole cards and
    #     count how often you win at showdown.
    #
    #   - Combine win probability with scoring:
    #       EV(CALL)  ~  P_win * payoff_if_win(CALL vs typical opp action)
    #                 + P_lose * payoff_if_lose(...)
    #                 + P_tie * payoff_if_tie(...)
    #
    #       EV(RAISE) ~ similar, but include higher rewards (and penalties).
    #
    #   - Use opponent raise/call/fold rates to approximate how often
    #     each action profile (CALL/CALL, RAISE/CALL, etc.) happens.
    #
    #   - Then pick the action with highest estimated EV.
    #
    # For now, we provide a *simple heuristic* baseline that you can
    # improve or completely replace.
    # ------------------------------------------------------------------

    # Simple heuristic strength bands (you should refine this)
    #  - Weak:  0 (High)
    #  - Medium: 1 (Pair), 2 (Flush)
    #  - Strong: 3 (Straight), 4 (Trips), 5 (Straight Flush)
    is_weak = (category == 0)
    is_medium = (category == 1 or category == 2)
    is_strong = (category >= 3)

    # Example logic outline (you can change this completely):
    #  1) Very strong hands: often RAISE (you gain more points when you win).
    #  2) Medium hands: often CALL, occasionally RAISE if opponent is timid.
    #  3) Weak hands: FOLD more against aggressive opponents, CALL more
    #     against extremely passive ones.

    # Thresholds you can tune:
    HIGH_RAISE_RATE = 0.60  # "opponent is very aggressive"
    HIGH_FOLD_RATE = 0.50   # "opponent folds quite often"

    # Example policy:

    # Strong hands: attack
    if is_strong:
        # If opponent raises a lot, you are still happy to RAISE
        # because a successful showdown can give +3 or +3/-3.
        return "RAISE"

    # Medium-strength hands
    if is_medium:
        # If opponent is extremely aggressive (raises a lot),
        # calling may be safer than raising.
        if opp_raise_rate > HIGH_RAISE_RATE:
            return "CALL"
        # If opponent folds a lot, raising has good upside:
        # they might fold and give you +3 (if they had chosen FOLD vs RAISE).
        if opp_fold_rate > HIGH_FOLD_RATE:
            return "RAISE"
        # Otherwise, just CALL as a middle-ground.
        return "CALL"

    # Weak hands (high card only)
    if is_weak:
        # Against a maniac (very aggressive), folding is often better:
        # you avoid big negative swings like -3 vs their raise.
        if opp_raise_rate > HIGH_RAISE_RATE:
            return "FOLD"
        # If opponent is very passive (rarely raises), you can afford
        # to CALL sometimes and hope they fold or have even worse.
        return "CALL"

    # Fallback (should never hit, but just in case)
    return "CALL"


# -----------------------------
# 5. I/O glue (do not touch)
# -----------------------------

def main():
    """
    DO NOT modify this unless you know what you're doing.

    It:
      - Reads one JSON object from stdin.
      - Calls decide_action(state).
      - Writes {"action": "..."} as JSON to stdout.
    """
    raw = sys.stdin.read().strip()
    try:
        state = json.loads(raw) if raw else {}
    except Exception:
        state = {}

    action = decide_action(state)

    # Safety check: default to CALL if something invalid is returned
    if action not in {"FOLD", "CALL", "RAISE"}:
        action = "CALL"

    sys.stdout.write(json.dumps({"action": action}))


if __name__ == "__main__":
    main()
