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
from random import sample

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
# 4. Monte Carlo Simulation
# -----------------------------------
def monte_carlo_sim(hole: List[str], table: str, n_simulations: int = 10000) -> Tuple[float, float, float]:
    """
    Running Monte Carlo simulation to estimate win/tie/lose probabilities.

    Returns: Win/Tie/Lose probabilities
    """

    my_category = hand_category(hole, table)
    SUITS = "CDHS"
    deck = [ r + s for r in RANKS for s in SUITS]
    known_cards = hole + [table]
    available_deck = [card for card in deck if card not in known_cards]
    wins = 0
    ties = 0

    for _ in range(n_simulations):
        opp_hole = sample(available_deck, 2)
        opp_category = hand_category(opp_hole, table)

        if my_category > opp_category:
            wins += 1
        elif my_category == opp_category:
            ties += 1
    
    prob_win = wins / n_simulations
    prob_tie = ties / n_simulations
    prob_lose = 1 - prob_win - prob_tie

    return prob_win, prob_lose, prob_tie



# -----------------------------------
# 5. Main strategy function to edit
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
    """

    # 1) Extract basic information
    hole = state["your_hole"]                 # e.g. ["AS", "TD"]
    table = state["table_card"]               # e.g. "7H"
    opp = state.get("opponent_stats") or {"fold": 0, "call": 0, "raise": 0}
    round_number = state.get("round", 1)

    total_opp_actions = opp["fold"] + opp["call"] + opp["raise"]
    if total_opp_actions == 0:
        opp_fold_rate = 0.33
        opp_raise_rate = 0.33
    else:
        opp_fold_rate = opp["fold"] / total_opp_actions
        opp_raise_rate = opp["raise"] / total_opp_actions
        opp_call_rate = 1 - opp_fold_rate - opp_raise_rate

    # 2) Basic hand strength (0..5)
    #    0: high, 1: pair, 2: flush, 3: straight, 4: trips, 5: straight flush
    category = hand_category(hole, table)

    prob_win, prob_tie, prob_lose = monte_carlo_sim(hole, table, n_simulations= 10000)

    ev_fold = -1
    ev_call = (
        opp_fold_rate * 2 +
        opp_call_rate * (prob_win * 2 + prob_lose * (-2)) +
        opp_raise_rate * (prob_win * 2 + prob_lose * (-3))
    )
    ev_raise = (
        opp_fold_rate * 3 +
        opp_call_rate * (prob_win * 3 + prob_lose * (-2)) +
        opp_raise_rate * (prob_win * 3 + prob_lose * (-3))
    )

    if category >= 4:
        if ev_raise >= ev_call or prob_win > 0.85:
            return "RAISE"
        else:
            return "CALL"
    
    if category == 3:
        if ev_raise >= max(ev_call, ev_fold):
            return "RAISE"
        elif ev_call > ev_fold:
            return "CALL"
        else:
            return "FOLD"
        
    if category == 2:
        if prob_win > 0.6 and ev_raise > ev_call:
            return "RAISE"
        elif prob_win > 0.45 or ev_call > 0:
            return "CALL"
        else:
            return "FOLD"
        
    if category == 1:
        if opp_raise_rate > 0.5 and prob_win < 0.5:
            if ev_call > 0:
                return "CALL"
            else:
                return "FOLD"
        elif prob_win > 0.55 or (ev_call > 0.5 and opp_fold_rate > 0.4):
            return "CALL"
        elif ev_fold >= ev_call:
            return "FOLD"
        else:
            return "CALL"
    
    if category == 0:
        if prob_win > 0.6 and opp_fold_rate > 0.4:
            return "CALL"
        elif ev_call > 0.3:
            return "CALL"
        else:
            return "FOLD"
        
    return "CALL"



# -----------------------------
# 6. I/O glue (do not touch)
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
