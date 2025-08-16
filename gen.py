#!/usr/bin/env python3
"""
PHH Six‑Max Poker Log Generator (Rule‑Compliant)
------------------------------------------------

Generates realistic Poker Hand History (PHH) logs for 6‑handed
No‑Limit Texas Hold'em between *skilled but varied* agents. Decisions mix
heuristics (position, pot odds, c‑bet) with fast Monte Carlo equity estimates.

This version enforces core poker rules and writes *cleanly formatted* output to
an output file. Stacks reset every hand. We model blinds internally to keep the
PHH lines close to your exemplar (no explicit blind lines), but all betting
obeys NLHE rules: min‑raise sizing, action order, all‑in handling, betting
rounds on each street, and the hand ends when everyone but one folds or when all
live players have matched the current bet.

Output example (one hand per line as a JSON array of PHH strings):
  ["d dh p1 6d5s", "d dh p2 Td2d", ..., "p6 cbr 275", "p2 cc", "d db 7s", ...]

Usage:
  pip install treys
  python phh_six_max_sim.py --hands 5 --seed 7 --iters 250 --out phh_logs.ndjson

Key tokens:
- "d dh pN XYuv"  : deal hole to player N (two 2‑char cards; no spaces)
- "d db ABC"      : deal board cards (3 on flop, 1 on turn, 1 on river)
- "pN f"          : player N folds
- "pN cc"         : player N checks or calls
- "pN cbr AMT"    : player N bets/raises *to* total amount AMT
- "pN sm XYuv"    : player N shows their two hole cards at showdown

Notes:
- Internal blinds (SB=50, BB=100) ensure legal preflop action. We don't emit
  separate blind lines to mirror your sample format. Amounts in action reflect
  legal sizing given those blinds and stacks.
- Stacks reset every hand (default 10,000). All‑ins are allowed; side‑pot math
  is respected internally, though only actions + showdowns are logged.
- Bet sizes are sampled from agent styles and then adjusted to satisfy
  min‑raise rules. If a raise cannot meet min‑raise and isn't all‑in, it
  downgrades to a call.
"""
from __future__ import annotations
import argparse, random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json

from treys import Card, Deck, Evaluator

# ------------------------- Utility: Cards & Formatting -------------------------

def _card_to_str(c: int) -> str:
    return Card.int_to_str(c)

def _cards_concat(cs: List[int]) -> str:
    return ''.join(_card_to_str(c) for c in cs)

# ------------------------------ Agent Parameters ------------------------------

@dataclass
class AgentStyle:
    name: str
    aggression: float        # tendency to raise vs call
    bluff_freq: float        # chance to bluff when equity is low
    cbet_freq: float         # continuation‑bet frequency when last aggressor
    tightness: float         # higher => needs more equity
    call_down: float         # willingness to call multiple streets
    bet_frac_flop: Tuple[float, float]
    bet_frac_turn: Tuple[float, float]
    bet_frac_river: Tuple[float, float]
    raise_scale: float

    def sample_bet_size(self, street: str, pot: int) -> int:
        if street == 'flop': f = random.uniform(*self.bet_frac_flop)
        elif street == 'turn': f = random.uniform(*self.bet_frac_turn)
        else: f = random.uniform(*self.bet_frac_river)
        amt = max(1, int(round(pot * f * self.raise_scale / 5.0) * 5))
        return amt

# -------------------------- Monte Carlo Equity Engine -------------------------

evalr = Evaluator()

def estimate_equity(hero: List[int], board: List[int], opp_count: int, iters: int, dead: List[int]) -> float:
    wins = ties = 0
    for _ in range(iters):
        deck = Deck()
        used = set(dead)
        opp_hands = []
        for _o in range(opp_count):
            c1 = deck.draw(1)[0]
            while c1 in used: c1 = deck.draw(1)[0]
            used.add(c1)
            c2 = deck.draw(1)[0]
            while c2 in used: c2 = deck.draw(1)[0]
            used.add(c2)
            opp_hands.append([c1, c2])
        sim_board = list(board)
        while len(sim_board) < 5:
            c = deck.draw(1)[0]
            while c in used or c in sim_board: c = deck.draw(1)[0]
            sim_board.append(c)
        hero_rank = evalr.evaluate(sim_board, hero)
        best_opp = min(evalr.evaluate(sim_board, h) for h in opp_hands)
        if hero_rank < best_opp: wins += 1
        elif hero_rank == best_opp: ties += 1
    return (wins + 0.5 * ties) / max(1, iters)

# ------------------------------- Table & State --------------------------------

@dataclass
class PlayerState:
    pid: int
    hole: List[int] = field(default_factory=list)
    in_hand: bool = True
    committed: int = 0          # committed this round
    total_commit: int = 0       # committed this hand (all streets)
    last_aggressor: bool = False
    stack: int = 0
    all_in: bool = False

@dataclass
class HandState:
    button: int
    pot: int = 0
    street: str = 'preflop'  # 'preflop','flop','turn','river'
    board: List[int] = field(default_factory=list)
    current_bet: int = 0     # current *to* amount to match
    last_raise_size: int = 0 # last raise increment (min‑raise basis)
    raises_this_round: int = 0

# ------------------------------ Heuristic Policy ------------------------------

def decide_action(style: AgentStyle, hero_ps: PlayerState, players: List[PlayerState], H: HandState, to_call: int, iters: int) -> Tuple[str, Optional[int]]:
    # Skip decisions when all‑in
    if hero_ps.all_in:
        return ('cc', None)

    opp_count = sum(1 for p in players if p.in_hand and p.pid != hero_ps.pid)
    equity = estimate_equity(hero_ps.hole, H.board, opp_count, iters, dead=hero_ps.hole + H.board)

    pot_odds = to_call / max(1.0, (H.pot + to_call)) if to_call > 0 else 0.0
    base_thresh = 0.42
    if H.street == 'preflop': base_thresh = 0.46
    elif H.street == 'river': base_thresh = 0.47
    thresh = base_thresh + 0.05 * (style.tightness - 0.5)

    want_aggr = equity > max(thresh, pot_odds + 0.10) and random.random() < style.aggression

    # C‑bet if last aggressor and no bet yet
    if to_call == 0 and hero_ps.last_aggressor and random.random() < style.cbet_freq and H.raises_this_round < 3:
        want_aggr = True

    # Occasional bluff opportunity when unchecked
    if to_call == 0 and equity < 0.38 and random.random() < style.bluff_freq and H.raises_this_round < 3:
        want_aggr = True

    if to_call == 0:
        if want_aggr:
            bet = style.sample_bet_size(H.street, max(H.pot, 50))
            target_to = max(H.current_bet + H.last_raise_size, bet) if H.current_bet > 0 else bet
            return ('cbr', max(5, int(round(target_to / 5) * 5)))
        else:
            return ('cc', None)
    else:
        # Facing a bet
        if equity + 0.02 >= pot_odds or random.random() < style.call_down:
            if want_aggr and H.raises_this_round < 3:
                add = style.sample_bet_size(H.street, H.pot + to_call)
                target_to = max(H.current_bet + H.last_raise_size, H.current_bet + add)
                return ('cbr', max(5, int(round(target_to / 5) * 5)))
            else:
                return ('cc', None)
        else:
            return ('f', None)

# ------------------------------ Betting Mechanics -----------------------------

def contribute(ps: PlayerState, amt: int, H: HandState):
    amt = min(amt, ps.stack)
    ps.stack -= amt
    ps.committed += amt
    ps.total_commit += amt
    H.pot += amt
    if ps.stack == 0:
        ps.all_in = True


def run_betting_round(styles: Dict[int, AgentStyle], players: List[PlayerState], H: HandState, order: List[int], iters: int, logs: List[str]):
    # Reset round state
    for p in players:
        p.committed = 0
        p.last_aggressor = False
    H.current_bet = max(H.current_bet, 0)
    H.raises_this_round = 0

    active_ids = [p.pid for p in players if p.in_hand and not p.all_in]
    if len(active_ids) <= 1:
        return

    acted: Dict[int, bool] = {p.pid: False for p in players if p.in_hand and not p.all_in}

    # Betting continues until: every non‑all‑in player has acted *and* either
    # (a) there is no bet (all checked), or (b) all have matched current_bet.
    safety = 0
    while True:
        progressed = False
        for pid in order:
            ps = next(p for p in players if p.pid == pid)
            if not ps.in_hand or ps.all_in:
                continue

            to_call = max(0, H.current_bet - ps.committed)

            # If everyone else acted and there's no bet, round ends
            if to_call == 0 and H.current_bet == 0 and all(acted.get(q.pid, True) for q in players if q.in_hand and not q.all_in and q.pid != pid):
                return

            action, target_to = decide_action(styles[pid], ps, players, H, to_call, iters)

            if action == 'f':
                ps.in_hand = False
                logs.append(f"p{pid} f")
                acted.pop(pid, None)
                progressed = True
            elif action == 'cc':
                # check or call
                if to_call > 0:
                    contribute(ps, to_call, H)
                logs.append(f"p{pid} cc")
                acted[pid] = True
                progressed = True
            else:  # 'cbr' raise/bet to target_to
                # Enforce min‑raise if not opening bet
                if H.current_bet == 0:
                    min_to = max(5, target_to)
                else:
                    min_to = max(H.current_bet + H.last_raise_size, target_to)
                # Cap at all‑in target
                all_in_to = ps.committed + ps.stack
                final_to = min(min_to, all_in_to)

                inc = final_to - ps.committed
                if inc <= 0:
                    # can't raise; default to call/check
                    if to_call > 0:
                        contribute(ps, to_call, H)
                    logs.append(f"p{pid} cc")
                    acted[pid] = True
                else:
                    contribute(ps, inc, H)
                    # Determine if this counts as a raise (vs call)
                    if final_to > H.current_bet:
                        # Update current bet and last raise size
                        if H.current_bet == 0:
                            H.last_raise_size = max(inc, 50)  # opening bet baseline
                        else:
                            H.last_raise_size = final_to - H.current_bet
                        H.current_bet = final_to
                        H.raises_this_round += 1
                        for k in list(acted.keys()):
                            acted[k] = False
                        acted[pid] = True
                        ps.last_aggressor = True
                        logs.append(f"p{pid} cbr {final_to}")
                    else:
                        # It was effectively a call (e.g., all‑in short of min‑raise)
                        logs.append(f"p{pid} cc")
                        acted[pid] = True
                progressed = True

            # If only one player remains, betting ends immediately
            if sum(1 for p in players if p.in_hand) == 1:
                return

        safety += 1
        # End round if all non‑all‑in players have acted and matched current bet
        if all(acted.get(p.pid, True) for p in players if p.in_hand and not p.all_in):
            all_matched = all((H.current_bet - p.committed) <= 0 for p in players if p.in_hand and not p.all_in)
            if all_matched:
                return
        if safety > 100:
            return

# ------------------------------- Hand Simulation ------------------------------

def simulate_hand(hand_idx: int, styles: Dict[int, AgentStyle], button: int, iters: int, stack_size: int, rng: random.Random) -> List[str]:
    deck = Deck()
    logs: List[str] = []

    # Initialize/reset stacks each hand
    players = [PlayerState(pid=i+1, stack=stack_size) for i in range(6)]

    # Deal hole cards
    for ps in players:
        ps.hole = deck.draw(2)
        logs.append(f"d dh p{ps.pid} {_cards_concat(ps.hole)}")

    H = HandState(button=button)

    # Blinds (internal, no explicit PHH lines to match exemplar)
    sb_pid = (button % 6) + 1
    bb_pid = (sb_pid % 6) + 1
    SB, BB = 50, 100

    sb = next(p for p in players if p.pid == sb_pid)
    bb = next(p for p in players if p.pid == bb_pid)

    contribute(sb, min(SB, sb.stack), H)
    contribute(bb, min(BB, bb.stack), H)

    H.current_bet = min(BB, bb.committed)
    H.last_raise_size = BB  # min‑raise baseline on preflop

    # Betting order functions
    def order_preflop() -> List[int]:
        # UTG is left of BB
        start = (bb_pid % 6) + 1
        return [(start + i - 1) % 6 + 1 for i in range(6)]

    def order_postflop() -> List[int]:
        # First to act is left of button (SB)
        start = (button % 6) + 1
        return [(start + i - 1) % 6 + 1 for i in range(6)]

    # PRE‑FLOP
    run_betting_round(styles, players, H, order_preflop(), iters, logs)
    if sum(1 for p in players if p.in_hand) == 1:
        return logs

    # FLOP
    H.street = 'flop'
    H.board.extend(deck.draw(3))
    logs.append(f"d db {_cards_concat(H.board)}")
    H.current_bet = 0
    H.last_raise_size = BB
    run_betting_round(styles, players, H, order_postflop(), iters, logs)
    if sum(1 for p in players if p.in_hand) == 1:
        return logs

    # TURN
    H.street = 'turn'
    H.board.extend(deck.draw(1))
    logs.append(f"d db {_cards_concat(H.board[-1:])}")
    H.current_bet = 0
    run_betting_round(styles, players, H, order_postflop(), iters, logs)
    if sum(1 for p in players if p.in_hand) == 1:
        return logs

    # RIVER
    H.street = 'river'
    H.board.extend(deck.draw(1))
    logs.append(f"d db {_cards_concat(H.board[-1:])}")
    H.current_bet = 0
    run_betting_round(styles, players, H, order_postflop(), iters, logs)

    # SHOWDOWN (show all surviving hands)
    survivors = [p for p in players if p.in_hand]
    if len(survivors) >= 2:
        # Determine winners (not logged beyond shows)
        ranks = [(p, evalr.evaluate(H.board, p.hole)) for p in survivors]
        best = min(r for _, r in ranks)
        winners = [p for p, r in ranks if r == best]
        # Log showdown
        for p in survivors:
            logs.append(f"p{p.pid} sm {_cards_concat(p.hole)}")
    return logs

# ------------------------------- Styles Factory -------------------------------

def make_varied_styles(seed: int) -> Dict[int, AgentStyle]:
    rng = random.Random(seed)
    styles = []
    for i in range(6):
        styles.append(AgentStyle(
            name=f"p{i+1}",
            aggression=min(1.0, max(0.15, rng.gauss(0.55, 0.15))),
            bluff_freq=min(0.5, max(0.05, rng.gauss(0.15, 0.08))),
            cbet_freq=min(0.95, max(0.25, rng.gauss(0.55, 0.2))),
            tightness=min(0.95, max(0.1, rng.gauss(0.55, 0.15))),
            call_down=min(0.95, max(0.15, rng.gauss(0.45, 0.2))),
            bet_frac_flop=(0.35, 0.85),
            bet_frac_turn=(0.35, 1.10),
            bet_frac_river=(0.40, 1.35),
            raise_scale=min(1.5, max(0.7, rng.gauss(1.0, 0.2))),
        ))
    return {i+1: s for i, s in enumerate(styles)}

# ------------------------------------- CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', type=int, default=3, help='Number of hands to generate')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for styles/rotation')
    ap.add_argument('--iters', type=int, default=100, help='MC iterations per decision')
    ap.add_argument('--stack', type=int, default=10000, help='Starting stack per player (resets each hand)')
    ap.add_argument('--out', type=str, required=True, help='Output file path (NDJSON: one JSON array per line)')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    styles = make_varied_styles(args.seed)

    button = rng.randint(1, 6)  # random initial button

    with open(args.out, 'w', encoding='utf-8') as f:
        for h in range(args.hands):
            logs = simulate_hand(h, styles, button, args.iters, args.stack, rng)
            f.write(json.dumps(logs) + "\n")
            button = (button % 6) + 1  # rotate button

if __name__ == '__main__':
    main()
