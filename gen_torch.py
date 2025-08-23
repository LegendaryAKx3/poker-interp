#!/usr/bin/env python3
"""
PHH Six‑Max Poker Log Generator (CUDA‑Accelerated Equity) with Periodic Save
-----------------------------------------------------------------------------

- CUDA‑accelerated, batched Monte Carlo equity engine (PyTorch)
- Exact 5‑card evaluation from 7 cards (vectorized) for equity
- CPU fallback using treys if --device=cpu
- Saves output to file every 100 hands and prints a progress message

Usage examples:
  pip install treys
  pip install torch --index-url https://download.pytorch.org/whl/cu121  # pick the CUDA build matching your system

  # GPU (recommended):
  python phh_six_max_sim_cuda.py --hands 50 --iters 8192 --device cuda --out phh_logs.ndjson

  # CPU fallback (still vectorized batching for MC draw logic):
  python phh_six_max_sim_cuda.py --hands 10 --iters 4096 --device cpu --out phh_logs.ndjson
"""
from __future__ import annotations
import argparse, random, json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from treys import Card, Deck, Evaluator

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# ------------------------- Utility: Cards & Formatting -------------------------

evalr = Evaluator()

RANKS = '23456789TJQKA'
SUITS = 'shdc'  # internal GPU order: 0:♠, 1:♥, 2:♦, 3:♣
RANK_TO_IDX = {ch:i for i,ch in enumerate(RANKS)}
SUIT_TO_IDX = {ch:i for i,ch in enumerate(SUITS)}


def _card_to_str(c: int) -> str:
    return Card.int_to_str(c)


def _cards_concat(cs: List[int]) -> str:
    return ''.join(_card_to_str(c) for c in cs)


def _treys_to_rs_ids(cs: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Convert treys ints to (ranks, suits, ids[0..51]) using internal mapping."""
    ranks, suits, ids = [], [], []
    for c in cs:
        s = Card.int_to_str(c)  # e.g., 'As'
        rch, sch = s[0], s[1]
        r = RANK_TO_IDX[rch]
        su = SUIT_TO_IDX[sch]
        ranks.append(r)
        suits.append(su)
        ids.append(su*13 + r)
    return ranks, suits, ids


# ------------------------------ Agent Parameters ------------------------------

@dataclass
class AgentStyle:
    name: str
    aggression: float        # tendency to raise vs call
    bluff_freq: float        # chance to bluff when equity is low
    cbet_freq: float        # continuation‑bet frequency when last aggressor
    tightness: float        # higher => needs more equity
    call_down: float        # willingness to call multiple streets
    bet_frac_flop: Tuple[float, float]
    bet_frac_turn: Tuple[float, float]
    bet_frac_river: Tuple[float, float]
    raise_scale: float

    def sample_bet_size(self, street: str, pot: int) -> int:
        import random as _r
        if street == 'flop': f = _r.uniform(*self.bet_frac_flop)
        elif street == 'turn': f = _r.uniform(*self.bet_frac_turn)
        else: f = _r.uniform(*self.bet_frac_river)
        amt = max(1, int(round(pot * f * self.raise_scale / 5.0) * 5))
        return amt

# -------------------------- CUDA / Vectorized Equity --------------------------

class CudaEquity:
    """Vectorized 7‑card evaluator & MC equity on CUDA (PyTorch).

    - Card ids: 0..51 = suit*13 + rank, rank 0..12 = 2..A, suit order 'shdc'.
    - Exact 5‑card evaluation per 7‑card hand via max over the 21 five‑card combos.
    - Returns fractional equity vs N opponents (ties split).
    """

    def __init__(self, device: str = 'cuda', mc_chunk: int = 8192):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed; install torch to use --device=cuda")
        self.device = torch.device(device)
        self.mc_chunk = mc_chunk
        # Precompute 21 combinations of 7 choose 5 as indices (21,5)
        combs = []
        idx = list(range(7))
        from itertools import combinations
        for c in combinations(idx, 5):
            combs.append(c)
        self.comb = torch.tensor(combs, dtype=torch.long, device=self.device)  # (21,5)
        self.rank_ar = torch.arange(13, device=self.device)

    # ---------------- 5‑card evaluator (vectorized; returns big int score) ---------------
    def _eval5(self, ranks: torch.Tensor, suits: torch.Tensor) -> torch.Tensor:
        """ranks,suits: (B,5) int64; returns (B,) int64 score; higher is better."""
        B = ranks.shape[0]
        ones = torch.ones((B, 5), dtype=torch.int32, device=self.device)
        counts = torch.zeros((B, 13), dtype=torch.int32, device=self.device)
        counts = counts.scatter_add(1, ranks, ones)  # (B,13)

        is_flush = (suits[:, 0:1] == suits).all(dim=1)
        present = counts > 0  # (B,13)

        # Straight detection
        win_sums = []
        for start in range(9):
            win = present[:, start:start+5].all(dim=1)
            win_sums.append((win, start))
        wheel = present[:, 12] & present[:, 0] & present[:, 1] & present[:, 2] & present[:, 3]
        straight_high = torch.full((B,), -1, dtype=torch.int32, device=self.device)
        for win, start in win_sums:
            high = torch.full((B,), start+4, dtype=torch.int32, device=self.device)
            straight_high = torch.where(win & (high > straight_high), high, straight_high)
        straight_high = torch.where(wheel & (straight_high < 3), torch.maximum(straight_high, torch.tensor(3, device=self.device, dtype=torch.int32)), straight_high)
        has_straight = straight_high >= 0

        r_idx = self.rank_ar.view(1, 13).expand(B, -1)
        key = counts * 100 + r_idx
        key_vals, key_idx = torch.sort(key, dim=1, descending=True)
        counts_sorted = counts.gather(1, key_idx)
        ranks_sorted = r_idx.gather(1, key_idx)

        def top_ranks_with_count(cnt: int, k: int) -> torch.Tensor:
            mask = counts_sorted == cnt
            filled = torch.where(mask, ranks_sorted, torch.full_like(ranks_sorted, -1))
            aux_key = torch.where(mask, ranks_sorted, torch.full_like(ranks_sorted, -1000))
            _, idxs = torch.sort(aux_key, dim=1, descending=True)
            top = ranks_sorted.gather(1, idxs)[:, :k]
            true_counts = mask.sum(dim=1, keepdim=True)
            rng = torch.arange(k, device=self.device).view(1, k).expand(B, -1)
            valid = rng < true_counts
            return torch.where(valid, top, torch.full_like(top, -1))

        quad_rank = top_ranks_with_count(4, 1).squeeze(1)
        trip_ranks = top_ranks_with_count(3, 2)
        pair_ranks = top_ranks_with_count(2, 2)
        single_mask = counts_sorted == 1
        aux_key = torch.where(single_mask, ranks_sorted, torch.full_like(ranks_sorted, -1000))
        _, idxs = torch.sort(aux_key, dim=1, descending=True)
        singles_sorted = ranks_sorted.gather(1, idxs)

        has_quads = quad_rank >= 0
        has_trips = trip_ranks[:, 0] >= 0
        has_two_trips = trip_ranks[:, 1] >= 0
        has_pair = pair_ranks[:, 0] >= 0
        has_two_pair = pair_ranks[:, 1] >= 0

        full_trip = trip_ranks[:, 0]
        full_pair = torch.where(has_two_trips, trip_ranks[:, 1], pair_ranks[:, 0])
        has_full = has_trips & ((has_pair) | (has_two_trips))

        is_straight_flush = is_flush & has_straight

        cat = torch.zeros((B,), dtype=torch.int64, device=self.device)
        cat = torch.where(is_straight_flush, torch.tensor(8, device=self.device), cat)
        cat = torch.where(~is_straight_flush & has_quads, torch.tensor(7, device=self.device), cat)
        cat = torch.where((cat == 0) & has_full, torch.tensor(6, device=self.device), cat)
        cat = torch.where((cat == 0) & is_flush, torch.tensor(5, device=self.device), cat)
        cat = torch.where((cat == 0) & has_straight, torch.tensor(4, device=self.device), cat)
        cat = torch.where((cat == 0) & has_trips, torch.tensor(3, device=self.device), cat)
        cat = torch.where((cat == 0) & has_two_pair, torch.tensor(2, device=self.device), cat)
        cat = torch.where((cat == 0) & has_pair, torch.tensor(1, device=self.device), cat)

        base = 15
        def pack(vals: List[torch.Tensor]) -> torch.Tensor:
            out = torch.zeros((B,), dtype=torch.int64, device=self.device)
            for v in vals:
                out = out * base + torch.clamp(v.to(torch.int64), min=0)
            return out

        r_sorted, _ = torch.sort(ranks, dim=1, descending=True)

        sf_hi = torch.where(is_straight_flush, straight_high.to(torch.int64), torch.zeros_like(straight_high, dtype=torch.int64))
        quad_kick_mask = (counts == 1)
        quads_kicker = torch.max(torch.where(quad_kick_mask, r_idx, torch.full_like(r_idx, -1000)), dim=1).values.to(torch.int64)

        full_trip_i = torch.where(has_full, full_trip.to(torch.int64), torch.zeros_like(full_trip, dtype=torch.int64))
        full_pair_i = torch.where(has_full, torch.where(has_two_trips, trip_ranks[:, 1].to(torch.int64), pair_ranks[:, 0].to(torch.int64)), torch.zeros_like(full_trip, dtype=torch.int64))

        st_hi = torch.where(has_straight, straight_high.to(torch.int64), torch.zeros_like(straight_high, dtype=torch.int64))

        trips_k1 = torch.where(has_trips, singles_sorted[:, 0].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))
        trips_k2 = torch.where(has_trips, singles_sorted[:, 1].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))

        tp_hi = torch.where(has_two_pair, pair_ranks[:, 0].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))
        tp_lo = torch.where(has_two_pair, pair_ranks[:, 1].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))
        tp_k = torch.where(has_two_pair, singles_sorted[:, 0].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))

        p_rank = torch.where(has_pair, pair_ranks[:, 0].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))
        p_k1 = torch.where(has_pair, singles_sorted[:, 0].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))
        p_k2 = torch.where(has_pair, singles_sorted[:, 1].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))
        p_k3 = torch.where(has_pair, singles_sorted[:, 2].to(torch.int64), torch.zeros((B,), dtype=torch.int64, device=self.device))

        top5 = r_sorted.to(torch.int64)

        tb = torch.zeros((B,), dtype=torch.int64, device=self.device)
        tb = torch.where(is_straight_flush, pack([sf_hi]), tb)
        tb = torch.where((cat == 7), pack([quad_rank.to(torch.int64), quads_kicker]), tb)
        tb = torch.where((cat == 6), pack([full_trip_i, full_pair_i]), tb)
        tb = torch.where((cat == 5), pack([top5[:, 0], top5[:, 1], top5[:, 2], top5[:, 3], top5[:, 4]]), tb)
        tb = torch.where((cat == 4), pack([st_hi]), tb)
        tb = torch.where((cat == 3), pack([trip_ranks[:, 0].to(torch.int64), trips_k1, trips_k2]), tb)
        tb = torch.where((cat == 2), pack([tp_hi, tp_lo, tp_k]), tb)
        tb = torch.where((cat == 1), pack([p_rank, p_k1, p_k2, p_k3]), tb)
        tb = torch.where((cat == 0), pack([top5[:, 0], top5[:, 1], top5[:, 2], top5[:, 3], top5[:, 4]]), tb)

        return cat * (base ** 6) + tb

    def _best7(self, ranks7: torch.Tensor, suits7: torch.Tensor) -> torch.Tensor:
        """ranks7,suits7: (N,7) -> returns (N,) int64 score (best 5‑card from 7)."""
        N = ranks7.shape[0]
        c = self.comb
        r5 = ranks7.unsqueeze(1).expand(-1, c.shape[0], -1).gather(2, c.unsqueeze(0).expand(N, -1, -1))
        s5 = suits7.unsqueeze(1).expand(-1, c.shape[0], -1).gather(2, c.unsqueeze(0).expand(N, -1, -1))
        B = N * c.shape[0]
        scores = self._eval5(r5.reshape(B, 5), s5.reshape(B, 5))
        scores = scores.view(N, c.shape[0]).max(dim=1).values
        return scores

    def equity_vs_n(self, hero_hole_ids: List[int], board_ids: List[int], opp_count: int, iters: int, dead_ids: List[int]) -> float:
        device = self.device
        total_sum = 0.0
        total_cnt = 0
        remaining = iters

        deck = torch.arange(52, device=device)
        dead_mask = torch.zeros(52, dtype=torch.bool, device=device)
        if dead_ids:
            dead_mask[torch.tensor(dead_ids, device=device)] = True
        avail = deck[~dead_mask]

        hero = torch.tensor(hero_hole_ids, device=device)
        board0 = torch.tensor(board_ids, device=device)
        needed_board = 5 - board0.numel()
        need = opp_count * 2 + needed_board

        while remaining > 0:
            B = min(self.mc_chunk, remaining)
            remaining -= B

            rand = torch.rand((B, avail.numel()), device=device)
            perm = torch.argsort(rand, dim=1)
            picks = avail[perm[:, :need]]  # (B, need)

            if opp_count > 0:
                opp = picks[:, :opp_count*2].reshape(B, opp_count, 2)
                board_rem = picks[:, opp_count*2:]
            else:
                opp = torch.empty((B, 0, 2), dtype=torch.long, device=device)
                board_rem = picks

            if needed_board > 0:
                board_full = torch.cat([board0.expand(B, -1), board_rem], dim=1)
            else:
                board_full = board0.expand(B, -1)

            hero7 = torch.cat([hero.expand(B, -1), board_full], dim=1)  # (B,7)
            hero_ranks = (hero7 % 13).to(torch.long)
            hero_suits = (hero7 // 13).to(torch.long)
            hero_scores = self._best7(hero_ranks, hero_suits)  # (B,)

            if opp_count > 0:
                opp7 = torch.cat([
                    opp.reshape(B*opp_count, 2),
                    board_full.repeat_interleave(opp_count, dim=0)
                ], dim=1)
                opp_r = (opp7 % 13).to(torch.long)
                opp_s = (opp7 // 13).to(torch.long)
                opp_scores = self._best7(opp_r.reshape(-1,7), opp_s.reshape(-1,7)).view(B, opp_count)
                opp_best = opp_scores.max(dim=1).values

                better = hero_scores > opp_best
                share = torch.zeros((B,), dtype=torch.float32, device=device)
                share = torch.where(better, torch.ones_like(share), share)

                not_better = ~better
                eq_counts = (opp_scores == hero_scores.unsqueeze(1)).sum(dim=1)
                hero_is_best_or_tie = hero_scores >= opp_best
                tie_rows = not_better & hero_is_best_or_tie
                share = torch.where(tie_rows, 1.0 / (eq_counts.float() + 1.0), share)
            else:
                share = torch.ones((B,), dtype=torch.float32, device=device)

            total_sum += float(share.sum().item())
            total_cnt += B

        return total_sum / max(1, total_cnt)


def autotune_mc_chunk(cuda_engine: CudaEquity, start: int = 8192, max_chunk: int = 1048576, factor: float = 2.0, safety: float = 0.85) -> Tuple[int, int]:
    """Probe the largest safe mc_chunk on current GPU.
    Returns (tuned_chunk, last_ok_chunk)."""
    if not _HAS_TORCH:
        return (start, start)
    if not torch.cuda.is_available():
        return (start, start)

    hero = [0, 13]  # 2♠, A♠
    board = []      # worst-case board fill (needs 5)
    dead = []
    opp = 5         # worst-case 6-max preflop opponents

    cand = max(1024, int(start))
    last_ok = cand
    while cand <= max_chunk:
        try:
            cuda_engine.mc_chunk = cand
            _ = cuda_engine.equity_vs_n(hero, board, opp, cand, dead)
            torch.cuda.synchronize()
            last_ok = cand
            cand = int(cand * factor)
        except RuntimeError as e:
            msg = str(e).lower()
            if 'out of memory' in msg or 'cublas' in msg or 'cuda error' in msg:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                break
            else:
                raise
    tuned = max(start, int(last_ok * safety))
    cuda_engine.mc_chunk = tuned
    return tuned, last_ok


# ------------------------------- Table & State --------------------------------

@dataclass
class PlayerState:
    pid: int
    hole: List[int] = field(default_factory=list)
    in_hand: bool = True
    committed: int = 0
    total_commit: int = 0
    last_aggressor: bool = False
    stack: int = 0
    all_in: bool = False

@dataclass
class HandState:
    button: int
    pot: int = 0
    street: str = 'preflop'
    board: List[int] = field(default_factory=list)
    current_bet: int = 0
    last_raise_size: int = 0
    raises_this_round: int = 0

# ------------------------------ Heuristic Policy ------------------------------

def _estimate_equity(hero: List[int], board: List[int], opp_count: int, iters: int, dead: List[int], device: str, cuda_engine: Optional[CudaEquity]) -> float:
    if device == 'cuda' and cuda_engine is not None:
        _, _, hero_ids = _treys_to_rs_ids(hero)
        _, _, board_ids = _treys_to_rs_ids(board)
        _, _, dead_ids = _treys_to_rs_ids(dead)
        return float(cuda_engine.equity_vs_n(hero_ids, board_ids, opp_count, iters, dead_ids))

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
        best_opp = min(evalr.evaluate(sim_board, h) for h in opp_hands) if opp_hands else 999999
        if hero_rank < best_opp: wins += 1
        elif hero_rank == best_opp: ties += 1
    return (wins + 0.5 * ties) / max(1, iters)


def decide_action(style: AgentStyle, hero_ps: PlayerState, players: List[PlayerState], H: HandState, to_call: int, iters: int, device: str, cuda_engine: Optional[CudaEquity]) -> Tuple[str, Optional[int]]:
    if hero_ps.all_in:
        return ('cc', None)

    opp_count = sum(1 for p in players if p.in_hand and p.pid != hero_ps.pid)
    equity = _estimate_equity(hero_ps.hole, H.board, opp_count, iters, dead=hero_ps.hole + H.board, device=device, cuda_engine=cuda_engine)

    pot_odds = to_call / max(1.0, (H.pot + to_call)) if to_call > 0 else 0.0
    base_thresh = 0.42
    if H.street == 'preflop': base_thresh = 0.46
    elif H.street == 'river': base_thresh = 0.47
    thresh = base_thresh + 0.05 * (style.tightness - 0.5)

    import random as _r
    want_aggr = equity > max(thresh, pot_odds + 0.10) and _r.random() < style.aggression

    if to_call == 0 and hero_ps.last_aggressor and _r.random() < style.cbet_freq and H.raises_this_round < 3:
        want_aggr = True

    if to_call == 0 and equity < 0.38 and _r.random() < style.bluff_freq and H.raises_this_round < 3:
        want_aggr = True

    if to_call == 0:
        if want_aggr:
            bet = style.sample_bet_size(H.street, max(H.pot, 50))
            target_to = max(H.current_bet + H.last_raise_size, bet) if H.current_bet > 0 else bet
            return ('cbr', max(5, int(round(target_to / 5) * 5)))
        else:
            return ('cc', None)
    else:
        if equity + 0.02 >= pot_odds or _r.random() < style.call_down:
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


def run_betting_round(styles: Dict[int, AgentStyle], players: List[PlayerState], H: HandState, order: List[int], iters: int, logs: List[str], device: str, cuda_engine: Optional[CudaEquity]):
    for p in players:
        p.committed = 0
        p.last_aggressor = False
    H.current_bet = max(H.current_bet, 0)
    H.raises_this_round = 0

    active_ids = [p.pid for p in players if p.in_hand and not p.all_in]
    if len(active_ids) <= 1:
        return

    acted: Dict[int, bool] = {p.pid: False for p in players if p.in_hand and not p.all_in}

    safety = 0
    while True:
        for pid in order:
            ps = next(p for p in players if p.pid == pid)
            if not ps.in_hand or ps.all_in:
                continue

            to_call = max(0, H.current_bet - ps.committed)

            if to_call == 0 and H.current_bet == 0 and all(acted.get(q.pid, True) for q in players if q.in_hand and not q.all_in and q.pid != pid):
                return

            action, target_to = decide_action(styles[pid], ps, players, H, to_call, iters, device, cuda_engine)

            if action == 'f':
                ps.in_hand = False
                logs.append(f"p{pid} f")
                acted.pop(pid, None)
            elif action == 'cc':
                if to_call > 0:
                    contribute(ps, to_call, H)
                logs.append(f"p{pid} cc")
                acted[pid] = True
            else:
                if H.current_bet == 0:
                    min_to = max(5, target_to)
                else:
                    min_to = max(H.current_bet + H.last_raise_size, target_to)
                all_in_to = ps.committed + ps.stack
                final_to = min(min_to, all_in_to)

                inc = final_to - ps.committed
                if inc <= 0:
                    if to_call > 0:
                        contribute(ps, to_call, H)
                    logs.append(f"p{pid} cc")
                    acted[pid] = True
                else:
                    contribute(ps, inc, H)
                    if final_to > H.current_bet:
                        if H.current_bet == 0:
                            H.last_raise_size = max(inc, 50)
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
                        logs.append(f"p{pid} cc")
                        acted[pid] = True

            if sum(1 for p in players if p.in_hand) == 1:
                return

        safety += 1
        if all(acted.get(p.pid, True) for p in players if p.in_hand and not p.all_in):
            all_matched = all((H.current_bet - p.committed) <= 0 for p in players if p.in_hand and not p.all_in)
            if all_matched:
                return
        if safety > 100:
            return

# ------------------------------- Hand Simulation ------------------------------

def simulate_hand(hand_idx: int, styles: Dict[int, AgentStyle], button: int, iters: int, stack_size: int, rng: random.Random, device: str, cuda_engine: Optional[CudaEquity]) -> List[str]:
    deck = Deck()
    logs: List[str] = []

    players = [PlayerState(pid=i+1, stack=stack_size) for i in range(6)]

    for ps in players:
        ps.hole = deck.draw(2)
        logs.append(f"d dh p{ps.pid} {_cards_concat(ps.hole)}")

    H = HandState(button=button)

    sb_pid = (button % 6) + 1
    bb_pid = (sb_pid % 6) + 1
    SB, BB = 50, 100

    sb = next(p for p in players if p.pid == sb_pid)
    bb = next(p for p in players if p.pid == bb_pid)

    contribute(sb, min(SB, sb.stack), H)
    contribute(bb, min(BB, bb.stack), H)

    H.current_bet = min(BB, bb.committed)
    H.last_raise_size = BB

    def order_preflop() -> List[int]:
        start = (bb_pid % 6) + 1
        return [(start + i - 1) % 6 + 1 for i in range(6)]

    def order_postflop() -> List[int]:
        start = (button % 6) + 1
        return [(start + i - 1) % 6 + 1 for i in range(6)]

    run_betting_round(styles, players, H, order_preflop(), iters, logs, device, cuda_engine)
    if sum(1 for p in players if p.in_hand) == 1:
        return logs

    H.street = 'flop'
    H.board.extend(deck.draw(3))
    logs.append(f"d db {_cards_concat(H.board)}")
    H.current_bet = 0
    H.last_raise_size = BB
    run_betting_round(styles, players, H, order_postflop(), iters, logs, device, cuda_engine)
    if sum(1 for p in players if p.in_hand) == 1:
        return logs

    H.street = 'turn'
    H.board.extend(deck.draw(1))
    logs.append(f"d db {_cards_concat(H.board[-1:])}")
    H.current_bet = 0
    run_betting_round(styles, players, H, order_postflop(), iters, logs, device, cuda_engine)
    if sum(1 for p in players if p.in_hand) == 1:
        return logs

    H.street = 'river'
    H.board.extend(deck.draw(1))
    logs.append(f"d db {_cards_concat(H.board[-1:])}")
    H.current_bet = 0
    run_betting_round(styles, players, H, order_postflop(), iters, logs, device, cuda_engine)

    survivors = [p for p in players if p.in_hand]
    if len(survivors) >= 2:
        ranks = [(p, evalr.evaluate(H.board, p.hole)) for p in survivors]
        best = min(r for _, r in ranks)
        winners = [p for p, r in ranks if r == best]
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
    ap.add_argument('--iters', type=int, default=200, help='MC iterations per decision')
    ap.add_argument('--stack', type=int, default=10000, help='Starting stack per player (resets each hand)')
    ap.add_argument('--out', type=str, required=True, help='Output file path (NDJSON: one JSON array per line)')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'], help='Equity device: cpu or cuda')
    ap.add_argument('--mc-chunk', type=int, default=8192, help='Batch size per CUDA chunk (memory control)')
    ap.add_argument('--autotune-chunk', action='store_true', help='Probe the largest safe mc_chunk on this GPU (CUDA only)')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    styles = make_varied_styles(args.seed)

    cuda_engine = None
    if args.device == 'cuda':
        if not _HAS_TORCH:
            raise SystemExit("PyTorch not installed. Install torch with CUDA to use --device=cuda.")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA not available. Install a CUDA‑enabled PyTorch build or select --device=cpu.")
        cuda_engine = CudaEquity('cuda', mc_chunk=args.mc_chunk)
        # Optional warm-up to initialize CUDA kernels
        try:
            _ = cuda_engine.equity_vs_n([0,13], [], 1, 1, [])
            torch.cuda.synchronize()
        except Exception:
            pass

        # Autotune mc_chunk if requested
        if args.autotune_chunk:
            tuned, last_ok = autotune_mc_chunk(cuda_engine, start=args.mc_chunk)
            print(f"[autotune] mc_chunk tuned to {tuned} (last OK: {last_ok})")

    button = rng.randint(1, 6)
    buffer_lines: List[str] = []
    hands_since_save = 0

    with open(args.out, 'w', encoding='utf-8') as f:
        for h in range(args.hands):
            logs = simulate_hand(h, styles, button, args.iters, args.stack, rng, args.device, cuda_engine)
            buffer_lines.append(json.dumps(logs) + "\n")
            button = (button % 6) + 1
            hands_since_save += 1

            if hands_since_save >= 100:
                f.writelines(buffer_lines)
                f.flush()
                buffer_lines.clear()
                hands_since_save = 0
                print(f"Saved {h+1} hands to {args.out}")

        if buffer_lines:
            f.writelines(buffer_lines)
            f.flush()
            print(f"Saved final {args.hands} hands to {args.out}")

if __name__ == '__main__':
    main()
