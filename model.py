"""
The bot, Stage 1: preflop charts by depth, then equity against an estimated
range for pot-odds calls and value bets. It does not bluff and learns nothing
between hands.

The charts and thresholds are starting values, tuned against heads-up play at
25 big blinds. With more players the same charts are tightened.
"""

import random
import time

from card_utils import Card, evaluate, full_deck
from game_state import GameStateView
from player_action import DEALER, PlayerAction
from preflop_table import PERCENTILE

SIM_CAP = 0.12  # seconds for one equity estimate, range building included
CHECK_EVERY = 50  # trials or hands between looks at the clock
SIM_TRIALS = 800

VALUE_BET = 0.60  # equity to bet when checked to
VALUE_RAISE = 0.75  # equity to raise a bet
CALL_MARGIN = 0.04  # equity above the pot odds to call a bet postflop
ALL_IN_MARGIN = 0.01  # the same preflop, where nothing is left to lose later
PREFLOP_SHOVE = 0.65  # equity to shove over a big preflop raise
ALL_IN_SHARE = 0.85  # a raise this close to the whole stack becomes all-in

OUTSIDE_RANGE = 0.02  # weight of hands outside the estimated preflop range
BIG_CALL = 0.33  # a preflop call above this share of the stack is priced

# Hands in the opponent's preflop range, by how many raises came before the
# hand was played on: their own last raise, or a raise of ours they called.
RAISED_RANGE = {1: 0.70, 2: 0.20, 3: 0.08}
CALLED_RANGE = {0: 1.00, 1: 0.60, 2: 0.30, 3: 0.12}
TIGHTEST_RANGE = 0.05
SHOVE_RANGE = 0.20  # a preflop all-in with at most one raise before it


def hand_key(first: Card, second: Card) -> str:
    """A starting hand's name in preflop_table: "AA", "AKs" or "AKo"."""
    high, low = sorted((first, second), key=lambda c: c.value, reverse=True)
    if high.rank == low.rank:
        return high.rank + low.rank
    return high.rank + low.rank + ("s" if high.suit == low.suit else "o")


def _all_combos() -> list[tuple[Card, Card, float]]:
    """Every two-card hand, with how strong it is before the flop."""
    deck = full_deck()
    return [
        (first, second, PERCENTILE[hand_key(first, second)])
        for i, first in enumerate(deck)
        for second in deck[i + 1:]
    ]


# Module constants come with the code, not the pickle.
DECK = full_deck()
COMBOS = _all_combos()


class Model:
    """
    Plays a hand in three steps: the chart picks preflop moves, a simulation
    against the opponent's likely hands picks postflop ones, and run checks
    every move before returning it.

    Attributes are plain data, as the model is pickled. errors counts the
    exceptions caught; with strict set they are raised again, so a test
    sees a bug that would otherwise look like a quiet check or fold.
    """

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.percentile = 1.0  # the hole cards' strength; 1.0 is the weakest
        self.log = []  # every action this hand, oldest first
        self.rng = random.Random(20261107)
        self.errors = 0
        self.strict = False

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self.log = []
        self.percentile = 1.0
        try:
            self.percentile = PERCENTILE[hand_key(hand[0], hand[1])]
        except Exception:
            self.errors += 1
            if self.strict:
                raise

    def run(
        self, actions: list[PlayerAction], game_state: GameStateView
    ) -> PlayerAction:
        try:
            self.log.extend(actions)
            action = self._decide(game_state)
            game_state.validate_action(action)
            return action
        except Exception:
            self.errors += 1
            if self.strict:
                raise
            return self._check_or_fold(game_state)

    # -- moves ---------------------------------------------------------------

    def _check_or_fold(self, gs: GameStateView) -> PlayerAction:
        """The safe move: an error should never pay for anything."""
        if gs.call_amount == 0:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)

    def _raise_to(self, gs: GameStateView, target: float) -> PlayerAction:
        """
        A legal raise as near target (a street total) as the rules allow,
        all-in if that is nearly the whole stack; a call if raising is not
        allowed. Every raise the bot makes comes through here.
        """
        if not gs.can_raise:
            return PlayerAction.call(self.player_id)
        low, high = gs.raise_range
        total = min(high, max(low, int(round(target))))
        if total >= ALL_IN_SHARE * high:
            total = high
        return PlayerAction.raise_to(self.player_id, total)

    def _shove(self, gs: GameStateView) -> PlayerAction:
        return self._raise_to(gs, gs.max_raise_to)

    def _pot_raise(self, gs: GameStateView, fraction: float) -> PlayerAction:
        """A bet or raise of fraction of the pot, counting our call in it."""
        pot = gs.pot + gs.call_amount
        return self._raise_to(gs, gs.current_bet + fraction * pot)

    def _decide(self, gs: GameStateView) -> PlayerAction:
        if gs.street == "preflop":
            return self._preflop(gs)
        return self._postflop(gs)

    # -- preflop -------------------------------------------------------------

    def _preflop(self, gs: GameStateView) -> PlayerAction:
        me = self.player_id
        others = [p for p in gs.players_in_hand if p != me]
        biggest = max(gs.stacks[p] + gs.bets[p] for p in others)
        depth = min(gs.stacks[me] + gs.bets[me], biggest) / gs.big_blind
        # The charts are heads-up: each extra player counts the hand weaker.
        hand = self.percentile / 0.6 ** (len(others) - 1)
        raises = sum(a.kind == "raise" for a in self.log)
        if raises == 0:
            if gs.call_amount == 0:
                return self._after_limps(gs, hand, depth)
            return self._first_in(gs, hand, depth)
        all_in = any(gs.status(p) == "all_in" for p in others)
        big = gs.call_amount > BIG_CALL * gs.stacks[me]
        if all_in or big or raises >= 3:
            return self._priced_call(gs)
        if raises == 1:
            return self._against_open(gs, hand, depth)
        return self._against_three_bet(gs, hand, depth)

    def _first_in(self, gs, hand: float, depth: float) -> PlayerAction:
        """Nobody has raised and we owe chips: raise or fold, never limp."""
        big_blind = gs.big_blind
        if depth <= 10:
            share = 0.85 if depth <= 5 else 0.68 if depth <= 8 else 0.58
            return self._shove(gs) if hand <= share else self._check_or_fold(gs)
        if depth <= 18:
            # Middling hands shove; the best min-raise and call a shove.
            shove_share = 0.30 - (depth - 10) * 0.15 / 8
            if 0.06 < hand <= shove_share:
                return self._shove(gs)
            if hand <= 0.70:
                return self._raise_to(gs, 2 * big_blind)
            return self._check_or_fold(gs)
        if hand <= 0.75:
            return self._raise_to(gs, 2.5 * big_blind)
        return self._check_or_fold(gs)

    def _after_limps(self, gs, hand: float, depth: float) -> PlayerAction:
        """Nobody has raised and checking is free."""
        big_blind = gs.big_blind
        if depth <= 10:
            if hand <= 0.30:
                return self._shove(gs)
        elif depth <= 18:
            if hand <= 0.25:
                return self._raise_to(gs, 3 * big_blind)
        elif hand <= 0.35:
            return self._raise_to(gs, 3.5 * big_blind)
        return PlayerAction.call(self.player_id)

    def _against_open(self, gs, hand: float, depth: float) -> PlayerAction:
        """One modest raise in front of us."""
        # The defending share is set for a 2.5 big blind open: a cheaper
        # raise is defended wider, a dearer one tighter.
        price = (0.30 / max(gs.pot_odds, 0.05)) ** 1.5
        call_share = min(0.85, max(0.05, price * 0.57))
        if depth <= 18:
            if hand <= 0.15:
                return self._shove(gs)
        elif depth <= 40:
            if hand <= 0.05:
                return self._raise_to(gs, 3.5 * gs.current_bet)
            if hand <= 0.12:
                return self._shove(gs)
        else:
            call_share = min(call_share, 0.45)
            if hand <= 0.08:
                return self._raise_to(gs, 3.5 * gs.current_bet)
        if hand <= call_share:
            return PlayerAction.call(self.player_id)
        return self._check_or_fold(gs)

    def _against_three_bet(self, gs, hand: float, depth: float) -> PlayerAction:
        """Our raise was raised, and calling is still cheap."""
        if depth > 40:
            if hand <= 0.04:
                return self._raise_to(gs, 2.3 * gs.current_bet)
            call_share = 0.22
        else:
            if hand <= 0.10:
                return self._shove(gs)
            call_share = 0.25
        if hand <= call_share:
            return PlayerAction.call(self.player_id)
        return self._check_or_fold(gs)

    def _priced_call(self, gs: GameStateView) -> PlayerAction:
        """A big preflop decision: our equity against their range decides."""
        equity = self._equity(gs)
        if equity >= PREFLOP_SHOVE and gs.can_raise:
            return self._shove(gs)
        if equity > gs.pot_odds + ALL_IN_MARGIN:
            return PlayerAction.call(self.player_id)
        return self._check_or_fold(gs)

    # -- postflop ------------------------------------------------------------

    def _postflop(self, gs: GameStateView) -> PlayerAction:
        equity = self._equity(gs)
        if gs.call_amount == 0:
            return self._checked_to(gs, equity)
        if equity >= VALUE_RAISE and gs.can_raise:
            if gs.spr(self.player_id) <= 1:
                return self._shove(gs)
            return self._pot_raise(gs, 0.75)
        if equity > gs.pot_odds + CALL_MARGIN:
            return PlayerAction.call(self.player_id)
        return self._check_or_fold(gs)

    def _checked_to(self, gs: GameStateView, equity: float) -> PlayerAction:
        if equity < VALUE_BET:
            return PlayerAction.call(self.player_id)
        if gs.spr(self.player_id) <= 1:
            return self._shove(gs)
        return self._pot_raise(gs, 0.75 if equity >= VALUE_RAISE else 0.5)

    # -- what the opponent holds ---------------------------------------------

    def _preflop_range(self) -> float:
        """The share of hands the opponent still has, from preflop betting:
        how many raises there were and who made the last one."""
        raises = 0
        last_raiser = None
        for action in self.log:
            if action.player_id == DEALER:
                break  # the flop: preflop betting is over
            if action.kind == "raise":
                raises += 1
                last_raiser = action.player_id
        if last_raiser == self.player_id or raises == 0:
            return CALLED_RANGE.get(raises, TIGHTEST_RANGE)
        return RAISED_RANGE.get(raises, TIGHTEST_RANGE)

    def _board_pressure(self, gs: GameStateView) -> tuple[float, float, float]:
        """
        What postflop betting says about the opponent, as (keep, rest, cap):
        the strongest keep share of their range stays at full weight, the
        hands below it are scaled by rest (draws and bluffs do not vanish),
        and the strongest cap share is halved, as a strong hand would
        usually have bet.
        """
        keep, rest, cap = 1.0, 1.0, 0.0
        street = 0
        reveals = 0
        they_bet = False  # on the street being read
        we_bet = False
        for action in self.log:
            if action.player_id == DEALER:
                reveals += 1
                if reveals >= 3:
                    street = reveals - 2
                    they_bet = we_bet = False
            elif street == 0:
                continue
            elif action.kind == "raise":
                if action.player_id == self.player_id:
                    we_bet = True
                else:
                    # A raise over a bet of ours says more than a first bet;
                    # an earlier street's bet is worth less than this one's.
                    keep *= 0.60 if we_bet else 0.80
                    rest = 0.15
                    they_bet = True
            elif action.kind == "call" and action.player_id != self.player_id:
                if we_bet:  # paying to continue folds out the worst hands
                    keep *= 0.85
                    rest = min(rest, 0.30)
                elif not they_bet and street == len(gs.board) - 2:
                    cap = 0.15  # checked to us on this street
        if gs.call_amount > 0:
            size = gs.call_amount / max(1, gs.pot - gs.call_amount)
            keep *= 0.75 if size <= 0.5 else 0.60
            cap = 0.0
        return max(keep, 0.12), rest, cap

    def _weighted_hands(
        self, gs: GameStateView, board: list[Card], deadline: float
    ) -> list[tuple[float, Card, Card, tuple | None]]:
        """The opponent's possible hands as (weight, first, second, rank):
        rank is theirs on this board, None before the flop."""
        dead = set(self.hand) | set(board)
        limit = self._range_limit(gs)
        hands = [
            (1.0 if strength <= limit else OUTSIDE_RANGE, first, second)
            for first, second, strength in COMBOS
            if first not in dead and second not in dead
        ]
        if not board:
            return [(w, first, second, None) for w, first, second in hands]
        self.rng.shuffle(hands)  # so running out of time cuts at random
        ranked = []
        for i, (weight, first, second) in enumerate(hands):
            if i and i % CHECK_EVERY == 0 and time.perf_counter() > deadline:
                break
            rank = evaluate([first, second] + board)
            ranked.append((weight, first, second, rank))
        ranked.sort(key=lambda hand: hand[3], reverse=True)
        keep, rest, cap = self._board_pressure(gs)
        total = sum(hand[0] for hand in ranked)
        adjusted = []
        seen = 0.0
        for weight, first, second, rank in ranked:
            share = seen / total
            seen += weight
            if share >= keep:
                weight *= rest
            elif share < cap:
                weight *= 0.5
            adjusted.append((weight, first, second, rank))
        return adjusted

    def _range_limit(self, gs: GameStateView) -> float:
        limit = self._preflop_range()
        if gs.street == "preflop":
            others = [p for p in gs.players_in_hand if p != self.player_id]
            shoved = any(gs.status(p) == "all_in" for p in others)
            raises = sum(a.kind == "raise" for a in self.log)
            if shoved and raises <= 2:
                return SHOVE_RANGE
        return limit

    def _equity(self, gs: GameStateView) -> float:
        """Our share of the pot against the opponent's weighted hands. With
        several opponents, roughly: we must beat each one."""
        start = time.perf_counter()
        board = list(gs.board)
        if len(board) == 5:  # ranking every hand is the whole job
            hands = self._weighted_hands(gs, board, start + SIM_CAP)
            equity = self._river_equity(hands, board)
        else:
            hands = self._weighted_hands(gs, board, start + SIM_CAP / 2)
            equity = self._simulate(hands, board, start + SIM_CAP)
        opponents = len(gs.players_in_hand) - 1
        return equity ** opponents

    def _river_equity(self, hands, board: list[Card]) -> float:
        """Exact: every hand they could hold is already ranked."""
        mine = evaluate(self.hand + board)
        won = total = 0.0
        for weight, _, _, rank in hands:
            total += weight
            won += weight if mine > rank else weight / 2 if mine == rank else 0
        return won / total

    def _simulate(self, hands, board: list[Card], deadline: float) -> float:
        """Deals out the board against hands drawn by weight, until
        SIM_TRIALS or the deadline, whichever comes first."""
        known = set(self.hand) | set(board)
        unseen = [card for card in DECK if card not in known]
        missing = 5 - len(board)
        picks = self.rng.choices(
            hands, weights=[hand[0] for hand in hands], k=SIM_TRIALS
        )
        won = 0.0
        played = 0
        for _, first, second, _ in picks:
            if played and played % CHECK_EVERY == 0:
                if time.perf_counter() > deadline:
                    break
            drawn = self.rng.sample(unseen, missing + 2)
            rest = [c for c in drawn if c != first and c != second][:missing]
            full_board = board + rest
            ours = evaluate(self.hand + full_board)
            theirs = evaluate([first, second] + full_board)
            won += 1.0 if ours > theirs else 0.5 if ours == theirs else 0.0
            played += 1
        return won / played
