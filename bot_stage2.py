"""
Stage 2, frozen as the harness's baseline. Not part of the submission. One
change from commit b6367fe: fresh randomness each hand, as model.py has, so
the two compare hand for hand.

The bot: preflop charts by depth, then equity against an estimated range for
pot-odds calls and value bets. It does not bluff. The only thing it learns
between hands is how often each opponent raises big preflop and how good
their raising hands turn out to be, and how often they bet after the flop.

The charts and thresholds are starting values, tuned against heads-up play at
25 big blinds. With more players the same charts are tightened.
"""

import random
import time

from card_utils import Card, evaluate, full_deck
from game_state import GameStateView
from player_action import DEALER, PlayerAction
from preflop_ranges import RANGE_EQUITY, RANGE_STEPS
from preflop_table import PERCENTILE

# Seconds for one equity estimate, range building included. The limit is 120
# ms: the clock is only read now and then, so the deadline sits below it.
SIM_CAP = 0.10
CHECK_EVERY = 50  # trials or hands between looks at the clock
SIM_TRIALS = 800

ALL_IN_MARGIN = 0.01  # the same preflop, where nothing is left to lose later
THREE_BET_MARGIN = 0.08  # the same for calling a three-bet, with streets to come
PREFLOP_SHOVE = 0.65  # equity to shove over a big preflop raise
ALL_IN_SHARE = 0.85  # a raise this close to the whole stack becomes all-in

OUTSIDE_RANGE = 0.02  # weight of hands outside the estimated preflop range
BIG_CALL = 0.33  # a preflop call above this share of the stack is priced

# Hands in the opponent's preflop range, by how many raises came before the
# hand was played on: their own last raise, or a raise of ours they called.
RAISED_RANGE = {1: 0.70, 2: 0.20, 3: 0.08}
CALLED_RANGE = {0: 1.00, 1: 0.60, 2: 0.30, 3: 0.12}
TIGHTEST_RANGE = 0.05

# How wide a big preflop raise is, learned per opponent: the share of hands
# they do it with, starting from a wide guess worth a few hands.
BIG_RAISE = 1 / 3  # a raise to this share of their stack counts as big
BIG_RAISE_PRIOR = 0.40
BIG_RAISE_WEIGHT = 5
SHOVE_HAPPY = 0.50  # from this rate on, raising small only gets shoved on

# How often says little about how good: a maniac shoves 15% of hands, not the
# best 15%. Hands shown after a raise correct the width. A range that is even
# over the top X% averages X / 2, so twice the average shown is the width.
SHOWN_WEIGHT = 3  # the rate's weight against shown hands
SMALL_RAISE_SHOWN = 0.25  # a hand shown after a raise that was not big

# A bet narrows a range less against someone who bets too often to be
# honest, or who shows weak hands after betting. How often alone is not
# enough: a bot that bets every good hand can bet half the time.
BET_RATE_PRIOR = 0.35
BET_RATE_WEIGHT = 10
HONEST_BET_RATE = 0.55  # above this the narrowing starts to go
RECKLESS_BET_RATE = 0.75  # at this it has gone
WEAK_SHOWN_PRIOR = 0.15
WEAK_SHOWN_WEIGHT = 6
RECKLESS_WEAK_SHOWN = 0.45  # share of weak hands at which it has gone


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
    Kept from hand to hand, per team: big_raises, [hands they acted in,
    hands they raised big in], and shown, [weight, weighted percentile] of
    the hands they showed after a preflop raise, bets, [postflop turns,
    turns they bet or raised on], and bet_shown, [hands shown after a
    postflop bet, those that were weak].
    """

    VALUE_BET = 0.54  # equity to bet when checked to
    VALUE_RAISE = 0.68  # equity to raise a bet
    CALL_MARGIN = 0.0  # equity above the pot odds to call a bet postflop
    CHART_SCALE = 1.25  # above 1 plays more hands preflop, below 1 fewer

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.hand_name = ""
        self.percentile = 1.0  # the hole cards' strength; 1.0 is the weakest
        self.log = []  # every action this hand, oldest first
        self.seat_chips = ()  # each seat's chips when the hand began
        self.big_raises = {}
        self.shown = {}
        self.bets = {}
        self.bet_shown = {}
        self.sim_trials = SIM_TRIALS
        self.rng = random.Random(20261107)
        self.hands_seen = 0
        self.errors = 0
        self.strict = False

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        # Fresh randomness each hand: what happens in one hand cannot shift
        # the next, so two versions compare hand for hand on the same decks.
        self.hands_seen += 1
        self.rng.seed(f"{self.hands_seen}:{hand}")
        self.log = []
        self.seat_chips = ()
        self.percentile = 1.0
        self.hand_name = ""
        try:
            self.hand_name = hand_key(hand[0], hand[1])
            self.percentile = PERCENTILE[self.hand_name]
        except Exception:
            self.errors += 1
            if self.strict:
                raise

    def run(
        self, actions: list[PlayerAction], game_state: GameStateView
    ) -> PlayerAction:
        try:
            self.log.extend(actions)
            if not self.seat_chips and game_state.street == "preflop":
                self.seat_chips = tuple(
                    stack + bet for stack, bet in
                    zip(game_state.stacks, game_state.bets)
                )
            action = self._decide(game_state)
            game_state.validate_action(action)
            return action
        except Exception:
            self.errors += 1
            if self.strict:
                raise
            return self._check_or_fold(game_state)

    def end_game(
        self, actions: list[PlayerAction], game_state: GameStateView
    ):
        """Counts this hand for every opponent: only here is it complete."""
        try:
            self.log.extend(actions)
            self._count_big_raises(game_state)
            self._count_shown_hands(game_state)
            self._count_bets(game_state)
        except Exception:
            self.errors += 1
            if self.strict:
                raise

    # -- what we keep between hands ------------------------------------------

    def _is_big(self, action: PlayerAction) -> bool:
        seat = action.player_id
        return (
            action.kind == "raise"
            and seat < len(self.seat_chips)
            and action.amount >= BIG_RAISE * self.seat_chips[seat]
        )

    def _is_all_in(self, action: PlayerAction) -> bool:
        seat = action.player_id
        return (
            action.kind == "raise"
            and seat < len(self.seat_chips)
            and action.amount >= self.seat_chips[seat]
        )

    def _count_big_raises(self, gs: GameStateView):
        acted = set()
        big = set()
        for action in self.log:
            if action.player_id == DEALER:
                break  # the flop: preflop betting is over
            if action.kind in ("fold", "call", "raise"):
                acted.add(action.player_id)
                if self._is_big(action):
                    big.add(action.player_id)
                if self._is_all_in(action):
                    break  # nobody after this could have raised
        for seat in acted:
            if seat != self.player_id:
                counts = self.big_raises.setdefault(gs.teams[seat], [0, 0])
                counts[0] += 1
                counts[1] += seat in big

    def _count_shown_hands(self, gs: GameStateView):
        raised = {}  # seat: how much a hand it shows says about big raises
        holes = {}
        preflop = True
        for action in self.log:
            seat = action.player_id
            if seat == DEALER:
                preflop = False
            elif action.kind == "raise" and preflop:
                weight = 1.0 if self._is_big(action) else SMALL_RAISE_SHOWN
                raised[seat] = max(raised.get(seat, 0.0), weight)
            elif action.kind == "reveal_card":
                holes.setdefault(seat, []).append(action.card)
        for seat, weight in raised.items():
            if seat != self.player_id and len(holes.get(seat, ())) == 2:
                percentile = PERCENTILE[hand_key(*holes[seat])]
                counts = self.shown.setdefault(gs.teams[seat], [0.0, 0.0])
                counts[0] += weight
                counts[1] += weight * percentile

    def _count_bets(self, gs: GameStateView):
        reveals = 0
        bettors = set()
        holes = {}
        for action in self.log:
            seat = action.player_id
            if seat == DEALER:
                reveals += 1
            elif action.kind == "reveal_card":
                holes.setdefault(seat, []).append(action.card)
            elif (reveals >= 3 and seat != self.player_id
                    and action.kind in ("fold", "call", "raise")):
                counts = self.bets.setdefault(gs.teams[seat], [0, 0])
                counts[0] += 1
                if action.kind == "raise":
                    counts[1] += 1
                    bettors.add(seat)
        board = list(gs.board)
        for seat in bettors:
            if len(holes.get(seat, ())) == 2 and len(board) == 5:
                counts = self.bet_shown.setdefault(gs.teams[seat], [0, 0])
                counts[0] += 1
                counts[1] += self._is_weak(holes[seat], board)

    def _is_weak(self, hole: list[Card], board: list[Card]) -> bool:
        """Nothing a bet could be for value: no pair, or under the top card."""
        rank = evaluate(hole + board)
        top = max(card.value for card in board)
        return rank[0] == 0 or (rank[0] == 1 and rank[1] < top)

    def _bet_rate(self, team: str) -> float:
        """The share of their postflop turns this team bets or raises on."""
        turns, bets = self.bets.get(team, (0, 0))
        prior = BET_RATE_PRIOR * BET_RATE_WEIGHT
        return (bets + prior) / (turns + BET_RATE_WEIGHT)

    def _weak_shown_rate(self, team: str) -> float:
        """The share of hands shown after a bet that were weak."""
        shows, weak = self.bet_shown.get(team, (0, 0))
        prior = WEAK_SHOWN_PRIOR * WEAK_SHOWN_WEIGHT
        return (weak + prior) / (shows + WEAK_SHOWN_WEIGHT)

    def _loosening(self, team: str) -> float:
        """How far to discount this team's bets, from 0 (not at all) to 1
        (they say nothing), by the stronger of the two signs."""
        often = (self._bet_rate(team) - HONEST_BET_RATE) / (
            RECKLESS_BET_RATE - HONEST_BET_RATE
        )
        weak = (self._weak_shown_rate(team) - WEAK_SHOWN_PRIOR) / (
            RECKLESS_WEAK_SHOWN - WEAK_SHOWN_PRIOR
        )
        return min(1.0, max(0.0, often, weak))

    def _big_raise_width(self, team: str) -> float:
        """The share of hands a big raise from this team comes from: as
        wide as they do it often, corrected by what they have shown."""
        weight, total = self.shown.get(team, (0.0, 0.0))
        rate = self._big_raise_rate(team)
        width = (rate * SHOWN_WEIGHT + 2 * total) / (SHOWN_WEIGHT + weight)
        return min(1.0, max(TIGHTEST_RANGE, width))

    def _big_raise_rate(self, team: str) -> float:
        """The share of hands this team raises big with, so far."""
        hands, big = self.big_raises.get(team, (0, 0))
        prior = BIG_RAISE_PRIOR * BIG_RAISE_WEIGHT
        return (big + prior) / (hands + BIG_RAISE_WEIGHT)

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
        hand = self.percentile / 0.6 ** (len(others) - 1) / self.CHART_SCALE
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
        others = [p for p in gs.players_in_hand if p != self.player_id]
        if len(others) == 1:
            team = gs.teams[others[0]]
            if self._big_raise_rate(team) >= SHOVE_HAPPY:
                return self._against_a_shover(gs, self._big_raise_width(team))
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

    def _against_a_shover(self, gs, width: float) -> PlayerAction:
        """Any raise of ours gets shoved on, so a hand that would fold to
        that folds now, for less, and one that would call may as well shove."""
        chips = gs.stacks[self.player_id] + gs.bets[self.player_id]
        price = gs.stacks[self.player_id] / (2 * chips)
        if self._equity(gs, width) > price + ALL_IN_MARGIN:
            return self._shove(gs)
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
        """Our raise was raised, and calling is still cheap: the price
        decides, so a min-raise is not folded to like a big three-bet."""
        if depth > 40:
            if hand <= 0.04:
                return self._raise_to(gs, 2.3 * gs.current_bet)
        elif hand <= 0.10:
            return self._shove(gs)
        equity = self._equity(gs, RAISED_RANGE[2])
        if equity > gs.pot_odds + THREE_BET_MARGIN:
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
        if equity >= self.VALUE_RAISE and gs.can_raise:
            if gs.spr(self.player_id) <= 1:
                return self._shove(gs)
            return self._pot_raise(gs, 0.75)
        if equity > gs.pot_odds + self.CALL_MARGIN:
            return PlayerAction.call(self.player_id)
        return self._check_or_fold(gs)

    def _checked_to(self, gs: GameStateView, equity: float) -> PlayerAction:
        if equity < self.VALUE_BET:
            return PlayerAction.call(self.player_id)
        if gs.spr(self.player_id) <= 1:
            return self._shove(gs)
        return self._pot_raise(
            gs, 0.75 if equity >= self.VALUE_RAISE else 0.5
        )

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
        loosening = max(
            self._loosening(gs.teams[p]) for p in gs.players_in_hand
            if p != self.player_id
        )
        keep += (1.0 - keep) * loosening
        return max(keep, 0.12), rest, cap

    def _weighted_hands(
        self,
        gs: GameStateView,
        board: list[Card],
        deadline: float,
        limit: float,
    ) -> list[tuple[float, Card, Card, tuple]]:
        """The opponent's possible hands as (weight, first, second, rank),
        rank being theirs on this board."""
        dead = set(self.hand) | set(board)
        hands = [
            (1.0 if strength <= limit else OUTSIDE_RANGE, first, second)
            for first, second, strength in COMBOS
            if first not in dead and second not in dead
        ]
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
        """The share of hands the opponent can hold: from how often they do
        this if they just raised big preflop, else from the betting."""
        if gs.street == "preflop":
            for action in reversed(self.log):
                if action.kind == "raise":
                    seat = action.player_id
                    if seat != self.player_id and (
                        self._is_big(action) or gs.status(seat) == "all_in"
                    ):
                        return self._big_raise_width(gs.teams[seat])
                    break
        return self._preflop_range()

    def _equity(self, gs: GameStateView, limit: float | None = None) -> float:
        """Our share of the pot against the opponent's weighted hands, whose
        range is limit wide (read from the betting if None). With several
        opponents, roughly: we must beat each one."""
        start = time.perf_counter()
        board = list(gs.board)
        if limit is None:
            limit = self._range_limit(gs)
        if not board:
            step = max(1, round(limit * RANGE_STEPS))
            equity = RANGE_EQUITY[self.hand_name][step - 1]
        elif len(board) == 5:  # ranking every hand is the whole job
            hands = self._weighted_hands(gs, board, start + SIM_CAP, limit)
            equity = self._river_equity(hands, board)
        else:
            hands = self._weighted_hands(gs, board, start + SIM_CAP / 2, limit)
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
        sim_trials or the deadline, whichever comes first."""
        known = set(self.hand) | set(board)
        unseen = [card for card in DECK if card not in known]
        missing = 5 - len(board)
        picks = self.rng.choices(
            hands, weights=[hand[0] for hand in hands], k=self.sim_trials
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
