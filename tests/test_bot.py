"""
Tests for the team's bot in model.py, beyond what tests/test_model.py checks.

    .venv/bin/python -m pytest tests/test_bot.py -q

test_model.py cannot see a bug that the bot's own try/except swallows: the
move just becomes a check or fold and counts as legal. So everything here
plays the model in-process with strict set, where a caught exception is
raised again and shows up as a fault.
"""

import pickle
import random
import shutil
import time

import pytest
from hypothesis import given, settings, strategies as st

import model as bot
from arena import vet
from arena.dummies import DUMMIES
from card_utils import Card, full_deck
from engine import play_hand
from game_state import GameState, GameStateView, InvalidAction
from model import Model, hand_key
from player_action import PlayerAction as A
from preflop_ranges import RANGE_EQUITY, RANGE_STEPS
from preflop_table import EQUITY, PERCENTILE


def cards(text: str) -> list[Card]:
    return [Card.parse(card) for card in text.split()]


def bot_with(hand: str, seat: int) -> Model:
    model = Model()
    model.strict = True
    model.seed(cards(hand), seat)
    return model


def table(*moves, stacks=(500, 500)) -> GameState:
    """A heads-up hand, button 0, after moves. A move is a PlayerAction or a
    string of board cards, dealt in turn."""
    state = GameState(list(stacks))
    for move in moves:
        if isinstance(move, str):
            for card in cards(move):
                state.process_action(A.reveal_card(card))
        else:
            state.process_action(move)
    return state


def log_of(*moves) -> list[A]:
    """moves as the log a model would have built up."""
    log = []
    for move in moves:
        if isinstance(move, str):
            log += [A.reveal_card(card) for card in cards(move)]
        else:
            log.append(move)
    return log


def decide(hand: str, seat: int, *moves, stacks=(500, 500)) -> A:
    """What the bot in seat plays after moves."""
    model = bot_with(hand, seat)
    state = table(*moves, stacks=stacks)
    assert state.to_act == seat
    return model.run(log_of(*moves), GameStateView(state))


# -- the table -----------------------------------------------------------------


def test_the_table_ranks_all_169_hands():
    assert len(EQUITY) == len(PERCENTILE) == 169
    values = list(EQUITY.values())
    assert values == sorted(values, reverse=True)
    assert list(PERCENTILE) == list(EQUITY)
    shares = list(PERCENTILE.values())
    assert shares == sorted(shares) and shares[-1] == 1.0
    assert list(EQUITY)[0] == "AA"
    assert EQUITY["AA"] == pytest.approx(0.852, abs=0.01)
    assert EQUITY["72o"] == pytest.approx(0.346, abs=0.01)
    assert PERCENTILE["AKs"] < PERCENTILE["T9s"] < PERCENTILE["72o"]


def test_the_range_table_covers_every_hand_and_width():
    assert set(RANGE_EQUITY) == set(EQUITY)
    assert all(len(row) == RANGE_STEPS for row in RANGE_EQUITY.values())
    for hand, row in RANGE_EQUITY.items():
        # The widest range is every hand: the same as against a random one.
        assert row[-1] == pytest.approx(EQUITY[hand], abs=0.025)
    # Strong hands lose equity as the range tightens.
    assert RANGE_EQUITY["K9o"][1] < RANGE_EQUITY["K9o"][-1] - 0.1
    assert RANGE_EQUITY["AA"][0] > 0.8


def test_hand_key_orders_ranks_and_reads_suits():
    assert hand_key(*cards("AS AH")) == "AA"
    assert hand_key(*cards("KD AD")) == "AKs"
    assert hand_key(*cards("2C 7D")) == "72o"
    assert len(bot.COMBOS) == 1326


# -- preflop -------------------------------------------------------------------


def test_the_button_opens_strong_hands_and_folds_the_worst():
    assert decide("AS AH", 0) == A.raise_to(0, 50)
    assert decide("7C 2D", 0) == A.fold(0)


def test_the_big_blind_checks_free_rather_than_folds():
    assert decide("7C 2D", 1, A.call(0)) == A.call(1)
    assert decide("AS AH", 1, A.call(0)) == A.raise_to(1, 70)


def test_the_big_blind_defends_by_hand_strength_and_price():
    assert decide("7C 2D", 1, A.raise_to(0, 50)) == A.fold(1)
    assert decide("KD 9C", 1, A.raise_to(0, 50)) == A.call(1)
    three_bet = decide("AS AH", 1, A.raise_to(0, 50))
    assert three_bet == A.raise_to(1, 175)


def test_all_ins_are_called_on_equity():
    shove = A.raise_to(0, 500)
    assert decide("AS AH", 1, shove) == A.call(1)
    assert decide("7C 2D", 1, shove) == A.fold(1)


def _seen_shoving(model: Model, team: str, shoves: int, hands: int):
    model.big_raises[team] = [hands, shoves]


def test_the_first_shove_gets_a_wide_prior_not_a_tight_range():
    assert 0.3 <= Model()._big_raise_rate("seat 0") <= 0.5
    limit = _shove_limit(Model())
    assert limit == pytest.approx(Model()._big_raise_rate("seat 0"))


def _shove_limit(model: Model) -> float:
    model.seed(cards("AS AH"), 1)
    moves = (A.raise_to(0, 500),)
    model.log = log_of(*moves)
    return model._range_limit(GameStateView(table(*moves)))


def test_the_range_follows_how_often_they_shove():
    always, never = Model(), Model()
    _seen_shoving(always, "seat 0", 40, 40)
    _seen_shoving(never, "seat 0", 0, 200)
    assert _shove_limit(always) > 0.9
    assert _shove_limit(never) == bot.TIGHTEST_RANGE


def test_shown_hands_correct_a_rate_that_says_nothing_about_strength():
    random_cards, premiums = Model(), Model()
    for model, average in ((random_cards, 0.5), (premiums, 0.02)):
        _seen_shoving(model, "seat 0", 15, 100)
        model.shown["seat 0"] = [12.0, 12 * average]
    assert _shove_limit(random_cards) > 0.8
    assert _shove_limit(premiums) < 0.08


def test_end_game_records_a_raiser_s_shown_hand():
    state = GameState([500, 500], teams=["us", "them"])
    model = bot_with("AS AH", 0)
    model.seat_chips = (500, 500)
    moves = [A.raise_to(0, 500), A.raise_to(1, 500) if False else A.call(1)]
    for move in moves:
        state.process_action(move)
    board = cards("2C 3D 4H 9S JC")
    deal = [A.reveal_card(card) for card in board]
    for action in deal:
        state.process_action(action)
    # They called, so nothing is learned about their raising.
    shown = [A.reveal_card(Card.parse("7C"), 1),
             A.reveal_card(Card.parse("2D"), 1)]
    model.end_game(moves + deal + shown, GameStateView(state))
    assert model.shown == {}

    model = bot_with("AS AH", 1)
    model.seat_chips = (500, 500)
    state = GameState([500, 500], teams=["them", "us"])
    moves = [A.raise_to(0, 500), A.call(1)]
    for move in moves:
        state.process_action(move)
    for action in deal:
        state.process_action(action)
    shown = [A.reveal_card(Card.parse("7C"), 0),
             A.reveal_card(Card.parse("2D"), 0)]
    model.end_game(moves + deal + shown, GameStateView(state))
    assert model.shown == {"them": [1.0, PERCENTILE["72o"]]}


def _call_share(hands_seen: int, shoves: int) -> float:
    """The share of hands that call a shove from the big blind."""
    shove = A.raise_to(0, 500)
    calling = 0
    for name, share in list(PERCENTILE.items())[::8]:
        model = Model()
        model.strict = True
        _seen_shoving(model, "seat 0", shoves, hands_seen)
        first = Card(name[0], "S")
        second = Card(name[1], "S" if name.endswith("s") else "H")
        model.seed([first, second], 1)
        action = model.run([shove], GameStateView(table(shove)))
        if action.kind == "call":
            calling = max(calling, share)
    return calling


def test_top_half_calls_an_always_shover_but_not_a_rare_one():
    assert 0.45 <= _call_share(40, 40) <= 0.65
    assert _call_share(200, 1) <= 0.12


def test_against_a_shover_the_button_shoves_or_folds_never_raises_small():
    def first_move(hand):
        model = bot_with(hand, 0)
        _seen_shoving(model, "seat 1", 40, 40)
        return model.run([], GameStateView(table()))

    assert first_move("AS KD") == A.raise_to(0, 500)
    assert first_move("7C 2D") == A.fold(0)


def test_end_game_counts_big_raises_per_team():
    state = GameState([500, 500], teams=["us", "them"])
    model = bot_with("AS AH", 0)
    shove = A.raise_to(1, 500)
    model.seat_chips = (500, 500)
    model.log = [A.raise_to(0, 50)]
    state.process_action(A.raise_to(0, 50))
    state.process_action(shove)
    state.process_action(A.fold(0))
    model.end_game([shove, A.fold(0), A.win(1, 550)], GameStateView(state))
    assert model.big_raises == {"them": [1, 1]}
    model.seed(cards("7C 2D"), 0)
    model.end_game([A.fold(0), A.win(1, 30)], GameStateView(state))
    assert model.big_raises == {"them": [1, 1]}  # they never had to act


def test_an_ordinary_raise_is_not_big():
    state = GameState([500, 500], teams=["them", "us"])
    model = bot_with("AS AH", 1)
    model.seat_chips = (500, 500)
    state.process_action(A.raise_to(0, 50))
    state.process_action(A.fold(1))
    model.end_game([A.raise_to(0, 50), A.fold(1), A.win(0, 70)],
                   GameStateView(state))
    assert model.big_raises == {"them": [1, 0]}


def test_our_own_shove_does_not_count_as_them_not_raising():
    state = GameState([500, 500], teams=["us", "them"])
    model = bot_with("AS AH", 0)
    model.seat_chips = (500, 500)
    shove = A.raise_to(0, 500)
    state.process_action(shove)
    state.process_action(A.fold(1))
    model.end_game([shove, A.fold(1), A.win(0, 520)], GameStateView(state))
    assert model.big_raises == {}


def test_a_learned_shover_is_found_within_a_hand_or_two_of_play(quick_sim):
    model = _strict_model()
    other = DUMMIES["shove"](seed=1)
    _play([model, other], 12, (500, 500), "learn")
    hands, big = model.big_raises["seat 1"]
    assert hands == big and hands > 0
    assert model._big_raise_rate("seat 1") > 0.6


def test_a_short_stack_shoves_or_folds():
    short = (120, 120)  # 6 big blinds
    assert decide("AS AH", 0, stacks=short) == A.raise_to(0, 120)
    assert decide("3C 2D", 0, stacks=short).kind == "fold"


def test_a_three_bet_is_shoved_called_or_folded():
    moves = (A.raise_to(0, 50), A.raise_to(1, 150))
    assert decide("AS AH", 0, *moves) == A.raise_to(0, 500)
    assert decide("7C 2D", 0, *moves) == A.fold(0)


def test_a_min_three_bet_is_priced_not_folded():
    moves = (A.raise_to(0, 50), A.raise_to(1, 80))
    assert decide("9C 8D", 0, *moves) == A.call(0)


# -- postflop ------------------------------------------------------------------


LIMPED = (A.call(0), A.call(1))


def test_the_nuts_on_the_river_bets_and_raises():
    river = (*LIMPED, "QS JS TS", A.call(1), A.call(0), "2C",
             A.call(1), A.call(0), "3D")
    assert decide("AS KS", 1, *river).kind == "raise"
    assert decide("AS KS", 0, *river, A.raise_to(1, 40)).kind == "raise"


def test_a_weak_hand_checks_and_folds_to_a_bet():
    flop = (*LIMPED, "AS KS QS")
    assert decide("7C 2D", 1, *flop) == A.call(1)
    assert decide("7C 2D", 0, *flop, A.raise_to(1, 40)) == A.fold(0)


def test_a_cheap_price_keeps_a_decent_hand_in():
    flop = (*LIMPED, "KC 7D 2H")
    assert decide("KD 9C", 0, *flop, A.raise_to(1, 20)).kind != "fold"


def test_a_board_that_plays_splits_on_the_river():
    moves = (*LIMPED, "AS KS QS", A.call(1), A.call(0), "JS",
             A.call(1), A.call(0), "TS")
    model = bot_with("2C 3D", 1)
    model.log = log_of(*moves)
    equity = model._equity(GameStateView(table(*moves)))
    assert equity == pytest.approx(0.5)


def test_a_bet_lowers_our_equity_against_the_range():
    flop = (*LIMPED, "KC 7D 2H")
    checked = (*flop, A.call(1))
    betting = (*flop, A.raise_to(1, 40))

    def equity(moves):
        model = bot_with("KD 9C", 0)
        model.log = log_of(*moves)
        return model._equity(GameStateView(table(*moves)))

    assert equity(betting) < equity(checked) - 0.05


def test_preflop_action_narrows_the_range():
    model = bot_with("AS AH", 0)
    model.log = log_of(A.raise_to(0, 50), A.raise_to(1, 150), A.call(0))
    assert model._preflop_range() == bot.RAISED_RANGE[2]
    model.log = log_of(A.raise_to(0, 50), A.call(1), "KC 7D 2H")
    assert model._preflop_range() == bot.CALLED_RANGE[1]
    model.log = log_of(*LIMPED)
    assert model._preflop_range() == 1.0


def test_several_opponents_lower_the_equity():
    moves = [A.call(0), A.call(1), A.call(2)]
    state = GameState([500, 500, 500])
    for move in moves:
        state.process_action(move)
    for card in cards("KC 7D 2H"):
        state.process_action(A.reveal_card(card))
    heads_up = table(*LIMPED, "KC 7D 2H")
    model = bot_with("KD 9C", 1)
    model.log = list(moves)
    three_way = model._equity(GameStateView(state))
    assert three_way < model._equity(GameStateView(heads_up))


# -- sizing --------------------------------------------------------------------


def _random_turn_states(seed: int, players: int):
    """Plays a hand of random legal moves, yielding the state at each turn."""
    rng = random.Random(seed)
    stacks = [rng.choice([25, 60, 200, 500, 2000]) for _ in range(players)]
    state = GameState(stacks)
    deck = full_deck()
    rng.shuffle(deck)
    while not state.is_hand_over:
        if state.is_street_closed:
            state.process_action(A.reveal_card(deck.pop()))
            continue
        yield state
        me = state.to_act
        roll = rng.random()
        if roll < 0.35 and state.can_raise:
            low, high = state.raise_range
            state.process_action(A.raise_to(me, rng.randint(low, high)))
        elif roll < 0.45:
            state.process_action(A.fold(me))
        else:
            state.process_action(A.call(me))


@settings(max_examples=150, deadline=None)
@given(st.integers(0, 10**9), st.integers(2, 6),
       st.floats(-1e6, 1e6, allow_nan=False))
def test_every_raise_is_legal_whatever_the_target(seed, players, target):
    for state in _random_turn_states(seed, players):
        model = Model()
        model.player_id = state.to_act
        view = GameStateView(state)
        action = model._raise_to(view, target)
        view.validate_action(action)  # raises InvalidAction if not
        assert type(action.amount) is int
        if not state.can_raise:
            assert action == A.call(state.to_act)
        else:
            assert action.kind == "raise"


def test_a_raise_near_the_whole_stack_goes_all_in():
    model = Model()
    model.player_id = 0
    view = GameStateView(table())
    assert model._raise_to(view, 430).amount == 500  # 86% of the stack
    assert model._raise_to(view, 400).amount == 400


def test_a_float_pot_fraction_becomes_a_legal_int():
    model = Model()
    model.player_id = 1
    state = table(*LIMPED, "KC 7D 2H")
    action = model._pot_raise(GameStateView(state), 0.75)
    assert action == A.raise_to(1, 30)


# -- safety --------------------------------------------------------------------


def _broken(self, gs):
    raise RuntimeError("a bug")


def _illegal(self, gs):
    return A.raise_to(self.player_id, 1)


@pytest.mark.parametrize("bug", [_broken, _illegal], ids=["raises", "illegal"])
def test_a_bug_becomes_a_check_or_fold_and_is_counted(monkeypatch, bug):
    monkeypatch.setattr(Model, "_decide", bug)
    model = Model()
    model.seed(cards("AS AH"), 1)
    free = table(A.call(0))
    owing = table(A.raise_to(0, 50))
    assert model.run([], GameStateView(free)) == A.call(1)
    assert model.run([], GameStateView(owing)) == A.fold(1)
    assert model.errors == 2


@pytest.mark.parametrize("bug, error", [(_broken, RuntimeError),
                                        (_illegal, InvalidAction)],
                         ids=["raises", "illegal"])
def test_strict_mode_raises_what_the_fallback_hides(monkeypatch, bug, error):
    monkeypatch.setattr(Model, "_decide", bug)
    model = bot_with("AS AH", 1)
    with pytest.raises(error):
        model.run([], GameStateView(table(A.call(0))))


def test_a_bad_hand_in_seed_is_counted_not_fatal():
    model = Model()
    model.seed(["not", "cards"], 0)
    assert model.errors == 1 and model.percentile == 1.0
    model.strict = True
    with pytest.raises(AttributeError):
        model.seed(["not", "cards"], 0)


def test_the_sim_stops_at_the_time_cap(monkeypatch):
    monkeypatch.setattr(bot, "SIM_TRIALS", 200_000)
    moves = (*LIMPED, "KC 7D 2H")
    model = bot_with("KD 9C", 1)
    model.log = log_of(*moves)
    start = time.perf_counter()
    equity = model._equity(GameStateView(table(*moves)))
    elapsed = time.perf_counter() - start
    assert 0.0 < equity < 1.0
    assert elapsed < bot.SIM_CAP + 0.1  # the list of picks costs a little


@pytest.mark.parametrize("streets", [0, 1, 2, 3],
                         ids=["preflop", "flop", "turn", "river"])
def test_no_time_at_all_still_gives_an_equity(monkeypatch, streets):
    monkeypatch.setattr(bot, "SIM_CAP", 0.0)
    moves = [*LIMPED]
    for number, cards_dealt in enumerate(["KC 7D 2H", "9S", "3C"][:streets]):
        if number:
            moves += [A.call(1), A.call(0)]
        moves.append(cards_dealt)
    state = table(*moves) if streets else table(A.call(0))
    moves = moves if streets else [A.call(0)]
    model = bot_with("KD 9C", 1)
    model.log = log_of(*moves)
    assert 0.0 <= model._equity(GameStateView(state)) <= 1.0


def test_a_real_decision_stays_near_the_cap():
    moves = (*LIMPED, "KC 7D 2H", A.raise_to(1, 40))
    model = bot_with("KD 9C", 0)
    state = table(*moves)
    slowest = 0.0
    for _ in range(5):
        start = time.perf_counter()
        model.log = log_of(*moves)
        model.run([], GameStateView(state))
        slowest = max(slowest, time.perf_counter() - start)
    assert slowest < 2 * bot.SIM_CAP


# -- whole hands, with strict set ----------------------------------------------


def _play(models, deals, stacks, label):
    for deal in range(deals):
        deck = full_deck()
        random.Random(f"{label}:{deal}").shuffle(deck)
        result = play_hand(models, deck, list(stacks),
                           button=deal % len(models))
        assert result.faults == {}, result.faults
    for model in models:
        if isinstance(model, Model):
            assert model.errors == 0


@pytest.fixture
def quick_sim(monkeypatch):
    """Fewer trials: these tests are about what happens, not how well."""
    monkeypatch.setattr(bot, "SIM_TRIALS", 50)


def _strict_model() -> Model:
    model = Model()
    model.strict = True
    return model


@pytest.mark.parametrize("stacks", [(500, 500), (60, 60), (2000, 2000),
                                    (500, 90)],
                         ids=["25bb", "3bb", "100bb", "uneven"])
@pytest.mark.parametrize("opponent", list(DUMMIES))
def test_strict_play_heads_up(quick_sim, opponent, stacks):
    other = DUMMIES[opponent](seed=1)
    label = f"{opponent}:{stacks}"
    _play([_strict_model(), other], 6, stacks, label)
    _play([other, _strict_model()], 6, stacks, label)


def test_strict_play_against_itself(quick_sim):
    _play([_strict_model(), _strict_model()], 20, (500, 500), "self")


@pytest.mark.parametrize("stacks", [(500,) * 4, (40, 500, 90, 1000),
                                    (500,) * 6],
                         ids=["four", "uneven", "six"])
def test_strict_play_at_a_table(quick_sim, stacks):
    names = ["random", "shove", "call", "tight", "fold"][:len(stacks) - 1]
    others = [DUMMIES[name](seed=2) for name in names]
    _play([_strict_model(), *others], 8, stacks, "table")
    _play([*others, _strict_model()], 8, stacks, "table2")


# -- the submission ------------------------------------------------------------


def test_the_pickle_holds_plain_data_and_plays(tmp_path):
    blob = pickle.dumps(Model())
    assert len(blob) < 50_000
    model = pickle.loads(blob)
    model.seed(cards("AS AH"), 0)
    assert model.run([], GameStateView(table())) == A.raise_to(0, 50)


def test_the_code_review_flags_nothing(tmp_path):
    path = tmp_path / "myteam.pkl"
    path.write_bytes(pickle.dumps(Model()))
    sources = []
    for name in ("model.py", "preflop_table.py", "preflop_ranges.py"):
        sources.append(shutil.copy(name, tmp_path / name))
    report = vet.vet(path, sources)
    assert report.ok, report.lines()
