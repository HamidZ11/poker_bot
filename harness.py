"""
A local tournament: your model against archetype bots, the sparring dummies
and variants of your own bot, in the format the organisers' tooling plays:
heads-up, 25 big blinds, 200 deals per seat rotation, every deal mirrored
(played once from each seat), and a fresh model for each rotation.

    .venv/bin/python harness.py                     # the whole pool
    .venv/bin/python harness.py --matches 25 --opponents shove maniac
    .venv/bin/python harness.py --model bot_stage1:Model

A match is 200 deals from both seats: 400 hands. --matches plays that many
with new decks and pools them, as one match is far too noisy to read. Your
model is pickled and played from that pickle in its own worker, as in the
arena. Nothing here is part of the submission.
"""

import argparse
import concurrent.futures
import importlib
import math
import os
import pickle
import random
import tempfile
from dataclasses import replace

from arena import dummies
from arena.dummies import _Bot
from arena.match import MatchConfig, play_match
from arena.players import Player, player
from card_utils import evaluate
from evals.score import score
from player_action import PlayerAction

DEALS = 200  # per seat rotation
STACK = 500  # 25 big blinds of 20


class ShoveAlways(dummies.ShoveBot):
    """All-in on every turn it can, otherwise calls."""


class CallAlways(dummies.CallBot):
    """Checks or calls everything, never raises."""


class MinRaiser(_Bot):
    """Min-raises whenever it may, whatever it holds; otherwise calls."""

    def run(self, actions, game_state):
        return self._raise_to(game_state, game_state.min_raise_to)


class Maniac(_Bot):
    """Plays any two cards hard: mostly pot-sized raises, some shoves, and
    it hardly ever folds."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = random.Random(seed)

    def run(self, actions, game_state):
        roll = self.rng.random()
        if roll < 0.15:
            return self._raise_to(game_state, game_state.max_raise_to)
        if roll < 0.70:
            target = game_state.current_bet + game_state.pot
            return self._raise_to(game_state, target)
        if roll < 0.95 or game_state.call_amount == 0:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)


class Nit(_Bot):
    """Plays about the top 8% of hands, and only continues after the flop
    with top pair or better."""

    def run(self, actions, game_state):
        low, high = sorted(card.value for card in self.hand)
        if game_state.street == "preflop":
            premium = (low == high and high >= 10) or (low >= 12 and high == 14)
            if not premium:
                return self._check_or_fold(game_state)
            if game_state.current_bet > game_state.big_blind:
                if low == high and high >= 12 or low == 13:
                    return self._raise_to(game_state, 3 * game_state.current_bet)
                return PlayerAction.call(self.player_id)
            return self._raise_to(game_state, 3 * game_state.big_blind)
        board = list(game_state.board)
        top = max(card.value for card in board)
        category = evaluate(self.hand + board)[0]
        strong = category >= 2 or (category == 1 and high >= top)
        if strong:
            return self._raise_to(game_state, game_state.current_bet
                                  + game_state.pot * 0.6)
        return self._check_or_fold(game_state)


class OverFolder(_Bot):
    """Sees a lot of cheap flops, then gives up to any bet or big raise
    without at least two pair."""

    def run(self, actions, game_state):
        owed = game_state.call_amount
        big_blind = game_state.big_blind
        if game_state.street == "preflop":
            low, high = sorted(card.value for card in self.hand)
            if low == high and high >= 11:
                return self._raise_to(game_state, 3 * big_blind)
            if owed <= 2 * big_blind:
                return PlayerAction.call(self.player_id)
            return self._check_or_fold(game_state)
        category = evaluate(self.hand + list(game_state.board))[0]
        if category >= 2:
            return self._raise_to(game_state, game_state.current_bet
                                  + game_state.pot * 0.5)
        return self._check_or_fold(game_state)


ARCHETYPES = {
    "always-shove": ShoveAlways,
    "always-call": CallAlways,
    "min-raiser": MinRaiser,
    "maniac": Maniac,
    "nit": Nit,
    "over-folder": OverFolder,
}
SPARRING = ("random", "tight", "equity")


def _variant(class_path: str, **values):
    """A model built as dummies are, with some class attributes changed."""
    module, _, name = class_path.partition(":")
    base = getattr(importlib.import_module(module), name)

    class Variant(base):
        def __init__(self, seed=None):
            super().__init__()
            self.rng = random.Random(seed)

    for key, value in values.items():
        setattr(Variant, key, value)
    return Variant


def variants() -> dict[str, type]:
    return {
        "bot-self": _variant("model:Model"),
        "bot-stage1": _variant("bot_stage1:Model"),
        "bot-tight": _variant("model:Model", VALUE_BET=0.68, VALUE_RAISE=0.82,
                              CALL_MARGIN=0.08, CHART_SCALE=0.7),
        "bot-loose": _variant("model:Model", VALUE_BET=0.54, VALUE_RAISE=0.68,
                              CALL_MARGIN=0.0, CHART_SCALE=1.25),
    }


def register():
    """Makes every bot here playable by name, as dummies are. Runs on import,
    so the worker processes of this harness have them too."""
    dummies.DUMMIES.update(ARCHETYPES)
    dummies.DUMMIES.update(variants())


register()


def play_one(pkl: str, opponent: str, seed: int):
    """One match: DEALS deals from both seats, returning its log."""
    config = MatchConfig(deals=DEALS, seed=seed, stack=STACK)
    return play_match([player(pkl), player(opponent)], config)


def build_pickle(spec: str, folder: str, trials: int | None) -> str:
    module, _, name = spec.partition(":")
    model = getattr(importlib.import_module(module), name or "Model")()
    if trials is not None:
        model.sim_trials = trials
    path = os.path.join(folder, "you.pkl")
    with open(path, "wb") as file:
        pickle.dump(model, file)
    return path


def main():
    pool = [*ARCHETYPES, *SPARRING, *variants()]
    parser = argparse.ArgumentParser(prog="harness.py")
    parser.add_argument("--model", default="model:Model")
    parser.add_argument("--opponents", nargs="+", default=pool)
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--trials", type=int, help="sim trials, to run faster")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as folder:
        pkl = build_pickle(args.model, folder, args.trials)
        with concurrent.futures.ProcessPoolExecutor(args.jobs) as executor:
            futures = {
                name: [
                    executor.submit(play_one, pkl, name, args.seed + m)
                    for m in range(args.matches)
                ]
                for name in args.opponents
            }
            print(f"{args.model}: {args.matches} matches of {DEALS} deals "
                  f"from both seats, {STACK // 20} bb")
            print(f"{'opponent':<16}{'bb/100':>18}{'verdict':>9}{'hands':>7}"
                  f"{'faults':>7}{'slowest':>9}")
            means, squares = [], 0.0
            for name in args.opponents:
                result = score([f.result() for f in futures[name]])
                bb, ci = result.bb_per_100[0], result.ci95[0]
                verdict = ("ahead" if bb - ci > 0
                           else "behind" if bb + ci < 0 else "unclear")
                print(f"{name:<16}{bb:>+10.1f} ± {ci:<5.1f}{verdict:>9}"
                      f"{result.hands:>7}{result.faults[0]:>7}"
                      f"{result.max_move_seconds[0]:>8.2f}s", flush=True)
                if result.first_faults[0] is not None:
                    print(f"    first fault: {result.first_faults[0]}")
                means.append(bb)
                squares += ci * ci
            count = len(means)
            print(f"{'pool average':<16}{sum(means) / count:>+10.1f} ± "
                  f"{math.sqrt(squares) / count:<5.1f}")


if __name__ == "__main__":
    main()
