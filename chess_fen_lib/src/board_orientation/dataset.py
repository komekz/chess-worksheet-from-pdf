import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src import common
from src.games import get_game
from src.pgn_parser import iter_pgn_games, parse_pgn_game, parse_pgn_tags, parse_variant_tag


def _parse_game_fens(args: tuple[str, int | None]) -> tuple[list[str], str | None]:
    game_text, max_positions_per_game = args
    try:
        from src.pgn_parser import parse_pgn_game
        fens = list(parse_pgn_game(game_text).fens)
        if max_positions_per_game is not None and len(fens) > max_positions_per_game:
            fens = random.sample(fens, max_positions_per_game)
        return fens, None
    except Exception as e:
        return [], str(e)


class BoardOrientationDataset(Dataset):

    def __init__(self, pgn_files: list[Path] | str | Path, game: str, rotate_probability=0.3, max=100000, max_positions_per_game: int | None = 10):
        self.game = get_game(game)
        self.board_list = []
        self.rotate_probability = rotate_probability

        if isinstance(pgn_files, (str, Path)):
            pgn_files = sorted(Path(pgn_files).glob("*.pgn"))

        def matching_games():
            for pgn_file in pgn_files:
                for game_text in iter_pgn_games(pgn_file):
                    tags = parse_pgn_tags(game_text)
                    try:
                        variant, _ = parse_variant_tag(tags.get("Variant", self.game.key))
                    except ValueError:
                        continue
                    if variant == self.game.key:
                        yield game_text

        num_workers = os.cpu_count() or 1
        batch_size = num_workers * 8

        games_attempted = 0
        games_failed = 0
        first_errors: list[str] = []

        with tqdm(total=max, desc="Loading positions", unit="pos", dynamic_ncols=True) as progress:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                game_iter = matching_games()
                done = False
                while not done:
                    batch = list(itertools.islice(game_iter, batch_size))
                    if not batch:
                        break
                    games_attempted += len(batch)
                    args = [(g, max_positions_per_game) for g in batch]
                    for fens, error in executor.map(_parse_game_fens, args):
                        if error is not None:
                            games_failed += 1
                            if len(first_errors) < 3:
                                first_errors.append(error)
                        for fen in fens:
                            position = common.position_from_notation(fen, game=self.game)
                            if position is not None:
                                self.board_list.append(position)
                                progress.update(1)
                            if len(self.board_list) >= max:
                                done = True
                                break
                        if done:
                            break

        random.shuffle(self.board_list)

        print(f"Found {len(self.board_list)} positions "
              f"({games_attempted} games attempted, {games_failed} failed to parse)")
        if games_failed > 0:
            print(f"WARNING: {games_failed}/{games_attempted} {self.game.key} games could not be parsed.")
            for err in first_errors:
                print(f"  Example error: {err}")

    def __len__(self):
        return len(self.board_list)

    def __getitem__(self, idx):
        rotate = random.uniform(0.0, 1.0) < self.rotate_probability

        input = common.position_to_tensor(self.board_list[idx])
        if rotate:
            input = common.rotate_tensor_180(input, self.game)

        return (input, torch.tensor([1.0 if rotate else 0.0]))


def test_data_set(game: str, pgn_dir: str = "resources/pychess_games"):
    spec = get_game(game)
    c = BoardOrientationDataset(pgn_dir, game=spec, max=1000)
    print("len(c):", len(c))
    for i in range(0, 100):
        input, target = c[i]

        assert not input.isnan().any()
        print(common.tensor_to_position(input, game=spec).fen())
        print("flipped:", target.item() == 1.0)
