import random
import torch

from src import common
from src.board_orientation.model import OrientationModel
from src.consts import DEFAULT_PGN_DIR, get_reserved_pgn_files
from src.games import get_game
from src.pgn_parser import iter_pgn_games, parse_pgn_game, parse_pgn_tags, parse_variant_tag


@torch.no_grad()
def show_wrong_orientation_evals(
    game: str,
    model_file,
    pgn_dir=DEFAULT_PGN_DIR,
    rotate_probability=0.5,
    no_rotate_bias=0.0,
):
    spec = get_game(game)
    model = OrientationModel(game=game)
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    model.eval()

    num_samples = 0
    num_correct = 0

    pgn_files = get_reserved_pgn_files(pgn_dir)
    if not pgn_files:
        print(f"No reserved PGN files found in {pgn_dir}")
        return

    for pgn_file in pgn_files:
        for game_text in iter_pgn_games(pgn_file):
            tags = parse_pgn_tags(game_text)
            variant, chess960 = parse_variant_tag(tags.get("Variant", spec.key))
            if variant != spec.key:
                continue

            try:
                parsed = parse_pgn_game(game_text)
            except Exception:
                continue
            fens = parsed.fens

            for fen in fens:
                position = common.position_from_notation(fen, game=spec)
                if position is None:
                    continue

                input_tensor = common.position_to_tensor(position)
                rotate = random.uniform(0.0, 1.0) < rotate_probability
                if rotate:
                    input_tensor = common.rotate_tensor_180(input_tensor, spec)
                target = 1.0 if rotate else 0.0

                output = model(input_tensor.unsqueeze(0)).squeeze(0)
                output -= no_rotate_bias

                print("Original:", position.fen())
                print("Input:   ", common.tensor_to_position(input_tensor, game=spec).fen())

                num_samples += 1
                if abs(output.item() - target) < 0.5:
                    num_correct += 1
                else:
                    print("Wrong: ", (output.item(), target))
                    print(num_correct / num_samples)
                    input("Press enter to continue ...")
