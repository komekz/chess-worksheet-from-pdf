import matplotlib.pyplot as plt
import torch

from src import common, consts
from src.fen_recognition import dataset
from src.fen_recognition.model import BoardRec
from src.games import get_game


def show_wrong_fens(
    model_path,
    game: str,
    eval_size: int = 1000,
    tile_size: int = consts.DEFAULT_TILE_SIZE,
    seed: int = 42,
):
    spec = get_game(game)

    test_set = dataset.generate_fixed_test_set(
        game=spec.key, size=eval_size, tile_size=tile_size, seed=seed,
    )

    model = BoardRec(game=spec.key, tile_size=tile_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    correct = 0
    total = 0
    for img, target in test_set:
        total += 1

        with torch.no_grad():
            output = model(img.unsqueeze(0)).squeeze(0)

        board = common.tensor_to_position(output, spec)
        true_board = common.tensor_to_position(target, spec)

        if board.piece_placement == true_board.piece_placement:
            correct += 1
            print("Correct:", board.fen())
            continue

        print("WRONG:")
        print(true_board.fen())
        print(board.fen())
        print(correct / total)

        pil_img = common.get_image(board, width=img.shape[2], height=img.shape[1])

        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(pil_img)
        ax[0].axis("off")

        ax[1].imshow((img.permute(1, 2, 0) - img.min()) / (img.max() - img.min()))
        ax[1].axis("off")
        plt.show()
