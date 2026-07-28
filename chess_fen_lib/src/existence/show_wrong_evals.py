import matplotlib.pyplot as plt
import torch

from src.existence import dataset
from src.existence.model import BoardExistence


def show_wrong_existence(
    model_path,
    game: str,
    eval_size: int = 1000,
    seed: int = 42,
):
    test_set = dataset.generate_fixed_test_set(game=game, size=eval_size, seed=seed)

    model = BoardExistence()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    correct = 0
    total = 0
    for img, target in test_set:
        total += 1

        with torch.no_grad():
            output = model(img.unsqueeze(0)).squeeze(0)

        pred = output.round()

        if pred == target:
            correct += 1
            continue

        print("WRONG:")
        print(f"(should be: {target.item()}, but is: {pred.item()})")
        print(correct / total)

        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        plt.axis("off")
        plt.show()
