import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset, TensorDataset
from torchvision.transforms import v2

from src import common, consts
from src.fen_recognition.generate_chessboards import BoardGenerator
from src.games import get_game

#
augment_transforms = torch.nn.Sequential(
    v2.RandomApply(
        [common.AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4
    ),
    v2.RandomApply(
        [
            v2.RandomChoice(
                [v2.ElasticTransform(alpha=30.0), v2.ElasticTransform(alpha=40.0)]
            )
        ],
        p=0.4,
    ),
    v2.RandomGrayscale(p=0.1),
    v2.RandomPosterize(bits=2, p=0.2),
    v2.RandomApply(
        [
            v2.RandomChoice(
                [
                    v2.ColorJitter(brightness=0.9, contrast=(0.1, 1.5), hue=0.3),
                    common.RandomColorTint(max_tint=0.7),
                ]
            )
        ],
        p=0.6,
    ),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3))], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5))], p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=10, p=0.1),
    v2.RandomEqualize(p=0.8),
)

affine_transforms = v2.RandomAffine(
    degrees=2.0, translate=(0.01, 0.01), scale=(0.99, 1.01), shear=1.8
)


def get_default_transforms(game: str, tile_size: int = consts.DEFAULT_TILE_SIZE):
    spec = get_game(game)
    board_h, board_w = consts.board_pixel_size(spec, tile_size)
    return torch.nn.Sequential(
        v2.ToDtype(torch.float32),
        v2.Resize(size=(board_h, board_w), interpolation=v2.InterpolationMode.BICUBIC),
        common.MinMaxMeanNormalization(),
    )


class GenerativeBoardDataset(IterableDataset):
    def __init__(
        self,
        game: str,
        tile_size: int = consts.DEFAULT_TILE_SIZE,
        augment_ratio: float = 0.5,
        affine_augment_ratio: float = 0.8,
    ):
        self.game = get_game(game)
        self.tile_size = tile_size
        self.board_h, self.board_w = consts.board_pixel_size(self.game, tile_size)
        self.generator = BoardGenerator(game, tile_size)
        self.default_transforms = torch.nn.Sequential(
            v2.ToDtype(torch.float32),
            v2.Resize(
                size=(self.board_h, self.board_w),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            common.MinMaxMeanNormalization(),
        )
        self.augments = torch.nn.Sequential(
            v2.RandomApply([affine_transforms], p=affine_augment_ratio),
            v2.RandomApply([augment_transforms], p=augment_ratio),
        )

    def __iter__(self):
        while True:
            image, position = self.generator.generate_one()
            input_img = common.to_rgb_tensor(image)
            target = common.position_to_tensor(position)

            while True:
                input_img = self.augments(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after augmentation. Trying again.")
                    continue
                input_img = self.default_transforms(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after default transform. Trying again.")
                    continue
                break

            yield input_img, target


def generate_fixed_test_set(
    game: str,
    size: int = 500,
    tile_size: int = consts.DEFAULT_TILE_SIZE,
    seed: int = 42,
) -> TensorDataset:
    spec = get_game(game)
    board_h, board_w = consts.board_pixel_size(spec, tile_size)
    default_transforms = torch.nn.Sequential(
        v2.ToDtype(torch.float32),
        v2.Resize(size=(board_h, board_w), interpolation=v2.InterpolationMode.BICUBIC),
        common.MinMaxMeanNormalization(),
    )
    augments = torch.nn.Sequential(
        v2.RandomApply([affine_transforms], p=0.8),
        v2.RandomApply([augment_transforms], p=0.5),
    )

    rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)

    generator = BoardGenerator(game, tile_size)
    images = []
    targets = []
    for _ in range(size):
        image, position = generator.generate_one()
        input_img = common.to_rgb_tensor(image)

        while True:
            input_img = augments(input_img)
            if input_img.isnan().any():
                print("WARNING: Found nan after augmentation. Trying again.")
                continue
            input_img = default_transforms(input_img)
            if input_img.isnan().any():
                print("WARNING: Found nan after default transform. Trying again.")
                continue
            break

        images.append(input_img)
        targets.append(common.position_to_tensor(position))

    random.setstate(rng_state)
    torch.random.set_rng_state(torch_rng_state)
    return TensorDataset(torch.stack(images), torch.stack(targets))


def test_data_set(game: str, size: int = 1000):
    ds = GenerativeBoardDataset(game=game)

    for i, (img, target) in enumerate(ds):
        if i >= size:
            break
        assert not img.isnan().any()

        pos = common.tensor_to_position(target, game=game)
        print(pos.fen())

        pos_img = common.get_image(pos, width=img.shape[2], height=img.shape[1])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
        ax1.imshow(img)
        ax2.imshow(pos_img)
        ax1.axis("off")
        ax2.axis("off")
        plt.show()
