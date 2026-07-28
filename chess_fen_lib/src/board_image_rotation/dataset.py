import random

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import IterableDataset, TensorDataset
from torchvision.transforms import v2

from src import common, consts
from src.fen_recognition.generate_chessboards import BoardGenerator

default_transforms = torch.nn.Sequential(
    v2.ToDtype(torch.float32),
    v2.Resize(
        size=(consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH),
        interpolation=v2.InterpolationMode.BICUBIC,
    ),
    common.MinMaxMeanNormalization(),
)

augment_transforms = torch.nn.Sequential(
    v2.RandomApply(
        [common.AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4
    ),
    v2.RandomApply(
        [v2.ElasticTransform(alpha=30.0), v2.ElasticTransform(alpha=40.0)], p=0.4
    ),
    v2.RandomGrayscale(p=0.4),
    v2.RandomPosterize(bits=2, p=0.2),
    v2.RandomApply(
        [v2.ColorJitter(brightness=0.9, contrast=(0.1, 1.5), hue=0.3)], p=0.3
    ),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3))], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5))], p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=10, p=0.1),
    v2.RandomEqualize(p=0.8),
)

affine_transforms = v2.RandomAffine(
    degrees=1.5, translate=(0.01, 0.01), scale=(0.99, 1.01), shear=1.5
)

ROTATIONS = [0, 90, 180, 270]


class GenerativeRotationDataset(IterableDataset):
    def __init__(
        self,
        game: str,
        augment_ratio: float = 0.5,
        affine_augment_ratio: float = 0.8,
    ):
        self.generator = BoardGenerator(game)
        self.augments = torch.nn.Sequential(
            v2.RandomApply([affine_transforms], p=affine_augment_ratio),
            v2.RandomApply([augment_transforms], p=augment_ratio),
        )

    def __iter__(self):
        while True:
            image, _ = self.generator.generate_one()
            target = random.randint(0, len(ROTATIONS) - 1)
            image = image.rotate(ROTATIONS[target], expand=True)

            input_img = common.to_rgb_tensor(image)

            while True:
                input_img = self.augments(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after augmentation. Trying again.")
                    continue
                input_img = default_transforms(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after default transform. Trying again.")
                    continue
                break

            yield input_img, target


def generate_fixed_test_set(
    game: str,
    size: int = 500,
    seed: int = 42,
) -> TensorDataset:
    rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)

    augments = torch.nn.Sequential(
        v2.RandomApply([affine_transforms], p=0.8),
        v2.RandomApply([augment_transforms], p=0.5),
    )

    generator = BoardGenerator(game)
    images = []
    targets = []
    for _ in range(size):
        image, _ = generator.generate_one()
        target = random.randint(0, len(ROTATIONS) - 1)
        image = image.rotate(ROTATIONS[target], expand=True)
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
        targets.append(target)

    random.setstate(rng_state)
    torch.random.set_rng_state(torch_rng_state)
    return TensorDataset(torch.stack(images), torch.tensor(targets))


def test_data_set(game: str, size: int = 1000):
    ds = GenerativeRotationDataset(game=game)

    for i, (img, target) in enumerate(ds):
        if i >= size:
            break
        assert not img.isnan().any()
        print(target, ROTATIONS[target])

        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.axis("off")
        plt.show()
