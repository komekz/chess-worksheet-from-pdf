import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset, TensorDataset
from torchvision.transforms import v2

from src.common import to_rgb_tensor, MinMaxMeanNormalization, AddGaussianNoise
from src import consts
from src.bounding_box.generate_chessboards_bbox import BboxGenerator
from src.existence.generate_existence import NoboardGenerator

ROTATIONS = [0, 90, 180, 270]


default_transforms = torch.nn.Sequential(
    v2.ToDtype(torch.float32),
    v2.Resize(
        size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
        interpolation=v2.InterpolationMode.BICUBIC,
    ),
    MinMaxMeanNormalization(),
)

augment_transforms = torch.nn.Sequential(
    v2.RandomApply([v2.RandomAffine(degrees=1.0, shear=1.0)], p=0.3),
    v2.RandomInvert(p=0.1),
    v2.RandomApply([AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4),
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


class GenerativeExistenceDataset(IterableDataset):
    def __init__(
        self,
        game: str,
        augment_ratio: float = 0.5,
        affine_augment_ratio: float = 0.8,
    ):
        self.bbox_generator = BboxGenerator(game)
        self.noboard_generator = NoboardGenerator()
        self.augments = torch.nn.Sequential(
            v2.RandomApply([affine_transforms], p=affine_augment_ratio),
            v2.RandomApply([augment_transforms], p=augment_ratio),
        )

    def __iter__(self):
        while True:
            if random.random() < 0.5:
                image, _ = self.bbox_generator.generate_one()
                target = 1.0
            else:
                image = self.noboard_generator.generate_one()
                target = 0.0

            angle = random.choice(ROTATIONS)
            if angle != 0:
                image = image.rotate(angle, expand=True)

            input_img = to_rgb_tensor(image)

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

            yield input_img, torch.tensor(target).unsqueeze(0)


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

    bbox_gen = BboxGenerator(game)
    noboard_gen = NoboardGenerator()
    images = []
    targets = []
    for i in range(size):
        if i % 2 == 0:
            image, _ = bbox_gen.generate_one()
            target = 1.0
        else:
            image = noboard_gen.generate_one()
            target = 0.0

        angle = random.choice(ROTATIONS)
        if angle != 0:
            image = image.rotate(angle, expand=True)

        input_img = to_rgb_tensor(image)

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
        targets.append(torch.tensor(target).unsqueeze(0))

    random.setstate(rng_state)
    torch.random.set_rng_state(torch_rng_state)
    return TensorDataset(torch.stack(images), torch.stack(targets))


def test_data_set(game: str, size: int = 1000):
    ds = GenerativeExistenceDataset(game=game, augment_ratio=0.5)

    for i, (img, target) in enumerate(ds):
        if i >= size:
            break
        assert not img.isnan().any()
        print(target)

        vis = img.clone()
        vis *= 255.0
        vis += 128.0
        vis = vis.to(torch.uint8)

        plt.imshow((vis.permute(1, 2, 0) - vis.min()) / (vis.max() - vis.min()))
        plt.show()
