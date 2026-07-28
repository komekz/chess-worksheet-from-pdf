import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image, ImageDraw
from torch.utils.data import IterableDataset
from torchvision.transforms import v2

from src import consts
from src.bounding_box.generate_chessboards_bbox import BboxGenerator
from src.common import AddGaussianNoise, to_rgb_tensor

ROTATIONS = [0, 90, 180, 270]


def _rotate_corners(
    corners: list[tuple[float, float]], angle: int
) -> list[tuple[float, float]]:
    """Rotate normalized [0,1] corners (TL, TR, BR, BL) for a square image.

    angle: one of 0, 90, 180, 270 (CCW, matching PIL.Image.rotate).
    """
    if angle == 0:
        return corners
    if angle == 90:
        # (x,y) -> (y, 1-x), reorder [TR, BR, BL, TL]
        reorder = [1, 2, 3, 0]
        transform = lambda x, y: (y, 1 - x)
    elif angle == 180:
        # (x,y) -> (1-x, 1-y), reorder [BR, BL, TL, TR]
        reorder = [2, 3, 0, 1]
        transform = lambda x, y: (1 - x, 1 - y)
    elif angle == 270:
        # (x,y) -> (1-y, x), reorder [BL, TL, TR, BR]
        reorder = [3, 0, 1, 2]
        transform = lambda x, y: (1 - y, x)
    else:
        raise ValueError(f"Unsupported angle: {angle}")

    return [transform(*corners[i]) for i in reorder]

augment_transforms = torch.nn.Sequential(
    v2.RandomInvert(p=0.1),
    v2.RandomApply([AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4),
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


def _normalize_to_01(img):
    """Normalize image to [0, 1] range."""
    mn, mx = img.min(), img.max()
    if mn >= mx:
        return torch.zeros_like(img)
    return (img - mn) / (mx - mn)


def corners_to_mask(corners_flat: torch.Tensor) -> torch.Tensor:
    """Convert normalized [8] corner coords to a filled polygon mask [1, H, W]."""
    s = consts.BBOX_IMAGE_SIZE
    # corners_flat: [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y] in [0,1]
    poly_pts = [(corners_flat[i].item() * s, corners_flat[i + 1].item() * s)
                for i in range(0, 8, 2)]
    mask_img = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask_img).polygon(poly_pts, fill=255)
    mask = torch.frombuffer(bytearray(mask_img.tobytes()), dtype=torch.uint8)
    return (mask.reshape(1, s, s).float() / 255.0)


def collate_fn(batch):
    """Stack images, corners, and masks into batched tensors."""
    images = torch.stack([item[0] for item in batch])
    corners = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])
    return images, corners, masks


class GenerativeBboxDataset(IterableDataset):
    def __init__(
        self,
        game: str,
        augment_ratio: float = 0.5,
    ):
        self.generator = BboxGenerator(game)
        self.augment_ratio = augment_ratio

    def __iter__(self):
        while True:
            image, corners = self.generator.generate_one()
            angle = random.choice(ROTATIONS)
            if angle != 0:
                image = image.rotate(angle, expand=True)
                corners = _rotate_corners(corners, angle)
            input_img = to_rgb_tensor(image)

            do_augment = self.augment_ratio > random.uniform(0, 1)

            while True:
                if do_augment:
                    input_img = augment_transforms(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after augmentation. Trying again.")
                    continue
                break

            input_img = v2.ToDtype(torch.float32)(input_img)
            input_img = v2.Resize(
                size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
                interpolation=v2.InterpolationMode.BICUBIC,
            )(input_img)
            if input_img.isnan().any():
                print("WARNING: Found nan after resize. Trying again.")
                continue

            input_img = _normalize_to_01(input_img)

            corners_flat = torch.tensor(
                [c for pt in corners for c in pt], dtype=torch.float32
            )
            mask = corners_to_mask(corners_flat)

            yield input_img, corners_flat, mask


def generate_fixed_test_set(
    game: str,
    size: int = 500,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)

    augments = v2.RandomApply([augment_transforms], p=0.5)

    generator = BboxGenerator(game)
    data = []
    for _ in range(size):
        image, corners = generator.generate_one()
        angle = random.choice(ROTATIONS)
        if angle != 0:
            image = image.rotate(angle, expand=True)
            corners = _rotate_corners(corners, angle)
        input_img = to_rgb_tensor(image)

        while True:
            input_img = augments(input_img)
            if input_img.isnan().any():
                print("WARNING: Found nan after augmentation. Trying again.")
                continue
            break

        input_img = v2.ToDtype(torch.float32)(input_img)
        input_img = v2.Resize(
            size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
            interpolation=v2.InterpolationMode.BICUBIC,
        )(input_img)
        input_img = _normalize_to_01(input_img)

        corners_flat = torch.tensor(
            [c for pt in corners for c in pt], dtype=torch.float32
        )
        mask = corners_to_mask(corners_flat)

        data.append((input_img, corners_flat, mask))

    random.setstate(rng_state)
    torch.random.set_rng_state(torch_rng_state)
    return data


def test_data_set(game: str, size: int = 1000):
    ds = GenerativeBboxDataset(game=game, augment_ratio=0.5)

    for i, (img, corners, mask) in enumerate(ds):
        if i >= size:
            break
        assert not img.isnan().any()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img.permute(1, 2, 0))

        c = corners * consts.BBOX_IMAGE_SIZE
        poly_pts = [(c[i].item(), c[i + 1].item()) for i in range(0, 8, 2)]
        polygon = patches.Polygon(
            poly_pts,
            closed=True,
            linewidth=2,
            edgecolor=(1.0, 0.0, 0.0, 0.8),
            facecolor=(1.0, 0.0, 0.0, 0.1),
            linestyle="--",
        )
        ax1.add_patch(polygon)
        ax2.imshow(mask.squeeze(0), cmap="gray")
        plt.show()
