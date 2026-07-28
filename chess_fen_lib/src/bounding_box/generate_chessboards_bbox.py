import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.transform import ProjectiveTransform

from src import common, consts
from src.fen_recognition.generate_chessboards import BoardGenerator

MIN_CROP_SIZE = consts.BOARD_PIXEL_WIDTH + 10
MAX_CROP_SIZE = consts.BOARD_PIXEL_WIDTH * 4

TARGET_SIZE = consts.BOARD_PIXEL_WIDTH * 2


def _perturb_corners(
    x: int,
    y: int,
    w: int,
    h: int,
    jitter_fraction: float = 0.15,
) -> list[tuple[float, float]]:
    """Return 4 corners of a perspective-distorted quad around (x, y, w, h).

    Corner order: TL, TR, BR, BL.
    """
    max_jitter = jitter_fraction * min(w, h)
    corners = [
        (x, y),  # TL
        (x + w, y),  # TR
        (x + w, y + h),  # BR
        (x, y + h),  # BL
    ]
    perturbed = []
    for cx, cy in corners:
        cx += random.uniform(-max_jitter, max_jitter)
        cy += random.uniform(-max_jitter, max_jitter)
        perturbed.append((cx, cy))
    return perturbed


def _quad_has_positive_area(corners: list[tuple[float, float]]) -> bool:
    """Check that the quad is non-degenerate (positive signed area via shoelace)."""
    n = len(corners)
    area = 0.0
    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area > 0


def _perspective_warp_image(
    source_image: Image.Image,
    dst_corners: list[tuple[float, float]],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    """Warp source_image so its rectangle maps to dst_corners on a canvas.

    Returns (warped_rgb, alpha_mask) both of canvas_size.
    """
    bw, bh = source_image.size
    src = np.array(
        [(0, 0), (bw, 0), (bw, bh), (0, bh)],
        dtype=np.float64,
    )
    dst = np.array(dst_corners, dtype=np.float64)

    t = ProjectiveTransform.from_estimate(dst, src)
    if not t:
        raise ValueError(
            f"Could not estimate projective transform from destination corners: {t}"
        )

    coeffs = t.params.flatten()[:8]
    canvas_w, canvas_h = canvas_size
    warped = source_image.convert("RGBA").transform(
        (canvas_w, canvas_h),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC,
    )
    alpha = warped.split()[-1]
    rgb = warped.convert("RGB")
    return rgb, alpha


class BboxGenerator:
    def __init__(
        self,
        game: str,
        background_root_dir: str = "resources/website_screenshots",
        board_middleground_probability: float = 0.4,
        perspective_probability: float = 0.2,
        jitter_fraction: float = 0.03,
    ):
        self.board_generator = BoardGenerator(game)
        self.board_middleground_probability = board_middleground_probability
        self.perspective_probability = perspective_probability
        self.jitter_fraction = jitter_fraction

        background_root_dir = Path(background_root_dir)
        assert background_root_dir.is_dir(), (
            f"With background_root_dir = {background_root_dir}"
        )
        self.background_image_files = common.glob_all_image_files_recursively(
            background_root_dir
        )
        assert len(self.background_image_files) > 0, "No background images found"

    def generate_one(self) -> tuple[Image.Image, list[tuple[float, float]]]:
        board_bg = self.board_generator.generate_board_background(
            self.board_generator.board_w, self.board_generator.board_h
        )
        board_image, _ = self.board_generator.generate_one(
            board_background=board_bg,
        )

        # 10% of the time: just the board, no website background
        board_only = random.random() < 0.1
        if board_only:
            img = board_image.convert("RGB")
            img = img.resize((TARGET_SIZE, TARGET_SIZE))
            normalized_corners = [
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ]
            return img, normalized_corners

        # Pick random background screenshot and crop
        bg_path = random.choice(self.background_image_files)
        img = Image.open(bg_path).convert("RGB")

        img_width, img_height = img.size
        max_size = min(img_width, img_height, MAX_CROP_SIZE)
        crop_width = random.randint(MIN_CROP_SIZE, max_size)
        crop_height = random.randint(MIN_CROP_SIZE, max_size)

        x = random.randint(0, img_width - crop_width)
        y = random.randint(0, img_height - crop_height)
        img = img.crop((x, y, x + crop_width, y + crop_height))

        # Resize board (and its background) to fit within crop
        board_image_width, board_image_height = board_image.size
        board_aspect = board_image_width / max(board_image_height, 1)
        max_w = min(crop_width, max(4, int(crop_height * board_aspect)))
        max_h = min(crop_height, max(4, int(crop_width / max(board_aspect, 1e-6))))
        new_width = random.randint(max(4, min(max_w, board_image_width // 2)), max_w)
        new_height = max(4, int(new_width / max(board_aspect, 1e-6)))
        if new_height > max_h:
            new_height = max_h
            new_width = max(4, int(new_height * board_aspect))
        board_image = board_image.resize((new_width, new_height))
        board_image_width, board_image_height = board_image.size
        board_bg = board_bg.resize((board_image_width, board_image_height))

        assert board_image_width <= crop_width
        assert board_image_height <= crop_height

        board_x = random.randint(0, crop_width - board_image_width)
        board_y = random.randint(0, crop_height - board_image_height)

        # Decide whether to apply perspective warp
        use_perspective = random.random() < self.perspective_probability

        if use_perspective:
            # Get perturbed destination corners on the crop canvas
            dst_corners = _perturb_corners(
                board_x,
                board_y,
                board_image_width,
                board_image_height,
                self.jitter_fraction,
            )
            # Retry if degenerate — fall back to axis-aligned
            if not _quad_has_positive_area(dst_corners):
                use_perspective = False

        if not use_perspective:
            dst_corners = [
                (board_x, board_y),
                (board_x + board_image_width, board_y),
                (board_x + board_image_width, board_y + board_image_height),
                (board_x, board_y + board_image_height),
            ]

        if random.uniform(0.0, 1.0) < self.board_middleground_probability:
            max_relative_size = random.uniform(0.1, 3.0)
            max_aspect_ratio = random.uniform(1.0, 4.0)
            while True:
                middleground_width = random.randint(
                    min(crop_width, 4, board_image_width // 2), crop_width
                )
                middleground_height = random.randint(
                    min(crop_height, 4, board_image_height // 2), crop_height
                )
                middleground_x = random.randint(0, crop_width - middleground_width)
                middleground_y = random.randint(0, crop_height - middleground_height)

                if (
                    middleground_x > board_x
                    and middleground_y > board_y
                    and middleground_x + middleground_width
                    < board_x + board_image_width
                    and middleground_y + middleground_height
                    < board_y + board_image_height
                ):
                    continue

                if (
                    board_image_width * board_image_height * max_relative_size
                    < middleground_height * middleground_width
                ):
                    continue

                if (
                    max(
                        middleground_height / middleground_width,
                        middleground_width / middleground_height,
                    )
                    > max_aspect_ratio
                ):
                    continue

                if (
                    middleground_x > board_x + board_image_width
                    or board_x > middleground_x + middleground_width
                ):
                    continue
                if (
                    middleground_y > board_y + board_image_height
                    or board_y > middleground_y + middleground_height
                ):
                    continue

                middleground_img, _ = self.board_generator.generate_one()
                middleground_img = middleground_img.resize(
                    (middleground_width, middleground_height)
                )
                img.paste(middleground_img, (middleground_x, middleground_y))
                break

        # Build a combined image of the board + surrounding strip tiles in
        # source (axis-aligned) space, then warp the whole thing together so
        # that strip edges stay flush with the board after perspective.
        if random.random() < 0.8:
            strip_w = random.randint(1, max(1, board_image_width // 5))
            strip_h = random.randint(1, max(1, board_image_height // 5))

            # Source composite: board + strips, sized to include strips on all sides
            comp_w = board_image_width + 2 * strip_w
            comp_h = board_image_height + 2 * strip_h
            composite = Image.new("RGBA", (comp_w, comp_h), (0, 0, 0, 0))

            bg_mirror = ImageOps.mirror(board_bg)
            bg_flip = ImageOps.flip(board_bg)
            bg_both = ImageOps.flip(bg_mirror)

            for tx in (-1, 0, 1):
                for ty in (-1, 0, 1):
                    if tx == 0 and ty == 0:
                        continue

                    if tx != 0 and ty != 0:
                        tile = bg_both
                    elif tx != 0:
                        tile = bg_mirror
                    else:
                        tile = bg_flip

                    if tx == -1:
                        comp_dst_x0 = 0
                        comp_dst_x1 = strip_w
                        src_x0 = board_image_width - strip_w
                    elif tx == 1:
                        comp_dst_x0 = strip_w + board_image_width
                        comp_dst_x1 = comp_w
                        src_x0 = 0
                    else:
                        comp_dst_x0 = strip_w
                        comp_dst_x1 = strip_w + board_image_width
                        src_x0 = 0

                    if ty == -1:
                        comp_dst_y0 = 0
                        comp_dst_y1 = strip_h
                        src_y0 = board_image_height - strip_h
                    elif ty == 1:
                        comp_dst_y0 = strip_h + board_image_height
                        comp_dst_y1 = comp_h
                        src_y0 = 0
                    else:
                        comp_dst_y0 = strip_h
                        comp_dst_y1 = strip_h + board_image_height
                        src_y0 = 0

                    tw = comp_dst_x1 - comp_dst_x0
                    th = comp_dst_y1 - comp_dst_y0
                    if tw <= 0 or th <= 0:
                        continue

                    src_crop = tile.crop((src_x0, src_y0, src_x0 + tw, src_y0 + th))
                    composite.paste(src_crop, (comp_dst_x0, comp_dst_y0))

            # Paste the board itself in the center of the composite
            composite.paste(board_image.convert("RGBA"), (strip_w, strip_h))

            # Compute destination corners for the composite (expanded by strips)
            # The board corners in composite space are at (strip_w, strip_h) to
            # (strip_w + board_w, strip_h + board_h). We need to map the full
            # composite corners to the canvas, expanding dst_corners outward by
            # the same relative amount.
            # For each edge, extend the corners outward proportionally.
            tl, tr, br, bl = dst_corners

            def _lerp(a, b, t):
                return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

            # Fractions that strips occupy relative to composite size
            fx0 = strip_w / comp_w
            fy0 = strip_h / comp_h
            fx1 = (strip_w + board_image_width) / comp_w
            fy1 = (strip_h + board_image_height) / comp_h

            # Composite corners: extrapolate from board quad corners
            # Top edge: tl---tr, extend left by fx0/(fx1-fx0) and right similarly
            def _extrapolate(inner_a, inner_b, frac_a, frac_b):
                """Given inner edge endpoints at fractional positions frac_a..frac_b,
                extrapolate to 0..1."""
                span = frac_b - frac_a
                if span < 1e-9:
                    return inner_a, inner_b
                outer_a = _lerp(inner_a, inner_b, -frac_a / span)
                outer_b = _lerp(inner_a, inner_b, (1.0 - frac_a) / span)
                return outer_a, outer_b

            # Extrapolate horizontally first
            ext_tl, ext_tr = _extrapolate(tl, tr, fx0, fx1)
            ext_bl, ext_br = _extrapolate(bl, br, fx0, fx1)
            # Then extrapolate vertically
            comp_tl, comp_bl = _extrapolate(ext_tl, ext_bl, fy0, fy1)
            comp_tr, comp_br = _extrapolate(ext_tr, ext_br, fy0, fy1)

            comp_dst_corners = [comp_tl, comp_tr, comp_br, comp_bl]

            warped_rgb, alpha_mask = _perspective_warp_image(
                composite.convert("RGB"), comp_dst_corners, (crop_width, crop_height)
            )
            _, comp_alpha = _perspective_warp_image(
                composite, comp_dst_corners, (crop_width, crop_height)
            )
            img.paste(warped_rgb, (0, 0), comp_alpha)
        else:
            # No strip tiling — just warp/paste the board alone
            if use_perspective:
                warped_rgb, alpha_mask = _perspective_warp_image(
                    board_image, dst_corners, (crop_width, crop_height)
                )
                img.paste(warped_rgb, (0, 0), alpha_mask)
            else:
                img.paste(board_image, (board_x, board_y))

        # Normalize to [0, 1] relative to crop dimensions and clamp
        img = img.resize((TARGET_SIZE, TARGET_SIZE))

        normalized_corners = [
            (
                max(0.0, min(1.0, cx / crop_width)),
                max(0.0, min(1.0, cy / crop_height)),
            )
            for cx, cy in dst_corners
        ]

        return img, normalized_corners
