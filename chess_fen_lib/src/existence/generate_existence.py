import random
from pathlib import Path

from PIL import Image

from src import consts, common

MIN_CROP_SIZE = consts.BOARD_PIXEL_WIDTH + 10
MAX_CROP_SIZE = consts.BOARD_PIXEL_WIDTH * 4

TARGET_SIZE = consts.BOARD_PIXEL_WIDTH * 2


class NoboardGenerator:
    def __init__(self, background_root_dir: str = "resources/website_screenshots"):
        background_root_dir = Path(background_root_dir)
        assert background_root_dir.is_dir(), f"With background_root_dir = {background_root_dir}"
        self.background_image_files = common.glob_all_image_files_recursively(background_root_dir)
        assert len(self.background_image_files) > 0, "No background images found"

    def generate_one(self) -> Image.Image:
        bg_path = random.choice(self.background_image_files)
        img = Image.open(bg_path).convert("RGB")

        img_width, img_height = img.size
        max_size = min(img_width, img_height, MAX_CROP_SIZE)
        crop_width = random.randint(MIN_CROP_SIZE, max_size)
        crop_height = random.randint(MIN_CROP_SIZE, max_size)

        x = random.randint(0, img_width - crop_width)
        y = random.randint(0, img_height - crop_height)

        img = img.crop((x, y, x + crop_width, y + crop_height))
        img = img.resize((TARGET_SIZE, TARGET_SIZE))
        return img
