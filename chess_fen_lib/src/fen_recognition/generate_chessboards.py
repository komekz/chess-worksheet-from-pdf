import random
import urllib.request
import warnings
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from cairosvg import svg2png
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pyfastnoiselite.pyfastnoiselite import (
    CellularDistanceFunction,
    CellularReturnType,
    DomainWarpType,
    FastNoiseLite,
    FractalType,
    NoiseType,
    RotationType3D,
)

from src import common, consts
from src.games import get_game
from src.render_config import (
    get_render_config,
    list_board_theme_paths,
    open_board_theme,
)


def _random_uniform_board(board_h: int, board_w: int) -> Image.Image:
    min_color = random.randint(0, 120)
    max_color = random.randint(140, 255)
    color = (
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        255,
    )
    return Image.new("RGBA", (board_w, board_h), color)


def _make_noise_generator() -> FastNoiseLite:
    noise = FastNoiseLite(seed=random.randint(0, 2**31 - 1))

    # --- Noise Type ---
    noise_type = random.choice(
        [
            NoiseType.NoiseType_OpenSimplex2,
            NoiseType.NoiseType_OpenSimplex2S,
            NoiseType.NoiseType_Cellular,
            NoiseType.NoiseType_Perlin,
            NoiseType.NoiseType_ValueCubic,
            NoiseType.NoiseType_Value,
        ]
    )
    noise.noise_type = noise_type
    noise.frequency = random.uniform(0.003, 0.06)

    # --- Rotation Type 3D ---
    noise.rotation_type_3d = random.choice(
        [
            RotationType3D.RotationType3D_None,
            RotationType3D.RotationType3D_ImproveXYPlanes,
            RotationType3D.RotationType3D_ImproveXZPlanes,
        ]
    )

    # --- Fractal Type ---
    fractal_type = random.choice(
        [
            FractalType.FractalType_None,
            FractalType.FractalType_FBm,
            FractalType.FractalType_Ridged,
            FractalType.FractalType_PingPong,
            FractalType.FractalType_DomainWarpProgressive,
            FractalType.FractalType_DomainWarpIndependent,
        ]
    )
    noise.fractal_type = fractal_type

    if fractal_type != FractalType.FractalType_None:
        noise.fractal_octaves = random.randint(2, 8)
        noise.fractal_lacunarity = random.uniform(1.5, 3.0)
        noise.fractal_gain = random.uniform(0.3, 0.7)
        noise.fractal_weighted_strength = random.uniform(0.0, 1.0)
        if fractal_type == FractalType.FractalType_PingPong:
            noise.fractal_ping_pong_strength = random.uniform(1.0, 5.0)

    # --- Cellular settings (only meaningful for Cellular noise) ---
    if noise_type == NoiseType.NoiseType_Cellular:
        noise.cellular_distance_function = random.choice(
            [
                CellularDistanceFunction.CellularDistanceFunction_Euclidean,
                CellularDistanceFunction.CellularDistanceFunction_EuclideanSq,
                CellularDistanceFunction.CellularDistanceFunction_Manhattan,
                CellularDistanceFunction.CellularDistanceFunction_Hybrid,
            ]
        )
        noise.cellular_return_type = random.choice(
            [
                CellularReturnType.CellularReturnType_CellValue,
                CellularReturnType.CellularReturnType_Distance,
                CellularReturnType.CellularReturnType_Distance2,
                CellularReturnType.CellularReturnType_Distance2Add,
                CellularReturnType.CellularReturnType_Distance2Sub,
                CellularReturnType.CellularReturnType_Distance2Mul,
                CellularReturnType.CellularReturnType_Distance2Div,
            ]
        )
        noise.cellular_jitter = random.uniform(0.2, 1.5)

    # --- Domain Warp ---
    if random.random() < 0.4:
        noise.domain_warp_type = random.choice(
            [
                DomainWarpType.DomainWarpType_OpenSimplex2,
                DomainWarpType.DomainWarpType_OpenSimplex2Reduced,
                DomainWarpType.DomainWarpType_BasicGrid,
            ]
        )
        noise.domain_warp_amp = random.uniform(10.0, 200.0)

    return noise


def _noisy_gray_board(board_h: int, board_w: int) -> Image.Image:
    noise = _make_noise_generator()
    ys, xs = np.mgrid[0:board_h, 0:board_w].astype(np.float32)
    coords = np.array([xs.reshape(-1), ys.reshape(-1)], dtype=np.float32)
    noise_array = noise.gen_from_coords(coords).reshape((board_h, board_w))
    noise_array = np.interp(
        noise_array, (noise_array.min(), noise_array.max()), (0, 255)
    ).astype(np.uint8)
    return Image.fromarray(noise_array, mode="L")


def _noisy_color_board(board_h: int, board_w: int) -> Image.Image:
    return Image.merge(
        "RGB",
        (
            _noisy_gray_board(board_h, board_w),
            _noisy_gray_board(board_h, board_w),
            _noisy_gray_board(board_h, board_w),
        ),
    )


_FONT_CACHE_DIR = Path("resources/fonts")

# Curated Google Fonts covering many scripts and styles.
# Format: (filename, Google Fonts GitHub path relative to google/fonts/main/)
_GOOGLE_FONTS = [
    # Latin / Latin Extended
    ("Roboto-Regular.ttf", "ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf"),
    ("OpenSans-Regular.ttf", "ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf"),
    ("Lora-Regular.ttf", "ofl/lora/Lora%5Bwght%5D.ttf"),
    ("Oswald-Regular.ttf", "ofl/oswald/Oswald%5Bwght%5D.ttf"),
    (
        "PlayfairDisplay-Regular.ttf",
        "ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    ),
    ("Pacifico-Regular.ttf", "ofl/pacifico/Pacifico-Regular.ttf"),
    ("IndieFlower-Regular.ttf", "ofl/indieflower/IndieFlower-Regular.ttf"),
    ("DancingScript-Regular.ttf", "ofl/dancingscript/DancingScript%5Bwght%5D.ttf"),
    ("Caveat-Regular.ttf", "ofl/caveat/Caveat%5Bwght%5D.ttf"),
    ("AbrilFatface-Regular.ttf", "ofl/abrilfatface/AbrilFatface-Regular.ttf"),
    ("PressStart2P-Regular.ttf", "ofl/pressstart2p/PressStart2P-Regular.ttf"),
    ("Bangers-Regular.ttf", "ofl/bangers/Bangers-Regular.ttf"),
    ("RubikMonoOne-Regular.ttf", "ofl/rubikmonoone/RubikMonoOne-Regular.ttf"),
    ("FredokaOne-Regular.ttf", "ofl/fredoka/Fredoka%5Bwdth%2Cwght%5D.ttf"),
    ("Orbitron-Regular.ttf", "ofl/orbitron/Orbitron%5Bwght%5D.ttf"),
    # Cyrillic
    ("Rubik-Regular.ttf", "ofl/rubik/Rubik%5Bwght%5D.ttf"),
    ("PTSerif-Regular.ttf", "ofl/ptserif/PT_Serif-Web-Regular.ttf"),
    ("Comfortaa-Regular.ttf", "ofl/comfortaa/Comfortaa%5Bwght%5D.ttf"),
    # Greek
    ("GFSDidot-Regular.ttf", "ofl/gfsdidot/GFSDidot-Regular.ttf"),
    # Arabic
    ("Amiri-Regular.ttf", "ofl/amiri/Amiri-Regular.ttf"),
    ("Cairo-Regular.ttf", "ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf"),
    ("Tajawal-Regular.ttf", "ofl/tajawal/Tajawal-Regular.ttf"),
    # Devanagari / Hindi
    ("Poppins-Regular.ttf", "ofl/poppins/Poppins-Regular.ttf"),
    (
        "NotoSansDevanagari-Regular.ttf",
        "ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
    ),
    # Bengali
    ("TiroBangla-Regular.ttf", "ofl/tirobangla/TiroBangla-Regular.ttf"),
    # Tamil
    (
        "NotoSansTamil-Regular.ttf",
        "ofl/notosanstamil/NotoSansTamil%5Bwdth%2Cwght%5D.ttf",
    ),
    # Thai
    ("Sarabun-Regular.ttf", "ofl/sarabun/Sarabun-Regular.ttf"),
    ("Kanit-Regular.ttf", "ofl/kanit/Kanit-Regular.ttf"),
    # Japanese
    ("NotoSansJP-Regular.ttf", "ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"),
    ("ZenMaruGothic-Regular.ttf", "ofl/zenmarugothic/ZenMaruGothic-Regular.ttf"),
    # Korean
    ("NotoSansKR-Regular.ttf", "ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf"),
    ("GamjaFlower-Regular.ttf", "ofl/gamjaflower/GamjaFlower-Regular.ttf"),
    # Simplified Chinese
    ("NotoSansSC-Regular.ttf", "ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"),
    # Traditional Chinese
    ("NotoSansTC-Regular.ttf", "ofl/notosanstc/NotoSansTC%5Bwght%5D.ttf"),
    # Georgian
    (
        "NotoSansGeorgian-Regular.ttf",
        "ofl/notosansgeorgian/NotoSansGeorgian%5Bwdth%2Cwght%5D.ttf",
    ),
    # Armenian
    (
        "NotoSansArmenian-Regular.ttf",
        "ofl/notosansarmenian/NotoSansArmenian%5Bwdth%2Cwght%5D.ttf",
    ),
    # Hebrew
    (
        "NotoSansHebrew-Regular.ttf",
        "ofl/notosanshebrew/NotoSansHebrew%5Bwdth%2Cwght%5D.ttf",
    ),
    # Ethiopic
    (
        "NotoSansEthiopic-Regular.ttf",
        "ofl/notosansethiopic/NotoSansEthiopic%5Bwdth%2Cwght%5D.ttf",
    ),
    # Symbols / Emoji
    (
        "NotoSansSymbols-Regular.ttf",
        "ofl/notosanssymbols/NotoSansSymbols%5Bwght%5D.ttf",
    ),
    (
        "NotoSansSymbols2-Regular.ttf",
        "ofl/notosanssymbols2/NotoSansSymbols2-Regular.ttf",
    ),
    ("NotoSansMath-Regular.ttf", "ofl/notosansmath/NotoSansMath-Regular.ttf"),
    # Decorative / Display
    ("Lobster-Regular.ttf", "ofl/lobster/Lobster-Regular.ttf"),
    ("RubikGlitch-Regular.ttf", "ofl/rubikglitch/RubikGlitch-Regular.ttf"),
    ("Silkscreen-Regular.ttf", "ofl/silkscreen/Silkscreen-Regular.ttf"),
    (
        "UnifrakturMaguntia-Regular.ttf",
        "ofl/unifrakturmaguntia/UnifrakturMaguntia-Book.ttf",
    ),
]

_GOOGLE_FONTS_BASE_URL = "https://github.com/google/fonts/raw/main/"


def _download_google_fonts() -> list[Path]:
    """Download curated Google Fonts to local cache. Skips already-cached files."""
    _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename, url_path in _GOOGLE_FONTS:
        dest = _FONT_CACHE_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            downloaded.append(dest)
            continue
        url = _GOOGLE_FONTS_BASE_URL + url_path
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded.append(dest)
        except Exception as e:
            warnings.warn(f"Failed to download font {filename}: {e}", stacklevel=2)
            if dest.exists():
                dest.unlink()
    return downloaded


@lru_cache(maxsize=1)
def _load_fonts() -> list[tuple[str, tuple[int, ...]]]:
    """Download (if needed) and load Google Fonts with their supported codepoints."""
    font_files = _download_google_fonts()

    fonts: list[tuple[str, tuple[int, ...]]] = []
    for path in font_files:
        try:
            tt = TTFont(str(path), fontNumber=0)
            cmap = tt.getBestCmap()
            if cmap is None:
                continue
            codepoints = tuple(
                cp for cp in cmap if cp >= 0x20 and chr(cp).isprintable()
            )
            if len(codepoints) >= 20:
                fonts.append((str(path), codepoints))
        except Exception:
            continue

    if len(fonts) < 20:
        warnings.warn(
            f"Only {len(fonts)} usable fonts loaded (expected ~{len(_GOOGLE_FONTS)}). "
            f"Random text overlays may lack variety.",
            stacklevel=2,
        )
    return fonts


def _sample_text_from_font(codepoints: tuple[int, ...], length: int) -> str:
    """Sample `length` random characters that the font is known to support."""
    return "".join(chr(random.choice(codepoints)) for _ in range(length))


def _overlay_random_text(image: Image.Image) -> Image.Image:
    """Overlay random unicode strings on the image with random fonts, sizes, rotations, colors, positions."""
    fonts = _load_fonts()
    if not fonts:
        return image

    w, h = image.size
    result = image.convert("RGBA") if image.mode != "RGBA" else image.copy()

    num_overlays = random.choice(
        [random.randint(1, 30), random.randint(1, 15), random.randint(1, 5)]
    )
    for _ in range(num_overlays):
        font_size = random.randint(max(4, h // 30), max(5, h // 6))
        font_path, codepoints = random.choice(fonts)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, Exception):
            continue

        # some single char, some multichar
        text_len = 1 if random.random() < 0.8 else random.randint(2, 8)
        text = _sample_text_from_font(codepoints, text_len)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Render text onto a temporary RGBA image
        dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0] + 4
        th = bbox[3] - bbox[1] + 4
        if tw <= 0 or th <= 0:
            continue

        txt_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((-bbox[0] + 2, -bbox[1] + 2), text, fill=(*color, 255), font=font)

        # rotate sometimes
        if random.random() < 0.5:
            angle = random.choice([random.uniform(0, 360), random.uniform(-20, 20)])
            txt_img = txt_img.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Paste at random position
        rw, rh = txt_img.size
        x = random.randint(-rw // 2, max(0, w - rw // 2))
        y = random.randint(-rh // 2, max(0, h - rh // 2))
        result.paste(txt_img, (x, y), txt_img)

    return result.convert(image.mode)


def _svg_to_image(svg_file: Path, tile_size: int) -> Image.Image:
    with open(svg_file, "rb") as f:
        svg_data = f.read()
    png_data = svg2png(
        bytestring=svg_data, output_width=tile_size, output_height=tile_size
    )
    return Image.open(BytesIO(png_data)).convert("RGBA")


def _symbol_to_asset_key(symbol: str) -> str:
    return ("w" if symbol.isupper() else "b") + symbol.lower()


@lru_cache(maxsize=None)
def _load_piece_images(game: str, tile_size: int) -> list[dict[str, Image.Image]]:
    spec = get_game(game)
    config = get_render_config(spec)

    all_sets: list[dict[str, Image.Image]] = []
    for piece_set in config.piece_sets:
        images: dict[str, Image.Image] = {}
        file_names = config.piece_file_names_by_provider.get(piece_set.provider, ())
        set_dir = Path(
            f"resources/pieces/{game}/{piece_set.provider}/{piece_set.set_name}"
        )
        for file_name in file_names:
            path = set_dir / file_name
            if not path.exists():
                continue
            if path.suffix.lower() == ".svg":
                img = _svg_to_image(path, tile_size)
            else:
                img = Image.open(path).convert("RGBA").resize((tile_size, tile_size))
            images[path.stem.lower()] = img

        # Verify all expected pieces are present.
        for symbol in spec.piece_symbols:
            key = _symbol_to_asset_key(symbol)
            if key not in images:
                raise FileNotFoundError(
                    f"Missing piece image for '{symbol}' (expected key '{key}') "
                    f"in {set_dir}"
                )

        all_sets.append(images)

    if not all_sets:
        raise FileNotFoundError(f"No piece sets found for game '{game}'")

    return all_sets


def _warp_piece_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    # Piecewise mesh warp gives stronger non-linear deformation.
    div_x = random.choice((2, 3))
    div_y = random.choice((2, 3))
    max_jitter = min(w, h) * random.uniform(0.0, 0.2)

    x_grid = [i * (w - 1) / div_x for i in range(div_x + 1)]
    y_grid = [i * (h - 1) / div_y for i in range(div_y + 1)]
    points: list[list[tuple[float, float]]] = []
    for gy, base_y in enumerate(y_grid):
        row: list[tuple[float, float]] = []
        for gx, base_x in enumerate(x_grid):
            is_border = gx == 0 or gx == div_x or gy == 0 or gy == div_y
            local_jitter = max_jitter * (0.7 if is_border else 1.0)
            x = np.clip(
                base_x + random.uniform(-local_jitter, local_jitter), 0.0, w - 1
            )
            y = np.clip(
                base_y + random.uniform(-local_jitter, local_jitter), 0.0, h - 1
            )
            row.append((float(x), float(y)))
        points.append(row)

    mesh = []
    for gy in range(div_y):
        for gx in range(div_x):
            left = int(round(x_grid[gx]))
            upper = int(round(y_grid[gy]))
            right = int(round(x_grid[gx + 1])) + 1
            lower = int(round(y_grid[gy + 1])) + 1
            if right <= left:
                right = min(w, left + 1)
            if lower <= upper:
                lower = min(h, upper + 1)
            if right <= left or lower <= upper:
                continue
            quad = (
                *points[gy][gx],
                *points[gy + 1][gx],
                *points[gy + 1][gx + 1],
                *points[gy][gx + 1],
            )
            mesh.append(((left, upper, right, lower), quad))

    return img.transform(
        (w, h),
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )


def _shift_hue(img: Image.Image, hue_shift: float) -> Image.Image:
    """Shift hue of an RGBA image by hue_shift in [0, 1)."""
    arr = np.array(img, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:]

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    s = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc != 0)

    h = np.zeros_like(r)
    mask = delta != 0
    m = mask & (maxc == r)
    h[m] = (g[m] - b[m]) / delta[m] % 6
    m = mask & (maxc == g)
    h[m] = (b[m] - r[m]) / delta[m] + 2
    m = mask & (maxc == b)
    h[m] = (r[m] - g[m]) / delta[m] + 4
    h = (h / 6.0 + hue_shift) % 1.0

    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r_new = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v]
    )
    g_new = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p]
    )
    b_new = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q]
    )

    arr_out = np.concatenate([np.stack([r_new, g_new, b_new], axis=-1), alpha], axis=-1)
    return Image.fromarray((np.clip(arr_out, 0, 1) * 255).astype(np.uint8), mode="RGBA")


def _board_to_image(
    position: common.Position,
    board_image: Image.Image,
    tile_size: int,
    piece_images: dict[str, Image.Image],
    random_offset: int,
) -> Image.Image:

    spec = get_game(position.game)
    img = board_image.copy()
    grid = common.parse_piece_placement(position.piece_placement, spec)
    use_warped_pieces = random.random() < 0.8
    per_board_piece_images = piece_images
    if use_warped_pieces:
        # Keep one warp per piece type (e.g. all pawns share the same warp on this board).
        warped_by_type: dict[str, Image.Image] = {}
        per_board_piece_images = {}
        for key, piece_img in piece_images.items():
            piece_type = key
            if piece_type not in warped_by_type:
                warped_by_type[piece_type] = _warp_piece_image(piece_img)
            per_board_piece_images[key] = warped_by_type[piece_type]

    if random.random() < 0.5:
        hue_shift = random.random()
        per_board_piece_images = {
            key: _shift_hue(piece_img, hue_shift)
            for key, piece_img in per_board_piece_images.items()
        }

    idx = 0
    for row in range(spec.board_rows):
        for col in range(spec.board_cols):
            piece = grid[idx]
            idx += 1
            if piece is None:
                continue

            piece_img = per_board_piece_images[_symbol_to_asset_key(piece)]
            
            if spec.opponent_pieces_may_be_rotated and random.random() < 0.2:
                piece_img = piece_img.rotate(180)

            x = col * tile_size + random.randint(-random_offset, random_offset)
            y = row * tile_size + random.randint(-random_offset, random_offset)
            img.paste(piece_img, (x, y), piece_img)

    return img


def _flip_piece_colors(position: common.Position) -> common.Position:
    spec = get_game(position.game)
    tensor = common.position_to_tensor(position)
    flipped = common.flip_color_tensor(tensor, spec)
    return common.tensor_to_position(flipped, spec, side_to_move=position.side_to_move)


def _random_position(game: str) -> common.Position:
    spec = get_game(game)
    grid = [None] * spec.num_squares
    approx_num_empty = random.randint(1, spec.num_squares)
    for square in range(spec.num_squares):
        if random.randint(1, spec.num_squares) < approx_num_empty:
            continue
        grid[square] = random.choice(spec.piece_symbols)

    return common.Position(
        game=spec.key, piece_placement=common.grid_to_piece_placement(grid, spec)
    )


class BoardGenerator:
    def __init__(self, game: str, tile_size: int = consts.DEFAULT_TILE_SIZE):
        self.spec = get_game(game)
        self.tile_size = tile_size
        self.board_h, self.board_w = consts.board_pixel_size(self.spec, tile_size)
        self.random_offset = max(1, tile_size // 30)
        self.piece_image_sets = _load_piece_images(self.spec.key, tile_size)
        self.disk_themes = []
        for path in list_board_theme_paths(self.spec.key):
            self.disk_themes.append(
                open_board_theme(path, board_w=self.board_w, board_h=self.board_h)
            )

    def generate_board_background(self, width: int, height: int) -> Image.Image:
        """Return a randomly styled board background (no pieces) at the given size."""
        theme_choice = random.choice(
            ["uniform", "noise_gray", "noise_color"]
            + 6 * (["disk"] if self.disk_themes else [])
        )
        if theme_choice == "uniform":
            bg = _random_uniform_board(height, width)
        elif theme_choice == "noise_gray":
            bg = _noisy_gray_board(height, width).convert("RGBA")
        elif theme_choice == "noise_color":
            bg = _noisy_color_board(height, width).convert("RGBA")
        else:
            bg = random.choice(self.disk_themes).copy().resize((width, height))

        if random.randint(0, 1) == 1:
            bg = ImageOps.mirror(bg)
        if random.randint(0, 1) == 1:
            bg = ImageOps.flip(bg)
        if random.randint(0, 1) == 1:
            noise = (
                _noisy_color_board(height, width)
                if random.randint(0, 1) == 1
                else _noisy_gray_board(height, width)
            )
            bg.paste(noise, mask=_noisy_gray_board(height, width))

        return bg.convert("RGBA")

    def generate_one(
        self,
        board_background: Image.Image | None = None,
    ) -> tuple[Image.Image, common.Position]:
        board_h, board_w = self.board_h, self.board_w

        current_board = (
            board_background
            if board_background is not None
            else self.generate_board_background(board_w, board_h)
        )

        piece_images = random.choice(self.piece_image_sets)
        position = _random_position(self.spec.key)
        image = _board_to_image(
            position, current_board, self.tile_size, piece_images, self.random_offset
        )

        image = image.convert("RGB")

        if random.random() < 0.1 and not self.spec.sides_may_use_distinct_piece_glyphs:
            if self.spec.color_encodes_piece_side:
                position = _flip_piece_colors(position)
            image = ImageOps.invert(image)

        # some chance for overlaying random unicode text
        if random.random() < 0.8:
            image = _overlay_random_text(image)

        return image, position
