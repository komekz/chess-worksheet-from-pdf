from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from cairosvg import svg2png
from PIL import Image

from src.games import GameSpec, get_game


# __file__ is src/render_config.py
# parent is src
# parent is chess_fen_lib
BASE_DIR = Path(__file__).resolve().parent.parent

PIECES_DIR = BASE_DIR / "resources" / "pieces" 
ALLOWED_EXTENSIONS = {".svg", ".png", ".webp", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class PieceSetConfig:
    provider: str
    set_name: str


@dataclass(frozen=True)
class GameRenderConfig:
    piece_sets: tuple[PieceSetConfig, ...]
    piece_file_names_by_provider: dict[str, tuple[str, ...]]


def _discover_render_config(game: str) -> GameRenderConfig:
    game_dir = PIECES_DIR / game
    if not game_dir.is_dir():
        return GameRenderConfig(piece_sets=(), piece_file_names_by_provider={})

    piece_sets: list[PieceSetConfig] = []
    file_names_by_provider: dict[str, set[str]] = {}

    for provider_dir in sorted(game_dir.iterdir()):
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name
        for set_dir in sorted(provider_dir.iterdir()):
            if not set_dir.is_dir():
                continue
            piece_sets.append(PieceSetConfig(provider, set_dir.name))
            for f in set_dir.iterdir():
                if f.suffix.lower() in ALLOWED_EXTENSIONS:
                    file_names_by_provider.setdefault(provider, set()).add(f.name)

    return GameRenderConfig(
        piece_sets=tuple(piece_sets),
        piece_file_names_by_provider={
            k: tuple(sorted(v)) for k, v in file_names_by_provider.items()
        },
    )


GAME_RENDER_CONFIGS: dict[str, GameRenderConfig] = {
    game_dir.name: _discover_render_config(game_dir.name)
    for game_dir in sorted(PIECES_DIR.iterdir())
    if game_dir.is_dir()
}


def get_render_config(game: str | GameSpec) -> GameRenderConfig:
    spec = get_game(game)
    if spec.key not in GAME_RENDER_CONFIGS:
        raise FileNotFoundError(
            f"No piece sets found for game '{spec.key}' in {PIECES_DIR / spec.key}"
        )
    return GAME_RENDER_CONFIGS[spec.key]


def list_board_theme_paths(game: str) -> list[Path]:
    game_dir = Path(f"resources/board_themes/{game}")
    if not game_dir.exists():
        return []
    return [
        p for p in game_dir.rglob("*")
        if p.suffix.lower() in ALLOWED_EXTENSIONS
    ]


def open_board_theme(path: Path, board_w: int, board_h: int) -> Image.Image:
    if path.suffix.lower() == ".svg":
        with open(path, "rb") as f:
            svg_data = f.read()
        png_data = svg2png(
            bytestring=svg_data, output_width=board_w
        )
        img = Image.open(BytesIO(png_data)).convert("RGBA")
    else:
        img = Image.open(path).convert("RGBA")
    img.putalpha(255)
    return img.resize((board_w, board_h))
