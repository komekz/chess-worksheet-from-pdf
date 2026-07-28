import re
from dataclasses import dataclass
from pathlib import Path

import pyffish as sf

@dataclass(frozen=True)
class ParsedPGN:
    tags: dict[str, str]
    moves: list[str]
    fens: tuple[str, ...]


RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}



def parse_variant_tag(raw_variant: str | None) -> tuple[str, bool]:
    raw = (raw_variant or "").strip().lower()
    if not raw:
        raise ValueError("PGN Variant tag is required")
    chess960 = "960" in raw or "random" in raw
    variant = raw.removesuffix("960")
    if variant == "caparandom":
        return "capablanca", True
    if variant == "fischerandom":
        return "chess", True
    if not variant:
        raise ValueError("PGN Variant tag is required")
    return variant, chess960


def parse_pgn_tags(pgn_text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    regex = re.compile(r'^\s*\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$', re.MULTILINE)
    for match in regex.finditer(pgn_text):
        tags[match.group(1)] = match.group(2).replace('\\"', '"')
    return tags


def _split_headers_and_body(pgn_text: str) -> tuple[dict[str, str], str]:
    tags: dict[str, str] = {}
    body_lines: list[str] = []
    headers_parsed = False

    for raw_line in pgn_text.splitlines():
        line = raw_line.strip()
        if not headers_parsed and line.startswith("["):
            match = re.match(r'^\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$', line)
            if match:
                tags[match.group(1)] = match.group(2).replace('\\"', '"')
            continue

        headers_parsed = True
        body_lines.append(raw_line)

    return tags, "\n".join(body_lines)


def _skip_comment(text: str, idx: int) -> int:
    end = text.find("}", idx)
    if end < 0:
        raise ValueError("Missing '}' for PGN comment")
    return end + 1


def _normalize_san_token(move: str) -> str:
    move = move.strip()
    move = move.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    move = re.sub(r"[?!]+$", "", move)
    move = re.sub(r"[+#]+$", "", move)
    return move


def _safe_get_san(variant: str, fen: str, move: str, chess960: bool, notation: int | None = None) -> str:
    if notation is None:
        try:
            return sf.get_san(variant, fen, move, chess960)
        except TypeError:
            return sf.get_san(variant, fen, move)
    return sf.get_san(variant, fen, move, chess960, notation)


def _legal_moves(
    variant: str,
    initial_fen: str,
    fen: str,
    history: list[str],
    chess960: bool,
) -> list[str]:
    # Janggi and Ataxx may require history for legal move generation.
    if variant in {"janggi", "ataxx"}:
        return list(sf.legal_moves(variant, initial_fen, history, chess960))
    return list(sf.legal_moves(variant, fen, [], chess960))


def _san_destination(san: str) -> str | None:
    """Extract the 2-char destination square from a normalized SAN token, if possible.

    Used to pre-filter legal moves before calling sf.get_san, reducing the number
    of expensive pyffish calls from O(legal_moves) to O(1-3) for most positions.
    """
    # Castling: we can't reliably map O-O / O-O-O to a UCI destination up front.
    if san.startswith("O-O"):
        return None
    # Drops (e.g. shogi "P*7f"): destination is the 2 chars after '*'.
    star = san.find("*")
    if star >= 0:
        rest = san[star + 1:]
        return rest[:2] if len(rest) >= 2 else None
    # Strip promotion suffix (e.g. "a8=Q" → "a8").
    eq = san.find("=")
    main = san[:eq] if eq >= 0 else san
    return main[-2:] if len(main) >= 2 else None


def _resolve_move_token(
    token: str,
    variant: str,
    initial_fen: str,
    fen: str,
    history: list[str],
    chess960: bool,
) -> str:
    legal = _legal_moves(variant, initial_fen, fen, history, chess960)

    # If PGN move text already contains UCI-like moves, accept directly.
    if token in legal:
        return token

    norm = _normalize_san_token(token)

    # Pre-filter legal moves by destination square to reduce sf.get_san calls.
    # UCI moves are typically "a2a4" format; destination is at indices [2:4].
    dest = _san_destination(norm)
    fast = [m for m in legal if len(m) >= 4 and m[2:4] == dest] if dest else []

    # For some variants (e.g. shogi) the pyffish default sf.get_san() produces a
    # notation that never matches pychess PGN (Japanese-style vs Western SAN).
    # Scanning all legal moves with the wrong notation wastes O(legal_moves) calls
    # per move, so we skip the default scan for those variants and go straight to
    # the notation that is known to match.
    _PRIMARY_NOTATION: dict[str, int | None] = {
        "shogi": getattr(sf, "NOTATION_SAN", None),
    }
    primary_notation = _PRIMARY_NOTATION.get(variant)

    # Fast path: use the primary (or default) notation, filtered by destination first.
    tried: set[str] = set()
    for move in (*fast, *legal):
        if move in tried:
            continue
        tried.add(move)
        try:
            if primary_notation is not None:
                san = _normalize_san_token(
                    _safe_get_san(variant, fen, move, chess960, notation=primary_notation)
                )
            else:
                san = _normalize_san_token(_safe_get_san(variant, fen, move, chess960))
        except Exception:
            continue
        if san == norm:
            return move

    # Fallback path: try remaining notations only when the fast path found no match.
    notation_candidates = (
        getattr(sf, "NOTATION_SAN", None),
        getattr(sf, "NOTATION_JANGGI", None),
        getattr(sf, "NOTATION_XIANGQI_WXF", None),
        getattr(sf, "NOTATION_SHOGI_HODGES_NUMBER", None),
    )
    for notation in notation_candidates:
        if notation is None or notation == primary_notation:
            continue
        for move in legal:
            try:
                san = _normalize_san_token(_safe_get_san(variant, fen, move, chess960, notation=notation))
            except Exception:
                continue
            if san == norm:
                return move

    raise ValueError(f"Illegal move token in PGN: '{token}'")


def extract_mainline_moves(move_text: str) -> list[str]:
    moves: list[str] = []
    idx = 0
    n = len(move_text)
    sorted_result_tokens = sorted(RESULT_TOKENS, key=len, reverse=True)
    while idx < n:
        ch = move_text[idx]

        if ch.isspace():
            idx += 1
            continue

        if ch == "{":
            idx = _skip_comment(move_text, idx)
            continue

        if ch == ";":
            end = move_text.find("\n", idx)
            idx = n if end < 0 else end + 1
            continue

        if ch == "(":
            depth = 1
            idx += 1
            while idx < n and depth > 0:
                if move_text[idx] == "{":
                    idx = _skip_comment(move_text, idx)
                    continue
                if move_text[idx] == "(":
                    depth += 1
                elif move_text[idx] == ")":
                    depth -= 1
                idx += 1
            continue

        if ch == "$":
            idx += 1
            while idx < n and move_text[idx].isdigit():
                idx += 1
            continue

        # Result tokens can start with a digit (e.g. "0-1"), so detect them
        # before treating leading digits as move numbers.
        matched_result = None
        for result_token in sorted_result_tokens:
            if move_text.startswith(result_token, idx):
                end = idx + len(result_token)
                if end == n or move_text[end].isspace() or move_text[end] in ")}]":
                    matched_result = result_token
                    break
        if matched_result is not None:
            break

        if ch.isdigit():
            while idx < n and move_text[idx].isdigit():
                idx += 1
            while idx < n and move_text[idx] in ". ":
                idx += 1
            continue

        end = idx
        while end < n and not move_text[end].isspace():
            end += 1
        token = move_text[idx:end]
        idx = end

        if token in RESULT_TOKENS:
            break

        token = _normalize_san_token(token)
        if token and token not in RESULT_TOKENS:
            moves.append(token)

    return moves


def read_game_pgn(pgn_text: str) -> ParsedPGN:
    tags, body = _split_headers_and_body(pgn_text)
    variant, chess960 = parse_variant_tag(tags.get("Variant"))
    initial_fen = tags.get("FEN") or sf.start_fen(variant)

    raw_tokens = extract_mainline_moves(body)
    fen = initial_fen
    history: list[str] = []
    moves: list[str] = []

    fens: list[str] = []
    for token in raw_tokens:
        move = _resolve_move_token(
            token,
            variant=variant,
            initial_fen=initial_fen,
            fen=fen,
            history=history,
            chess960=chess960,
        )
        moves.append(move)
        history.append(move)
        try:
            fen = sf.get_fen(variant, fen, [move], chess960, False, False, 0)
        except TypeError:
            fen = sf.get_fen(variant, fen, [move], chess960)
        fens.append(fen)

    return ParsedPGN(tags=tags, moves=moves, fens=tuple(fens))


def parse_pgn_game(pgn_text: str) -> ParsedPGN:
    return read_game_pgn(pgn_text)


def replay_moves_to_fens(
    moves: list[str], variant: str, initial_fen: str | None = None, chess960: bool = False
) -> list[str]:
    fen = initial_fen or sf.start_fen(variant)
    fens: list[str] = []
    for move in moves:
        try:
            fen = sf.get_fen(variant, fen, [move], chess960, False, False, 0)
        except TypeError:
            fen = sf.get_fen(variant, fen, [move], chess960)
        fens.append(fen)
    return fens


def iter_pgn_games(file_path: str | Path):
    current_lines: list[str] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("[Event ") and current_lines:
                game_text = "".join(current_lines).strip()
                if game_text:
                    yield game_text
                current_lines = [line]
                continue
            current_lines.append(line)
    if current_lines:
        game_text = "".join(current_lines).strip()
        if game_text:
            yield game_text
