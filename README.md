# Convert PDFs to Chess Worksheets

Convert multi page PDF into individual chess worksheets with 6 puzzles each. The PDF must also have 6 puzzles each page.

# Setup

Install all dependencies with one of the following:
```shell
uv sync --extra cpu
uv sync --extra cuda
```

Download the pretrained models (from the root):
```shell
./chess_fen_lib/download_models.sh
```