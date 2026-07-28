# Convert PDFs to Chess Worksheets

Convert multi page PDF into individual chess worksheets with 6 puzzles each. The PDF must also have 6 puzzles each page.

# Setup

Install all dependencies with one of the following:
```shell
uv sync --extra cpu
uv sync --extra cuda
```

Download the pretrained models (from root):
```shell
./chess_fen_lib/download_models.sh
```

Run playwright install:
```shell
uv run playwright install chromium
```

Download Cairo (macos):
```shell
brew install cairo
```

To run the script:
```shell
uv run python main.py <PATH/TO/INPUT.pdf> -n <WORKSHEET_NAME>
```
