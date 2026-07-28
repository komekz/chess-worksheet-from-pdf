import torch
import torch.nn as nn

from src.games import get_game, GameSpec


class OrientationModel(nn.Module):
    def __init__(self, game: str | GameSpec = "chess"):
        super().__init__()
        spec = get_game(game)
        self.board_rows = spec.board_rows
        self.board_cols = spec.board_cols
        in_channels = len(spec.piece_symbols) + 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        batch = x.size(0)
        # x: [batch, num_squares, channels] -> [batch, channels, rows, cols]
        x = x.view(batch, self.board_rows, self.board_cols, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.view(batch, -1)
        x = self.head(x)
        return x
