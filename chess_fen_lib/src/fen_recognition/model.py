import torch
import torch.nn as nn
import torchvision.models as models

from src import consts
from src.games import get_game

TILE_EMBED_DIM = 512
FULL_EMBED_DIM = 512


def get_tile_model():
    result = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    result.classifier = nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=TILE_EMBED_DIM),
        nn.ReLU(),
    )
    return result


def get_full_img_model():
    result = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    result.classifier = nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=FULL_EMBED_DIM),
        nn.ReLU(),
    )
    return result


class BoardRec(nn.Module):
    def __init__(
        self, game: str, tile_size: int = consts.DEFAULT_TILE_SIZE, dropout: float = 0.1
    ):
        super().__init__()
        self.game = get_game(game)
        self.tile_size = tile_size
        self.board_h, self.board_w = consts.board_pixel_size(self.game, tile_size)
        self.num_squares = self.game.num_squares
        self.out_channels = len(self.game.piece_symbols) + 1

        self.tile = get_tile_model()
        self.full = get_full_img_model()

        # Positional embeddings added to tile features before concatenation
        self.tile_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_squares, TILE_EMBED_DIM)
        )
        nn.init.trunc_normal_(self.tile_pos_embed, std=0.02)

        # Projection from concatenated features to attention dimension
        self.embed_dim = 512
        self.pre_attn_dense = nn.Sequential(
            nn.Linear(TILE_EMBED_DIM + FULL_EMBED_DIM, self.embed_dim),
            nn.ReLU(),
        )

        # Transformer block: Attention → Add&Norm → FFN → Add&Norm
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(self.embed_dim)

        # Classification head
        self.dense = nn.Sequential(
            nn.Linear(self.embed_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.out_channels),
        )

    def forward(self, img):
        batch_size, ch, h, w = img.shape

        assert h == self.board_h
        assert w == self.board_w
        assert ch == 3

        x = img
        x = x.unfold(2, self.tile_size, self.tile_size)
        x = x.unfold(3, self.tile_size, self.tile_size)
        x = x.permute(0, 2, 3, 1, 4, 5)

        x = x.reshape(
            batch_size * self.game.board_rows * self.game.board_cols,
            ch,
            self.tile_size,
            self.tile_size,
        )

        x = self.tile(x)
        x = x.reshape(batch_size, self.num_squares, -1)

        # Add positional embeddings to tile features so each square has a unique identity
        x = x + self.tile_pos_embed

        z = self.full(img)
        z = z.reshape(batch_size, 1, -1)
        z = z.expand(-1, self.num_squares, -1)

        # Concatenate position-aware tile features with global image features
        x = torch.cat((x, z), dim=-1)

        # Project to attention dimension
        x = self.pre_attn_dense(x)

        # Transformer block
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        # Per-square classification
        x = x.reshape(batch_size * self.num_squares, -1)
        x = self.dense(x)
        x = x.reshape(batch_size, self.num_squares, self.out_channels)

        return x

if __name__ == "__main__":
    model = BoardRec(game="xiangqi")
    print("CUDA Available:", torch.cuda.is_available())
