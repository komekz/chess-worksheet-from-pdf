import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPPHead


class BoardQuad(nn.Module):
    def __init__(self):
        super(BoardQuad, self).__init__()

        self.model = lraspp_mobilenet_v3_large(weights="DEFAULT")
        self.model.classifier = LRASPPHead(40, 960, 1, 128)

    def forward(self, x):
        return self.model(x)["out"]  # [B, 1, H, W] raw logits


if __name__ == "__main__":
    model = BoardQuad()
    print(model)
