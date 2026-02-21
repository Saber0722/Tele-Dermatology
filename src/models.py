import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet34_backbone(pretrained=True):
    model = models.resnet34(
        weights="IMAGENET1K_V1" if pretrained else None
    )
    layers = list(model.children())[:-2]
    return nn.Sequential(*layers)


class SegmentationModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = get_resnet34_backbone(pretrained)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        mask = self.decoder(features)
        return mask