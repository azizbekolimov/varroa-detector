import torch
from torch import nn
from torchvision import models


def build_model(num_classes=2, unfreeze_last_n=0):
    """Return MobileNetV2 with a new head and optionally unfrozen last blocks.

    build_model() -> 2,562 trainable params (head only)
    build_model(unfreeze_last_n=1) -> head + last feature block trainable
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        for param in model.features[-unfreeze_last_n:].parameters():
            param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    return model


def main():
    for n in [0, 1, 2]:
        model = build_model(unfreeze_last_n=n)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tensors = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"unfreeze_last_n={n}: {trainable:,} params in {tensors} tensors")


if __name__ == "__main__":
    main()