"""Model adapter copied next to the pinned upstream scripts at runtime."""
from functools import reduce

import torch
from torchvision import models

# Use explicit weight enums; DEFAULT may silently change between versions.
MODEL_SPECS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, ("fc",)),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, ("classifier",)),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, ("classifier", "1")),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, ("classifier", "2")),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, ("heads", "head")),
}


def make_model(architecture, n_classes, pretrained):
    build, weights, path = MODEL_SPECS[architecture]
    model = build(weights=weights if pretrained else None)
    parent = reduce(getattr, path[:-1], model)
    previous = getattr(parent, path[-1])
    head = torch.nn.Linear(previous.in_features, n_classes)
    # torchvision's ViT initializes its replacement head to zero before training.
    if architecture == "vit_b_16":
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    setattr(parent, path[-1], head)
    return model


def get_embedding(model, images, architecture):
    head = reduce(getattr, MODEL_SPECS[architecture][2], model)
    features = []
    hook = head.register_forward_pre_hook(lambda _module, args: features.append(args[0]))
    try:
        model(images)
    finally:
        hook.remove()
    return features[0]
