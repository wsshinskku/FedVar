from torch import nn


def build_model(name, channels, classes, image_size):
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * image_size * image_size, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(channels, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )
    if name not in ("tinynet_a", "ghostnet_100", "mobilenetv3_small_100"):
        raise ValueError("Unknown model")
    try:
        import timm
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Mobile architectures require pip install -e '.[mobile]'") from exc
    return timm.create_model(name, pretrained=False, in_chans=channels, num_classes=classes)
