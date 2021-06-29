from .resnet18 import ResNet18
from .wide_resnet50 import WideResNet50
from .vanilla_vae import VanillaVAE

backbone_models = {
    "resnet18": ResNet18,
    "wide_resnet50": WideResNet50,
    "vanilla_vae": VanillaVAE
}

backbone_kinds = {
    "resnet18": "pretrained_imagenet",
    "wide_resnet50": "pretrained_imagenet",
    "vanilla_vae": "vae"
}
