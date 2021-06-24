from .resnet18 import ResNet18
from .wide_resnet50 import WideResNet50

backbone_models = {
    "resnet18": ResNet18,
    "wide_resnet50": WideResNet50
}
