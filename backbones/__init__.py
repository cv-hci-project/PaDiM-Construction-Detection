from .resnet18 import ResNet18
from .wide_resnet50 import WideResNet50
from .efficientnet import EffNet

backbone_models = {
    "resnet18": ResNet18,
    "wide_resnet50": WideResNet50,
    "efficientnet": EffNet
}
