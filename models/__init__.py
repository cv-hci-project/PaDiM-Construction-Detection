from .padim_base import PaDiMBase
from .padim import PaDiM
from .padim_shared import PaDiMShared

registered_padim_models = {
    "vanilla": PaDiM,
    "shared": PaDiMShared
}
