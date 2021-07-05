from .padim_base import PaDiMBase
from .padim import PaDiM
from .padim_shared import PaDiMShared
from .padim_per_category import PaDiMPerCategory

registered_padim_models = {
    "vanilla": PaDiM,
    "shared": PaDiMShared,
    "per_category": PaDiMPerCategory
}
