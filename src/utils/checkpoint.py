from logging import getLogger
from typing import Dict

import torch.nn as nn
from torch import Tensor

logger = getLogger(__name__)


def get_weights_to_load(model: nn.Module, ckpt: Dict[str, Tensor]) -> Dict[str, Tensor]:
    model_dict = model.state_dict()
    for ckpt_key, ckpt_weight in ckpt.items():
        if ckpt_key not in model_dict:
            logger.info(
                f"  skip load weight: {ckpt_key} "
                f"(not defined in {model.__class__.__name__})"
            )
        else:
            if ckpt_weight.size() != model_dict[ckpt_key].size():
                logger.info(
                    f"  skip load weight: {ckpt_key} "
                    f"({ckpt_weight.size()} in ckpt != "
                    f"{model_dict[ckpt_key].size()} in {model.__class__.__name__})"
                )
            else:
                model_dict[ckpt_key] = ckpt_weight
    return model_dict
