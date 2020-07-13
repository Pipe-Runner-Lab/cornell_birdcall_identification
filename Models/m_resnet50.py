import torch
import torchvision.models as models
from pathlib import Path
from torch import nn

from utils.paths import WEIGHTS_DIR
from Models.utils import get_default_fc

full_wt_path = Path(WEIGHTS_DIR) / "m_resnet_50.pth"

def m_resnet50(adjusted_classes, config):
    # only for loading
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    
    weights_dict = torch.load(full_wt_path)
    model.load_state_dict(weights_dict["model"])

    model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

    return model