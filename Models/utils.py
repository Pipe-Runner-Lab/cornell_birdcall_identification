from torch import nn

def get_default_fc(num_ftrs,adjusted_classes, params):
    return nn.Sequential(
        nn.Linear(num_ftrs, 1024, bias=True),
        nn.ReLU(),
        nn.Dropout(p=params.fc_drop_out_0),
        nn.Linear(1024, 1024, bias=True),
        nn.ReLU(),
        nn.Dropout(p=params.fc_drop_out_1),
        nn.Linear(1024, adjusted_classes, bias=True)
    )
