from torch.optim import (RMSprop, Adam, AdamW)


def _validate_config(config):
    if not isinstance(config.model.name, str):
        raise Exception("optimiser name invalid!")


def get(config=None, params=None, model_params=None):
    _validate_config(config)

    name = config.optimiser.name
    lr = config.optimiser.params.lr

    # ===========================================================================
    #                             Optimiser list
    # ===========================================================================

    if name == 'RMSprop':
        optimiser = RMSprop(
            model_params,
            lr=lr
        )
    elif name == 'Adam':
        optimiser = Adam(
            model_params,
            lr=lr
        )
    elif name == 'AdamW':
        optimiser = AdamW(
            model_params,
            lr=lr,
            weight_decay=config.optimiser.params.weight_decay
        )
    else:
        raise Exception("optimiser not in list!")

    print("[ Optimiser : {} / Accum: {} ]".format(name, config.train.grad_accum))

    return optimiser
