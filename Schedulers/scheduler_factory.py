from torch.optim.lr_scheduler import (StepLR, CosineAnnealingLR, ReduceLROnPlateau)
from transformers import get_cosine_schedule_with_warmup


def get(config=None, optimiser=None, iter_per_epoch=None):
    name = config.scheduler.name
    epochs = config.train.epochs

    # ===========================================================================
    #                             Scheduler list
    # ===========================================================================

    if name == 'Step':
        scheduler = StepLR(
            optimiser,
            config.scheduler.params.steps,
            config.scheduler.params.lr_decay
        )
    elif name == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(
            optimiser,
            # T_max=epochs*iter_per_epoch
            T_max = config.scheduler.params.T_max,
            eta_min = config.scheduler.params.eta_min
        )
    elif name == 'CosineAnnealing-warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimiser,
            num_warmup_steps=iter_per_epoch * 5,
            num_training_steps=iter_per_epoch * epochs
        )
    elif name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimiser, 
            mode=config.scheduler.params.mode, 
            factor=config.scheduler.params.factor, 
            patience=config.scheduler.params.patience,
            threshold=config.scheduler.params.threshold,
            min_lr=0
        )
    else:
        scheduler = None

    print("[ Scheduler : {} ]".format(name))

    return scheduler
