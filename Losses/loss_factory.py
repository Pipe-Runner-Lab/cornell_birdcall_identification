from torch.nn import (CrossEntropyLoss, NLLLoss, MSELoss, BCEWithLogitsLoss)
from Losses.focal_loss import FocalLoss
from Losses.arcface_loss import ArcfaceLoss
from Losses.utils import (ClassificationLossWrapper,
                          RegressionLossWrapper, MixedLossWrapper)


def _get_pure(function_name, params=None):
    # ===========================================================================
    #                                Loss list
    # ===========================================================================
    
    loss_function = None
    
    if function_name == 'focal':
        if params:
            loss_function = FocalLoss(
                size_average=params['size_average']
            )
        else:
            loss_function = FocalLoss()

    elif function_name == 'cross-entropy':
        loss_function = CrossEntropyLoss(reduction="mean")

    elif function_name == 'negative-log-likelihood':
        loss_function = NLLLoss()

    elif function_name == 'mean-squared-error':
        loss_function = MSELoss()

    elif function_name == 'arcface':
        loss_function = ArcfaceLoss()

    elif function_name == 'binary-cross-entropy':
        loss_function = BCEWithLogitsLoss(reduction="mean")

    else:
        raise Exception("loss function not in list!")

    print("[ Loss : {} ]".format(function_name))

    return loss_function

def _validate_config(config):
    if not isinstance(config.loss.name, str):
        raise Exception("loss name invalid!")

def get(config=None):
    _validate_config(config)

    name = config.loss.name
    pred_type = config.model.params.pred_type

    if pred_type == 'REG':
        wrapped_loss_function = RegressionLossWrapper(
            _get_pure(name, config.loss.params)
        )
    elif pred_type == 'CLS':
        wrapped_loss_function = ClassificationLossWrapper(
            _get_pure(name, config.loss.params)
        )
    elif pred_type == 'MIX':
        wrapped_loss_function = MixedLossWrapper(
            _get_pure(
                name,
                config.loss.params
            ),
            _get_pure(
                config.loss.params.classification_loss,
                config.loss.params
            ),
            config.loss.params.classification_coefficient
        )

    return wrapped_loss_function
