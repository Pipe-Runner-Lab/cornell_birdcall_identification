import torch
from torch import nn
from os import path
import pretrainedmodels
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

from Models.m_resnet50 import m_resnet50
from Models.utils import get_default_fc
from Models.layer_utils import GeM

def get(config=None):
    name = config.model.name
    classes = config.classes
    pred_type = config.model.params.pred_type
    tune_type = config.model.params.tune_type

    adjusted_classes = classes
    if pred_type == 'REG':
        adjusted_classes = 1
    elif pred_type == 'MIX':
        adjusted_classes = classes + 1

    # ===========================================================================
    #                                 Model list
    # ===========================================================================

    if name == 'densenet161':
        model = models.densenet161(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'densenet201':
        model = models.densenet201(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'resnet101':
        model = models.resnet101(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'resnet152':
        model = models.resnet152(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'wide_resnet101_2':
        model = models.wide_resnet101_2(pretrained=True)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model._fc.in_features
        model._fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'efficientnet-b5':
        model = EfficientNet.from_pretrained('efficientnet-b5')
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model._fc.in_features
        model._fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    elif name == 'm-resnet50':
        model = m_resnet50(adjusted_classes, config)
        if tune_type == 'FE':
            for param in model.parameters():
                param.requires_grad = False

        model.avgpool = GeM()

        num_ftrs = model.fc.in_features
        model.fc = get_default_fc(num_ftrs, adjusted_classes, config.model.params)
    else:
        raise Exception("model not in list!")

    print("[ Model : {} ]".format(name))
    print("↳ [ Prediction type : {} ]".format(pred_type))
    print("↳ [ Adjusted classes : {} ]".format(adjusted_classes))
    if config.mode != "PRD":
        print("↳ [ Tuning type : {} ]".format(tune_type))
    return model
