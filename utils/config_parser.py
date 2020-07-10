import yaml
import copy 
from os import path
from easydict import EasyDict as edict

from utils.paths import (CONFIG_DIR)
from utils.print_util import cprint

def _hydrate_aux(config):
    for config_dict in config.session_list:
        orig_config = load(config_dict.session.path)
        session_data = config_dict.session
        cfg = edict(orig_config)

        # ==============================================================
        #                Add auxilary conditions here
        # ==============================================================

        # if no params passes
        if "params" not in session_data.keys():
            session_data.params = edict()

        # basics
        cfg.mode = "PRD"

        # pred
        cfg.pred = edict(config.pred)
        if "tta" in session_data.params.keys():
            cfg.pred.tta = session_data.params.tta
        if "runs" in session_data.params.keys():
            cfg.pred.runs = session_data.params.runs
        if "batch" in session_data.params.keys():
            cfg.pred.batch = session_data.params.batch

        # pred data
        cfg.pred_data = edict(config.pred_data)
        # data_params
        if "resize" in session_data.params.keys():
            if session_data.params.resize == "ORG":
                cfg.pred_data.params.resize = cfg.train_data.params.resize
            else:
                cfg.pred_data.params.resize = session_data.params.resize
        if "transformer" in session_data.params.keys():
            if session_data.params.transformer == "ORG":
                cfg.pred_data.params.transformer = cfg.train_data.params.transformer
            else:
                cfg.pred_data.params.transformer = session_data.params.transformer

        # checkpoint
        cfg.checkpoint = edict()
        cfg.checkpoint.type = "PRD"
        # checkpoint.params
        cfg.checkpoint.params = edict()
        cfg.checkpoint.params.path = session_data.wt_path

        config_dict.session.aux_config = copy.deepcopy(cfg)

    return config

def _verify_config(config):
    if config.mode == "TRA":
        pass
    elif config.mode == "PRD":
        pass


def _get_default_config(mode):
    config = edict()

    # common
    config.seed = 10
    config.classes = 4

    if mode == "TRA":
        # train
        config.train = edict()
        config.train.epochs = 5
        config.train.batch = 2
        config.train.workers = 2
        config.train.grad_accum = None

        # val
        config.val = edict()
        config.val.freq = None
        config.val.batch = 10
        config.val.workers = 2

        # train_data
        config.train_data = edict()
        config.train_data.name = None
        # train_data.params
        config.train_data.params = edict()
        config.train_data.params.fold = None
        config.train_data.params.transformer = "DFLT"
        config.train_data.params.resize = None

        # val_data
        config.val_data = edict()
        config.val_data.name = None
        # val_data.params
        config.val_data.params = edict()
        config.train_data.params.fold = None
        config.train_data.params.transformer = "DFLT"

        # checkpoint
        config.checkpoint = edict()
        config.checkpoint.name = None
        # checkpoint.params
        config.checkpoint.type = "RSM"

        # model
        config.model = edict()
        config.model.name = None
        # model.params
        config.model.params = edict()
        config.model.params.pred_type = "CLS"
        config.model.params.tune_type = "FT"

        # optimiser
        config.optimiser = edict()
        config.optimiser.name = None
        # optimiser.params
        config.optimiser.params = edict()
        config.optimiser.params.lr = 0.01

        # scheduler
        config.scheduler = edict()
        config.scheduler.name = None
        # scheduler.params
        config.scheduler.params = edict()

        # loss
        config.loss = edict()
        config.loss.name = None
        # loss.params
        config.loss.params = edict()

        # checkpoint
        config.checkpoint = edict()
        config.checkpoint.type = None
        # checkpoint.params
        config.checkpoint.params = edict()
        config.checkpoint.params.path = 'chkp_wt.pth'
    
    elif mode == "PRD":
        # pred
        config.pred = edict()
        config.pred.batch = 10
        config.pred.workers = 2
        config.pred.ensemble = False
        config.pred.tta = None
        
        # pred_data
        config.pred_data = edict()
        config.pred_data.name = None
        # pred_data.params
        config.pred_data.params = edict()
        config.pred_data.params.fold = None
        config.pred_data.params.transformer = "DFLT"
        config.pred_data.params.resize = None

        # session_list
        config.session_list = []
    else:
        raise Exception("[mode] missing (TRA/PRD)")

    return config


def _get_yml_config(config_path):

    # adding file name as attribute
    session_name = config_path.split('.')[0]

    config_path = path.join(CONFIG_DIR, config_path)

    with open(config_path, 'r') as fid:
        yml_config = edict(yaml.safe_load(fid))

    yml_config.session_name = session_name

    return yml_config


def _merge(yml_config, default_config):
    """Merges yml config and default config into one tree

    Args:
        yml_config (edict): yml config sub tree
        default_config (edict): yml config sub tree

    Returns:
        edict: merged config
    """

    if not isinstance(yml_config, edict):
        return

    for key, value in yml_config.items():
        if isinstance(value, edict):
            _merge(yml_config[key], default_config[key])
        else:
            default_config[key] = value

    return default_config


def load(config_path):
    """Generates a config tree based on yml path

    Args:
        config_path (string): path of the yml file to be parsed

    Returns:
        edict: hydratd config tree
    """

    yml_config = _get_yml_config(config_path)
    default_config = _get_default_config(yml_config.mode)

    config = _merge(yml_config, default_config)

    if config.mode == "PRD":
        config = _hydrate_aux(config)

    _verify_config(config)

    return config
