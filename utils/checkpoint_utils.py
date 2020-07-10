import torch
from os import path

from utils.paths import RESULTS_ROOT_DIR
from utils.print_util import cprint


def load(config, model, optimiser=None, scheduler=None):
    if config.checkpoint.type == "RSM":
        full_wt_path = path.join(
            RESULTS_ROOT_DIR,
            config.session_name,
            config.checkpoint.params.path
        )

        if path.exists(full_wt_path):
            weights_dict = torch.load(full_wt_path)

            # load model weight
            model.load_state_dict(
                weights_dict["model"]
                # ,map_location={'cuda:1': 'cuda:0'}
            )

            if optimiser:
                # load oprimiser weight
                optimiser.load_state_dict(weights_dict["optimiser"])

            if scheduler:
                # load scheduler weight
                scheduler.load_state_dict(weights_dict["scheduler"])
        else:
            raise Exception("{} missing!".format(full_wt_path))

        print("[ Checkpoint : {} ({}/{}) ]".format(
            full_wt_path,
            config.checkpoint.type,
            weights_dict["epoch"]
        ))

        return weights_dict["epoch"]
    elif config.checkpoint.type == "PRG":
        if config.checkpoint.params.path.split("/")[0] == config.session_name:
            raise Exception(
                "Progressive training cannot be done on same session")

        full_wt_path = path.join(
            RESULTS_ROOT_DIR,
            config.checkpoint.params.path
        )

        if path.exists(full_wt_path):
            weights_dict = torch.load(full_wt_path)

            # load model weight
            model.load_state_dict(weights_dict["model"])
        else:
            raise Exception("{} missing!".format(full_wt_path))

        print("[ Checkpoint : {} ({}/{}) ]".format(
            full_wt_path,
            config.checkpoint.type,
            weights_dict["epoch"]
        ))

        return 0
    elif config.checkpoint.type == "PRD":
        full_wt_path = path.join(
            RESULTS_ROOT_DIR,
            config.session_name,
            config.checkpoint.params.path
        )

        if path.exists(full_wt_path):
            weights_dict = torch.load(full_wt_path)

            # load model weight
            model.load_state_dict(weights_dict["model"])
        else:
            raise Exception("{} missing!".format(full_wt_path))

        print("[ Checkpoint : {} ({}/{}) ]".format(
            full_wt_path,
            config.checkpoint.type,
            weights_dict["epoch"]
        ))

        return 0
    else:
        return 0
