import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.seed_backend import seed_all
from utils.prediction_utils import (PredictionHelper, vote)
from utils.check_gpu import get_training_device
import utils.checkpoint_utils as checkpoint
from utils.print_util import cprint

from Transformers import transformer_factory
from Datasets import dataset_factory
from Models import model_factory


def predict_all(model, input):
    output = model.forward(input)
    return output


def _predict_single_ep(config, dataloader, model, device):
    # set model to evaluation mode
    model.eval()

    log_dict = {
        "output_list": [],
    }

    for _, sample in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            input = sample
            input = input.to(device)

            # forward
            output = predict_all(model, input)

            log_dict["output_list"].append(output.detach().cpu())

    return log_dict


def _predict(config, dataloader, model, pred_helper, device):

    if config.pred.tta:
        cprint("[ TTA : {} ]".format(config.pred.tta), type="success")

    pred_output_list = []

    for tta_idx in range(config.pred.tta if config.pred.tta else 1):
        if config.pred.tta:
            cprint("↳ [ TTA idx : {}/{} ]".format(tta_idx +
                                                  1, config.pred.tta), type="info1")

        predict_log_dict = _predict_single_ep(
            config, dataloader, model, device)

        predict_log_dict["output_list"] = torch.cat(
            predict_log_dict["output_list"],
            dim=0
        )
        pred_output_list.append(
            predict_log_dict["output_list"]
        )

    pred_output_list = torch.stack(pred_output_list, dim=2)

    pred_helper.post_process(
        config,
        pred_output_list,
        dataloader.dataset.get_csv_path()
    )


def run(config):
    pred_helper = PredictionHelper(config)

    seed_all(
        seed=config.seed
    )

    device = get_training_device()

    for config_dict in config.session_list:
        aux_config = config_dict.session.aux_config

        cprint("↳ [ Aux Session : {} ]".format(
            aux_config.session_name), type="info2")

        transformer = transformer_factory.get(aux_config.pred_data.params)

        dataset = dataset_factory.get(
            config=config, mode="PRD", transformer=transformer)

        dataloader = DataLoader(
            dataset,
            batch_size=aux_config.pred.batch,
            num_workers=aux_config.pred.workers,
            pin_memory=True,
            shuffle=False
        )

        model = model_factory.get(
            config=aux_config
        )
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        checkpoint.load(
            aux_config,
            model
        )

        if "runs" in aux_config.pred.keys():
            cprint("[ Runs : {} ]".format(
                aux_config.pred.runs), type="success")

        for run_idx in range(aux_config.pred.runs if "runs" in aux_config.pred.keys() else 1):

            if "runs" in aux_config.pred.keys():
                cprint("↳ [ Run idx : {}/{} ]".format(run_idx +
                                                      1, aux_config.pred.runs), type="info1")

            _predict(aux_config, dataloader, model, pred_helper, device)

    if config.vote:
        cprint("[ Voting ]", type="success")
        vote(config)
