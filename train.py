import torch
from os import path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.seed_backend import seed_all
from utils.custom_bar import CustomBar
from utils.check_gpu import get_training_device
from utils.experiment_utils import ExperimentHelper
import utils.checkpoint_utils as checkpoint
from utils.paths import TB_DIR

from Models import model_factory
from Optimisers import optimiser_factory
from Losses import loss_factory
from Schedulers import scheduler_factory
from Transformers import transformer_factory
from Datasets import dataset_factory


def predict_all(model, input):
    output = model.forward(input)
    return output


def _validate_single_ep(config, dataloader, model, criterion, device):
    # set model to evaluation mode
    model.eval()

    log_dict = {
        "loss": 0,
        "output_list": [],
        "target_list": []
    }

    for _, sample in enumerate(dataloader):
        with torch.no_grad():
            input, target = sample
            input = input.to(device)
            target = target.to(device)

            # forward
            output = predict_all(model, input)

            # loss calculation
            loss = criterion(
                output,
                target
            )

            log_dict["loss"] += (loss.item() * input.shape[0])
            log_dict["output_list"].append(output.detach().cpu())
            log_dict["target_list"].append(target.cpu())

    return log_dict


def _train_single_ep(config, dataloader, model, optimiser, criterion, device, progress_bar, is_val):
    # set model to training mode
    model.train()

    log_dict = {
        "loss": 0,
        "output_list": [],
        "target_list": []
    }

    # flush accumulators before training starts
    optimiser.zero_grad()

    for batch_ndx, sample in enumerate(dataloader):
        # progress bar update
        progress_bar.update_batch_info(batch_ndx)

        input, target = sample
        input = input.to(device)
        target = target.to(device)

        # forward pass
        # TODO: add space for aux logit
        output = predict_all(model, input)

        # loss calculation
        # TODO: add space for auxilary loss
        loss = criterion(
            output,
            target
        )

        # backward pass
        loss.backward()

        # log training values
        if is_val:
            log_dict["loss"] += (loss.item() * input.shape[0])
            log_dict["output_list"].append(output.detach().cpu())
            log_dict["target_list"].append(target.cpu())

        # ==================================================
        #        update (may lead to loss tampering)
        # ==================================================
        if config.train.grad_accum:
            if (batch_ndx + 1) % config.train.grad_accum == 0:
                # average accumulated gradient
                loss /= config.train.grad_accum
                
                # update and flush
                optimiser.step()
                optimiser.zero_grad()
        else:
            # update and flush
            optimiser.step()
            optimiser.zero_grad()

        # progress bar update
        progress_bar.step()

    return log_dict


def _train(config, dataloaders, model, optimiser, scheduler, criterion, exp_helper, device, start_epoch=0):
    epochs = config.train.epochs
    batch = config.train.batch
    train_size = len(dataloaders["TRA"].dataset)
    val_size = len(dataloaders["VAL"].dataset)

    with CustomBar(start_epoch, epochs, len(dataloaders["TRA"].dataset), batch) as progress_bar:
        for i in range(start_epoch, epochs):
            is_val = exp_helper.should_trigger(i)

            # progress bar update
            progress_bar.update_epoch_info(i)

            # train single
            train_log_dict = _train_single_ep(
                config,
                dataloaders["TRA"],
                model,
                optimiser,
                criterion,
                device,
                progress_bar,
                is_val
            )

            if is_val:
                # validate single
                val_log_dict = _validate_single_ep(
                    config,
                    dataloaders["VAL"],
                    model,
                    criterion,
                    device
                )

                # post process dict
                train_log_dict["output_list"] = torch.cat(
                    train_log_dict["output_list"], dim=0)
                train_log_dict["target_list"] = torch.cat(
                    train_log_dict["target_list"], dim=0)
                train_log_dict["loss"] /= train_size
                val_log_dict["output_list"] = torch.cat(
                    val_log_dict["output_list"], dim=0)
                val_log_dict["target_list"] = torch.cat(
                    val_log_dict["target_list"], dim=0)
                val_log_dict["loss"] /= val_size

                # adding lr to logs
                train_log_dict["lr"] = optimiser.param_groups[0]['lr']

                # validate and save checkpoint internally
                result_dict = exp_helper.validate(
                    train_log_dict,
                    val_log_dict,
                    i + 1,
                    weights_dict={
                        'model': model.state_dict(),
                        'optimiser': optimiser.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'epoch': i + 1
                    }
                )

            if scheduler:
                if config.scheduler.name == 'ReduceLROnPlateau':
                    scheduler.step(result_dict["val/acc"])
                else:
                    scheduler.step()

            # Manually save training state
            exp_helper.save_checkpoint(
                weights_dict={
                    'model': model.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'epoch': i + 1
                }
            )


def run(config):
    """Generates pipeline dependency instances by
       calling factory methods and feed them to 
       train

    Args:
        config (edict): parsed cofig dict corresponding
                        to training yml

    Returns:
        [type]: [description]
    """

    writer = SummaryWriter(
        log_dir=path.join(
            TB_DIR,
            config.session_name
        )
    )

    exp_helper = ExperimentHelper(
        config,
        tb_writer=writer
    )

    seed_all(
        seed=config.seed
    )

    device = get_training_device()

    transformers = {
        "TRA": transformer_factory.get(config.train_data.params),
        "VAL": transformer_factory.get(config.val_data.params)
    }

    datasets = {
        mode: dataset_factory.get(
            config=config, mode=mode, transformer=transformers[mode])
        for mode in ["TRA", "VAL"]
    }

    dataloaders = {
        "TRA": DataLoader(
            datasets["TRA"],
            batch_size=config.train.batch,
            num_workers=config.train.workers,
            pin_memory=True,
            shuffle=True
        ),
        "VAL": DataLoader(
            datasets["VAL"],
            batch_size=config.val.batch,
            num_workers=config.val.workers,
            pin_memory=True,
            shuffle=False
        )
    }

    model = model_factory.get(
        config=config
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimiser = optimiser_factory.get(
        config=config,
        model_params=model.parameters(),
    )

    scheduler = scheduler_factory.get(
        config=config,
        optimiser=optimiser,
        iter_per_epoch=len(dataloaders["TRA"].dataset)/config.train.batch
    )

    criterion = loss_factory.get(
        config=config
    )

    start_epoch = checkpoint.load(
        config,
        model,
        optimiser,
        scheduler
    )

    _train(
        config,
        dataloaders,
        model,
        optimiser,
        scheduler,
        criterion,
        exp_helper,
        device,
        start_epoch
    )

    return exp_helper.best_scores
