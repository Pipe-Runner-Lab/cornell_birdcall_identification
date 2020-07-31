import wandb
import torch

from utils.print_util import cprint
from utils.paths import CONFIG_DIR


def wandb_init(config):
    cprint("[ Logging started on W&B https://app.wandb.ai/humble_d/cornell-birdcall-identification ]", type="info3")

    wandb.init(project="cornell-birdcall-identification")

    wandb.config.session = config.session_name
    wandb.config.seed = config.seed
    wandb.config.model = config.model.name
    wandb.config.pred_type = config.model.params.pred_type
    wandb.config.optim = config.optimiser.name
    wandb.config.lr = config.optimiser.params.lr
    wandb.config.loss = config.loss.name
    wandb.config.resize_dims = config.train_data.params.resize
    wandb.config.epochs = config.train.epochs
    wandb.config.batch = config.train.batch

    # saving config files to W&B
    wandb.save('./{}/{}.yml'.format(CONFIG_DIR, config.session_name))
    return True


def publish_intermediate(result_dict, epoch, best_scores, output_list, target_list):
    wandb.run.summary["b/val/loss"] = best_scores["val/loss"]
    wandb.run.summary["b/val/acc"] = best_scores["val/acc"]
    wandb.run.summary["b/val/f1"] = best_scores["val/f1"]

    # saving confusion matrix (image)
    # wandb.sklearn.plot_confusion_matrix(
    #     torch.argmax(target_list, dim=1).numpy(),
    #     torch.argmax(output_list, dim=1).numpy(),
    #     ['H', 'MD', 'R', 'S']
    # )

    result_dict["epoch"] = epoch

    return wandb.log(result_dict)
