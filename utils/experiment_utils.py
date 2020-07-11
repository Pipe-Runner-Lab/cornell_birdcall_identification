import torch
from os import (makedirs, path, environ)
from shutil import rmtree
import pandas as pd
import math

from utils.paths import RESULTS_ROOT_DIR
from utils.regression_utils import covert_to_classification
import utils.metric as metric
from utils.print_util import cprint
from utils.wandb_update import (wandb_init, publish_intermediate)


class ExperimentHelper:
    def __init__(self, config, tb_writer=None):
        self.publish = config.publish
        self.session_name = config.session_name
        self.tb_writer = tb_writer
        self.freq = None if config.scheduler.name == "ReduceLROnPlateau" else config.val.freq
        self.pred_type = config.model.params.pred_type
        self.classes = config.classes

        self.best_scores = {
            "val/loss": float('inf'),
            "val/aroc": 0,
            "val/acc": 0
        }

        if config.checkpoint.type != "RSM":
            if path.exists(path.join(RESULTS_ROOT_DIR, self.session_name)) == False:
                makedirs(path.join(RESULTS_ROOT_DIR, self.session_name))
                cprint("[ Session : {} ]".format(
                    self.session_name), type="success")
            else:
                if config.dev:
                    cprint("[ <", self.session_name,
                           "> output exists - Overwriting! ]", type="warn")
                    rmtree(path.join(RESULTS_ROOT_DIR, self.session_name))
                    makedirs(path.join(RESULTS_ROOT_DIR, self.session_name))
                else:
                    cprint("[ <", self.session_name,
                           "> output exists - Manual deletion needed ]", type="warn")
                    exit()
        else:
            cprint("[ Session : {} ]".format(
                self.session_name), type="success")

        self.publish and wandb_init(config)

    def should_trigger(self, i):
        if self.freq:
            return (i + 1) % self.freq == 0
        return True

    def save_checkpoint(self, weights_dict):
        torch.save(
            weights_dict,
            path.join(RESULTS_ROOT_DIR, self.session_name, 'chkp_wt.pth')
        )

    def save_checkpoint_conditional(self, result_dict, weights_dict):
        # storing loss for check
        if self.best_scores["val/loss"] >= result_dict["val/loss"]:
            self.best_scores["val/loss"] = result_dict["val/loss"]
            torch.save(
                weights_dict,
                path.join(RESULTS_ROOT_DIR, self.session_name, 'loss_wt.pth')
            )

        # storing aroc for check
        if self.best_scores["val/aroc"] <= result_dict["val/aroc"]:
            self.best_scores["val/aroc"] = result_dict["val/aroc"]
            torch.save(
                weights_dict,
                path.join(RESULTS_ROOT_DIR, self.session_name, 'aroc_wt.pth')
            )

        # storing acc for check
        if self.best_scores["val/acc"] <= result_dict["val/acc"]:
            self.best_scores["val/acc"] = result_dict["val/acc"]
            torch.save(
                weights_dict,
                path.join(RESULTS_ROOT_DIR, self.session_name, 'acc_wt.pth')
            )

    def validate(self, train_log_dict, val_log_dict, epoch, weights_dict):
        result_dict = {}

        result_dict["train/lr"] = train_log_dict["lr"]

        train_output_list = train_log_dict["output_list"]
        train_target_list = train_log_dict["target_list"]
        result_dict["train/loss"] = train_log_dict["loss"]
        val_output_list = val_log_dict["output_list"]
        val_target_list = val_log_dict["target_list"]
        result_dict["val/loss"] = val_log_dict["loss"]

        if self.pred_type == 'REG' or self.pred_type == 'MIX':
            train_output_list = covert_to_classification(
                train_output_list,
                self.classes,
            )
            val_output_list = covert_to_classification(
                val_output_list,
                self.classes,
            )

        # generating accuracy scores
        result_dict["train/acc"] = metric.acc(
            train_output_list,
            train_target_list
        )
        result_dict["val/acc"] = metric.acc(
            val_output_list,
            val_target_list
        )

        # # generating aroc scores
        # result_dict["train/aroc"] = metric.aroc(
        #     train_output_list, train_target_list)
        # result_dict["val/aroc"] = metric.aroc(
        #     val_output_list, val_target_list)

        # # generate confusion matrix (validation only)
        # metric.confusion_matrix_generator(
        #     val_output_list,
        #     val_target_list,
        #     self.session_name
        # )

        # saving results to csv
        self.save_results(result_dict, epoch)

        # creating tensorboard events
        self.save_tb_event(result_dict, epoch)

        # check for progress and save
        self.save_checkpoint_conditional(result_dict, weights_dict)

        # publish intermediate results
        self.publish and self.publish_intermediate(
            result_dict,
            epoch,
            val_output_list,
            val_target_list
        )

        return result_dict

    def save_results(self, result_dict, epoch):
        header = ["epoch"]
        values = [epoch]

        for key, value in result_dict.items():
            header.append(key)
            values.append(value)

        df = pd.DataFrame(
            [values])
        result_path = path.join(
            RESULTS_ROOT_DIR, self.session_name, 'result.csv')

        if not path.isfile(result_path):
            df.to_csv(
                result_path,
                header=header,
                index=False
            )
        else:
            df.to_csv(result_path, mode='a', header=False, index=False)

    def save_tb_event(self, result_dict, epoch):
        for key, value in result_dict.items():
            self.tb_writer.add_scalar(
                key,
                value,
                epoch
            )

    def publish_intermediate(self, result_dict, epoch, output_list, target_list):
        # wandb
        publish_intermediate(
            result_dict,
            epoch,
            self.best_scores,
            output_list,
            target_list
        )
