import torch
from torch.nn.functional import softmax
from os import (makedirs, path, listdir)
from shutil import rmtree
import pandas as pd
import numpy as np
import time

from utils.paths import RESULTS_ROOT_DIR
from utils.regression_utils import covert_to_classification
from utils.print_util import cprint
from utils.metric import (post_process_output, onehot_converter)
from utils.submission_utils import output_header


def vote(config):
    csv_list = listdir(path.join(RESULTS_ROOT_DIR, config.session_name))
    result_path = path.join(
        RESULTS_ROOT_DIR,
        config.session_name,
        "voted.csv"
    )

    one_hot = None
    header_np = None
    for csv_file in csv_list:
        csv_path = path.join(RESULTS_ROOT_DIR, config.session_name, csv_file)
        df = pd.read_csv(csv_path)

        if one_hot is not None:
            one_hot += onehot_converter(
                torch.tensor(df.iloc[:, 1:].to_numpy()),
                config.classes
            )
        else:
            one_hot = onehot_converter(
                torch.tensor(df.iloc[:, 1:].to_numpy()),
                config.classes
            )
            header_np = df.iloc[:, 0].to_numpy()

    output = torch.argmax(one_hot, dim=1).view(-1, 1).numpy()
    header_np = header_np.reshape(-1, 1)

    df = pd.DataFrame(
        np.hstack(
            (
                header_np,
                output
            )
        )
    )

    df.to_csv(
        result_path,
        header=output_header,
        index=False
    )


class PredictionHelper:
    def __init__(self, config):
        self.session_name = config.session_name

        if path.exists(path.join(RESULTS_ROOT_DIR, self.session_name)) == False:
            makedirs(path.join(RESULTS_ROOT_DIR, self.session_name))
            cprint("[ Session : {} ]".format(
                self.session_name
            ), type="success")
        else:
            if config.dev:
                cprint("[ <", self.session_name,
                       "> output exists - Overwriting! ]", type="warn")
                rmtree(path.join(RESULTS_ROOT_DIR, self.session_name))
                makedirs(path.join(RESULTS_ROOT_DIR, self.session_name))
            else:
                if config.vote:
                    cprint(
                        "[ < {} > output exists - Voting ]".format(self.session_name), type="success")
                    vote(config)
                else:
                    cprint("[ <", self.session_name,
                           "> output exists - Manual deletion needed ]", type="warn")
                exit()

        self.is_ensemble = config.pred.ensemble
        self.ensemble_list = []

    def post_process(self, aux_config, pred_output_list, test_csv_path):
        pred_type = aux_config.model.params.pred_type
        classes = aux_config.classes
        aux_session_name = aux_config.session_name

        if pred_type == 'CLS':
            pred_output_list = post_process_output(pred_output_list)
        elif pred_type == 'REG' or pred_type == 'MIX':
            pred_output_list = covert_to_classification(
                pred_output_list,
                classes,
            )

        # averaging TTA output if any
        pred_output = torch.mean(pred_output_list, dim=2)

        # problem specific: argmax to convert to single class output
        pred_output = torch.argmax(pred_output, dim=1).view(-1, 1)

        adjusted_path = aux_session_name + "_" + str(int(time.time())) + ".csv"

        result_path = path.join(
            RESULTS_ROOT_DIR,
            self.session_name,
            adjusted_path
        )

        cprint("↳ [ Path : {} ]".format(result_path), type="warn")

        test_df = pd.read_csv(test_csv_path)

        df = pd.DataFrame(
            np.hstack(
                (
                    test_df.to_numpy(),
                    pred_output.numpy()
                )
            )
        )

        df.to_csv(
            result_path,
            header=output_header,
            index=False
        )

        # self.data_frame = pd.read_csv(self.csv_path)

    # def ensemble(self, test_csv_path, type = "softmax"):
    #     cprint("[ Ensembling results ]", type="success")

    #     self.ensemble_list = torch.stack(self.ensemble_list, dim=2)

    #     if self.ensemble_list.size()[2] < 3:
    #         cprint("[ Too few experiments for ensembling ]", type="warn")
    #         exit()
    #     elif self.ensemble_list.size()[2] % 2 == 0:
    #         cprint("[ Ensemble logic needs odd majority < ",
    #                str(self.ensemble_list.size()[2]), " > ]", type="warn")
    #         exit()

    #     # Voting logic
    #     if type == "softmax":
    #         self.results = torch.sum(self.ensemble_list, dim=2)
    #         self.results = torch.softmax(self.results, dim=1)
    #     elif type == "thresholding":
    #         self.results = torch.mean(self.ensemble_list, dim=2)
    #         OHV_target = torch.zeros(self.results.size()).to(
    #             self.results.get_device())
    #         OHV_target[range(self.results.size()[0]),
    #                    torch.argmax(self.results, dim=1)] = 1
    #         self.results = OHV_target
    #     elif type == "mean":
    #         self.results = torch.mean(self.ensemble_list, dim=2)

    #     result_path = path.join(
    #         'results', self.experiment_name, 'ensembled.csv')
    #     ensemble_df = pd.read_csv(test_csv_path)

    #     with torch.no_grad():
    #         # saving results to csv
    #         df = pd.DataFrame(
    #             np.hstack(
    #                 (
    #                     ensemble_df.to_numpy(),
    #                     self.results.cpu().numpy()
    #                 )
    #             )
    #         )

    #         df.to_csv(
    #             result_path,
    #             header=kaggle_output_header,
    #             index=False
    #         )
