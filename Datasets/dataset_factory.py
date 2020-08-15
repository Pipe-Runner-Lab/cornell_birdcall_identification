from os import path
from pathlib import Path

from utils.paths import DATA_ROOT_DIR

# list of datasets
from Datasets.bird_song import Bird_Song_Dataset


def get(config=None, mode=None, transformer=None):

    if mode == "TRA":
        data_config = config.train_data
    elif mode == "VAL":
        data_config = config.val_data
    elif mode == "PRD":
        data_config = config.pred_data
    else:
        raise Exception("Wrong mode passes!")

    data_name = data_config.name
    fold = data_config.params.fold
    data_path = Path(DATA_ROOT_DIR) / data_name

    # ===========================================================================
    #                                Dataset list
    # ===========================================================================

    if data_name == "bird_song":
        dataset = Bird_Song_Dataset(
            mode,
            data_path,
            transformer,
            fold,
            noise=True,
            mel_spec=True,
            multi_label=False
        )
    else:
        raise Exception("dataset not in list!")

    print("[ Dataset : {} <{}/{}> ]".format(
        data_name,
        mode,
        "raw" if fold is None else fold
    ))
    print("↳ [ Image Transformer : {} ]".format(
        str(transformer["image"])
    ))

    return dataset
