from os import path

from utils.paths import DATA_ROOT_DIR

# list of datasets
from Datasets.jh_2020_dataset import JH_2020_Dataset


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
    data_path = path.join(DATA_ROOT_DIR, data_name)

    # ===========================================================================
    #                                Dataset list
    # ===========================================================================

    if data_name == "jh_2020":
        dataset = JH_2020_Dataset(
            mode,
            data_path,
            transformer,
            fold
        )
    else:
        raise Exception("dataset not in list!")

    print("[ Dataset : {} <{}/{}> ]".format(
        data_name,
        mode,
        "raw" if fold is None else fold
    ))
    print("↳ [ Transformer : {} ]".format(
        str(transformer)
    ))

    return dataset
