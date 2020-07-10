import torch
import pandas as pd
from skimage import io
from os import path
import numpy as np

from torch.utils.data import Dataset

from Datasets.utils import (fold_creator)

NUMBER_OF_FOLDS = 5
DATASET_NAME = 'bird_song_v2'
SHOULD_CACHE = True


class Bird_Song_v2_Dataset(Dataset):
    """Read audio files and cache them if no audio transformer has been passed
       else cache them as image files for use later

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, mode, data_path, transformer=None, fold_number=None):
        if transformer is None:
            raise Exception("transformer missing!")

        if fold_number is not None and fold_number >= NUMBER_OF_FOLDS:
            raise Exception("fold limit exceeded!")

        if mode == "TRA":
            self.mode = "train"
        elif mode == "VAL":
            self.mode = "val"
        elif mode == "PRD":
            self.mode = "test"
        self.transformer = transformer
        self.data_path = data_path

        if fold_number is None:
            # If fold not selected
            self.csv_path = path.join(data_path, self.mode + ".csv")
            self.image_dir = path.join(data_path, "images")
        else:
            # if fold selected
            self.create_folds()
            self.csv_path = path.join(
                "folds", DATASET_NAME, str(fold_number), self.mode + ".csv")
            self.image_dir = path.join(
                "folds", DATASET_NAME, str(fold_number), self.mode)

        self.data_frame = pd.read_csv(self.csv_path)

    def create_folds(self):
        fold_creator(
            self.data_path,
            path.join("folds", DATASET_NAME),
            NUMBER_OF_FOLDS
        )

    def get_csv_path(self):
        return self.csv_path

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # caching mechanism
        if SHOULD_CACHE:
            pass

        # image_name = str(self.data_frame.iloc[idx, 0]) + ".jpg"
        image_name = str(self.data_frame.iloc[idx, 0])
        image_path = path.join(self.image_dir, image_name)
        image = io.imread(image_path)

        if self.mode == "test":
            return self.transformer(image)
        else:
            label = torch.tensor(
                self.data_frame.iloc[idx, 1:].to_numpy(dtype=np.int64)
            ).item()
            return self.transformer(image), label
            
