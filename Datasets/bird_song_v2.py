import torch
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path

from torch.utils.data import Dataset

from Datasets.utils import (fold_creator)
from utils.submission_utils import BIRD_CODE

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
            self.csv_path = Path(data_path) / self.mode + ".csv"
            self.data_dir = Path(data_path) / self.mode
        else:
            # if fold selected
            self.create_folds()
            self.csv_path = Path("folds") / DATASET_NAME / \
                str(fold_number) / self.mode + ".csv"
            self.data_dir = Path("folds") / DATASET_NAME / \
                str(fold_number) / self.mode

        self.data_frame = pd.read_csv(self.csv_path)

        # create auxilary data directory if caching possible
        if SHOULD_CACHE and self.transformer['audio'] is None:
            self.aux_data_dir = Path(data_path) / "spectrograms"
            self.aux_data_dir.mkdir(parents=True, exist_ok=True)

    def create_folds(self):
        fold_creator(
            self.data_path,
            Path("folds") / DATASET_NAME,
            NUMBER_OF_FOLDS
        )

    def get_csv_path(self):
        return self.csv_path

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get basic data path
        ebird_code = self.data_frame["ebird_code"][idx]
        filename = self.data_frame["filename"][idx]
        
        # generate file path
        audio_filepath = self.data_dir / ebird_code / filename

        # caching mechanism
        if SHOULD_CACHE and self.transformer['audio'] is None:
            # check if png exists
            # if yes, then use it
            # else read audio + generate png it
            pass
        else:
            # read audio
            # generate intermediate png
            # dont save it
            pass

        # image_name = str(self.data_frame.iloc[idx, 0]) + ".jpg"
        # image_name = str(self.data_frame.iloc[idx, 0])
        # image_path = path.join(self.data_dir, image_name)
        # image = io.imread(image_path)

        if self.mode == "test":
            return self.transformer(image)
        else:
            # converting to one hotvector
            label = np.zeros(len(BIRD_CODE), dtype="f")
            label[BIRD_CODE[ebird_code]] = 1
            return self.transformer(image), label
