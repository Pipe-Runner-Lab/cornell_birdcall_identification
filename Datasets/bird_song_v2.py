import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from skimage import io
from pathlib import Path

from torch.utils.data import Dataset

from Datasets.utils import (fold_creator, mono_to_color)
from utils.submission_utils import BIRD_CODE

NUMBER_OF_FOLDS = 5
DATASET_NAME = 'bird_song_v2'
SHOULD_CACHE = False
PERIOD = 5
MEL_PARAMS = {'n_mels': 128, 'fmin': 20, 'fmax': 16000}


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
        else:
            raise Exception(
                "{} not available for {}".format(mode, DATASET_NAME))
        
        self.transformer = transformer
        self.data_path = data_path

        if fold_number is None:
            # If fold not selected
            self.csv_path = Path(data_path) / (self.mode + ".csv")
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
            y, sr = sf.read(audio_filepath)

            length = y.shape[0]
            effective_length = sr * 5

            if length < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - length)
                new_y[start:start + length] = y
                y = new_y.astype(np.float32)
            elif length > effective_length:
                start = np.random.randint(length - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

            melspec = librosa.feature.melspectrogram(y, sr=sr, **MEL_PARAMS)
            melspec = librosa.power_to_db(melspec).astype(np.float32)

            image = mono_to_color(melspec)

        # converting to one hotvector
        label = np.zeros(len(BIRD_CODE), dtype="f")
        label[BIRD_CODE[ebird_code]] = 1
        return self.transformer["image"](image), label
