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
DATASET_NAME = 'bird_song_v2_test'
PERIOD = 5
MEL_PARAMS = {'n_mels': 128, 'fmin': 20, 'fmax': 16000}
SR = 32000


class Bird_Song_v2_Test_Dataset(Dataset):
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

        if mode == "VAL":
            self.mode = "val"
        else:
            raise Exception(
                "{} not available for {}".format(mode, DATASET_NAME))
        
        self.transformer = transformer
        self.data_path = data_path

        
        self.csv_path = Path(data_path) / (self.mode + ".csv")
        self.data_dir = Path(data_path) / self.mode

        self.data_frame = pd.read_csv(self.csv_path)

    def get_csv_path(self):
        return self.csv_path

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get basic data path
        row_id = self.data_frame["filename_seconds"][idx]
        seconds = self.data_frame["seconds"][idx]
        filename = self.data_frame["filename"][idx]
        try:
            ebird_codes = self.data_frame["birds"][idx].split(' ')
        except:
            ebird_codes = []

        # generate file path
        audio_filepath = self.data_dir / (filename + '.wav')

        y, _ = sf.read(audio_filepath)

        # trim audio
        end_seconds = int(seconds)
        start_seconds = int(end_seconds - 5)
            
        start_index = SR * start_seconds
        end_index = SR * end_seconds

        y = y[start_index:end_index].astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=SR, **MEL_PARAMS)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        image = mono_to_color(melspec)

        # converting to one hotvector
        label = np.zeros(len(BIRD_CODE), dtype="f")
        for code in ebird_codes:
            try:
                index = BIRD_CODE[code]
            except:
                index = 0
            label[index] = 1
        return self.transformer["image"](image), label
