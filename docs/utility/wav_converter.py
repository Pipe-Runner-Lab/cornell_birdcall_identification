import argparse
import soundfile as sf
import warnings

import librosa
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from joblib import delayed, Parallel


def resample(df: pd.DataFrame, target_sr: int):
    org_dir = Path("../../__data__/bird_song/train_audio")
    resample_dir = Path("../../__data__/bird_song.v2/train_audio")
    
    resample_dir.mkdir(exist_ok=True, parents=True)
    warnings.simplefilter("ignore")

    for i, row in df.iterrows():
        ebird_code = row.ebird_code
        filename = row.filename
        ebird_dir = resample_dir / ebird_code
        if not ebird_dir.exists():
            ebird_dir.mkdir(exist_ok=True, parents=True)

        try:
            y, _ = librosa.load(
                org_dir / ebird_code / filename,
                sr=target_sr, 
                mono=True, 
                res_type="kaiser_fast"
            )

            filename = filename.replace(".mp3", ".wav")
            sf.write(ebird_dir / filename, y, samplerate=target_sr)
        except Exception:
            with open("skipped.txt", "a") as f:
                file_path = str(org_dir / ebird_code / filename)
                f.write(file_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", default=32000, type=int)
    parser.add_argument("--n_splits", default=12, type=int)
    args = parser.parse_args()

    target_sr = args.sr

    print("Resampling rate : {}KHz".format(target_sr/1000))

    train = pd.read_csv("../../__data__/bird_song/train.csv")
    dfs = []
    for i in range(args.n_splits):
        if i == args.n_splits - 1:
            start = i * (len(train) // args.n_splits)
            df = train.iloc[start:, :].reset_index(drop=True)
            dfs.append(df)
        else:
            start = i * (len(train) // args.n_splits)
            end = (i + 1) * (len(train) // args.n_splits)
            df = train.iloc[start:end, :].reset_index(drop=True)
            dfs.append(df)

    Parallel(
        n_jobs=args.n_splits,
        verbose=10
    )(
        delayed(resample)(df, args.sr) for df in dfs
    )

    clean_csv = train.drop(columns=[
        "rating","playback_used","channels",
        "pitch","speed", "description", "file_type", "volume", 
        "xc_id", "author", "url","length", "recordist",
        "title", "bird_seen", "sci_name", "location", "license"
    ])

    clean_csv.to_csv("../../__data__/bird_song.v2/train.csv", index=False)
