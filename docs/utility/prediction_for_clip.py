import torch
import pandas as pd
import librosa

from pann import AttBlock, Pann_Cnn14_Attn


def prediction_for_clip(test_df, audio_id, clip, model, threshold=0.5):
    """
    [ Function that takes a long audio clip, 
      generates prediction for each row-id related
      to that clip in the test_df
    ]

    Args:
        test_df : Original test df
        clip : Long clip on which tagging needs to be done
        model : The model that will be responsible for prediction
        threshold : Used to generate classification output. Defaults to 0.5.

    Returns:
        dataframe : Modified dataframe with row-id and corresponding space
                    seperated audio tags
    """
    model.eval()

    return "Something"


if __name__ == '__main__':

    # Model
    model = Pann_Cnn14_Attn(pretrained=True)
    model.att_block = AttBlock(2048, 264, activation='linear')
    model = model.to(torch.device("cuda:0"))

    # test df
    test_df = pd.read_csv("./test_data/test.csv")

    # audio clip id
    unique_audio_id = test_df.audio_id.unique()

    y, sr = librosa.load("./test_data/test_audio/{}.mp3".format(unique_audio_id),
                         sr=32000, mono=True, res_type="kaiser_fast")

    # predicting only for a single clip
    output =  prediction_for_clip(test_df, unique_audio_id[0], y, model, threshold=0.5)

    print(output)
