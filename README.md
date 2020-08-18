# Cornell Birdcall Identification 🐦

<img src="./docs/img/header.png">
Competition Link : https://www.kaggle.com/c/birdsong-recognition/

## Setup
```
apt-get update
apt install unzip libgtk2.0-dev libsndfile1

pip install --upgrade torch torchvision
pip install --user wandb easydict blessed tensorboard pandas sklearn matplotlib seaborn pretrainedmodels efficientnet_pytorch transformers albumentations librosa inquirer

```

## Bashrc update
Add this at the end of **.bashrc**

```
export PATH="$PATH:/root/.local/bin"
```

## Login
```
wandb login
```

## Dataset download and upzip
```
cd __data__/bird_song/audio

kaggle datasets download -d ttahara/birdsong-resampled-train-audio-00 && unzip birdsong-resampled-train-audio-00.zip && rm -rf birdsong-resampled-train-audio-00.zip
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-01 && unzip birdsong-resampled-train-audio-01.zip && rm -rf birdsong-resampled-train-audio-01.zip
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-02 && unzip birdsong-resampled-train-audio-02.zip && rm -rf birdsong-resampled-train-audio-02.zip
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-03 && unzip birdsong-resampled-train-audio-03.zip && rm -rf birdsong-resampled-train-audio-03.zip
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-04 && unzip birdsong-resampled-train-audio-04.zip && rm -rf birdsong-resampled-train-audio-04.zip

cd __noise__/random
kaggle datasets download -d kuldeepvansadia/noise-nature-white && unzip noise-nature-white.zip && rm -rf noise-nature-white.zip
```
