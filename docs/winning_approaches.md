## 1st place [https://www.kaggle.com/c/birdsong-recognition/discussion/183208]

#### Data Aug:
- pink noise
- gaussian noise
- gaussian SNR
- Gain( Vol Adj)

#### Models:
- Used SED model
- Changed feature extractor to Densenet121 (using smaller model to avoid over fitting due to less data)
- Reduced attention block size to 1024
- Used densenet as it has been proven to work in audio comp. 
- Changed clamp to tanh
- 4 fold, 5 fold models without mixup, 4 fold models with mixup

#### Training:
- cosine Annealing Scheduler with warmup
- batch-size 28
- mixup (on 4 of the final models)
- 50 epochs for non-mixup models and 100 epochs for mixup models
- AdamW with weight_decay 0.01
- SpecAugmentation enabled
- 30 second audio clips during training and evaluating on 2 30 second clips per audio.

#### Loss:
- BCELoss. 
- different loss function for 2 of the non-mixup models (randomly removing the primary label predictions from the loss function, 
to try increase the secondary_label predictions)

#### Threshold
- 0.3 on the framewise_output and 0.3 on the clipwise_output to reduce the impact of false positives
- [So if the 30 second clip contained a bird according to the clipwise prediction and the 
5 second interval based on framewise prediction also said it had the same bird then it would be a valid prediction.]
- 10TTA by adding same audio sample 10 times in the batch and enabling spec aug

#### Ensemble
- voting to ensemble the models. Voting selection was based on LB score so in total he had 13 models with 4 votes to consider if the bird existed or not.

***

## 2nd [https://www.kaggle.com/c/birdsong-recognition/discussion/183269]
- saved the Mel spectrograms and later worked with them
 IMPORTANT! While training different architectures, I manually went through 20 thousand training files and deleted large segments without the target bird. If necessary, I can put them in a separate dataset.
- mixed 1 to 3 file
- For contrast, I raised the image to a power of 0.5 to 3. at 0.5, the background noise is closer to the birds, and at 3, on the contrary, the quiet sounds become even quieter.
- Slightly accelerated / slowed down recording
IMPORTANT! Add a different sound without birds(rain, noise, conversations, etc.)
- Added white, pink, and band noise. Increasing the noise level increases recall, but reduces precision.
IMPORTANT! With a probability of 0.5 lowered the upper frequencies. In the real world, the upper frequencies fade faster with distance
- Used BCEWithLogitsLoss. For the main birds, the label was 1. For birds in the background 0.3.
- I didn't look at metrics on training records, but only on validation files similar to the test sample (see dataset). They worked well.
- Added 265 class nocall, but it didn't help much
- The final solution consisted of an ensemble of 6 models, one of which trained on 2.5-second recordings, and one of which only trained on 150 classes. But this model did not work much better than an ensemble of 3 models, where everyone studied in 5 seconds and 265 classes.
- My best solution was sent 3 weeks ago and would have given me first place=)
- Model predictions were squared, averaged, and the root was extracted. The rating slightly increased, compared to simple averaging.
- All models gave similar quality, but the best was efficientnet-b0, resnet50, densenet121.
- Pre-trained models work better
- Spectrogram worked slightly worse than melspectrograms
- Large networks worked slightly worse than small ones
- n_fft = 892, sr = 21952, hop_length=245, n_mels = 224, len_chack 448(or 224), image_size = 224*448
IMPORTANT! If there was a bird in the segment, I increased the probability of finding it in the entire file.
- I tried pseudo-labels, predict classes on training files, and train using new labels, but the quality decreased slightly
- A small learning rate reduced the rating

***

## 3rd [https://www.kaggle.com/c/birdsong-recognition/discussion/183199]
## 4th [https://www.kaggle.com/c/birdsong-recognition/discussion/183339]
## 5th [https://www.kaggle.com/c/birdsong-recognition/discussion/183300]
## 6th [https://www.kaggle.com/c/birdsong-recognition/discussion/183204]
## 8th [https://www.kaggle.com/c/birdsong-recognition/discussion/183223]
## 10th [https://www.kaggle.com/c/birdsong-recognition/discussion/183407]
## 13th [https://www.kaggle.com/c/birdsong-recognition/discussion/183436]
## 18th [https://www.kaggle.com/c/birdsong-recognition/discussion/183219]
