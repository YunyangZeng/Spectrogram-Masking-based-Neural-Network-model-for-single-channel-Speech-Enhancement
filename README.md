# Spectrogram-Masking-based-Neural-Network-model-for-single-channel-Speech-Enhancement
Implementation(unofficial) is based on models proposed by Google Audio team: [Exploring Tradeoffs in Models for Low-latency Speech Enhancement](https://arxiv.org/abs/1811.07030)

---

## Dependencies
- python 3.9.7
- tensorflow 2.6.0
- keras 2.6.0

---

## Dataset
- **Clean speech:** This project uses clean speech from [Librispeech](https://www.openslr.org/12/). To reproduce the results, download 'train-clean-360.tar.gz' and 'test-clean.tar.gz ' from Librispeech. Unzip the training clean files to "./data/clean/clean_train/" and unzip the testing clean files to "./data/clean/clean_test/". 
- **Noise:** The noise sounds are obtained from [Microsoft Scalable Noisy Speech Dataset (MS-SNSD)](https://github.com/microsoft/MS-SNSD). Download the noise_train and noise_test folder from [Microsoft Scalable Noisy Speech Dataset (MS-SNSD)](https://github.com/microsoft/MS-SNSD) and put them under "./data/noise/noise_train" and "./data/noise/noise_test".
- **Use sounds from other sources:** Put your training and testing clean speech files(.wav or .flac) under "./data/clean/clean_train/" and "./data/clean/clean_test/" correspondingly. 
Put your training and testing noise files(.wav or .flac) under "./data/noise/noise_train/" and "./data/noise/noise_test/" correspondingly. 

---

## Prepare data
The noisy sounds are synthesized by adding the clean speech and noise at a certain SNR level. Before synthesizing, all sounds will be resampled at the sampling rate specified in "./data/config.ini"

- Open the config.ini file under "./data/". 
- Configure your data path.
- **sample_rate**: all the sounds files will be resampled at this sample rate.
- **file_length**: The length of your training and testing sound files in seconds, default is 3s.
- **total_training_files**: total number of training files that will be synthesized.
- **total_testing_files**: total number of testing files that will be synthesized.
- **SNR_lower**: The lower bound of your SNR level
- **SNR_lower**: The upper bound of your SNR level
- **total_snrlevels**: Number of SNR levels between the lower bound and upper bound

After configuration, run "./data/data_prep.py", 6 folders will be created under "./data/":
- **CleanSpeech_training**: All the normalized clean speech training files used for synthesizing the noisy sounds, which will be used to evaluate the SNRi and SDR performance of the model. Name of each sound file has the form :  "clnsp***n***.wav"
- **CleanSpeech_testing**: All the normalized clean speech testing files used for synthesizing the noisy sounds, which will be used to evaluate the SNRi and SDR performance of the model. Name of each sound file has the form :  "clnsp***n***.wav"
- **Noise_training**: All the normalized noise speech training files used for synthesizing the noisy sounds, which will be used to evaluate the SNRi and SDR performance of the model. Name of each sound file has the form :  "noisy***n***_SNRdb _***snr_level***.wav"
- **Noise_testing**: All the normalized noise speech testing files used for synthesizing the noisy sounds, which will be used to evaluate the SNRi and SDR performance of the model. Name of each sound file has the form :  "noisy***n***_SNRdb _***snr_level***.wav"
- **NoisySpeech_training**: All the noisy speech training files synthesized. Name of each sound file has the form :  "noisy***n***_SNRdb _***snr_level*** _clnsp***n***.wav"
- **NoisySpeech_testing**: All the noisy speech training files synthesized. Name of each sound file has the form :  "noisy***n***_SNRdb _***snr_level*** _clnsp***n***.wav"

---

## STFT
In order to speed up training, the STFT of all training and testing sound files are calculated a proir. Edit "config.ini" in the main directory to configure your STFT settings. According to the paper, a hann window with 25ms window length and 10ms hop size is used, number of frequency bins is set to 512.

Run "STFT.py", two folders named **Training_STFT** and **Testing_STFT** will be created under the main directory.

## Train
Run "train.py" will train the model. Model parameters are saved in "cp". If you want to resume training from last checkpoint, modify the last line of "train.py", change it as
```
trainer.train(from_checkpoint=True,save_weights = True).
```
Training and testing results will be saved every each epoch, you can use tensorboard to visualize the results.
```
python -m tensorboard.main --logdir=*your_dir*\logs\gradient_tape
```
The enhanced testing files are saved under "./enhanced" 

---

## System architecture
**Delta phase is not used in this implementation because I haven't figured out the correct way to implement it**.
![](./assets/System_architecture.PNG)

---

## Result
- Trained on an NVIDIA RTX 3080 GPU for 24 hours.
### SDR
|    Metric   |  value  |
|-------------|---------|
| AVG SDR     | 11.65db |
| AVG SNRi    | 9.96db  |

![](./assets/Testing_SDR.PNG)
![](./assets/Testing_SNRi.PNG)



