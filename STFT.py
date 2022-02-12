# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 19:16:07 2022

@author: Yunyang Zeng
"""
import os
import configparser
import soundfile as sf
import librosa

import scipy.io
from tqdm import tqdm
import time

def find_WAV_file(file_dir):
    file_list=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames :
            if file[-4:].upper()==".WAV":
                file_list.append(os.path.join(dirpath, file))
    return file_list
def STFT(cfg):
    CleanSpeech_training_dir = cfg.get('STFT','CleanSpeech_training_dir')
    CleanSpeech_testing_dir = cfg.get('STFT','CleanSpeech_testing_dir')
    NoisySpeech_training_dir = cfg.get('STFT','NoisySpeech_training_dir')
    NoisySpeech_testing_dir = cfg.get('STFT','NoisySpeech_testing_dir')
    Noise_training_dir = cfg.get('STFT','Noise_training_dir')
    Noise_testing_dir = cfg.get('STFT','Noise_testing_dir')
    
    
    
    sample_rate = cfg.getfloat('STFT','sample_rate')
    file_length = cfg.getfloat('STFT','file_length')
    window_type = cfg.get('STFT','window_type')
    window_length = cfg.getfloat('STFT','window_length')
    hop_size = cfg.getfloat('STFT','hop_size')
    n_fft = cfg.getint('STFT','n_fft')
    
    dir_list={'Training':[CleanSpeech_training_dir,NoisySpeech_training_dir,Noise_training_dir],'Testing':[CleanSpeech_testing_dir, NoisySpeech_testing_dir,Noise_testing_dir]}

    for i in dir_list:
        noisy_file_list=find_WAV_file(dir_list[i][1])
        for noisy_file in tqdm(noisy_file_list):
            #print(noisy_file)
            clean_file=os.path.join(dir_list[i][0],"clnsp"+noisy_file.split('\\')[-1].split("_clnsp")[-1]).replace("\\","/")
            noise_file=os.path.join(dir_list[i][2],noisy_file.split('\\')[-1].split("_clnsp")[0]+".wav").replace("\\","/")
            
            noisy_wav_file, sr = sf.read(noisy_file)
            clean_wav_file, sr = sf.read(clean_file)
            noise_wav_file, sr = sf.read(noise_file)
            
            if len(noisy_wav_file) > (file_length*sr): 
                noisy_wav_file = noisy_wav_file[:int(file_length*sr)]
                clean_wav_file = clean_wav_file[:int(file_length*sr)]
                noise_wav_file = noise_wav_file[:int(file_length*sr)]
                flag = True
            else:
                flag = False
            noisy_stft_file = librosa.stft(noisy_wav_file,n_fft=n_fft,hop_length=int(hop_size*sr),win_length=int(window_length*sr),window=window_type)
            clean_stft_file = librosa.stft(clean_wav_file,n_fft=n_fft,hop_length=int(hop_size*sr),win_length=int(window_length*sr),window=window_type)
            noise_stft_file = librosa.stft(noise_wav_file,n_fft=n_fft,hop_length=int(hop_size*sr),win_length=int(window_length*sr),window=window_type)
            if not os.path.exists(os.path.join(os.getcwd(),i)+"_STFT"):
                os.makedirs(os.path.join(os.getcwd(),i)+"_STFT")
            

            if not os.path.exists(os.path.join(os.getcwd(),i+"_STFT",(noisy_file.split('\\')[-1][:-4]+".mat"))):
                scipy.io.savemat(os.path.join(os.getcwd(),i+"_STFT",(noisy_file.split('\\')[-1][:-4]+".mat")),{"noisy_stft": noisy_stft_file,"clean_stft": clean_stft_file,"noise_stft": noise_stft_file})
            if flag:
                sf.write(noisy_file, noisy_wav_file, sr)
                sf.write(clean_file, clean_wav_file, sr)
                sf.write(noise_file, noise_wav_file, sr)
            time.sleep(0.05)
            
            
       
    



if __name__ == "__main__":
    cfg=configparser.ConfigParser()
    cfg.read('.\\config.ini',encoding='utf8')
    STFT(cfg)
    
    
    