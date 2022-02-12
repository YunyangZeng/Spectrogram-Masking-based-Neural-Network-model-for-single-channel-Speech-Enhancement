# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 23:24:06 2022

@author: Yunyang Zeng
"""
import os
import soundfile as sf
import librosa
import numpy as np
import random
import configparser
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def resample(input_dir ,output_dir, target_sr, orig_sr = None):
    input_file, orig_sr = sf.read(input_dir)
    if orig_sr !=16000:
        print(orig_sr)
    output_file = librosa.resample(input_file, orig_sr, target_sr)
    sf.write(output_dir, output_file, target_sr)
def find_WAV_file(file_dir):
    file_list=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames :
            if file[-5:].upper()==".FLAC" or file[-4:].upper() == '.WAV':
                file_list.append(os.path.join(dirpath, file))
    return file_list
def SNR_mix(clean, noise, snr_db):
    #snr = 20*np.log(((clean**2).mean()**0.5)/((noise**2).mean()**0.5))
    P_clean = np.mean(clean**2)
    P_noise = np.mean(noise**2)
    
    clean_normalized = clean/(P_clean**0.5)
    noise_normalized = noise/(P_noise**0.5)
    
    noisy = clean_normalized + noise_normalized/10**(snr_db/20)
    
    if max(abs(noisy)) >=1:
        
        clean_normalized  = clean_normalized /max(abs(noisy))
        noise_normalized = noise_normalized/max(abs(noisy))   
        noisy = noisy/max(abs(noisy))
        
    return noisy, clean_normalized , noise_normalized/10**(snr_db/20)
    

    
    
    
    
if __name__ == "__main__":
    
    cfg=configparser.ConfigParser()
    cfg.read('.\config.ini',encoding='utf8')
    clean_train_path = cfg.get('config', "clean_train_path")
    noise_train_path = cfg.get('config', "noise_train_path")
    clean_test_path = cfg.get('config', "clean_test_path")
    noise_test_path = cfg.get('config', "noise_test_path")    
    target_sr = cfg.getint('config', "sample_rate")
    file_len = cfg.getint('config', "file_length")
    #train_test_ratio =cfg.getfloat('config', "train_test_ratio")
    snr_lower = cfg.getint('config',"SNR_lower")
    snr_upper = cfg.getint('config',"SNR_upper")
    
    total_snrlevels= cfg.getint('config',"total_snrlevels")
    total_training_files =cfg.getint('config','total_training_files')
    total_testing_files = cfg.getint('config','total_testing_files')
    snr_range = np.linspace(snr_lower, snr_upper, total_snrlevels, dtype=float)
    
    if not os.path.exists(r'./NoisySpeech_training'):
        os.makedirs(r'./NoisySpeech_training')
    if not os.path.exists(r'./NoisySpeech_testing'):
        os.makedirs(r'./NoisySpeech_testing')
        
    if not os.path.exists(r'./CleanSpeech_training'):
        os.makedirs(r'./CleanSpeech_training')
    if not os.path.exists(r'./CleanSpeech_testing'):
        os.makedirs(r'./CleanSpeech_testing') 
        
    if not os.path.exists(r'./Noise_training'):
        os.makedirs(r'./Noise_training')
    if not os.path.exists(r'./Noise_testing'):
        os.makedirs(r'./Noise_testing')
    
    
    clean_train_list = find_WAV_file(clean_train_path)
    noise_train_list = find_WAV_file(noise_train_path)
    clean_test_list = find_WAV_file(clean_test_path)
    noise_test_list = find_WAV_file(noise_test_path)  
    
    
    random.shuffle(clean_train_list)
    random.shuffle(noise_train_list)
    random.shuffle(clean_test_list)
    random.shuffle(noise_test_list)
    
    modes = ['training', 'testing']
    print("###################################################")
    print("Making training files")
    for i in tqdm(range(total_training_files)):
        clean_train_index =random.choice(range(len(clean_train_list)))
        clean_train, sr_c = sf.read(clean_train_list[clean_train_index])
        if sr_c != target_sr :
            clean_train = librosa.resample(clean_train, sr_c, target_sr)
            
        if len(clean_train) >=  file_len*target_sr:
            start_index_c = random.randint(0,len(clean_train)-target_sr*file_len)
            clean_train = clean_train[start_index_c:start_index_c+target_sr*file_len]
            
            noise_train_index = random.choice(range(len(noise_train_list)))
            noise_train, sr_n = sf.read(noise_train_list[noise_train_index])

            if sr_n !=target_sr:
                noise_train = librosa.resample(noise_train, sr_n, target_sr)
            if len(noise_train) >= file_len*target_sr:
                start_index_n = random.randint(0,len(noise_train)-target_sr*file_len)
                noise_train= noise_train[start_index_n:start_index_n+target_sr*file_len]
                snr_chosen = random.choice(snr_range)
                noisy_train, clean_train_new, noise_train_new= SNR_mix(clean_train, noise_train, snr_chosen)
                #chosen_mode = np.random.choice(modes,p=[1-1/(train_test_ratio+1),1/(train_test_ratio+1)]) 
                sf.write('./NoisySpeech_{mode}/noisy{noisy_ind}_SNRdb_{snr_db:.2f}_clnsp{clean_ind}.wav'.format(mode= 'training', noisy_ind=str(i+1),\
                         snr_db=snr_chosen,clean_ind=str(i+1)), noisy_train, target_sr)
                sf.write('./Noise_{mode}/noisy{noisy_ind}_SNRdb_{snr_db:.2f}.wav'.format(mode= 'training', noisy_ind=str(i+1),\
                         snr_db=snr_chosen), noise_train_new, target_sr)
                sf.write('./CleanSpeech_{mode}/clnsp{clean_ind}.wav'.format(mode= 'training', \
                         clean_ind=str(i+1)), clean_train_new, target_sr)
                
            else: noise_train_list.pop(noise_train_index)
        else:
            clean_train_list.pop(clean_train_index)
        
        time.sleep(0.05)
    print("###################################################")
    print("Making testing files")            
    for j in tqdm(range(total_testing_files)):
        clean_test_index =random.choice(range(len(clean_test_list)))
        clean_test, sr_c = sf.read(clean_test_list[clean_test_index])
        if sr_c != target_sr :
            clean_test = librosa.resample(clean_test, sr_c, target_sr)
            
        if len(clean_test) >=  file_len*target_sr:
            start_index_c = random.randint(0,len(clean_test)-target_sr*file_len)
            clean_test = clean_test[start_index_c:start_index_c+target_sr*file_len]
            
            noise_test_index = random.choice(range(len(noise_test_list)))
            noise_test, sr_n = sf.read(noise_test_list[noise_test_index])

            if sr_n !=target_sr:
                noise_test = librosa.resample(noise_test, sr_n, target_sr)
            if len(noise_test) >= file_len*target_sr:
                start_index_n = random.randint(0,len(noise_test)-target_sr*file_len)
                noise_test= noise_test[start_index_n:start_index_n+target_sr*file_len]
                snr_chosen = random.choice(snr_range)
                noisy_test, clean_test_new, noise_test_new= SNR_mix(clean_test, noise_test, snr_chosen)
                #chosen_mode = np.random.choice(modes,p=[1-1/(train_test_ratio+1),1/(train_test_ratio+1)]) 
                sf.write('./NoisySpeech_{mode}/noisy{noisy_ind}_SNRdb_{snr_db:.2f}_clnsp{clean_ind}.wav'.format(mode= 'testing', noisy_ind=str(j+1),\
                         snr_db=snr_chosen,clean_ind=str(j+1)), noisy_test, target_sr)
                sf.write('./Noise_{mode}/noisy{noisy_ind}_SNRdb_{snr_db:.2f}.wav'.format(mode= 'testing', noisy_ind=str(j+1),\
                         snr_db=snr_chosen), noise_test_new, target_sr)
                sf.write('./CleanSpeech_{mode}/clnsp{clean_ind}.wav'.format(mode= 'testing', \
                         clean_ind=str(j+1)), clean_test_new, target_sr)
                
            else: noise_test_list.pop(noise_test_index)
        else:
            clean_test_list.pop(clean_test_index)
        
        time.sleep(0.05)      
            
            
            
            
        
    '''   
        noisy_stft_file = librosa.stft(noisy,n_fft=512,hop_length=int(0.01*16000),win_length=int(0.025*16000),window='hann')
        clean_stft_file = librosa.stft(clean,n_fft=512,hop_length=int(0.01*16000),win_length=int(0.025*16000),window='hann')
        noise_stft_file = librosa.stft(noise,n_fft=512,hop_length=int(0.01*16000),win_length=int(0.025*16000),window='hann')
        clean_new_stft_file = librosa.stft(clean_new,n_fft=512,hop_length=int(0.01*16000),win_length=int(0.025*16000),window='hann')
        noise_new_stft_file = librosa.stft(noise_new,n_fft=512,hop_length=int(0.01*16000),win_length=int(0.025*16000),window='hann')
        plt.imshow(20*np.log(np.transpose(np.abs(noisy_stft_file))+1e-9),cmap='jet')
        plt.title('noisy %d' %i)
        plt.gca().invert_yaxis()
        plt.show()
        plt.imshow(20*np.log(np.transpose(np.abs(clean_new_stft_file))+1e-9),cmap='jet')
        plt.title('clean_new %d' %i)
        plt.gca().invert_yaxis()
        plt.show()
        plt.imshow(20*np.log(np.transpose(np.abs(noise_new_stft_file))+1e-9),cmap='jet')
        plt.title('noise_new %d' %i)
        plt.gca().invert_yaxis()
        plt.show()
        plt.imshow(20*np.log(np.transpose(np.abs(clean_stft_file))+1e-9),cmap='jet')
        plt.title('clean %d' %i)
        plt.gca().invert_yaxis()
        plt.show()
        plt.imshow(20*np.log(np.transpose(np.abs(noise_stft_file))+1e-9),cmap='jet')
        plt.title('noise %d' %i)
        plt.gca().invert_yaxis()
        plt.show()
        '''
        
        
        
            
            
                
            
                
                
                
        
        
  