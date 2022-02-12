# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:50:24 2022

@author: Yunyang Zeng
"""
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os



class loader:
    def __init__(self, train_stft_dir: str, test_stft_dir: str):
        self.train_stft_dir = train_stft_dir
        self.test_stft_dir = test_stft_dir
        
    def load_train(self, file_name: str):
        file = scipy.io.loadmat(os.path.join(self.train_stft_dir,file_name))
        return file
    
    def load_test(self, file_name: str):
        file = scipy.io.loadmat(os.path.join(self.test_stft_dir,file_name))
        return file
    
    def get_MAT_files(self, file_dir: str):
        file_list=[]
        for dirpath, dirnames, filenames in os.walk(file_dir):
            for file in filenames :
                if file[-4:].upper()==".MAT":
                    file_list.append(os.path.join(dirpath, file))
        return file_list      
    def get_file(self, file_dir, shuffle: bool):
        file_list = self.get_MAT_files(file_dir)
        if shuffle:
            np.random.shuffle(file_list)
        return file_list
        
    def get_batch(self, file_list: list, noisy_dir,clean_dir, noise_dir):
        
        batch_clean = []
        batch_noisy = []
        batch_noise = []
        noisy_file_dirs=[]
        clean_file_dirs=[]
        noise_file_dirs=[]
        for i in file_list:
            file = scipy.io.loadmat(i)
            clean = file['clean_stft']
            noisy = file['noisy_stft']
            noise = file['noise_stft']
            batch_clean.append(clean)
            batch_noisy.append(noisy)
            batch_noise.append(noise)
            noisy_file_dirs.append(os.path.join(noisy_dir,i.split('\\')[-1].split('.mat')[0]+'.wav').replace("\\","/"))
            clean_file_dirs.append(os.path.join(clean_dir,"clnsp"+i.split('\\')[-1].split("_clnsp")[-1].split(".mat")[0]+'.wav').replace("\\","/"))
            noise_file_dirs.append(os.path.join(noise_dir,i.split('\\')[-1].split("_clnsp")[0]+".wav").replace("\\","/"))
            
            
        return np.asarray(batch_noisy), np.asarray(batch_clean), np.asarray(batch_noise),noisy_file_dirs, clean_file_dirs, noise_file_dirs
        


