# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:48:59 2022

@author: Yunyang Zeng

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from model import MyModel
import numpy as np
from loader import loader
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import matplotlib.pyplot as plt
import datetime
from mir_eval.separation import bss_eval_sources
import librosa
import soundfile as sf
import configparser
from tqdm import tqdm
import time


class trainer:
    def __init__(self,cfg, model, loader, train_file_path:str, test_file_path:str,n_epoch=5, batch_size=10, learn_rate=0.00021, Lambda = 0.113):
        self.cfg=cfg
        self.n_epoch = n_epoch
        self.batch_size=batch_size
        self.learn_rate = learn_rate
        self.Lambda = Lambda
        self.train_file_path=train_file_path
        self.test_file_path=test_file_path
        self.loader = loader(self.train_file_path,self.test_file_path)
        self.input_shape =(301, 257, 2)
        self.model = model(self.input_shape)
        self.model.compute_output_shape(input_shape=(None, 301, 257, 2))
        self.sr = self.cfg.getint('STFT','sample_rate')
        self.window_length = self.cfg.getfloat('STFT','window_length')
        self.hop_size=self.cfg.getfloat('STFT','hop_size')
        self.n_fft=self.cfg.getint('STFT','n_fft')
        self.window_type=self.cfg.get('STFT','window_type')
    def loss(self, enhanced_compressed, clean_compressed):
        mag_loss = tf.math.pow(tf.norm(tf.math.abs(enhanced_compressed)-tf.math.abs(clean_compressed),ord='euclidean',axis=[-2,-1]),2)
        complex_loss = self.Lambda*tf.math.real(tf.math.pow(tf.norm(enhanced_compressed-clean_compressed,ord='euclidean',axis=[-2,-1]),2))
        loss = mag_loss + complex_loss
        loss = tf.math.reduce_mean(loss)
        return loss
    
    def get_SNRi(self,enhanced, enhanced_clean, noisy, clean, noise):
        snr_orig = np.mean(clean**2)/np.mean(noise**2)
        snr_orig_db = 10*np.log10(snr_orig)
        
        residual_noise = enhanced-enhanced_clean
        snr_enhanced = np.mean(enhanced_clean**2)/np.mean(residual_noise**2)
        snr_enhanced_db = 10*np.log10(snr_enhanced)
        SNRi = snr_enhanced_db-snr_orig_db
        return SNRi, snr_enhanced_db, snr_orig_db
    
    def get_SDR(self,enhanced, clean):

        sdr = bss_eval_sources(clean, enhanced, False)[0][0]
        return sdr
    def iSTFT(self,STFT_file):
        wav_file = librosa.istft(STFT_file, int(self.hop_size*self.sr), int(self.window_length*self.sr), self.n_fft, self.window_type)
        return wav_file
    
    def istft(self, mag, phase):
       stft_matrix = mag * np.exp(1j*phase)
       return librosa.istft(stft_matrix,
                            hop_length=int(self.hop_size*self.sr),
                            win_length=int(self.window_length*self.sr))
   
    def delta_phase(self,batch_spectrogram):
        batch_delta_phase = np.zeros_like(batch_spectrogram)
        for i in range(batch_spectrogram.shape[0]):
            spectrogram=batch_spectrogram[i,:,:]
            delta_phase = np.zeros_like(spectrogram)
            delta_phase[:,0]=np.angle(spectrogram[:,0])
            for f in range(1, spectrogram.shape[-1]):
                delta_phase[:,f]=np.angle(np.divide(spectrogram[:,f],spectrogram[:,f-1]))
            batch_delta_phase[i,:,:] = delta_phase
        return batch_delta_phase
    
    def test(self, train= True, get_wav= False):
        weights_save_dir = "./cp/my_checkpoint"
        if not train:
            self.model.load_weights(weights_save_dir)
        noisy_dir = self.cfg.get('STFT','NoisySpeech_testing_dir')
        clean_dir = self.cfg.get('STFT','CleanSpeech_testing_dir')
        noise_dir = self.cfg.get('STFT','Noise_testing_dir') 
        files = self.loader.get_file(file_dir = self.test_file_path, shuffle = True)
        num_files = len(files)
        num_batch = num_files//self.batch_size
        epoch_SNRi = 0
        epoch_SDR = 0
        epoch_orig_SDR = 0
        print("Testing")
        for i in tqdm(range(num_batch)):
            batch_noisy_matfile_dirs = files[i*self.batch_size : (i+1)*self.batch_size]
            batch_noisy, batch_clean, batch_noise, noisy_wavfile_dirs, clean_wavfile_dirs, noise_wavfile_dirs = self.loader.get_batch(batch_noisy_matfile_dirs, noisy_dir, clean_dir, noise_dir)
            batch_delta_phase = self.delta_phase(np.transpose(batch_noisy,(0,2,1)))
            batch_noisy = tf.transpose(batch_noisy,perm=[0,2,1])
            batch_clean = tf.transpose(batch_clean,perm=[0,2,1])
            batch_noisy_compressed = tf.math.pow(batch_noisy,0.3)
            batch_clean_compressed = tf.math.pow(batch_clean,0.3)
            batch_noisy_compressed_r = tf.math.real(batch_noisy_compressed)
            batch_noisy_compressed_i = tf.math.imag(batch_noisy_compressed)
            batch_noisy_compressed_r = tf.expand_dims(batch_noisy_compressed_r, axis=-1)
            batch_noisy_compressed_i = tf.expand_dims(batch_noisy_compressed_i, axis=-1)
            batch_delta_phase = tf.expand_dims(tf.cast(batch_delta_phase, 'float64'), axis=-1)
            batch_noisy_compressed_ri = tf.concat([batch_noisy_compressed_r, batch_noisy_compressed_i], axis=-1)
            mask = tf.cast(self.model.call(batch_noisy_compressed_ri),'float64')               
            batch_enhanced = tf.cast(tf.math.multiply(batch_noisy_compressed_ri, mask),'complex128')
            batch_enhanced = batch_enhanced[:,:,:,0]+batch_enhanced[:,:,:,1]*1j
            batch_clean_compressed_r = tf.math.real(batch_clean_compressed)
            batch_clean_compressed_i = tf.math.imag(batch_clean_compressed)
            batch_clean_compressed_r = tf.expand_dims(batch_clean_compressed_r, axis=-1)
            batch_clean_compressed_i = tf.expand_dims(batch_clean_compressed_i, axis=-1)
            batch_clean_compressed_ri = tf.concat([batch_clean_compressed_r, batch_clean_compressed_i], axis=-1)
            batch_clean_enhanced = tf.cast(tf.math.multiply(batch_clean_compressed_ri, tf.cast(self.model.call(batch_clean_compressed_ri),'float64')),'complex128')
            batch_clean_enhanced = batch_clean_enhanced[:,:,:,0]+batch_clean_enhanced[:,:,:,1]*1j
            loss = self.loss(tf.pow(batch_enhanced,0.3), batch_clean_compressed
            for ind in range(self.batch_size):
                enhanced_istft = self.istft(np.abs(np.transpose(batch_enhanced[ind,:,:].numpy())), np.angle(np.transpose(batch_clean[ind,:,:])))
                enhanced_clean_istft = self.istft(np.abs(np.transpose(batch_clean_enhanced[ind,:,:].numpy())), np.angle(np.transpose(batch_clean[ind,:,:])))                        
                noisy_istft,_ = sf.read(noisy_wavfile_dirs[ind])
                clean_istft,_ = sf.read(clean_wavfile_dirs[ind])
                noise_istft,_ = sf.read(noise_wavfile_dirs[ind])
                if get_wav:
                    if not os.path.exists(r'./enhanced/batch{n}/result{ind}'.format(n=i, ind=str(ind))):
                        os.makedirs(r'./enhanced/batch{n}/result{ind}'.format(n=i,ind=str(ind)))
                    sf.write('./enhanced/batch{n}/result{ind}/enhanced.wav'.format(n=i,ind=str(ind)), enhanced_istft/max(enhanced_istft), self.sr)
                    sf.write('./enhanced/batch{n}/result{ind}/noisy.wav'.format(n=i,ind=str(ind)), noisy_istft, self.sr)
                    sf.write('./enhanced/batch{n}/result{ind}/clean.wav'.format(n=i,ind=str(ind)), clean_istft, self.sr)
                SNRi,snr_enhanced_db, snr_orig_db = self.get_SNRi(enhanced_istft,enhanced_clean_istft,noisy_istft,clean_istft,noise_istft)
                SDR = self.get_SDR(enhanced_istft,enhanced_clean_istft)
                orig_SDR = self.get_SDR(noisy_istft, clean_istft)
                epoch_SNRi += SNRi
                epoch_SDR += SDR
                epoch_orig_SDR += orig_SDR
        avg_epoch_SNRi = epoch_SNRi/(num_batch*self.batch_size)   
        avg_epoch_SDR = epoch_SDR/(num_batch*self.batch_size)
        avg_epoch_orig_SDR = epoch_orig_SDR/(num_batch*self.batch_size)
        return loss, avg_epoch_SNRi, avg_epoch_SDR, avg_epoch_orig_SDR
    
    def train(self,from_checkpoint = False, save_weights=True):
        noisy_dir = self.cfg.get('STFT','NoisySpeech_training_dir')
        clean_dir = self.cfg.get('STFT','CleanSpeech_training_dir')
        noise_dir = self.cfg.get('STFT','Noise_training_dir')       
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)
        weights_save_dir = "./cp/my_checkpoint"
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('./logs/gradient_tape/'):
            os.makedirs('./logs/gradient_tape/')
        train_log_dir = './logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)   
        if from_checkpoint:
            self.model.load_weights(weights_save_dir)
        self.model.summary()
        for e in range(self.n_epoch):
            files = self.loader.get_file(file_dir = self.train_file_path, shuffle = True)
            num_files = len(files)
            num_batch = num_files//self.batch_size
            epoch_SNRi = 0
            epoch_SDR = 0
            epoch_orig_SDR=0
            print("Training epoch "+str(e+1))
            for i in tqdm(range(num_batch)):
                batch_noisy_matfile_dirs = files[i*self.batch_size : (i+1)*self.batch_size]
                batch_noisy, batch_clean, batch_noise, noisy_wavfile_dirs, clean_wavfile_dirs, noise_wavfile_dirs = self.loader.get_batch(batch_noisy_matfile_dirs, noisy_dir, clean_dir, noise_dir)
                with tf.GradientTape() as tape:
                    batch_noisy = tf.transpose(batch_noisy,perm=[0,2,1])
                    batch_clean = tf.transpose(batch_clean,perm=[0,2,1])
                    batch_noisy_compressed = tf.math.pow(batch_noisy,0.3)
                    batch_clean_compressed = tf.math.pow(batch_clean,0.3)
                    batch_noisy_compressed_r = tf.math.real(batch_noisy_compressed)
                    batch_noisy_compressed_i = tf.math.imag(batch_noisy_compressed)                  
                    batch_noisy_compressed_r = tf.expand_dims(batch_noisy_compressed_r, axis=-1)
                    batch_noisy_compressed_i = tf.expand_dims(batch_noisy_compressed_i, axis=-1)
                    batch_noisy_compressed_ri = tf.concat([batch_noisy_compressed_r, batch_noisy_compressed_i], axis=-1)       
                    mask = tf.cast(self.model.call(batch_noisy_compressed_ri),'float64')                     
                    batch_enhanced = tf.cast(tf.math.multiply(batch_noisy_compressed_ri, mask),'complex128')
                    batch_enhanced = batch_enhanced[:,:,:,0]+batch_enhanced[:,:,:,1]*1j     
                    batch_clean_compressed_r = tf.math.real(batch_clean_compressed)
                    batch_clean_compressed_i = tf.math.imag(batch_clean_compressed)
                    batch_clean_compressed_r = tf.expand_dims(batch_clean_compressed_r, axis=-1)
                    batch_clean_compressed_i = tf.expand_dims(batch_clean_compressed_i, axis=-1)
                    batch_clean_compressed_ri = tf.concat([batch_clean_compressed_r, batch_clean_compressed_i], axis=-1)
                    batch_clean_enhanced = tf.cast(tf.math.multiply(batch_clean_compressed_ri, tf.cast(self.model.call(batch_clean_compressed_ri),'float64')),'complex128')
                    batch_clean_enhanced = batch_clean_enhanced[:,:,:,0]+batch_clean_enhanced[:,:,:,1]*1j
                    loss = self.loss(tf.pow(batch_enhanced,0.3), batch_clean_compressed)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.trainable_variables))
                time.sleep(0.05)
                
                for ind in range(self.batch_size):
                    enhanced_istft = self.istft(np.abs(np.transpose(batch_enhanced[ind,:,:].numpy())), np.angle(np.transpose(batch_clean[ind,:,:])))
                    enhanced_clean_istft = self.istft(np.abs(np.transpose(batch_clean_enhanced[ind,:,:].numpy())), np.angle(np.transpose(batch_clean[ind,:,:])))
                    noisy_istft,_ = sf.read(noisy_wavfile_dirs[ind])
                    clean_istft,_ = sf.read(clean_wavfile_dirs[ind])
                    noise_istft,_ = sf.read(noise_wavfile_dirs[ind])
                    SNRi,snr_enhanced_db, snr_orig_db = self.get_SNRi(enhanced_istft, enhanced_clean_istft, noisy_istft,clean_istft,noise_istft)
                    SDR = self.get_SDR(enhanced_istft, enhanced_clean_istft)
                    orig_SDR = self.get_SDR(noisy_istft, clean_istft)
                    epoch_SNRi += SNRi
                    epoch_SDR += SDR
                    epoch_orig_SDR += orig_SDR
            if not os.path.exists(weights_save_dir+'/cp{cpn}'.format(cpn=e)):
                os.makedirs(weights_save_dir+'/cp{cpn}'.format(cpn=e))
            if save_weights:
                self.model.save_weights(weights_save_dir+'/cp{cpn}/'.format(cpn=e))     
                self.model.save_weights(weights_save_dir)
            avg_epoch_SNRi = epoch_SNRi/(num_batch*self.batch_size)   
            avg_epoch_SDR = epoch_SDR/(num_batch*self.batch_size)
            avg_epoch_orig_SDR = epoch_orig_SDR/(num_batch*self.batch_size)
            testing_loss,avg_testing_epoch_SNRi, avg_testing_epoch_SDR, avg_original_testing_SDR=self.test(train=True, get_wav=True)
            print("""Epoch: {epoch}||training_Loss: {loss:.2f}||training_SNRi: {SNRi:.2f} db||avg_original_training_SDR:{avg_original_training_SDR:.2f}db||training_SDR:{SDR:.2f} db
                  ||testing_Loss: {testing_Loss:.2f}||testing_SNRi: {testing_SNRi:.2f}db||avg_original_testing_SDR:{avg_original_testing_SDR:.2f}db||testing_SDR:{testing_SDR:.2f} db"""\
                  .format(epoch=e+1, loss=loss.numpy(), SNRi=avg_epoch_SNRi, avg_original_training_SDR=avg_epoch_orig_SDR , SDR=avg_epoch_SDR, \
                  testing_Loss=testing_loss, testing_SNRi=avg_testing_epoch_SNRi,avg_original_testing_SDR=avg_original_testing_SDR, testing_SDR=avg_testing_epoch_SDR   ))

            with train_summary_writer.as_default():
                tf.summary.scalar('Training Loss', loss, step=e)
                tf.summary.scalar('Training SNRi', avg_epoch_SNRi, step=e)
                tf.summary.scalar('Training SDR', avg_epoch_SDR, step=e)
                tf.summary.scalar('Testing Loss', testing_loss, step=e)
                tf.summary.scalar('Testing SNRi', avg_testing_epoch_SNRi, step=e)
                tf.summary.scalar('Testing SDR', avg_testing_epoch_SDR, step=e)

        print("##################")
        print("Training finished!")     
    
        
if __name__ == "__main__":
    cfg=configparser.ConfigParser()
    cfg.read('.\\config.ini',encoding='utf8')
    trainer = trainer(cfg, MyModel, loader, '.\Training_STFT', '.\Testing_STFT',n_epoch=1000, batch_size=10, Lambda = 0.113)
    trainer.train(from_checkpoint=False,save_weights = True)
