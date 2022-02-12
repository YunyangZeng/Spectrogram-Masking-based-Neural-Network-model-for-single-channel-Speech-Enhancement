# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:50:24 2022

@author: Yunyang Zeng
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf




class MyModel(tf.keras.Model):
    def __init__(self,input_shape):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [1, 7],
            padding='same',
            dilation_rate = (1, 1),  
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 2]
            )
        
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [7, 1],
            padding='same',
            dilation_rate = (1, 1),  
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 32]
            )
        
        self.conv3 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [5, 5],
            padding='same',
            dilation_rate = (1, 1),  
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 32]
            )
        
        self.conv4 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [5, 5],
            padding='same',
            dilation_rate = (2, 1), 
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 32]
            )  
        
        self.conv5 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [5, 5],
            padding='same',
            dilation_rate = (4, 1),  
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 32]
            )
        self.conv6 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [5, 5],
            padding='same',
            dilation_rate = (8, 1),  
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 32]
            )
        self.conv7 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = [5, 5],
            padding='same',
            dilation_rate = (16, 1), 
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 32]
            )
        self.conv8 = tf.keras.layers.Conv2D(
            filters = 8,
            kernel_size = [1, 1],
            padding='same',
            dilation_rate = (1, 1),  
            activation=tf.nn.relu,
            #input_shape = [None, 301, 257, 32]
            )
        
        #forward_layer = tf.keras.layers.LSTM(1023,return_sequences=True)
        #backward_layer = tf.keras.layers.LSTM(1023,return_sequences=True, go_backwards=True)
        
        self.blstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1023,return_sequences=True),   
        backward_layer = tf.keras.layers.LSTM(1023,return_sequences=True, go_backwards=True))       
        '''
        self.blstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1023,return_sequences=True),  
        backward_layer = tf.keras.layers.LSTM(1023,return_sequences=True, go_backwards=True)) 
        
        self.blstm3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1023,return_sequences=True),
        backward_layer = tf.keras.layers.LSTM(1023,return_sequences=True, go_backwards=True)) 
        '''
        
        self.dense1 = tf.keras.layers.Dense(
            units = 873,
            activation = tf.nn.relu,
            #input_shape = [None, 301, 1023*2]
                                           )
        self.dense2 = tf.keras.layers.Dense(
            units = 514,
            activation = tf.nn.sigmoid,
            #input_shape = [None, 301, 873]
                                           )
        self.build((None,)+input_shape)
        
        
    def call(self, X):
        X=self.conv1(X)
        X=self.conv2(X)
        X=self.conv3(X)
        X=self.conv4(X)
        X=self.conv5(X)
        X=self.conv6(X)
        X=self.conv7(X)
        X=self.conv8(X)
        X_ = X[:,:,:,0]
        for i in range(8-1):
            X_=tf.concat([X_, X[:,:,:,i+1]],axis=-1)
        X=X_
        #X=tf.reshape(X,[X.shape[0], X.shape[1], X.shape[2]*X.shape[3]])
        X=self.blstm1(X)
        #X=self.blstm2(X)
        #X=self.blstm3(X)
        X=self.dense1(X)
        X=self.dense2(X)
        #X = tf.reshape(X,[X.shape[0], X.shape[1], X.shape[2]//2, 2])
        
        X = tf.concat([tf.expand_dims(X[:,:,0:257],axis=-1), tf.expand_dims(X[:,:,257:514],axis=-1)],axis=-1)
        return X