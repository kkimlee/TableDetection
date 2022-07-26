import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow import keras

import cv2
import os
#%% 데이터 불러오기
data_path = '..\\data'
train_path = os.path.join(data_path, 'train')
x_train_path = os.path.join(train_path, 'input')
y_train_path = os.path.join(train_path, 'output')

x_train = [cv2.imread(os.path.join(x_train_path, f)) for f in os.listdir(x_train_path)]
y_train = [cv2.imread(os.path.join(y_train_path, f)) for f in os.listdir(y_train_path)]

#%% 데이터 전처리
# width = x_train[0].shape[1]
# height = x_train[0].shape[0]

width = 896
height = 896

x_train = [cv2.resize(img, (height, width)) for img in x_train]
y_train = [cv2.resize(img, (height, width)) for img in y_train]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train.astype('float32') / 255.
y_train = y_train.astype('float32') / 255.

x_train = x_train[:-10]
y_train = y_train[:-10]

x_test = x_train[-10:]
y_test = y_train[-10:]
#%% 데이터 시각화
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.title("blueprint")
    plt.imshow(x_train[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("mask")
    plt.imshow(y_train[i])
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()

#%% 모델 정의
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(height, width, 3)),
            layers.Conv2D(16, (9, 9), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (7, 7), activation='relu', padding='same', strides=2),
            layers.Conv2D(4, (5, 5), activation='relu', padding='same', strides=2),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same', strides=2)
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(2, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(4, kernel_size=5, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(8, kernel_size=7, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=9, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = Autoencoder()



#%% 모델 초기화
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#%%
keras.utils.plot_model(autoencoder.encoder)
plt.show()
#%% 모델 학습
autoencoder.fit(x_train, y_train,
                validation_data = (x_test, y_test),
                epochs=1000,
                batch_size=32,
                shuffle=True)

#%%
autoencoder.encoder.summary()
autoencoder.decoder.summary()

#%%


#%%
autoencoder.encoder.save('encoder.h5')
autoencoder.decoder.save('decoder.h5')

#%% 테스트 결과
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

#%% 테스트 결과 시각화
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.title("blueprint")
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("mask")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()

#%%
cv2.imshow('result', decoded_imgs[4])
cv2.waitKey(0)

cv2.imshow('result2', x_test[4])
cv2.waitKey(0)
