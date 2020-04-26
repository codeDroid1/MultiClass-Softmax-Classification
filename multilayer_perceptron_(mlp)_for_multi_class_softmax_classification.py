# -*- coding: utf-8 -*-
"""Multilayer Perceptron (MLP) for multi-class softmax classification.ipynb
# Multilayer Perceptron (MLP) for multi-class softmax classification
"""

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import Adam

#Generate Dummy Data 
import numpy as np
# train Data
train_x = np.random.random((1000,20))      #Return random floats in the half-open interval [0.0, 1.0).
train_y = np.random.randint(10,size=(1000,1))
# test Data
test_x = np.random.random((100,20))
test_y = np.random.randint(10,size=(100,1))

#one hot encoding
train_y = keras.utils.to_categorical(train_y)
test_y  = keras.utils.to_categorical(test_y)

model = Sequential()

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first laye, you must specify the excepted input data type

model.add(Dense(64,activation='relu',input_dim=20))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_x,train_y,epochs=20,batch_size=128)

score = model.evaluate(test_x,test_y,batch_size=128)


