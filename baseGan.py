# -*- coding: utf-8 -*-
"""
BASE GAN: A fairly simple and flexible gan to build new projects from
Training data should be stored in a file in the same directory as BaseGAN.
This program was developed around the AADB dataset and thus is limited to
a maximum resolution of 256,256. Scale denotes magnitude of downscaling
against original value.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from keras.layers import Input, Activation
from keras.models import Sequential, Model
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

resolution = 64
scale = int(256/resolution)

adam = Adam(lr=0.001, beta_1=0.5)

xTrain = np.load("trainX_scale%d_.npy" %scale)

xTrain = (xTrain.astype(np.float32)/127.5)-1 #Normalising
randomDim = 16

#For building GANs that generate images from noise
def buildConvGenFromNoise(inputDimension, maxFilters = 128, outputDimension = 32, isColour = True, reLuLeakage = 0.1):
    startDim = int(outputDimension/8)
    if isColour == True:
        colourDim = 3
    else:
        colourDim = 1
        
    model = Sequential()
    model.add(Dense((3*startDim**2), input_dim = inputDimension, kernel_initializer= initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(reLuLeakage))
    model.add(Reshape((startDim,startDim,3)))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(int(maxFilters), kernel_size=(5,5), padding=('same')))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(reLuLeakage))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(int(maxFilters/2), kernel_size=(5,5), padding=('same')))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(reLuLeakage))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(int(maxFilters/4), kernel_size=(5,5), padding=('same')))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(reLuLeakage))
    model.add(Conv2D(colourDim, kernel_size=(5,5), padding=('same'), activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer=adam)
    
    return model

#Discriminator is independent of generator input
def buildConvDisc(inputResolution = 32, maxFilters = 128, reLuLeakage = 0.1, dropout=0.3):        
    model = Sequential()
    model.add(Conv2D(int(maxFilters/8), kernel_size=(5,5), strides=(2, 2), padding=('same'), input_shape=(inputResolution, inputResolution, 3), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(reLuLeakage))
    model.add(Dropout(dropout))
    model.add(Conv2D(int(maxFilters/4), kernel_size=(5,5), strides=(2, 2), padding=('same')))
    model.add(LeakyReLU(reLuLeakage))
    model.add(Dropout(dropout))
    model.add(Conv2D(int(maxFilters/2), kernel_size=(5,5), strides=(2, 2), padding=('same')))
    model.add(LeakyReLU(reLuLeakage))
    model.add(Dropout(dropout))
    model.add(Conv2D(maxFilters, kernel_size=(5,5), strides=(2, 2), padding=('same')))
    model.add(LeakyReLU(reLuLeakage))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam)
    
    return model

#Change GAN parameters here to allow easy return to defaults
def buildGan(inputDimension):
    global generator, discriminator, gan
    generator = buildConvGenFromNoise(inputDimension, outputDimension = resolution)
    discriminator = buildConvDisc(inputResolution = resolution)
    discriminator.trainable = False
    ganInput = Input(shape=(inputDimension,))
    genOutput = generator(ganInput)
    discOutput = discriminator(genOutput)
    gan = Model(inputs = ganInput, outputs = discOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)
    
#Generates a sweet looking grid of images
def plotGeneratedImages(epoch, examples=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, resolution, resolution, 3)
    generatedImages = ((generatedImages+1)/2)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/aesGAN_epoch_%d.png' % epoch)

def saveModels(epoch):
    generator.save('models/aesGAN_G_scale_%d_epoch_%d.h5' %(scale,epoch))
    discriminator.save('models/aesGAN_D_scale_%d_epoch_%d.h5' %(scale,epoch))
    
def loadModels(epoch):
    generator.load_weights('models/aesGAN_G_scale_%d_epoch_%d.h5' %(scale,epoch))
    discriminator.load_weights('models/aesGAN_D_scale_%d_epoch_%d.h5' %(scale,epoch))

def printModels():
    generator.summary()
    discriminator.summary()
    gan.summary()
    
def train(epochs=1, batchSize=128, startPoint=0):
    global dLosses, gLosses
    dLosses = []
    gLosses = []
    batchCount = int(xTrain.shape[0] / batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    if startPoint > 0:
        loadModels(startPoint)
    for e in range(startPoint+1, startPoint+epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            randSelection = np.random.randint(0, xTrain.shape[0], size=batchSize)
            imageBatch = xTrain[randSelection].reshape(batchSize,resolution,resolution,3)

            generatedImages = generator.predict(noise)

            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.random.rand(batchSize*2)*0.2

            # One-sided label smoothing
            yDis[0:batchSize] = (np.random.rand(batchSize)*0.5)+0.7

            # Train discriminator

            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)
            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 5 == 0:
            plotGeneratedImages(e)
            
    saveModels(e)

"""
To start training from scratch, set start-point to 0. Otherwise, resume training
to a specified save point with the corresponding epoch.
"""        
if __name__ == '__main__':
    buildGan(randomDim)
    #printModels() #Uncomment this to view layer breakdown at start of training
    train(50, 64, 50)