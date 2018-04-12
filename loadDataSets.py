# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:38:11 2018

@author: Harrison Rose
"""
import numpy as np
import os
import cv2
from scipy import ndimage

def imageToArray(scaling=1, flipH = False):
    rawData = open("Labels\imgListTrainRegression_score.txt","r")
    splitData = rawData.read().split(",")
    setLen = len(splitData)
    arraySize = int(256*scaling)
    if flipH == True:
        trainY = np.zeros(setLen*2, dtype=np.float16)
        trainX = np.zeros((setLen*2, arraySize, arraySize, 3), dtype=np.int16)
    else:
        trainY = np.zeros(setLen, dtype=np.float16)
        trainX = np.zeros((setLen, arraySize, arraySize, 3), dtype=np.int16)
    for i in range(len(splitData)):
        if i%100 == 0:
            print(i)
        currentLine = splitData[i].split(" ")
        trainY[i] = float(currentLine[1])
        if scaling != 1:
            trainX[i] = cv2.resize(ndimage.imread(os.path.join("Images", currentLine[0]),mode = 'RGB'),(0,0), fx = scaling, fy = scaling).astype(int)
        else:
            trainX[i] = ndimage.imread(os.path.join("Images", currentLine[0]),mode = 'RGB').astype(int)
        if flipH == True:
            trainY[i+setLen] = float(currentLine[1])
            trainX[i+setLen] = np.flip(trainX[i],1)

    np.save("trainX_scale%d_.npy" %(1/scaling) , trainX)
    np.save("trainY.npy",trainY)

imageToArray(0.5,True)