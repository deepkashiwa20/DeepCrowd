import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
from Param import *

def getXSYS(allData):
    XS, YS = [], []
    for i in range(allData.shape[0] - TIMESTEP):
        x = allData[i:i+TIMESTEP, :, :, :]
        y = allData[i+TIMESTEP, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    if MODELNAME == 'CNN':
        XS = XS.swapaxes(4, 1)
        XS = np.squeeze(XS)
    else:
        pass
    return XS, YS

def testModel(name, testData):
    print('Model Evaluation Started ...', time.ctime())
    XS, YS = getXSYS(testData)
    print(XS.shape, YS.shape)
    predYS = XS[:, -1, :, :, :]
    keras_score = np.mean((YS - predYS) ** 2)
    rescaled_MSE = keras_score * MAX_DENSITY * MAX_DENSITY

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on testData, %f\n" % keras_score)
    f.write("Rescaled MSE on testData, %f\n" % rescaled_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)
    print('Model Evaluation Ended ...', time.ctime())

    pred = predYS * MAX_DENSITY
    groundtruth = YS * MAX_DENSITY
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', groundtruth)

def trainModel(name, trainvalidateData):
    print('Model Training Started ...', time.ctime())
    XS, YS = getXSYS(trainvalidateData)
    print(XS.shape, YS.shape)
    predYS = XS[:, -1, :, :, :]
    keras_score = np.mean((YS - predYS) ** 2)
    rescaled_MSE = keras_score * MAX_DENSITY * MAX_DENSITY

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on trainData, %f\n" % keras_score)
    f.write("Rescaled MSE on trainData, %f\n" % rescaled_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)
    print('Model Training Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'CopyLastFrame'
KEYWORD = 'preddensity_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../' + KEYWORD
################# Parameter Setting #######################

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)

    data = np.load(dataFile)
    print('data.shape', data.shape)
    train_Num = int(data.shape[0] * trainRatio)

    print(KEYWORD, 'training started', time.ctime())
    trainvalidateData = data[:train_Num, :, :, :]
    print('trainvalidateData.shape', trainvalidateData.shape)
    trainvalidateData = trainvalidateData / MAX_DENSITY
    trainModel(MODELNAME, trainvalidateData)

    print(KEYWORD, 'testing started', time.ctime())
    testData = data[train_Num:, :, :, :]
    print('testData.shape', testData.shape)
    testData = testData / MAX_DENSITY
    testModel(MODELNAME, testData)


if __name__ == '__main__':
    main()