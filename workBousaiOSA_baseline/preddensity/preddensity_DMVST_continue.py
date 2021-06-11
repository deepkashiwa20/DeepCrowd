import pandas as pd
import datetime
import sys
import shutil

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard

import model_structure
from load_data import data_generator, test_generator, get_test_true
from Param_DMVST import *


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_model_structure(name):
    model = model_structure.get_model(name)
    model.summary()

    # model_json = model.to_json()
    # with open('model_data/structure/' + name + '.json', "w") as json_file:
    #     json_file.write(model_json)

    return model


def get_data(model_name):
    print('loading data...')
    if model_name == 'density':
        region_window = np.load(local_density_path)
        topo_data = np.loadtxt(topo_density_path, skiprows=1, usecols=range(1, toponet_len + 1))
    temporal_data = np.loadtxt(temporal_path, skiprows=1, delimiter=',')
    print('population data', region_window.shape)
    print('temporal data', temporal_data.shape)
    print('topo data', topo_data.shape)

    region_window = region_window / MAX_VALUE

    startIndex, endIndex = 0, int(int(region_window.shape[0] * trainRatio) * (1 - SPLIT))
    trainData = region_window[startIndex:endIndex, :, :, :, :]
    trainTemporal = temporal_data[startIndex:endIndex]
    print('train data', trainData.shape)

    startIndex, endIndex = int(int(region_window.shape[0] * trainRatio) * (1 - SPLIT)), int(region_window.shape[0] * trainRatio)
    validData = region_window[startIndex:endIndex, :, :, :, :]
    validTemporal = temporal_data[startIndex:endIndex]
    print('valid data', validData.shape)

    startIndex, endIndex = int(region_window.shape[0] * trainRatio), region_window.shape[0]
    testData = region_window[startIndex:endIndex, :, :, :, :]
    testTemporal = temporal_data[startIndex:endIndex]
    print('test data', testData.shape)
    print('load finished')

    return trainData, validData, testData, trainTemporal, validTemporal, testTemporal, topo_data


def model_train(model_name, train_data, valid_data, trainTemporal, validTemporal, topo_data):
    # set callbacks
    csv_logger = CSVLogger(PATH + '/' + MODELNAME + '.log')
    checkpointer_path = PATH + '/' + MODELNAME + '.h5'
    checkpointer = ModelCheckpoint(filepath=checkpointer_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    LearnRate = LearningRateScheduler(lambda epoch: LR)

    # data generator
    train_generator = data_generator(train_data, trainTemporal, topo_data, BATCHSIZE, TIMESTEP, model_name)
    val_generator = data_generator(valid_data, validTemporal, topo_data, BATCHSIZE, TIMESTEP, model_name)
    sep = (train_data.shape[0] - TIMESTEP) * train_data.shape[1] // BATCHSIZE
    val_sep = (valid_data.shape[0] - TIMESTEP) * valid_data.shape[1] // BATCHSIZE

    # train model
    model = get_model_structure(model_name)
    # model = multi_gpu_model(model, gpus=2)  # gpu parallel
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.fit_generator(train_generator, steps_per_epoch=sep, epochs=EPOCH,
                        validation_data=val_generator, validation_steps=val_sep,
                        callbacks=[csv_logger, checkpointer, LearnRate, early_stopping])

    # compute mse
    val_nolabel_generator = test_generator(valid_data, validTemporal, topo_data, BATCHSIZE, TIMESTEP)
    val_predY = model.predict_generator(val_nolabel_generator, steps=val_sep)
    valY = get_test_true(valid_data, TIMESTEP, model_name)
    # mse
    scaled_valY = np.reshape(valY, ((valid_data.shape[0] - TIMESTEP), HEIGHT, WIDTH))
    scaled_predValY = np.reshape(val_predY, ((valid_data.shape[0] - TIMESTEP), HEIGHT, WIDTH))
    print('val scale shape: ', scaled_predValY.shape)
    val_scale_MSE = np.mean((scaled_valY - scaled_predValY) ** 2)
    print("Model val scaled MSE", val_scale_MSE)
    # rescale mse
    val_rescale_MSE = val_scale_MSE * MAX_VALUE ** 2
    print("Model val rescaled MSE", val_rescale_MSE)

    # write record
    with open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a') as wf:
        wf.write('train start time: {}\n'.format(StartTime))
        wf.write('train end time:   {}\n'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        wf.write("Keras MSE on trainData, %f\n" % val_scale_MSE)
        wf.write("Rescaled MSE on trainData, %f\n" % val_rescale_MSE)


def model_pred(model_name, test, testTemporal, topo_data):
    # test generator
    test_gene = test_generator(test, testTemporal, topo_data, BATCHSIZE, TIMESTEP)
    test_sep = (test.shape[0] - TIMESTEP) * test.shape[1] // BATCHSIZE

    # get predict
    model = get_model_structure(model_name)
    # model = multi_gpu_model(model, gpus=2)  # gpu parallel
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.load_weights(PATH + '/' + MODELNAME + '.h5')
    predY = model.predict_generator(test_gene, steps=test_sep)

    # ground truth
    testY = get_test_true(test, TIMESTEP, model_name)

    # compute mse
    scaled_testY = np.reshape(testY, ((test.shape[0] - TIMESTEP), HEIGHT, WIDTH))
    scaled_predTestY = np.reshape(predY, ((test.shape[0] - TIMESTEP), HEIGHT, WIDTH))
    print('test scale shape: ', scaled_predTestY.shape)
    scale_MSE = np.mean((scaled_testY - scaled_predTestY) ** 2)
    print("Model scaled MSE", scale_MSE)

    rescale_MSE = scale_MSE * MAX_VALUE ** 2
    print("Model rescaled MSE", rescale_MSE)

    with open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a') as wf:
        wf.write("Keras MSE on testData, %f\n" % scale_MSE)
        wf.write("Rescaled MSE on testData, %f\n" % rescale_MSE)

    np.save(PATH + '/' + MODELNAME + '_prediction.npy', scaled_predTestY * MAX_VALUE)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', scaled_testY * MAX_VALUE)


################# Path Setting #######################
MODELNAME = 'DMVST'
# KEYWORD = 'preddensity_' + MODELNAME + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
# KEYWORD = 'preddensity_DMVST_1907310039'
# KEYWORD = 'preddensity_DMVST_1908010150'
KEYWORD = 'preddensity_DMVST_1908221706'
PATH = '../' + KEYWORD
###########################Reproducible#############################
import numpy as np
import random
from keras import backend as K
import os
import tensorflow as tf

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3

tf.set_random_seed(100)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.45
session_conf.gpu_options.visible_device_list = '2'
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###################################################################

if __name__ == '__main__':

    mkdir(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param_DMVST.py', PATH)
    shutil.copy2('model_structure.py', PATH)
    shutil.copy2('load_data.py', PATH)
    shutil.copy2('preprocess.py', PATH)
    StartTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    choose = 0
    if choose == 0:
        # density
        model_name = 'density'

    print('#' * 50)
    print('start running at {}'.format(StartTime))
    print('model name: {}'.format(model_name))
    print('#' * 50, '\n')

    train_data, valid_data, test_data, trainTemporal, validTemporal, testTemporal, topo_data = get_data(model_name)
    model_pred(model_name, test_data, testTemporal, topo_data)
