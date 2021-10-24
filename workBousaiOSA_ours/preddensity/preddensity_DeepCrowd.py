import sys
import shutil
import os
import time
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import h5py
from copy import copy
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Activation, Flatten, Dense, Reshape, Concatenate, Add, Lambda, Layer, add, multiply, \
    Bidirectional, TimeDistributed, UpSampling2D, concatenate
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import keras.backend as K
from Param import *


def getXSYS_CPT_D(mode, allData, trainData, dayinfo):
    len_c, len_p, len_t = TIMESTEP, 1, 1
    interval_p, interval_t = 1, 7

    stepC = list(range(1, len_c + 1))
    periods, trends = [interval_p * DAYTIMESTEP * i for i in range(1, len_p + 1)], \
                      [interval_t * DAYTIMESTEP * i for i in range(1, len_t + 1)]
    stepP, stepT = [], []
    for p in periods:
        stepP.extend(list(range(p, p + len_c)))
    for t in trends:
        stepT.extend(list(range(t, t + len_c)))
    depends = [stepC, stepP, stepT]

    if mode == 'train':
        start = max(stepT)
        end = trainData.shape[0]
    elif mode == 'test':
        start = trainData.shape[0] + len_c
        end = allData.shape[0]
    else:
        assert False, 'invalid mode...'

    XC, XP, XT, YS, YD = [], [], [], [], []
    for i in range(start, end):
        x_c = [allData[i - j][np.newaxis, :, :, :] for j in depends[0]]
        x_p = [allData[i - j][np.newaxis, :, :, :] for j in depends[1]]
        x_t = [allData[i - j][np.newaxis, :, :, :] for j in depends[2]]
        x_c = np.concatenate(x_c, axis=0)
        x_p = np.concatenate(x_p, axis=0)
        x_t = np.concatenate(x_t, axis=0)
        x_c = x_c[::-1, :, :, :]
        x_p = x_p[::-1, :, :, :]
        x_t = x_t[::-1, :, :, :]
        d = dayinfo[i]
        y = allData[i]
        XC.append(x_c)
        XP.append(x_p)
        XT.append(x_t)
        YS.append(y)
        YD.append(d)
    XC, XP, XT, YS, YD = np.array(XC), np.array(XP), np.array(XT), np.array(YS), np.array(YD)

    return XC, XP, XT, YS, YD


def ConvLSTMs(x_dim):
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         input_shape=x_dim))
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    return model


def BidConvLSTMs(x_dim):
    model = Sequential()
    model.add(Bidirectional(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                       padding='same', return_sequences=True,
                                       input_shape=x_dim)))
    model.add(Bidirectional(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                       padding='same', return_sequences=True)))
    model.add(Bidirectional(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                                       padding='same', return_sequences=True)))
    return model


def PyConvLSTMs(x_input):
    x1 = ConvLSTM2D(32, (3, 3), strides=(2, 2), padding='same', return_sequences=True)(x_input)
    x2 = ConvLSTM2D(64, (3, 3), strides=(2, 2), padding='same', return_sequences=True)(x1)
    x3 = ConvLSTM2D(128, (3, 3), strides=(3, 3), padding='same', return_sequences=True)(x2)

    y3 = TimeDistributed(UpSampling2D(size=(3, 3)))(x3)
    z3 = ConvLSTM2D(128, kernel_size=(1, 1), padding='same', return_sequences=True)(x2)
    p3 = add([y3, z3])

    y2 = TimeDistributed(UpSampling2D(size=(2, 2)))(p3)
    z2 = ConvLSTM2D(128, kernel_size=(1, 1), padding='same', return_sequences=True)(x1)
    p2 = add([y2, z2])

    y1 = TimeDistributed(UpSampling2D(size=(2, 2)))(p2)
    z1 = ConvLSTM2D(128, kernel_size=(1, 1), padding='same', return_sequences=True)(x_input)
    p1 = add([y1, z1])
    return p1


class Attention(Layer):
    def __init__(self, bias=True, **kwargs):
        self.supports_masking = True
        self.name = 'Attention'
        self.bias = bias
        self.step_dim = 0
        self.features_dim = 0
        self.Height = 0
        self.Width = 0
        self.Filter = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 5

        self.W = self.add_weight(shape=input_shape[2:],
                                 initializer='glorot_normal',
                                 name='{}_W'.format(self.name))
        self.step_dim = input_shape[1]
        self.features_dim = input_shape[2] * input_shape[3] * input_shape[4]
        self.Height, self.Width, self.Filter = input_shape[2], input_shape[3], input_shape[4]
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name))
        else:
            self.b = None
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, step_dim, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.tile(a[:, :, np.newaxis, np.newaxis, np.newaxis], (1, 1, self.Height, self.Width, self.Filter))
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.Height, self.Width, self.Filter


def getModel(x_dim, meta_dim):
    # metadata fusion
    Xmeta = Input(shape=(meta_dim,))
    dens1 = Dense(units=10, activation='relu')(Xmeta)
    dens2 = Dense(units=TIMESTEP * WIDTH * HEIGHT * 1, activation='relu')(dens1)
    hmeta = Reshape((TIMESTEP, WIDTH, HEIGHT, 1))(dens2)

    XC = Input(shape=x_dim)
    XP = Input(shape=x_dim)
    XT = Input(shape=x_dim)
    XC_c = concatenate([XC, hmeta], axis=-1)
    XP_c = concatenate([XP, hmeta], axis=-1)
    XT_c = concatenate([XT, hmeta], axis=-1)
    hc = PyConvLSTMs(XC_c)
    hp = PyConvLSTMs(XP_c)
    ht = PyConvLSTMs(XT_c)
    a_hc = Attention()(hc)
    a_hp = Attention()(hp)
    a_ht = Attention()(ht)
    x = Lambda(lambda l: K.concatenate([i[:, np.newaxis, :, :, :] for i in l], axis=1))([a_hc, a_hp, a_ht])
    x = Attention()(x)
    X_hat = Conv2D(CHANNEL, (1, 1), padding='same', activation='relu')(x)

    # add2 = Add()([x, hmeta])
    # X_hat = Activation('relu')(x)

    model = Model(inputs=[XC, XP, XT, Xmeta], outputs=X_hat)
    return model


def testModel(name, allData, trainData, dayinfo):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = load_model(PATH + '/' + name + '.h5', custom_objects={'Attention': Attention})
    model.summary()

    XC, XP, XT, YS, YD = getXSYS_CPT_D('test', allData, trainData, dayinfo)
    print(XC.shape, XP.shape, XT.shape, YS.shape, YD.shape)

    keras_score = model.evaluate(x=[XC, XP, XT, YD], y=YS, verbose=1)
    rescale_MSE = keras_score * MAX_DENSITY * MAX_DENSITY

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on testData, %f\n" % keras_score)
    f.write("Rescaled MSE on testData, %f\n" % rescale_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescale_MSE)
    print('Model Evaluation Ended ...', time.ctime())

    pred = model.predict([XC, XP, XT, YD], verbose=1, batch_size=BATCHSIZE) * MAX_DENSITY
    groundtruth = YS * MAX_DENSITY
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', groundtruth)


def trainModel(name, allData, trainData, dayinfo):
    print('Model Training Started ...', time.ctime())
    XC, XP, XT, YS, YD = getXSYS_CPT_D('train', allData, trainData, dayinfo)
    print(XC.shape, XP.shape, XT.shape, YS.shape, YD.shape)

    model = getModel((TIMESTEP, HEIGHT, WIDTH, CHANNEL), dayinfo.shape[1])
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(x=[XC, XP, XT, YD], y=YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR, early_stopping], validation_split=SPLIT)

    keras_score = model.evaluate(x=[XC, XP, XT, YD], y=YS, verbose=1)
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
MODELNAME = 'VLUC-final'
KEYWORD = 'preddensity_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../' + KEYWORD
################# Parameter Setting #######################

###########################Reproducible#############################
import random

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.set_random_seed(100)
###################################################################

def main():
    param = sys.argv
    if len(param) == 2:
        GPU = param[-1]
    else:
        GPU = '0'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
    set_session(tf.Session(graph=tf.get_default_graph(), config=config))

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)

    data = np.load(dataFile)
    data = data / MAX_DENSITY
    dayinfo = np.genfromtxt(dataPath + '/day_information_onehot.csv', delimiter=',', skip_header=1)
    print('data.shape, dayinfo.shape', data.shape, dayinfo.shape)
    train_Num = int(data.shape[0] * trainRatio)

    print(KEYWORD, 'training started', time.ctime())
    trainvalidateData = data[:train_Num, :, :, :]
    print('trainvalidateData.shape', trainvalidateData.shape)
    trainModel(MODELNAME, data, trainvalidateData, dayinfo)

    print(KEYWORD, 'testing started', time.ctime())
    testData = data[train_Num:, :, :, :]
    print('testData.shape', testData.shape)
    testModel(MODELNAME, data, trainvalidateData, dayinfo)


if __name__ == '__main__':
    main()
