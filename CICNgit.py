import os
import numpy as np
from keras.preprocessing.image import *
from keras.models import Model,Sequential
from keras.layers import Dense,Dropout,Activation,Conv2D, MaxPooling2D,Flatten,Input,BatchNormalization,AveragePooling2D,concatenate,ZeroPadding2D,add
from keras. optimizers import SGD
from keras.utils import np_utils,multi_gpu_model
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
import csv
import pandas as pd
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def load_img_label_data(path0,path1):
    img0=os.listdir(path0)
    img1=os.listdir(path1)
    acetic=[]
    label=[]
    for i in range(len(img0)):
        my_img = load_img(path0+'/' + img0[i])
        interimage = img_to_array(my_img, data_format='channels_last')


        acetic.append(interimage)
        label.append(int(0))
    for i in range(len(img1)):
        my_img = load_img(path1+'/' + img1[i])
        interimage = img_to_array(my_img, data_format='channels_last')

        acetic.append(interimage)
        label.append(int(1))

    print(label)
    print(len(label))
    acetic = np.array(acetic)
    print(acetic.shape)
    y_pre=label
    label = np_utils.to_categorical(label, 2)
    acetic = acetic.astype('float32')
    acetic /= 255

    x_train,x_test,y_train,y_test=train_test_split(acetic,label,test_size=0.3, random_state=1)

    x_train1, x_test1, y_train1, y_test1 = train_test_split(acetic, y_pre, test_size=0.3, random_state=1)

    return x_train,x_test,y_train,y_test,y_test1


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def cnn_model(x_train,x_test,y_train,y_test):
    datagen=ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range = 0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               )
    datagen.fit(x_train)

    # model = multi_gpu_model(model, gpus=2)

    inpt = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model = multi_gpu_model(model, gpus=2)

    # model = multi_gpu_model(model,4)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),steps_per_epoch=100,epochs=30,validation_data=(x_test,y_test),verbose=1)

    model.save('/home/som/lab/seed-yzj/paper3/newmodel/resacetic_model.h5')

    sscores = model.predict(x_test, batch_size=16)
    return history,sscores

def roc(lstm_predictions,y_pre_temp):
    lstm_predictions=lstm_predictions[:,1]
    print(lstm_predictions)
    fpr, tpr, threshold = roc_curve(y_pre_temp, lstm_predictions)
    roc_auc = auc(fpr, tpr)

    csv_file1 = open('/home/som/lab/seed-yzj/paper3/result/classification/resfprcsv.csv', 'w', newline='')
    writer = csv.writer(csv_file1)
    writer.writerow(fpr)
    csv_file1.close()

    csv_file2 = open('/home/som/lab/seed-yzj/paper3/result/classification/restprcsv.csv', 'w', newline='')
    writer = csv.writer(csv_file2)

    writer.writerow(tpr)
    csv_file2.close()
    print(roc_auc)


def sen_spe(y_test,y_score,n_classes,save_path= None):
    # 计算每一类的ROC
    sen = [] #特异性
    spe = []  #敏感性
    y_pre = score_to_pre(y_score)
    for i in range(n_classes):
        temp_sen, temp_spe= cal_sen_spe(y_test[:, i], y_pre[:, i])
        sen.append(temp_sen)
        spe.append(temp_spe)
    avg_sen = sum(sen)/n_classes
    avg_spe = sum(spe)/n_classes
    print('sen:', sen)
    print('spe:', spe)
    print('\n avg_spe:%.6f, \n' % (avg_spe))
    print('\n avg_sen:%.6f, \n' % (avg_sen))
    if save_path == None:
        #excel 保存数据，后期画图
        pass
    else:
        pass
    return spe,sen,avg_sen,avg_spe

def cal_sen_spe(y_test, y_pre):
    T = sum(y_test)
    F = len(y_test) - T
    sen = sum(list(map(lambda x, y: 1 if x == y and x==1 else 0,y_test, y_pre)))/ T
    spe = sum(map(lambda x, y: 1 if x == y and x == 0 else 0,y_test, y_pre)) / F
    return sen ,spe

def score_to_pre(y):
    """
    根据概率计算预测值
    :param y:
    :return:
    """
    tmax = np.argmax(y,1)
    y_score = np.zeros_like(y)
    for i in range(len(tmax)):
        y_score[i][tmax[i]] = 1
    return y_score

def drawlines(history):
    history_dict=history.history
    csv_file = open('/home/som/lab/seed-yzj/paper3/result/classification/res.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    for key in history_dict:
        writer.writerow([key, history_dict[key]])
    csv_file.close()

def metrics(y_true,y_pred):

    y_final=[]
    for i in range(y_pred.shape[0]):
        if y_pred[i][0]>y_pred[i][1]:
            y_final.append(int(0))
        else:
            y_final.append(int(1))
    print(y_final)
    classify_report = classification.classification_report(y_true, y_final)

    print('classify_report : \n', classify_report)

    return 0

if __name__ == '__main__':
    path0 = '/home/som/lab/seed-yzj/paper3/cervicaldata/movecervixbinary/0/'
    path1='/home/som/lab/seed-yzj/paper3/cervicaldata/movecervixbinary/1/'


    x_train, x_test, y_train, y_test ,y_pre= load_img_label_data(path0,path1)

    history,y_prediction=cnn_model(x_train,x_test,y_train,y_test)
    spe, sen, avg_sen, avg_spe=sen_spe(y_test,y_prediction,2)
    drawlines(history)
    roc(y_prediction,y_pre)
    metrics(y_pre, y_prediction)