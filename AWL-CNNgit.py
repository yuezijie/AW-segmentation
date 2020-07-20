import os
import numpy as np
from keras.preprocessing.image import *
from keras.models import Model,Sequential
from keras.layers import *
from keras. optimizers import SGD
from keras.utils import np_utils,multi_gpu_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
import csv
import pandas as pd
from keras import regularizers
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



def load_data(cindatapath,cingtpath,cinheatpath):
    # normalimage=os.listdir(normaldatapath)
    cinimage=os.listdir(cindatapath)
    acetic = []
    groundtruth=[]
    heatmap=[]



    for i in range(len(cinimage)):
        my_img = load_img(cindatapath + '/' + cinimage[i])
        my_img= my_img.resize((256,256))
        interimage = img_to_array(my_img, data_format='channels_last')
        acetic.append(interimage)

        groundtruthimg=load_img(cingtpath+ cinimage[i],color_mode='grayscale')
        groundtruthimg = groundtruthimg.resize((256, 256))
        groundtruthimg = img_to_array(groundtruthimg)

        groundtruthimg[groundtruthimg > 0.5] = 1
        groundtruthimg[groundtruthimg <= 0.5] = 0


        groundtruth.append(groundtruthimg)

        heat=load_img(cinheatpath+ cinimage[i],color_mode='grayscale')
        heat = heat.resize((256, 256))
        heatnpy = img_to_array(heat)
        # heatnpy=1-heatnpy
        heatmap.append(heatnpy)

    acetic = np.array(acetic)
    originalacetic=acetic
    groundtruth=np.array(groundtruth)
    heatmap=np.array(heatmap)

    acetic = acetic.astype('float32')
    groundtruth = groundtruth.astype('float32')
    heatmap = heatmap.astype('float32')
    acetic /= 255
    heatmap /=  255

    return acetic,groundtruth,heatmap,originalacetic


def resunet(acetic, groundtruth,valacetic,valgroundtruth, heatmap,valheatmap):
    data_gen= ImageDataGenerator(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    # seed = 1
    heatinput=heatmap
    heattestinput=valheatmap

    def gen_flow_for_two_inputs(X1, X2, y):
        genX1 = data_gen.flow(X1, y, batch_size=8, seed=1)
        genX2 = data_gen.flow(X1, X2, batch_size=8, seed=1)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            # Assert arrays are equal - this was for peace of mind, but slows down training
            # np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

    gen_flow = gen_flow_for_two_inputs(acetic, heatinput, groundtruth)

    #input
    inputs = Input(shape=(256, 256, 3))
    heatmapinput=Input(shape=(256, 256,1))

    #block 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(inputs)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation(activation='relu')(bn1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(act1)


    heat1=Multiply()([heatmapinput,conv2])


    shotcutconv1=Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(inputs)
    shortcutbn1 = BatchNormalization()(shotcutconv1)
    add1=add([shortcutbn1, conv2,heat1])

    #block2

    bn2=BatchNormalization()(add1)
    act2=Activation(activation='relu')(bn2)
    conv3=Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(2, 2))(act2)
    bn3=BatchNormalization()(conv3)
    act3=Activation(activation='relu')(bn3)
    conv4=Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(act3)

    heatmap2=MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(heatmapinput)
    heat2 = Multiply()([heatmap2, conv4])

    shotcutconv2 = Conv2D(128, kernel_size=(1, 1), strides=(2, 2))(add1)
    shortcutbn2 = BatchNormalization()(shotcutconv2)
    add2 = add([shortcutbn2, conv4,heat2])

    # block3

    bn4 = BatchNormalization()(add2)
    act4 = Activation(activation='relu')(bn4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(2, 2))(act4)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation(activation='relu')(bn5)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1))(act5)

    heatmap3=MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(heatmap2)
    heat3 = Multiply()([heatmap3, conv6])

    shotcutconv3 = Conv2D(256, kernel_size=(1, 1), strides=(2, 2))(add2)
    shortcutbn3 = BatchNormalization()(shotcutconv3)
    add3 = add([shortcutbn3, conv6,heat3])

    # block4

    bn6 = BatchNormalization()(add3)
    act6 = Activation(activation='relu')(bn6)
    conv7 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(2, 2))(act6)
    bn7 = BatchNormalization()(conv7)
    act7 = Activation(activation='relu')(bn7)
    conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1, 1))(act7)


    heatmap2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(heatmapinput)
    heatmap3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(heatmap2)
    heatmap4=MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(heatmap3)
    heat4 = Multiply()([heatmap4, conv8])

    shotcutconv4 = Conv2D(512, kernel_size=(1, 1), strides=(2, 2))(add3)
    shortcutbn4 = BatchNormalization()(shotcutconv4)
    add4 = add([shortcutbn4, conv8,heat4])

    #block5
    ups1 = UpSampling2D(size=(2, 2))(add4)
    conca1 = concatenate([ups1, add3], axis=3)

    bn8 = BatchNormalization()(conca1)
    act8 = Activation(activation='relu')(bn8)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1))(act8)
    bn9 = BatchNormalization()(conv9)
    act9 = Activation(activation='relu')(bn9)
    conv10 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1))(act9)

    heat5 = Multiply()([heatmap3, conv10,])


    shotcutconv5 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1))(conca1)
    shortcutbn5 = BatchNormalization()(shotcutconv5)
    add5 = add([shortcutbn5, conv10,heat5])


    #block6
    ups2 = UpSampling2D(size=(2, 2))(add5)
    conca2 = concatenate([ups2, add2], axis=3)

    bn10 = BatchNormalization()(conca2)
    act10 = Activation(activation='relu')(bn10)
    conv11 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(act10)
    bn11 = BatchNormalization()(conv11)
    act11 = Activation(activation='relu')(bn11)
    conv12 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(act11)

    heat6 = Multiply()([heatmap2, conv12])

    shotcutconv6 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1))(conca2)
    shortcutbn6 = BatchNormalization()(shotcutconv6)
    add6 = add([shortcutbn6, conv12,heat6])

    #block7
    ups3 = UpSampling2D(size=(2, 2))(add6)
    conca3 = concatenate([ups3, add1], axis=3)

    bn12 = BatchNormalization()(conca3)
    act12 = Activation(activation='relu')(bn12)
    conv13 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(act12)
    bn13 = BatchNormalization()(conv13)
    act13 = Activation(activation='relu')(bn13)
    conv14 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(act13)

    heat7 = Multiply()([heatmapinput, conv14])

    shotcutconv7 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(conca3)
    shortcutbn7 = BatchNormalization()(shotcutconv7)
    add7 = add([shortcutbn7, conv14,heat7])

    out = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(add7)

    model = Model(input=[inputs,heatmapinput], output=out)

    model = multi_gpu_model(model, gpus=2)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint('/home/som/lab/seed-yzj/paper3/newmodel/mynet.hdf5', monitor='loss', verbose=1, save_best_only=True)

    history=model.fit_generator(gen_flow,steps_per_epoch=300,epochs=10, callbacks=[model_checkpoint],validation_data=([valacetic,valheatmap],valgroundtruth))

    pred=model.predict([valacetic,valheatmap])
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return history,pred



def dice(y_true, y_pred):
    y_true=y_true.reshape(y_true.shape[0],y_true.shape[1],y_true.shape[2])
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], y_pred.shape[2])
    avgcross=0
    for i in range(y_true.shape[0]):
        cross=truenumber=prednumber=0
        for j in range(y_true.shape[1]):
            for m in range(y_true.shape[2]):
                if y_true[i][j][m]==y_pred[i][j][m]==1:
                    cross+=1
                if y_true[i][j][m]==1:
                    truenumber+=1
                if y_pred[i][j][m]==1:
                    prednumber+=1
        dicenumber=(2*cross)/(truenumber+prednumber)
        avgcross+=dicenumber
        print("dice系数：",dicenumber)
    avgcross=avgcross/y_true.shape[0]
    print("平均dice系数",avgcross)

def iou(y_true, y_pred):
    y_true=y_true.reshape(y_true.shape[0],y_true.shape[1],y_true.shape[2])
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], y_pred.shape[2])
    avgiou = 0

    for i in range(y_true.shape[0]):
        Intersection = Union = 0

        for j in range(y_true.shape[1]):
            for m in range(y_true.shape[2]):
                if y_true[i][j][m]==y_pred[i][j][m]==1:
                    Intersection+=1
                if y_true[i][j][m]==1 or pred[i][j][m]==1:
                    Union+=1

        totaliou=Intersection/Union
        avgiou+=totaliou
        print("IOU值：", totaliou)

    avgiou=avgiou/y_true.shape[0]
    print("平均IOU", avgiou)



def drawlines(history):
    history_dict=history.history
    csv_file = open('/home/som/lab/seed-yzj/paper3/result/mynet.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    for key in history_dict:
        writer.writerow([key, history_dict[key]])
    csv_file.close()


def savepred(testacetic,pred,save_path):
    pred *= 255

    for i in range(testacetic.shape[0]):

        acetic = cv2.cvtColor(testacetic[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + str(i) + '.jpg', acetic)
        # cv2.imwrite(save_path + str(i) + '.png', pred[i])
        cv2.imwrite(save_path + str(i) + '.png', pred[i])



if __name__ == '__main__':
    # normaldatapath = '/home/som/lab/seed-yzj/paper3/cervicaldata/binary/data/0/'
    cindatapath='/home/som/lab/seed-yzj/paper3/cervicaldata/newbinary/train/data/1/'
    # normalgtpath='/home/som/lab/seed-yzj/paper3/cervicaldata/binary/groundtruth/0/'
    cingtpath='/home/som/lab/seed-yzj/paper3/cervicaldata/newbinary/train/groundtruth/1/'
    # normalmaskpath='/home/som/lab/seed-yzj/paper3/cervicaldata/binary/mask/0/'
    cinheatpath='/home/som/lab/seed-yzj/paper3/cervicaldata/newbinary/train/heatmap/1/'

    testdatapath='/home/som/lab/seed-yzj/paper3/cervicaldata/newbinary/test/data/1/'
    testgtpath='/home/som/lab/seed-yzj/paper3/cervicaldata/newbinary/test/groundtruth/1/'
    testheatpath='/home/som/lab/seed-yzj/paper3/cervicaldata/newbinary/test/heatmap/1/'


    save_path='/home/som/lab/seed-yzj/paper3/result/mynet_pred/'
    acetic, groundtruth, heatmap,originalacetic=load_data(cindatapath,cingtpath,cinheatpath)

    valacetic,valgroundtruth,valheatmap,originalvalacetic=load_data(testdatapath,testgtpath,testheatpath)


    history, pred=resunet(acetic, groundtruth,valacetic,valgroundtruth, heatmap,valheatmap)
    drawlines(history)
    dice(testgroundtruth, pred)
    iou(testgroundtruth, pred)
