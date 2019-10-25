
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt 

import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
import keras
import datetime
import sys  
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.activations import relu, tanh, elu
from keras.optimizers import Adagrad, Adam, Nadam, SGD
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os
import gc


import cv2
from cv2 import resize

from keras.metrics import top_k_categorical_accuracy

from sklearn.metrics import accuracy_score

from main_functions import Pretrained_ResNet50, Pretrained_Xception, plot_confusion_matrix, Pretrained_MobileNet, preprocess_input

#####################################

# Parameters

classes = 10
weight_decay=1e-5
lr = 1e-4
epochs = 250
decay = lr/epochs

np.random.seed(7)


#####################################

# The test set used for all the models 

data_generator_test = ImageDataGenerator(
    featurewise_center=False,  
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
     data_format='channels_last',
   preprocessing_function=preprocess_input
)


test_generator = data_generator_test.flow_from_directory(
    '/home/basmatamer/Desktop/Project1_AyaBasma/new_data_set/testing', shuffle=False, target_size = (250,250), batch_size = 20)  

#####################################

# Get all the models

MobileNet_model = Pretrained_MobileNet()
Xception_model = Pretrained_Xception()
ResNet50_model = Pretrained_ResNet50()


#######################################

# Get individual Predictions and average for Stacking ensemble 

DATA_DIR = '/home/basmatamer/Desktop/Project1_AyaBasma/new_data_set/testing'

y_test = []
y_pred = []


for class_name in os.listdir(DATA_DIR):
    print(class_name)
    for image_name in os.listdir(os.path.join(DATA_DIR, class_name)):
        image = cv2.imread(os.path.join(DATA_DIR, class_name, image_name)).astype('float')
        
        smaller_image = resize(image, (299, 299))
                
        mn_image = np.expand_dims(preprocess_input(resize(smaller_image, (224, 224))), 0)
        
        xc_image = np.expand_dims(preprocess_input(smaller_image), 0)
        
        resnet_image = np.expand_dims(resize(smaller_image, (224, 224)), 0)        
                
        
        mn_pred = MobileNet_model.predict(mn_image)
        xc_pred = Xception_model.predict(xc_image)
        resnet_pred = ResNet50_model.predict(resnet_image)
        
        preds = (mn_pred + xc_pred + resnet_pred) / 3
        
        
        prediction = np.argmax(preds, axis=1)[0]
        y_pred.append(prediction)
        y_test.append(test_generator.class_indices[class_name])
        

y_pred = np.array(y_pred)
y_test = np.array(y_test)

print(accuracy_score(y_test, y_pred))

###########################################

# Plot the confusion matrix 

CM = confusion_matrix(y_test, y_pred).astype('float')
np.set_printoptions(precision=2)
plt.figure(figsize=(20,20))

CM = CM.astype('float') / CM.sum(axis = 1)[:, np.newaxis]
CM = np.round_(CM, decimals=2)

classes_names = ["Abstract_Expressionism", "Art_Nouveau_Modern", "Color_Field_Painting", "Cubism", "Fauvism",
                "Impressionism", "Naive_Art_Primitivism", "Romanticism", "Symbolism", "Ukiyo_e"]
plot_confusion_matrix(CM, classes_names)

plt.show()




