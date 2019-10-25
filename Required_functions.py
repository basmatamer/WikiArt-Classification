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
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.activations import relu, tanh, elu
from keras.optimizers import Adagrad, Adam, Nadam, SGD
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def Pretrained_ResNet50():    
    # Constructing the model and loading the weights 
    name = "ResNet50_2"
    classes = 10
    weight_decay=1e-5
    lr = 3e-4
    epochs = 50
    decay = lr/epochs
    batch_size = 100
    image_size = 224  
    
    feature_extractor= ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet',
    pooling = 'avg', classes=10)

    for layer in feature_extractor.layers[:]:
        layer.trainable = False

    classifier = feature_extractor.output
    classifier = Dropout(0.5)(classifier)
    logits1 = Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (classifier)
    logits2 = Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (logits1)
    logits3 = Dense(classes, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (logits2)
    probabilities = Activation('softmax') (logits3)

    full_model = Model(feature_extractor.input, probabilities)
    
    optimizer=Nadam(lr=lr,beta_1=0.9, beta_2 = 0.999, epsilon = 1e-8, schedule_decay=decay)
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy', top_2_accuracy])
    
    full_model.load_weights("my_best_weights_ResNet50_2_training.hdf5")

    
    return full_model



def loss_for_Xception(y_true, y_pred):
    entropy = -K.mean(K.sum(y_pred*K.log(y_pred), 1))
    beta = 0.1
    return categorical_crossentropy(y_true, y_pred) - beta*entropy




def Pretrained_Xception():
    # Constructing the Model and loading the weights
    classes = 10
    weight_decay=1e-5
    lr = 1e-3
    epochs = 50
    decay = lr/epochs
    batch_size = 50
    image_size = 299  
    feature_extractor= Xception(input_shape=(299, 299, 3), include_top=False, weights='imagenet',
    pooling = 'avg', classes=10)

    for layer in feature_extractor.layers[:]:
        layer.trainable = False

    classifier = feature_extractor.output
    classifier = Dropout(0.5)(classifier)
    logits1 = Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (classifier)
    logits2 = Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (logits1)
    logits3 = Dense(classes, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (logits2)
    probabilities = Activation('softmax') (logits3)

    full_model = Model(feature_extractor.input, probabilities)
    
    optimizer=SGD(lr=lr, momentum=0.9, nesterov=True)
    full_model.compile(loss=loss_for_Xception, optimizer=optimizer,   
                   metrics=['accuracy', top_2_accuracy])
    
    full_model.load_weights("my_best_weights_Xception_training.hdf5")

    
    return full_model


def Pretrained_MobileNet():   
    # Constructing the Model and loading the weights
    name = "MobileNet"
    classes = 10
    weight_decay=1e-5
    lr = 1e-3
    epochs = 50
    decay = lr/epochs
    batch_size = 50
    image_size = 224
    feature_extractor= MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet',
    pooling = 'avg', classes=10)

    for layer in feature_extractor.layers[:]:
        layer.trainable = False

    classifier = feature_extractor.output
    classifier = Dropout(0.5)(classifier)
    logits1 = Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (classifier)
    logits2 = Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (logits1)
    logits3 = Dense(classes, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)) (logits2)
    probabilities = Activation('softmax') (logits3)

    full_model = Model(feature_extractor.input, probabilities)
    
    optimizer=SGD(lr=lr, momentum=0.9, nesterov=True)
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy', top_2_accuracy])
    
    
    full_model.load_weights("my_best_weights_MobileNet_training.hdf5")

    
    return full_model
    
    

# Function that plots and prints the confusion matrix 
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap= plt.cm.Blues ):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    #print (cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 #transform = ax.transAxes,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    




def preprocess_input(x):
    x_new = x.copy()
    x_new /= 255.0
    x_new -= 0.5
    x_new *= 2.0
    return x_new


