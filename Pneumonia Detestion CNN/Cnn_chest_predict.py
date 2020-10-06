# %%
import re
import os
import numpy as np

from tensorflow.python import keras
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

# %%
BATCH_SIZE = 32
IMAGE_SIZE = 250
EPOCHS = 200

TRAIN_DIR = "E:\\NEU\\ALY6020\\Final Project\\chest_xray\\train"
TEST_DIR = "E:\\NEU\ALY6020\\Final Project\\chest_xray\\test"
VAL_DIR = "E:\\NEU\ALY6020\\Final Project\\chest_xray\\val"
print(os.listdir(TRAIN_DIR))


"""
    ###########################################
    ### ===== Image Data Augmentation ===== ###
              - Image Preprocessing -
    ###########################################
"""

# ImageDataGerator for image preprocessing as image generator
train_datagen = ImageDataGenerator(rescale = 1./255,     # pixel value from 1 to 255
                              zca_whitening = True,      # Apply ZCA whitening with default
                              width_shift_range = 0.1,   # 0.1 fraction of total width
                              height_shift_range = 0.1,  # 0.1 fraction of total height
                              shear_range = 0.2,         # Shear Intensity 
                              #horizontal_flip = True,
                              fill_mode ='nearest') 

test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

# Apply flow_from_direction() function to data preprocessing
# Take the path to a directory & generates batches of augmented data.
# Return: 
#      A DirectoryIterator yielding tuples of (x, y) 
#      where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) 
#      and y is a numpy array of corresponding labels.
train_datagen = train_datagen.flow_from_directory(TRAIN_DIR,
                                            target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                            color_mode = 'grayscale', 
                                            # color_mode = 'rgb',
                                            class_mode = 'categorical',   # return one-hot encoder
                                            batch_size = BATCH_SIZE)

test_datagen = test_datagen.flow_from_directory(TEST_DIR,
                                            target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                            color_mode = 'grayscale',
                                            # color_mode = 'rgb',
                                            class_mode = 'categorical',   # return one-hot encoder
                                            batch_size = BATCH_SIZE)

val_datagen = val_datagen.flow_from_directory(directory = VAL_DIR,
                                            target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                            color_mode = 'grayscale',
                                            # color_mode = 'rgb',
                                            class_mode = 'categorical',   # return one-hot encoder
                                            batch_size = BATCH_SIZE)


"""
    #########################################################################
    #### =========== Build Convolution Neural Network Model ============ ####
    #########################################################################
"""

# Define the steps per epoch
train_steps = train_datagen.samples // BATCH_SIZE
test_steps = test_datagen.samples // BATCH_SIZE
val_steps = val_datagen.samples // BATCH_SIZE

# accurate the input shape
if K.image_data_format() == 'channels_first':
    input_shape = (1, IMAGE_SIZE, IMAGE_SIZE)
    # input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
else:
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  
    # input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)  


def conv_model():
    '''
    Model for convolution neural network.
    @Output: output_model, the cnn model with input and output
    '''
    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    # img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    model = layers.Conv2D(64, 3, activation='relu', padding='same')(img_input)
    model = layers.Conv2D(64, 3, activation='relu', padding='same')(model)
    model = layers.MaxPooling2D()(model)

    # model = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(model)
    # model = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(model)
    # model = layers.BatchNormalization()(model)
    # model = layers.MaxPooling2D()(model)

    model = layers.Dropout(0.2)(model)
    model = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(model)
    model = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)

    model = layers.Dropout(0.3)(model)
    model = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(model)
    model = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)

    model = layers.Dropout(0.5)(model)
    model = layers.SeparableConv2D(256, 3, activation='relu', padding='same')(model)
    model = layers.SeparableConv2D(256, 3, activation='relu', padding='same')(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model = layers.Dropout(0.5)(model)

    model = layers.Flatten()(model)
    model = layers.Dense(512, activation='relu')(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(0.5)(model)

    # model = layers.Dense(128, activation='relu')(model)
    model = layers.Dense(512, activation='relu')(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(0.5)(model)

    model = layers.Dense(64, activation='relu')(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(0.5)(model)

    # model = layers.Dense(2, activation='sigmoid')(model)
    model = layers.Dense(2, activation='softmax')(model)
    
    output_model = Model(img_input, model)

    return output_model


# Correct the data imbalance
NOR_DIR = "E:\\NEU\\ALY6020\\Final Project\\chest_xray\\train\\NORMAL"
PNE_DIR = "E:\\NEU\\ALY6020\\Final Project\\chest_xray\\train\\PNEUMONIA"
nor_count = len(os.listdir(NOR_DIR))
pne_count = len(os.listdir(PNE_DIR))

initial_bias = np.log(pne_count / nor_count)
print('Model Initial Bias: {:f}'.format(initial_bias))


# Find weight of different categories to reduce the initial bias in the model
weight_for_0 = (1 / nor_count) * (train_datagen.samples) / 2.0 
weight_for_1 = (1 / pne_count) * (train_datagen.samples) / 2.0

class_w = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0 (Normal): {:.2f}'.format(weight_for_0))  # Normal
print('Weight for class 1 (Pneumonia): {:.2f}'.format(weight_for_1))  # Pneumonia


"""
    #######################################################################
    #### ============= Compiling and Training CNN Model ============== ####
    #######################################################################
"""

model = conv_model()

# Define evaluate metrics
METRICS = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
]

# Compile model with learning rate = 0.0001
model.compile(
    optimizer='adam',
    # optimizer = Adam(lr=0.00001),
    loss= 'categorical_crossentropy',
    #'binary_crossentropy',
    metrics=METRICS
)


# Training model
history = model.fit_generator(
    train_datagen,
    steps_per_epoch = train_steps,
    epochs = EPOCHS,
    validation_data = val_datagen,
    validation_steps = val_steps,
    class_weight = class_w,
)


fig, ax = plt.subplots(2, 2, figsize=(20, 15))
ax = ax.ravel()

for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])






# %%
