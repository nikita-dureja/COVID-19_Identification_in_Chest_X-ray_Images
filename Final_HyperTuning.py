#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Batch_size: 32, optimiser: Adam, epochs: 100, learning_rate: 0.001 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ## Pre-Trained model

# In[2]:


WEIGHTS_PATH = ('/home/srenchi/.keras/models/'
               'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('/home/srenchi/.keras/models/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def vgg16(image_height, image_width, channels, NUM_CLASSES, include_top=True, weights='imagenet', pooling=None):
    model = tf.keras.Sequential()
    # 1
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     activation=tf.keras.activations.relu,
                                     name='block1_conv1',
                                     input_shape=(image_height, image_width, channels)))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block1_conv2',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        name='block1_pool',
                                        padding='same'))

    # 2
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block2_conv1',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block2_conv2',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        name='block2_pool',
                                        padding='same'))

    # 3
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block3_conv1',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block3_conv2',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block3_conv3',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        name='block3_pool',
                                        padding='same'))

    # 4
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block4_conv1',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block4_conv2',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block4_conv3',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        name='block4_pool',
                                        padding='same'))

    # 5
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block5_conv1',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block5_conv2',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     name='block5_conv3',
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        name='block5_pool',
                                        padding='same'))

    if include_top:
        # Classification block
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=4096,
                                        activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(units=4096,
                                        activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dropout(rate=0.5))

        model.add(tf.keras.layers.Dense(units=NUM_CLASSES,
                                    activation=tf.keras.activations.softmax))
    else:
        if pooling == 'avg':
            model.add(tf.keras.layers.GlobalAveragePooling2D())
        elif pooling == 'max':
            model.add(tf.keras.layers.GlobalMaxPooling2D())

    if weights == 'imagenet':
        if include_top:
            weights_path = data_utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = data_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


# ## Loading training and validation data

# In[2]:

#Using same augmentation parameters that are used in data augmentation file
train_datagen = ImageDataGenerator(rescale = 1./255,#rescaling
                                  rotation_range=15, # rotation
                                  zoom_range=0.2, # zoom
                                  horizontal_flip=True, # horizontal flip
                                  width_shift_range=0.2, # horizontal shift
                                  height_shift_range=0.2, # vertical shift
                                  brightness_range=[0.2,1.2]) # brightness
#Loading training images
train_generator = train_datagen.flow_from_directory(
        'train/',  
        target_size=(224, 224),  
        batch_size= 32,
        class_mode='categorical', 
        shuffle= True,
        seed = 42)



# In[3]:


train_generator.class_indices #Indices for different image categories


# ## Training the model

# In[5]:


baseModel = vgg16(224, 224, 3, 3, include_top=False) #Calling pre-trained model
headModel = baseModel.output

#Adding layers
headModel = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = tf.keras.layers.Flatten(name='flatten')(headModel)
headModel = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(headModel)
headModel = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(headModel)
headModel = tf.keras.layers.Dropout(0.5)(headModel)
headModel = tf.keras.layers.Dense(3, activation='softmax')(headModel)

# create a model object
model = Model(inputs=baseModel.input, outputs=headModel)

# Existing weights not trained
for layer in baseModel.layers:
    layer.trainable = False

#structure of the model    
model.summary()


# ## Fitting the model on our dataset images

# In[6]:
opt = keras.optimizers.Adam(learning_rate=0.001)
#Loss and optimization method on model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#Fit the model on traing data
hist = model.fit_generator(
  train_generator,
  epochs=100)


# save model
print("saving model...")
model.save('Covid19_model.h5')


# The SGD optimizer does not work well for almost all hyper parameters and seems very sensitive to the learning rate.
# ADAM optimizer gives better accuracy than the SGD optimizer in all cases.
# ADAM optimizer performs the best for hyperparameters: batch_size : 32, Learning Rate : 0.001, Epochs : 100, Optimizer : adam
# giving us a validation accuracy of approximately 85% percent 
# when the number of epochs increases from 80 to 100 then ADAM optimizer tends to perform better with lower learning rate
# Adam optimizer is not very sensitive to learning rate as the accuracy changes only few percent when learning rate is changed
# the best hyperparamaters from the tuning is
# hyperparameters: batch_size : 32, Learning Rate : 0.001, Epochs : 100, Optimizer : adam

