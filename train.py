import joblib
import os
import sys
from glob import glob

import numpy as np
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.models import load_model

def train():
    train_datagen = ImageDataGenerator(
            #rotation_range=40,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            rescale=1./255,
            #shear_range=0.2,
            zoom_range=0.2,
            #horizontal_flip=True,
            fill_mode='nearest')

    # Figure out how many classes there should be.
    nb_class = len(os.listdir('/volumes/data/data/train'))

    if nb_class < 2:
        print('There should be at least 2 classes to predict on.')
        sys.exit(1)


    vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)

    custom_vgg_model = Model(vgg_model.input, out)

    for layer in vgg_model.layers:
        layer.trainable = False
    
    custom_vgg_model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

    batch_size = 15

    train_generator = train_datagen.flow_from_directory(
            '/volumes/data/data/train',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical')

    custom_vgg_model.fit_generator(
            train_generator,
            steps_per_epoch=15 // batch_size,
            epochs=100)

    joblib.dump(train_generator.class_indices, '/volumes/data/class_indices.dict')

    custom_vgg_model.save('/volumes/data/vgg_model.h5')


if __name__ == '__main__':
    train()
    
