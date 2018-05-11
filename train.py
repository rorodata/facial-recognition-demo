import cv2 as cv
from glob import glob
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils

TRAINING_DIR = os.getenv("TRAINING_DIR", "/volumes/data/training")
MODEL_FILE = os.getenv("MODEL_PATH", "/volumes/data/vgg_features.model")
LABEL_FILE = os.getenv("LABEL_PATH", "/volumes/data/label_vectors.dict")

def process_image(image):
    img = cv.resize(image, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=2)
    return img

def labels_to_images(path_to_images):
    files = glob(path_to_images + '/*')
    label_files = {os.path.basename(f).split('.')[0] : f for f in files}

    label_images = {label : cv.imread(path) 
            for label, path in label_files.items()}

    return label_images

def labels_to_vector(label_images):
    label_images = {label : process_image(img) 
            for label, img in label_images.items()}

    label_vector = {label : vgg_features.predict(img)
            for label, img in label_images.items()}

    return label_vector

def extract_vectors():
    # Convolution Features
    vgg_features = VGGFace(
        model='resnet50',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg') 

    labels_images = labels_to_images(TRAINING_DIR)
    labels_vectors = labels_to_vector(labels_images)

    joblib.dump(labels_vectors, LABEL_FILE)
    joblib.dump(vgg_features, MODEL_FILE)

if __name__ == '__main__':

    extract_vectors()

