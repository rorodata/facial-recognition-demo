from __future__ import print_function
import joblib
import os
import sys
from keras_vggface import utils
from keras.models import load_model
import cv2 as cv
import numpy as np
from PIL import Image
import base64
import io
import binascii

MODEL_FILE = os.getenv("MODEL_PATH", "vgg_model.h5")
CLASS_INDEX_FILE = os.getenv("CLASS_INDEX_PATH", "class_indices.dict")

model = None
class_indices = None

def get_model():
    global model
    if model is None:
        model = load_keras_model()
    return model

def get_class_indices():
    global class_indices
    if class_indices is None:
        class_indices = load_class_indices()
    return class_indices

def load_class_indices():
    if not os.path.exists(CLASS_INDEX_FILE):
        print("Unable to find the class index file class_indices.dict", file=sys.stderr)
        return None
    return joblib.load(CLASS_INDEX_FILE)
        
def load_keras_model():
    if not os.path.exists(MODEL_FILE):
        print("Unable to find the model file vgg_model.h5", file=sys.stderr)
        return None
    return load_model(MODEL_FILE)

def process_image(image):
    img = cv.resize(image, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=2)
    return img

def get_prediction_label(prediction):
    global class_indices
    #if class_indices is None:
    #    class_indices = load_class_indices()
        
    index_to_label = {value : key for key, value in class_indices.items()}
    
    prediction_index = np.argmax(prediction)
    return index_to_label[prediction_index]

def predict(ascii_image_data):
    
    binary_image_data = binascii.a2b_base64(ascii_image_data)
    #base64_decoded_image = base64.b64decode(binary_image_data)
    pil_image = Image.open(fp=io.BytesIO(binary_image_data)) 
    opencv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
    
    model = get_model()
    global class_indices
    class_indices = load_class_indices()
    
    if not model:
        return 'error-no-model'
    if not class_indices:
        return 'error-no-class_indices'

    test_image_processed = process_image(opencv_image)
    
    prediction = model.predict(test_image_processed)
    
    class_name = get_prediction_label(prediction)
    
    print("predict: {}".format(class_name))
    
    return {'label': class_name, 'softmax_probabilities': prediction.tolist()}

if __name__ == '__main__':
    test_image_path = '/volumes/data/test_image.png'
    test_image = open(test_image_path, 'rb').read()
    test_image = binascii.b2a_base64(test_image)
    test_image = str(test_image, encoding='utf-8')
    
    result = predict(test_image)
    
    print(result['label'])
    print(result['softmax_probabilities'])