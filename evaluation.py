import cv2 as cv
from glob import glob
import os
import numpy as np
import joblib
from PIL import Image
import io
import binascii
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
from sklearn.metrics.pairwise import cosine_similarity

MODEL_FILE = os.getenv("MODEL_PATH", "/voluems/data/vgg_features.model")
LABEL_FILE = os.getenv("LABEL_PATH", "/voluems/data/label_vectors.dict")
CASCADE_FILE = os.getenv("CASCADE_PATH", "/voluems/data/haarcascade_frontalface_alt.xml")

model = None
label_vectors = None
haar_face_cascade = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

def load_model():
    if not os.path.exists(MODEL_FILE):
        print("Unable to find the model file vgg_features.model", file=sys.stderr)
        return None
    return joblib.load(MODEL_FILE)

def get_labels():
    global label_vectors
    if label_vectors is None:
        label_vectors = load_label_vectors()
    return label_vectors

def load_label_vectors():
    if not os.path.exists(LABEL_FILE):
        print("Unable to find the label vectors file label_vectors.dict", file=sys.stderr)
        return None
    return joblib.load(LABEL_FILE)

def get_cascade():
    global haar_face_cascade
    if haar_face_cascade is None:
        haar_face_cascade = load_cascade()
    return haar_face_cascade

def load_cascade():
    if not os.path.exists(CASCADE_FILE):
        print("Unable to find the cascade file haarcascade_frontalface_alt.xml", file=sys.stderr)
        return None
    return cv.CascadeClassifier(CASCADE_FILE)

def process_image(image):
    img = cv.resize(image, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=2)
    return img

def extract_bb_faces(f_cascade, colored_img, scaleFactor=1.1):
    img_copy = colored_img.copy()
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)          
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          

    bb_faces = {}

    for (x, y, w, h) in faces:
        face = colored_img[y:y+h, x:x+w]
        bb_faces[(x, y, w, h)] = face

    return bb_faces


def image_to_label(image):
    global label_vectors 
    global model

    image = process_image(image)
    feature_vector = model.predict(image)

    for label, vector in label_vectors.items():
        if cosine_similarity(feature_vector, vector) > 0.4:
            return label

    return 'UNKNOWN'
        
def tag_faces(ascii_image_data):
    model = get_model()
    if not model:
        return 'error-no-model'

    label_vectors = get_labels()
    if not label_vectors:
        return 'error-no-label-vectors'

    haar_face_cascade = get_cascade()
    if not haar_face_cascade:
        return 'error-no-haar-cascade'

    binary_image_data = binascii.a2b_base64(ascii_image_data)
    pil_image = Image.open(fp=io.BytesIO(binary_image_data))
    opencv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    bb_faces = extract_bb_faces(haar_face_cascade, opencv_image)

    bb_face_labels = {bb : image_to_label(image) for bb, image in bb_faces.items()}

    image_copy = opencv_image.copy()

    font = cv.FONT_HERSHEY_SIMPLEX

    for bb, label in bb_face_labels.items():
        x, y, w, h = bb
        cv.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(image_copy, label, (x, y), font, 1 ,(0, 0, 255), 2, cv.LINE_AA)

    cv.imwrite('./temp.jpg', image_copy)

    image_copy = open('./temp.jpg', 'rb').read()
    os.remove('./temp.jpg')
    byte_data = binascii.b2a_base64(image_copy)
    return str(byte_data, encoding='utf-8')

if __name__ == '__main__':
    image_data = open('./world_leaders.jpg', 'rb').read()
    image_data = binascii.b2a_base64(image_data)
    image_data = str(image_data, encoding='utf-8')

    new_image = tag_faces(image_data)

    new_image = binascii.a2b_base64(new_image)
    pil_image = Image.open(fp=io.BytesIO(new_image))
    opencv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    cv.imwrite('new_image.jpg', opencv_image)





    

