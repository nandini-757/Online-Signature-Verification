import os
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "model", "static_embedding_model.keras")
)

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 96

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def get_embedding(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image")

    img = preprocess(img)
    emb = model.predict(img)[0]

    # normalize here
    emb = emb / np.linalg.norm(emb)

    return emb.tolist()