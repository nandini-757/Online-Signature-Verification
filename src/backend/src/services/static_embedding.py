import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# ==========================================
# CUSTOM LAYER (required to load model)
# ==========================================

class L2Normalize(Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)

# ==========================================
# PATH TO MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "model", "final_signature_embedding_model.keras")
)

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"L2Normalize": L2Normalize}
)

print("✅ Static embedding model loaded")

IMG_SIZE = 96

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess(img):

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ==========================================
# GET EMBEDDING
# ==========================================

def get_embedding(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Invalid image")

    # Handle RGBA images
    if len(img.shape) == 3 and img.shape[2] == 4:

        alpha = img[:, :, 3]
        rgb = img[:, :, :3]

        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        mask = alpha > 0

        white_bg[mask] = rgb[mask]

        img = cv2.cvtColor(white_bg, cv2.COLOR_BGR2GRAY)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = preprocess(img)

    embedding = model.predict(img, verbose=0)[0]

    return embedding.tolist()