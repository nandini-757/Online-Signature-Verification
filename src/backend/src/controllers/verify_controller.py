import base64
import numpy as np
from PIL import Image
import io
import uuid
import os
from db.mongo import users
from services.static_matcher import compute_static_score
from services.dynamic_matcher import compute_dynamic_score
from services.static_embedding import get_embedding
from services.dynamic_features import extract_physics_features_from_strokes





def save_temp_image(base64_string):

    # Handle data URL
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    img_bytes = base64.b64decode(base64_string)

    # 🔥 Absolute uploads path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "uploads")
    UPLOAD_FOLDER = os.path.abspath(UPLOAD_FOLDER)

    # Ensure folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = f"verify_{uuid.uuid4()}.png"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as f:
        f.write(img_bytes)

    print("Saved to:", file_path)
    print("Exists:", os.path.exists(file_path))
    print("Size:", os.path.getsize(file_path))

    return file_path




def verify_signature(data):

    # ===============================
    # VALIDATE PAYLOAD
    # ===============================
    if not isinstance(data, dict):
        return {"error": "Invalid JSON"}, 400

    user_id = data.get("user_id")
    signature = data.get("signature", {})

    image_b64 = signature.get("image")
    strokes = signature.get("strokes")

    if not user_id or not image_b64 or not strokes:
        return {"error": "Missing required fields"}, 400

    # ===============================
    # FETCH USER
    # ===============================
    user = users.find_one({"user_id": user_id})
    if not user:
        return {"error": "User not found"}, 404

    try:
        # ===============================
        # DECODE BASE64 IMAGE
        # ===============================
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        img_path = save_temp_image(image_b64)

        # ===============================
        # STATIC EMBEDDING
        # ===============================
        new_embedding = get_embedding(img_path)

        # ===============================
        # DYNAMIC FEATURES
        # ===============================
        new_dynamic = extract_physics_features_from_strokes(strokes)

        if new_dynamic is None:
            return {"error": "Invalid stroke data"}, 400

        # convert numpy → list safe
        new_embedding = np.array(new_embedding).astype(float).tolist()
        new_dynamic = np.array(new_dynamic).astype(float).tolist()

        # ===============================
        # COMPUTE SCORES
        # ===============================
        static_score = compute_static_score(
            new_embedding,
            user["static_embeddings"]
        )

        dynamic_score = compute_dynamic_score(
            new_dynamic,
            user["dynamic_features"]
        )

        # ===============================
        # FUSION
        # ===============================
        w1, w2 = 0.7, 0.3
        final_score = (w1 * static_score) + (w2 * dynamic_score)

        threshold = float(user["threshold"])

        verified = final_score >= threshold
        os.remove(img_path)


        # ===============================
        # RESPONSE
        # ===============================
        return {
            "hi"
            "verified": bool(verified),
            "static_score": round(float(static_score), 3),
            "dynamic_score": round(float(dynamic_score), 3),
            "final_score": round(float(final_score), 3),
            "threshold": round(threshold, 3)
        }, 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500
