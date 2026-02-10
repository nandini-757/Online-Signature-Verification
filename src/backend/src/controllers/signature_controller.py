import os
import uuid
import base64

from db.mongo import users
from model.user_model import create_user_doc
from services.static_embedding import get_embedding
from services.dynamic_features import extract_physics_features_from_strokes
from services.threshold import compute_threshold

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_base64_image(base64_string):
    # remove data:image/png;base64,
    img_data = base64_string.split(",")[1]
    img_bytes = base64.b64decode(img_data)

    filename = f"{uuid.uuid4()}.png"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(img_bytes)

    return file_path


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(i) for i in obj]
    elif hasattr(obj, "item"):
        return obj.item()   # converts numpy scalars
    else:
        return obj



def register_signature(data):
    user_id = str(uuid.uuid4())[:8]
    user_doc = create_user_doc(user_id)

    for sig in data["signatures"]:
        # ✅ FIX: frontend sends base64 image
        base64_image = sig.get("image")
        strokes = sig.get("strokes")

        if not base64_image or not strokes:
            return {"error": "Invalid signature payload"}, 400

        # convert base64 → image path
        img_path = save_base64_image(base64_image)

        # 🔥 ORIGINAL LOGIC (UNCHANGED)
        emb = get_embedding(img_path)
        dyn = extract_physics_features_from_strokes(strokes)

        user_doc["static_embeddings"].append(emb)
        user_doc["dynamic_features"].append(dyn)

    # compute threshold
    user_doc["threshold"] = compute_threshold(
        user_doc["static_embeddings"]
    )



    clean_doc = to_python(user_doc)
    users.insert_one(clean_doc)

    return {
        "message": "Signature registered successfully",
        "user_id": user_id,
        "samples": len(user_doc["static_embeddings"])
    }, 200