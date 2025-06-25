from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from io import BytesIO
from PIL import Image

app = FastAPI()

# Initialize InsightFace
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

# Pydantic model for request payload
class FaceVerificationRequest(BaseModel):
    img1: str  # base64 image 1
    img2: str  # base64 image 2

def decode_base64_image(b64_str: str):
    try:
        decoded = base64.b64decode(b64_str)
        img = Image.open(BytesIO(decoded)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError("Invalid base64 image format")

def get_embedding(img: np.ndarray):
    faces = face_app.get(img)
    if not faces:
        return None
    return faces[0].embedding

def cosine_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

@app.post("/verify")
async def verify_faces(payload: FaceVerificationRequest):
    try:
        img1 = decode_base64_image(payload.img1)
        img2 = decode_base64_image(payload.img2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    if emb1 is None or emb2 is None:
        raise HTTPException(status_code=400, detail="Face not detected in one or both images.")

    similarity = cosine_similarity(emb1, emb2)
    is_match = similarity > 0.5  # tweakable threshold

    return {
        "match": is_match,
        "similarity": similarity
  }
