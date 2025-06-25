# 🧠 Face Verification API (FastAPI + InsightFace)

This project provides a face verification API using InsightFace (ArcFace) to verify whether two base64-encoded images are of the same person. It is designed for scalable deployment in a production environment.

---

## ⚙️ Requirements

- Python 3.8+
- Uvicorn
- FastAPI
- InsightFace (CPU version)
- OpenCV, NumPy, Pillow
- Optional: Gunicorn + Nginx (for production)
- OS: Ubuntu 20.04+ or equivalent

---

## 📁 Project Structure


---

## 🚀 Deployment Instructions

### 1.  to install the dependancies required by our main.py 

```bash
pip install fastapi uvicorn insightface numpy opencv-python pillow python-multipart
```

### 2.  run the fast api swagger for the main.py file 
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
