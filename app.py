from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from facenet_pytorch import MTCNN
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
import cv2
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware

# Initialize FastAPI
app = FastAPI()

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# Dynamic blacklist for IPs
blacklist = set()

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    client_ip = get_remote_address(request)
    # Log the blocked IP
    logger.warning(f"Rate limit exceeded for IP: {client_ip}")
    # Add to blacklist
    blacklist.add(client_ip)
    return PlainTextResponse("Too Many Requests. Your IP has been temporarily blocked.", status_code=429)

@app.middleware("http")
async def ip_block_middleware(request: Request, call_next):
    client_ip = get_remote_address(request)
    if client_ip in blacklist:
        logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
        return PlainTextResponse("Your IP has been blocked due to suspicious activity.", status_code=403)
    return await call_next(request)

@app.on_event("startup")
async def startup_event():
    app.state.limiter = limiter

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    response = await app.state.limiter.middleware(request, call_next)
    return response

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load TensorFlow model on startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = tf.keras.models.load_model("model/expression_detection_new.h5")
    logger.info("Model loaded successfully.")

# Initialize MTCNN detector
detector = MTCNN(keep_all=True)

# Emotion labels
emotion_labels = ["Angry", "Fear", "Happy", "Neutral", "Sad"]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "Server is running smoothly"}

# Detect expression and return processed image
@app.post("/detect-expression/", dependencies=[limiter.limit("10/minute")])
async def detect_expression(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG files are supported.")
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        boxes, _ = detector.detect(image)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=400, detail="No faces detected.")

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image_np[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))
            predictions = model.predict(face_reshaped)
            emotion = emotion_labels[np.argmax(predictions)]

            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image_np, emotion, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    except Exception as e:
        logger.error(f"Error in detect-expression: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Detect expression and return labels only
@app.post("/get-expression-label/", dependencies=[limiter.limit("10/minute")])
async def get_expression_label(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG files are supported.")
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        boxes, _ = detector.detect(image)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=400, detail="No faces detected.")

        expression_results = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image_np[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))
            predictions = model.predict(face_reshaped)
            emotion = emotion_labels[np.argmax(predictions)]
            expression_results.append(emotion)

        return JSONResponse({"emotions": expression_results}, status_code=200)

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    except Exception as e:
        logger.error(f"Error in get-expression-label: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Customize OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Expression Detection API",
        version="1.0.0",
        description="API for detecting facial expressions using MTCNN and TensorFlow.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
