from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_model("./emotion.h5")
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

def preprocess_image(image_bytes) -> np.ndarray:
    # Open the image from bytes
    image = Image.open(BytesIO(image_bytes))
    # Resize the image to 48x48
    image_resized = image.resize((48, 48))
    # Convert the image to grayscale
    image_bw = image_resized.convert('L')  # 'L' mode stands for grayscale
    # Convert the image to numpy array
    image_array = np.array(image_bw)
    return image_array

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = await file.read()
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    print("Processed Image Shape:", processed_image.shape)
    print("Processed Image Data:", processed_image)
    
    img_batch = np.expand_dims(processed_image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
