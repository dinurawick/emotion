from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import keras as keras

app = FastAPI()

@app.get("/retard")
async def hello():
   return "hola" 

def read_file_as_image(data) -> np.ndarray:
   Image = np.array(Image.open(BytesIO(data)))


@app.get("/predict")
async def predict(
    file: UploadFile = File(...)  
):
    image = read_file_as_image(await file.read())
    

if __name__ == "__main__":
   uvicorn.run(app, host="localhost", port=8000)
