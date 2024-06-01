import tensorflow as tf
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from PIL import Image 
import io

import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True
)

def load_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "..", "mnist_model.h5")
    return tf.keras.models.load_model(model_path)

def predict(x):
    return model.predict(x)

def preprocess_image(image):
    image = image.convert("L")          # convert to gray scale
    image = image.resize((28, 28))      # resize to 28x28, since the model was trained on 28x28 images
    image = np.array(image)             # convert to numpy array
    image = image / 255.0               # normalize the image
    image = image.reshape(1, 28, 28, 1) # reshape to (1, 28, 28, 1) to match the input shape of the model
    return image


@app.post("/predict/")
async def predict_api(file:UploadFile = File(...)):
    try:

        if not file.file:
            return JSONResponse(status_code=400, content={"message": "Invalid file"})

        image = Image.open(io.BytesIO(await file.read()))
        preproccessed_image = preprocess_image(image)

        prediction = predict(preproccessed_image)
        predicted_class = np.argmax(prediction)

        return JSONResponse(content={"predicted_class": int(predicted_class)}, status_code=200)

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})


if __name__ == "__main__":
    model = load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)