from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()



MODEL=tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Early Blight","Late Blight" , "Healthy"]

@app.get("/ping") 
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data)->np.ndarray:
    image = Image.open(BytesIO(data))
    img_batch = np.expand_dims(image,0)
    predictions=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    print(predicted_class)
    return {
        'class': predicted_class,
        'confidence':float(confidence)
    }
    # return image

@app.post("/predict") 
async def predict(file:UploadFile=File(...)):
    image = read_file_as_image(await file.read())
    return


if __name__ == "__main__": 
    uvicorn.run(app,host='localhost',port=8000)