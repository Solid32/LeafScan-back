
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from leafscan.main import pred
from  PIL import Image
import numpy as np
import cv2
from numpy import asarray




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

@app.post('/predict')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_expended = np.expand_dims(cv2_img, axis=0)

    prediction = pred(img_expended) #imput shape (1,256, 256, 3)
    return round(prediction.max()*100,3)
