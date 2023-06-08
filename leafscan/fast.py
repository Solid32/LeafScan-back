
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from main import pred
from  PIL import Image
import numpy as np



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
async def create_upload_file(file: UploadFile):
    image=Image.open(file)
    data = np.array(image)
    return {"np.shape": data.shape}
    #loaded_img_array = np.asarray(file.resize((256, 256)))[..., :3]
    #pred_result = pred(loaded_img_array)
    #return pred_result
