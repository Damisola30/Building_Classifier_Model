from typing import Union
from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import pickle
import os
app = FastAPI()

# Recreate the model
from model.model import CNN 
model_path = os.path.abspath("../Building_-Classification_ML_FPro/experimentation/model.pth")

model = CNN()

# Load the weights
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Define the preprocessing steps
transform = transforms.Compose([
    transforms.Resize((400, 300)),  # Match training data size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)

    prediction = output.argmax(dim=1).item()
    return {"prediction": prediction}
