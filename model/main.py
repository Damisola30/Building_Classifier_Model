from typing import Union
from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = torch.load("../model/model.pth", map_location=torch.device("cpu"))
model.eval()

# Define the preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard for ImageNet models
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
