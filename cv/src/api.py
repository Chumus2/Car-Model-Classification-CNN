import os
import torch
import configs
from PIL import Image
from io import BytesIO
from model import CarModelClassifierTL
from colorama import Fore, Style, init
from fastapi import FastAPI, File, UploadFile, HTTPException


init(autoreset=True)
app = FastAPI(title="Car Classifier")

model_path = os.path.join(configs.MODEL_DIR, "best_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{Fore.RED}[Error]{Style.RESET_ALL} Checkpoint not found at {model_path}")

checkpoint = torch.load(model_path, map_location=configs.DEVICE, weights_only=False)
classes = checkpoint.get("classes", configs.CLASSES)

model = CarModelClassifierTL(output_size=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(configs.DEVICE)
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail=f"File is not a valid image")

    input_tensor = configs.VAL_TRANSFORMS(image).unsqueeze(0).to(configs.DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    pred_label = classes[predicted.item()]
    conf_score = confidence.item() * 100

    return {
        "filename": file.filename,
        "pred_label": pred_label,
        "confidence": conf_score,
    }