import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import os

# --- Setup ---
WEIGHTS_PATH = "wildlife_detector_v5.pth"  # must be in the same folder as this script
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Rebuild model architecture, then load your trained weights ---
model = models.resnet18(weights=None)  # no need to redownload ImageNet weights, we're loading our own
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Prediction function ---
def predict(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        pred = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1)[0][pred].item()
    label = "Animal detected" if pred == 1 else "Empty frame"
    return f"{label} ({conf:.1%} confidence)"

# --- Launch local interface ---
demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text",
                     title="Camera Trap Wildlife Detector",
                     description="Upload a camera trap image to check if it contains an animal.")
demo.launch()