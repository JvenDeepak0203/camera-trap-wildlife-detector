import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

WEIGHTS_PATH = "wildlife_detector_v2.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("Camera Trap Wildlife Detector")
st.write("Upload a camera-trap image to check whether it contains an animal.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        pred = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1)[0][pred].item()

    label = "Animal detected" if pred == 1 else "Empty frame"
    st.subheader(f"{label} ({conf:.1%} confidence)")