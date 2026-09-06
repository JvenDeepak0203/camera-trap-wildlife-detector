import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import open_clip
import os
import urllib.request

WEIGHTS_PATH = "wildlife_detector_v5.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MD_WEIGHTS_PATH = "md_v5a.0.0.pt"
MD_WEIGHTS_URL = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
MEGADETECTOR_AVAILABLE = True  # will flip to False below if loading genuinely fails

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

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    return model, activations, gradients

model, activations, gradients = load_model()

def overlay_heatmap(cam, original_image, alpha=0.45):
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, resample=Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0
    heatmap = (cm.jet(cam_arr)[:, :, :3] * 255).astype(np.uint8)
    original_arr = np.array(original_image.convert('RGB'))
    overlay = (heatmap * alpha + original_arr * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)

@st.cache_resource
def load_clip():
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    return clip_model, clip_preprocess, tokenizer

clip_model, clip_preprocess, tokenizer = load_clip()

CANDIDATE_PHRASES = [
    "a close-up photo of an animal's eyes",
    "a photo showing a clear animal silhouette",
    "a photo showing fur or skin texture of an animal",
    "a wide shot of an animal's full body in a natural landscape",
    "a photograph of a wild animal walking or moving in its habitat",
    "a photo showing multiple animals together",
    "a black and white night-vision photo of an animal",
    "a photo of a person, not an animal",
    "a photo of a vehicle, not an animal",
    "a blurry photo with no visible animal, just background",
    "a photo of an inanimate object, not an animal",
]

animal_phrases = {
    "a close-up photo of an animal's eyes",
    "a photo showing a clear animal silhouette",
    "a photo showing fur or skin texture of an animal",
    "a wide shot of an animal's full body in a natural landscape",
    "a photograph of a wild animal walking or moving in its habitat",
    "a photo showing multiple animals together",
    "a black and white night-vision photo of an animal",
}

def get_clip_explanation(pil_image):
    image_input = clip_preprocess(pil_image).unsqueeze(0)
    text_input = tokenizer(CANDIDATE_PHRASES)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    scores = similarity[0].tolist()
    ranked = sorted(zip(CANDIDATE_PHRASES, scores), key=lambda x: -x[1])
    return ranked

def draw_detection_boxes(pil_image, detections):
    img_copy = pil_image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    for d in detections:
        left, top, right, bottom = float(d[0]), float(d[1]), float(d[2]), float(d[3])
        conf = float(d[4])
        draw.rectangle([left, top, right, bottom], outline='#ff3b30', width=max(3, w // 150))
        draw.text((left + 4, max(top - 18, 0)), f"{conf:.0%}", fill='#ff3b30')
    return img_copy

if MEGADETECTOR_AVAILABLE:
    @st.cache_resource
    def load_megadetector():
        try:
            if not os.path.exists(MD_WEIGHTS_PATH):
                urllib.request.urlretrieve(MD_WEIGHTS_URL, MD_WEIGHTS_PATH)
            return torch.hub.load('ultralytics/yolov5', 'custom', path=MD_WEIGHTS_PATH, trust_repo=True)
        except Exception as e:
            st.error(f"MegaDetector failed to load: {e}")
            return None
    md_model = load_megadetector()
    if md_model is None:
        MEGADETECTOR_AVAILABLE = False
else:
    md_model = None

st.title("Camera Trap Wildlife Detector")
st.write("Upload a camera-trap image to check whether it contains an animal.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    progress = st.progress(0, text="Running ResNet18...")

    # ============================================================
    # STEP 1 — Run all three models silently, collect every result
    # ============================================================

    # --- Model 1: ResNet ---
    img_t = transform(image).unsqueeze(0).to(device)
    img_t.requires_grad_()

    output = model(img_t)
    TEMPERATURE = 1.785
    probs = torch.softmax(output / TEMPERATURE, dim=1)
    pred = output.argmax(dim=1).item()
    resnet_conf = probs[0, pred].item()
    resnet_label = "Animal detected" if pred == 1 else "Empty frame"
    resnet_animal_prob = probs[0, 1].item()

    model.zero_grad()
    output[0, pred].backward()

    acts = activations['value'][0]
    grads = gradients['value'][0]
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().cpu().numpy()

    h, w = cam.shape
    peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)
    vertical = "top" if peak_y < h/3 else "bottom" if peak_y > 2*h/3 else "middle"
    horizontal = "left" if peak_x < w/3 else "right" if peak_x > 2*w/3 else "center"
    focus_area = "the center of the image" if (vertical == "middle" and horizontal == "center") else f"the {vertical}-{horizontal} region of the image"

    hot_fraction = (cam > 0.7).sum() / cam.size
    if hot_fraction < 0.05:
        spread_desc = "a small, tightly localized area"
    elif hot_fraction < 0.20:
        spread_desc = "a moderately sized, fairly focused area"
    else:
        spread_desc = "a large, spread-out area with no single clear focal point"

    resnet_detail = (
        f"The model's attention was concentrated on **{spread_desc}**, centered around **{focus_area}**. "
        + ("This suggests a strong, spatially specific visual pattern drove the decision."
           if hot_fraction < 0.20 and resnet_conf > 0.85
           else "This suggests the attention pattern is broad or the confidence is only moderate — a sign the image may not closely resemble typical training examples."
           if resnet_conf < 0.85 or hot_fraction >= 0.20
           else "This suggests a reasonably confident decision based on a moderately localized region.")
    )

    overlay_image = overlay_heatmap(cam, image)

    progress.progress(33, text="Running CLIP...")

    # --- Model 2: CLIP ---
    clip_results = get_clip_explanation(image)
    clip_animal_prob = sum(score for phrase, score in clip_results if phrase in animal_phrases)

    progress.progress(66, text="Running MegaDetector..." if MEGADETECTOR_AVAILABLE else "Skipping MegaDetector (unavailable)...")

    # --- Model 3: MegaDetector ---
    if MEGADETECTOR_AVAILABLE:
        md_results = md_model(image)
        detections = md_results.xyxy[0]
        CONFIDENCE_THRESHOLD = 0.5
        # MegaDetector's classes: 0 = animal, 1 = person, 2 = vehicle
        md_animal_detections = [d for d in detections if int(d[5]) == 0 and float(d[4]) > CONFIDENCE_THRESHOLD]
        md_animal_prob = max([float(d[4]) for d in md_animal_detections], default=0.0)
        md_boxed_image = draw_detection_boxes(image, md_animal_detections) if len(md_animal_detections) > 0 else None
    else:
        md_animal_detections = []
        md_animal_prob = None
        md_boxed_image = None

    progress.progress(100, text="Done!")
    progress.empty()

    # ============================================================
    # STEP 2 — Combine into final verdict
    # ============================================================

    resnet_leans_animal = resnet_animal_prob > 0.5
    clip_leans_animal = clip_animal_prob > 0.5

    votes = [resnet_leans_animal, clip_leans_animal]
    if md_animal_prob is not None:
        votes.append(md_animal_prob > 0.5)

    final_label = "Animal detected" if sum(votes) > len(votes) / 2 else "Empty frame"

    prob_sum = resnet_animal_prob + clip_animal_prob + (md_animal_prob if md_animal_prob is not None else 0)
    prob_count = 3 if md_animal_prob is not None else 2
    final_conf = prob_sum / prob_count
    if final_label == "Empty frame":
        final_conf = 1 - final_conf

    models_agree = len(set(votes)) == 1

    # ============================================================
    # STEP 3 — Display: big verdict first, then each model in order
    # ============================================================

    verdict_color = "#2e7d32" if final_label == "Animal detected" else "#616161"
    st.markdown(
        f"<h1 style='text-align:center; color:{verdict_color};'>{final_label}</h1>"
        f"<p style='text-align:center; font-size:1.3rem;'>{final_conf:.1%} confidence</p>",
        unsafe_allow_html=True
    )
    if models_agree:
        st.markdown(f"<p style='text-align:center;'>✅ All {len(votes)} available models agree</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center;'>⚠️ Models disagree — see breakdown below</p>", unsafe_allow_html=True)

    st.divider()

    # --- Model 1: ResNet ---
    st.subheader("Model 1: ResNet18")
    st.caption("A convolutional neural network fine-tuned on labeled camera-trap images to distinguish animal frames from empty ones.")
    st.markdown(f"**Result:** {resnet_label} ({resnet_conf:.1%} confidence)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(overlay_image, caption="Where the model looked", use_container_width=True)
    st.caption(resnet_detail)
    st.caption(
        "Red/yellow regions show where the model focused most. This doesn't mean "
        "the model 'knows' concepts like eyes or silhouettes — it shows which "
        "pixels most influenced the output."
    )

    st.divider()

    # --- Model 2: CLIP ---
    st.subheader("Model 2: CLIP")
    st.caption("A general-purpose vision-language model that scores how well the image matches a set of descriptive phrases.")
    st.markdown(f"**Result:** {'Animal detected' if clip_leans_animal else 'Empty frame'} ({(clip_animal_prob if clip_leans_animal else 1 - clip_animal_prob):.1%} confidence)")

    fig, ax = plt.subplots(figsize=(6, 4))
    phrases = [p for p, _ in clip_results]
    scores = [s * 100 for _, s in clip_results]
    ax.barh(phrases, scores, color='#4a90d9')
    ax.set_xlabel('Match %')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # --- Model 3: MegaDetector ---
    st.subheader("Model 3: MegaDetector")
    st.caption("A detection model built specifically for camera-trap images by conservation-tech researchers, locating animals, people, and vehicles.")
    if MEGADETECTOR_AVAILABLE:
        md_leans_animal = md_animal_prob > 0.5
        st.markdown(f"**Result:** {'Animal detected' if md_leans_animal else 'Empty frame'} ({(md_animal_prob if md_leans_animal else 1 - md_animal_prob):.1%} confidence)")
        if len(md_animal_detections) > 0:
            st.image(md_boxed_image, caption=f"{len(md_animal_detections)} animal region(s) detected", use_container_width=True)
        else:
            st.image(image, caption="No animal regions detected", use_container_width=True)
    else:
        st.markdown("**Result:** Not used")
        st.caption("MegaDetector was unavailable in this environment, so this model was skipped. The final verdict above is based on the remaining available models.")

    st.divider()

    # --- Summary of models used ---
    md_icon = "✅" if MEGADETECTOR_AVAILABLE else "❌"
    st.markdown(f"**Models used for this analysis:** ResNet18 (✅), CLIP (✅), MegaDetector ({md_icon})")