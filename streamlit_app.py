import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import open_clip

try:
    from megadetector.detection import run_detector
    MEGADETECTOR_AVAILABLE = True
except ImportError:
    MEGADETECTOR_AVAILABLE = False

WEIGHTS_PATH = "wildlife_detector_v5.pth"
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
    "a blurry photo with no visible animal, just background",
    "a photo of an inanimate object, not an animal",
]

animal_phrases = {
    "a close-up photo of an animal's eyes",
    "a photo showing a clear animal silhouette",
    "a photo showing fur or skin texture of an animal",
    "a wide shot of an animal's full body in a natural landscape",
    "a photograph of a wild animal walking or moving in its habitat",
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

if MEGADETECTOR_AVAILABLE:
    @st.cache_resource
    def load_megadetector():
        return run_detector.load_detector('MDV5A')
    md_model = load_megadetector()
else:
    md_model = None

st.title("Camera Trap Wildlife Detector")
st.write("Upload a camera-trap image to check whether it contains an animal.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)
    img_t.requires_grad_()

    output = model(img_t)
    TEMPERATURE = 1.785
    probs = torch.softmax(output / TEMPERATURE, dim=1)
    pred = output.argmax(dim=1).item()
    conf = probs[0, pred].item()
    label = "Animal detected" if pred == 1 else "Empty frame"

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
    if vertical == "middle" and horizontal == "center":
        focus_area = "the center of the image"
    else:
        focus_area = f"the {vertical}-{horizontal} region of the image"

    hot_fraction = (cam > 0.7).sum() / cam.size
    if hot_fraction < 0.05:
        spread_desc = "a small, tightly localized area"
    elif hot_fraction < 0.20:
        spread_desc = "a moderately sized, fairly focused area"
    else:
        spread_desc = "a large, spread-out area with no single clear focal point"

    st.markdown(f"""
    **Detail:** the model's attention was concentrated on **{spread_desc}**, centered around
    **{focus_area}**. Combined with a **{conf:.1%}** confidence score, this suggests the model
    {"found a strong, spatially specific visual pattern to base its decision on" if hot_fraction < 0.20 and conf > 0.85
     else "found some signal, but the attention pattern is broad or the confidence is only moderate — a sign the image may not closely resemble typical training examples" if conf < 0.85 or hot_fraction >= 0.20
     else "made a reasonably confident decision based on a moderately localized region"}.
    """)

    overlay_image = overlay_heatmap(cam, image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(overlay_image, caption="Where the model looked", use_container_width=True)

    st.subheader(f"{label} ({conf:.1%} confidence)")
    st.caption(
        "Red/yellow regions show where the model focused most when making this "
        "prediction. This doesn't mean the model 'knows' concepts like eyes or "
        "silhouettes — it shows which pixels most influenced the output."
    )

    clip_results = get_clip_explanation(image)
    st.markdown("**Visual features detected (via CLIP, a separate model trained on image-text pairs):**")
    for phrase, score in clip_results[:3]:
        st.write(f"- {phrase}: {score:.1%} match")

    resnet_animal_prob = probs[0, 1].item()
    clip_animal_prob = sum(score for phrase, score in clip_results if phrase in animal_phrases)

    if MEGADETECTOR_AVAILABLE:
        md_result = md_model.generate_detections_one_image(image)
        md_detections = md_result['detections']
        CONFIDENCE_THRESHOLD = 0.5
        animal_detections = [d for d in md_detections if d['category'] == '1' and d['conf'] > CONFIDENCE_THRESHOLD]
        md_animal_prob = max([d['conf'] for d in animal_detections], default=0.0)

        st.markdown("**MegaDetector (purpose-built camera-trap detection model):**")
        if animal_detections:
            st.write(f"- Detected {len(animal_detections)} animal region(s), highest confidence: {md_animal_prob:.1%}")
        else:
            st.write("- No animal regions detected")
    else:
        st.caption("MegaDetector unavailable in this environment — verdict based on ResNet + CLIP only.")
        md_animal_prob = None

    debug_md = f"{md_animal_prob:.1%}" if md_animal_prob is not None else "N/A"
    st.write(f"Debug — ResNet: {resnet_animal_prob:.1%}, CLIP: {clip_animal_prob:.1%}, MegaDetector: {debug_md}")

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

    st.divider()
    st.subheader(f"Combined verdict: {final_label} ({final_conf:.1%} confidence)")
    if models_agree:
        st.write(f"✅ All {len(votes)} available models agree on this prediction.")
    else:
        md_leans_animal = md_animal_prob > 0.5 if md_animal_prob is not None else None
        st.write("⚠️ The models disagree — treat this result with caution. "
                 f"ResNet: {'animal' if resnet_leans_animal else 'empty'} ({resnet_animal_prob:.1%}), "
                 f"CLIP: {'animal' if clip_leans_animal else 'empty'} ({clip_animal_prob:.1%})"
                 + (f", MegaDetector: {'animal' if md_leans_animal else 'empty'} ({md_animal_prob:.1%})." if md_animal_prob is not None else "."))