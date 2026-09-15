import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import open_clip
import textwrap
import hashlib
import base64
import glob
import io
import os
import urllib.parse
import urllib.request

AUTHOR = "Jven Deepak"
REPO_URL = "https://github.com/JvenDeepak0203/camera-trap-detector"

WEIGHTS_PATH = "v6_320.pth"
INPUT_SIZE = 320
TEMPERATURE = 1.656
W_RESNET = 0.3
W_MD = 0.7
DEFAULT_THRESHOLD = 0.275

MD_WEIGHTS_PATH = "md_v5a.0.0.pt"
MD_WEIGHTS_URL = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"

EXCLUDE = {
    "images (4).jpg",
    "images (9).png",
    "images (13).png",
    "sample_gecko.png",
    "stock-photo-random-pictures-cute-and-funny-2286554497.jpg",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.set_page_config(page_title="Camera Trap Wildlife Detector",
                   layout="wide", initial_sidebar_state="collapsed")

CARD, EDGE, TEXT, MUTED = "#1b2027", "#2c333d", "#e6eaf0", "#8a95a5"
GREEN, AMBER, GREY, RED, BLUE, TEAL = "#4caf7d", "#d9a441", "#6b7684", "#e05c5c", "#6a7fd4", "#3fa8a0"

st.markdown(f"""
<style>
.card {{ background:{CARD}; border:1px solid {EDGE}; border-radius:10px;
         padding:1rem 1.2rem; margin-bottom:0.8rem; }}
.row {{ padding:0.5rem 0; border-bottom:1px solid {EDGE}; }}
.row:last-of-type {{ border-bottom:none; }}
.rowtop {{ display:flex; justify-content:space-between; align-items:baseline; }}
.name {{ color:{TEXT}; font-size:0.92rem; }}
.val {{ color:{TEXT}; font-weight:700; font-size:1.05rem; }}
.tag {{ font-size:0.72rem; letter-spacing:0.02em; }}
.verdict {{ font-size:1.8rem; font-weight:700; margin:0; line-height:1.2; }}
.sub {{ color:{MUTED}; font-size:0.85rem; }}
.rank {{ font-size:1.05rem; color:{TEXT}; margin:0.2rem 0 0.5rem 0; }}
.tile {{ font-size:0.78rem; text-align:center; padding:0.25rem 0 0.7rem 0; }}
.grid {{ display:grid; grid-template-columns:repeat(6, 1fr); gap:10px;
         padding-bottom:1.5rem; }}
.grid a {{ display:block; line-height:0; border-radius:8px; overflow:hidden;
           border:2px solid transparent; transition:border-color .15s, transform .15s; }}
.grid a:hover {{ border-color:{GREEN}; transform:translateY(-2px); }}
.grid img {{ width:100%; height:auto; display:block; }}
.meth h4 {{ color:{TEXT}; font-size:0.95rem; margin:1rem 0 0.3rem 0; }}
.meth p {{ color:{MUTED}; font-size:0.85rem; line-height:1.55; margin:0 0 0.4rem 0; }}
.meth b {{ color:{TEXT}; }}
.foot {{ color:{MUTED}; font-size:0.8rem; line-height:1.7; padding:1.2rem 0 2rem 0;
         border-top:1px solid {EDGE}; margin-top:1.5rem; }}
.foot a {{ color:{TEAL}; text-decoration:none; }}
.foot a:hover {{ text-decoration:underline; }}
</style>
""", unsafe_allow_html=True)

matplotlib.rcParams.update({
    'figure.facecolor': CARD, 'axes.facecolor': CARD, 'text.color': TEXT,
    'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': EDGE, 'savefig.facecolor': CARD,
})

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ANIMAL_PHRASES = [
    "a camera trap photo of a wild animal",
    "a mammal walking through forest undergrowth",
    "an animal standing in vegetation",
    "a night-vision photo of an animal",
    "livestock or cattle in an open area",
    "an animal's legs and body among tree trunks",
    "the back or side of a large animal",
    "fur or hide visible against the forest floor",
    "an animal close to the camera lens",
    "an animal partially hidden behind leaves",
]
EMPTY_PHRASES = [
    "an empty forest scene with no animals",
    "empty vegetation moving in the wind",
    "a dark empty night photo",
    "an empty dirt trail with no animal",
    "blurry out of focus foliage, no subject",
]
ALL_PHRASES = ANIMAL_PHRASES + EMPTY_PHRASES
PANELS = ["Attention map", "Detection boxes", "Scene description"]


@st.cache_resource
def load_resnet():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    m.to(device).eval()
    acts, grads = {}, {}
    m.layer4[-1].register_forward_hook(lambda mo, i, o: acts.__setitem__('v', o))
    m.layer4[-1].register_full_backward_hook(lambda mo, gi, go: grads.__setitem__('v', go[0]))
    return m, acts, grads


@st.cache_resource
def load_clip():
    m, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    m.eval()
    return m, pre, open_clip.get_tokenizer('ViT-B-32')


@st.cache_resource
def load_md():
    try:
        if not os.path.exists(MD_WEIGHTS_PATH):
            urllib.request.urlretrieve(MD_WEIGHTS_URL, MD_WEIGHTS_PATH)
        return torch.hub.load('ultralytics/yolov5', 'custom',
                              path=MD_WEIGHTS_PATH, trust_repo=True)
    except Exception as e:
        st.error(f"MegaDetector failed to load: {e}")
        return None


model, activations, gradients = load_resnet()
clip_model, clip_pre, tokenizer = load_clip()
md_model = load_md()


@st.cache_data
def thumb_uri(path, size=(320, 240)):
    im = ImageOps.fit(Image.open(path).convert("RGB"), size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def thumb_img(img, size=(300, 225)):
    return ImageOps.fit(img, size, Image.LANCZOS)


def band(s, thr):
    if s < thr * 0.36: return "Very likely empty", GREEN
    if s < thr: return "Likely empty", GREEN
    if s < thr + 0.175: return "Uncertain - worth reviewing", AMBER
    if s < 0.75: return "Likely contains an animal", GREEN
    return "Very likely contains an animal", GREEN


def overlay_heatmap(cam, img, alpha=0.45):
    ci = Image.fromarray(np.uint8(cam * 255)).resize(img.size, resample=Image.BILINEAR)
    heat = (cm.jet(np.array(ci) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray((heat * alpha + np.array(img.convert('RGB')) * (1 - alpha)).astype(np.uint8))


def draw_boxes(img, dets):
    out = img.copy()
    d = ImageDraw.Draw(out)
    w = out.size[0]
    for b in dets:
        l, t, r, bo, c = b[:5]
        d.rectangle([l, t, r, bo], outline="#ff3b30", width=max(3, w // 160))
        d.text((l + 4, max(t - 16, 0)), f"{c:.0%}", fill="#ff3b30")
    return out


def run_resnet(image):
    x = transform(image).unsqueeze(0).to(device)
    x.requires_grad_()
    out = model(x)
    rp = torch.softmax(out / TEMPERATURE, dim=1)[0, 1].item()
    model.zero_grad()
    out[0, out.argmax(1).item()].backward()
    a, g = activations['v'][0], gradients['v'][0]
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = ((cam - cam.min()) / (cam.max() + 1e-8)).detach().cpu().numpy()
    return rp, cam


def run_md(image):
    if md_model is None:
        return [], None
    dets = md_model(image).xyxy[0]
    mdd = [[float(v) for v in b[:5]] for b in dets
           if int(b[5]) == 0 and float(b[4]) > 0.1]
    return mdd, max([b[4] for b in mdd], default=0.0)


def run_clip(image):
    xi = clip_pre(image).unsqueeze(0)
    with torch.no_grad():
        fi = clip_model.encode_image(xi)
        ft = clip_model.encode_text(tokenizer(ALL_PHRASES))
        fi = fi / fi.norm(dim=-1, keepdim=True)
        ft = ft / ft.norm(dim=-1, keepdim=True)
        sim = (100.0 * fi @ ft.T).softmax(dim=-1)[0].tolist()
    ranked = sorted(zip(ALL_PHRASES, sim), key=lambda p: -p[1])
    return ranked, sum(s for p, s in ranked if p in ANIMAL_PHRASES)


def analyse(image, status=None):
    if status: status.update(label="Running ResNet18 and computing attention map...")
    rp, cam = run_resnet(image)
    if status: status.update(label="Running MegaDetector - locating candidate regions...")
    mdd, mp = run_md(image)
    if status: status.update(label="Running CLIP - matching scene descriptions...")
    ranked, cp = run_clip(image)
    blend = (W_RESNET * rp + W_MD * mp) if mp is not None else rp
    return {'resnet': rp, 'cam': cam, 'dets': mdd, 'md': mp,
            'ranked': ranked, 'clip': cp, 'blend': blend}


def model_row(name, tag, tagcolour, value):
    return (f"<div class='row'><div class='rowtop'>"
            f"<span class='name'>{name}</span>"
            f"<span class='val'>{value}</span></div>"
            f"<div class='tag' style='color:{tagcolour};'>{tag}</div></div>")


def library():
    files = [p for p in sorted(glob.glob("Images/*.jpg") + glob.glob("Images/*.jpeg") +
                               glob.glob("Images/*.png"))
             if os.path.basename(p) not in EXCLUDE]
    seen, lib = set(), []
    for p in files:
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        lib.append(p)
    return lib


def picker():
    lib = library()
    if not lib:
        return
    cells = "".join(
        f"<a href='?pick={urllib.parse.quote(p)}' target='_self'>"
        f"<img src='{thumb_uri(p)}'></a>" for p in lib)
    with st.expander("No photo to hand? Try one of these", expanded=False):
        st.markdown(f"<div class='grid'>{cells}</div>", unsafe_allow_html=True)


def methodology():
    with st.expander("How this works, and how the 96.2% was measured", expanded=False):
        st.markdown("""
<div class="meth">
<p style="font-size:0.92rem;">Two models score every photograph and their scores are
blended. The result is right <b>96.2%</b> of the time on 2,961 images from 314 camera
locations that were held out of training and tuning entirely.</p>
</div>
""", unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["The models", "The 96.2%",
                                  "The threshold", "Limitations"])

        with t1:
            st.markdown("""
<div class="meth">
<p><b>ResNet18 v6</b> is a classifier fine-tuned on 8,000 images from the
Wildlife Conservation Society camera-trap dataset (1.37M images, 675 species, 12
countries). Answers one question: is there an animal here? <b>30% of the verdict.</b></p>
<p><b>MegaDetector v5a</b> is a detector built by the conservation-technology
community for camera-trap imagery. Draws boxes around animals, people and vehicles.
Its highest animal-box confidence is <b>70% of the verdict.</b></p>
<p>The 30/70 split was found by sweeping every weighting. Neither model alone matches
the blend: ResNet reaches 0.967 AUC, MegaDetector 0.969, the blend <b>0.991</b>. They
fail on different images, which is the whole reason to combine them.</p>
<p><b>CLIP</b> is shown for interpretation only and takes no part in the verdict.
After a search over 32 candidate phrases it reached 0.897 AUC, well short of the
other two.</p>
</div>
""", unsafe_allow_html=True)

        with t2:
            st.markdown("""
<div class="meth">
<p>Photographs from one camera share background, vegetation and lighting, and often
arrive in bursts seconds apart. Split such a dataset at random and near-identical
frames land on both sides, and a model can then score well by recognising
backgrounds rather than animals.</p>
<p>So this was evaluated on a <b>location-disjoint</b> split: 2,961 images from
<b>314 cameras that appear nowhere in training</b>, and which were never used to tune
the weights, the threshold or the calibration either.</p>
<p>The gap this closes is not small. An earlier version of this project reported
87&ndash;88% on a random split. Re-measured properly, that same model scored
<b>78.8%</b>.</p>
</div>
""", unsafe_allow_html=True)

        with t3:
            st.markdown("""
<div class="meth">
<p>MegaDetector returns a hard zero on roughly half of all frames. When it finds no
box, it contributes nothing rather than a low number. At 70% weight that
drags the combined score down even for real animals, so a 50% cutoff would discard
genuine detections.</p>
<p>Every value from 10% to 70% was swept, and 27.5% gave the best balance between
catching animals and correctly rejecting empty frames: <b>96.2%</b> accuracy, against
94.3% at 50%. The slider on the analysis page lets you move it and watch the verdicts
change.</p>
</div>
""", unsafe_allow_html=True)

        with t4:
            st.markdown("""
<div class="meth">
<p><b>Small animals.</b> Recall falls from about 0.90 for animals filling over 5% of
the frame to <b>0.485</b> below 1%, and small animals account for roughly 73% of all
misses. The cause is the resize: an animal filling 1% of a frame survives the resize
to 320&times;320 as about a 30-pixel patch, smaller than one cell of the network's
final feature map.</p>
<p><b>People and vehicles</b> count as <i>not animals</i>, following the source
dataset's convention.</p>
<p><b>Non-camera-trap photographs</b> such as macro shots, landscapes and studio
images give unreliable results, since nothing resembling them appears in training.</p>
</div>
""", unsafe_allow_html=True)


def footer():
    st.markdown(f"""
<div class="foot">
Built by <b style="color:{TEXT};">{AUTHOR}</b> &nbsp;&middot;&nbsp;
<a href="{REPO_URL}" target="_blank">source and development report</a><br>
Uses <a href="https://github.com/agentmorris/MegaDetector" target="_blank">MegaDetector
v5a</a> by Dan Morris and contributors, developed at Microsoft AI for Earth and now
maintained by the conservation-technology community, and
<a href="https://github.com/mlfoundations/open_clip" target="_blank">OpenCLIP</a>
(ViT-B-32) by LAION and contributors. Trained on
<a href="https://lila.science/datasets/wcscameratraps" target="_blank">WCS Camera
Traps</a>, contributed by the
<a href="https://www.wcs.org/" target="_blank">Wildlife Conservation Society</a> and
hosted by <a href="https://lila.science/" target="_blank">LILA BC</a> under the
Community Data License Agreement; earlier versions used
<a href="https://www.kaggle.com/c/iwildcam-2020-fgvc7" target="_blank">iWildCam
2020</a>. Built with <a href="https://pytorch.org/" target="_blank">PyTorch</a> and
<a href="https://streamlit.io/" target="_blank">Streamlit</a>.<br>
<span style="font-size:0.75rem;">A research and portfolio project, not a production
conservation tool. Accuracy may not transfer to other regions or species.</span>
</div>
""", unsafe_allow_html=True)


picked = st.query_params.get("pick")
if picked and os.path.exists(picked):
    st.session_state.source = ('lib', picked)
    st.query_params.clear()

st.markdown("## Camera Trap Wildlife Detector")
st.markdown(
    f"<p class='sub'>Most camera-trap photographs are empty. This tool tells you which "
    f"ones aren't, and it is <b style='color:{TEXT};'>right 96.2% of the time</b> on "
    f"cameras it has never seen.</p>", unsafe_allow_html=True)

methodology()

tab_single, tab_batch = st.tabs(["Single image", "Batch"])

with tab_single:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if up is not None:
        st.session_state.source = ('upload', hashlib.md5(up.getvalue()).hexdigest())

    src, key = None, None
    s = st.session_state.get('source')
    if s and s[0] == 'lib':
        src, key = Image.open(s[1]).convert("RGB"), s[1]
    elif s and s[0] == 'upload' and up is not None:
        src, key = Image.open(up).convert("RGB"), s[1]

    if src is None:
        picker()
    else:
        if st.session_state.get('key') != key:
            with st.status("Starting...", expanded=False) as stt:
                st.session_state.res = analyse(src, status=stt)
                stt.update(label="Done", state="complete")
            st.session_state.key = key
            st.session_state.img = src
            st.session_state.panel = 0

        r, img = st.session_state.res, st.session_state.img
        blend, rp, mp, cp = r['blend'], r['resnet'], r['md'], r['clip']

        thr = st.slider("Decision threshold - scores above this are flagged as animal",
                        min_value=0.05, max_value=0.75, value=DEFAULT_THRESHOLD,
                        step=0.005, format="%.3f", key='thr')
        if abs(thr - DEFAULT_THRESHOLD) > 1e-6:
            st.caption(f"Measured optimum is {DEFAULT_THRESHOLD:.1%}. "
                       f"You are currently at {thr:.1%}.")

        verdict = "Animal detected" if blend > thr else "Empty frame"
        phrase, colour = band(blend, thr)
        idx = st.session_state.get('panel', 0)

        left, right = st.columns([5, 4], gap="large")

        with left:
            if idx == 0:
                st.markdown(f"<div class='rank'><b>ResNet18 v6</b> ranks this "
                            f"<b style='color:{colour};'>{rp:.1%}</b> animal.</div>",
                            unsafe_allow_html=True)
                st.image(overlay_heatmap(r['cam'], img), width='stretch')
                cap = ("Brighter regions influenced the classifier most. Not proof it "
                       "recognises anatomy - only where its activations concentrated.")
            elif idx == 1:
                st.markdown(f"<div class='rank'><b>MegaDetector v5a</b> ranks this "
                            f"<b style='color:{colour};'>{mp:.1%}</b> animal.</div>"
                            if mp is not None else
                            "<div class='rank'><b>MegaDetector</b> is unavailable.</div>",
                            unsafe_allow_html=True)
                st.image(draw_boxes(img, r['dets']) if r['dets'] else img, width='stretch')
                cap = (f"{len(r['dets'])} candidate region(s) from MegaDetector."
                       if r['dets'] else "No candidate regions above minimum confidence.")
            else:
                st.markdown(f"<div class='rank'><b>CLIP</b> ranks this "
                            f"<b style='color:{colour};'>{cp:.1%}</b> animal.</div>",
                            unsafe_allow_html=True)
                top = r['ranked'][:5]
                fig, ax = plt.subplots(figsize=(6.2, 2.6))
                ax.barh(['\n'.join(textwrap.wrap(p, 34)) for p, _ in top],
                        [s_ * 100 for _, s_ in top],
                        color=[TEAL if p in ANIMAL_PHRASES else GREY for p, _ in top])
                ax.set_xlabel('Match %', fontsize=8)
                ax.invert_yaxis()
                ax.tick_params(axis='y', labelsize=6.5)
                ax.tick_params(axis='x', labelsize=7)
                for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                cap = "Teal = animal phrases, grey = empty-scene phrases."

            n1, n2, n3 = st.columns([1, 3, 1])
            if n1.button("<", width='stretch', key="prev"):
                st.session_state.panel = (idx - 1) % 3
                st.rerun()
            if n3.button(">", width='stretch', key="next"):
                st.session_state.panel = (idx + 1) % 3
                st.rerun()
            n2.markdown(f"<div style='text-align:center;color:{MUTED};font-size:0.82rem;"
                        f"padding-top:0.45rem;'>{PANELS[idx]} &nbsp; "
                        f"{' '.join('&#9679;' if i==idx else '&#9675;' for i in range(3))}"
                        f"</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='sub'>{cap}</p>", unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="card">
              <div class="verdict" style="color:{colour};">{verdict}</div>
              <div class="sub">{phrase}</div>
              <div style="margin:0.85rem 0 0.3rem 0;position:relative;height:10px;
                          background:{EDGE};border-radius:5px;">
                <div style="width:{blend*100:.1f}%;height:100%;background:{colour};
                            border-radius:5px;"></div>
                <div style="position:absolute;left:{thr*100:.1f}%;top:-4px;
                            height:18px;width:2px;background:{RED};"></div>
              </div>
              <div class="sub" style="display:flex;justify-content:space-between;">
                <span>combined animal score <b style="color:{TEXT};">{blend:.1%}</b></span>
                <span style="color:{RED};">decision line {thr:.1%}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            agree = (mp is not None) and ((rp > 0.5) == (mp > 0.5))
            note = ("Both scoring models point the same way." if agree else
                    "The scoring models disagree. MegaDetector carries more weight, so "
                    "it decides - but these are the frames worth a human look.")
            st.markdown(
                "<div class='card'>"
                + model_row("ResNet18 v6 ranks it", "30% of verdict", BLUE, f"{rp:.1%}")
                + model_row("MegaDetector v5a ranks it", "70% of verdict", TEAL,
                            f"{mp:.1%}" if mp is not None else "n/a")
                + model_row("CLIP ranks it", "not counted", MUTED, f"{cp:.1%}")
                + f"<div class='sub' style='margin-top:0.7rem;'>{note}</div></div>",
                unsafe_allow_html=True)

            st.markdown(
                "<div class='card'><div class='sub'>Animals filling under 5% of the frame "
                "are detected substantially less reliably. Full methodology is in the "
                "panel at the top of the page.</div></div>",
                unsafe_allow_html=True)

        st.divider()
        picker()

with tab_batch:
    st.markdown("<p class='sub'>Upload a folder's worth of frames. Roughly 5 seconds per "
                "image on CPU.</p>", unsafe_allow_html=True)
    ups = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"],
                           accept_multiple_files=True, key="batch")
    if ups and st.button(f"Analyse {len(ups)} images", key="runbatch"):
        tiles, bar = [], st.progress(0.0, text="Starting...")
        for i, f in enumerate(ups):
            bar.progress(i / len(ups), text=f"Image {i+1} of {len(ups)} - {f.name}")
            im = Image.open(f).convert("RGB")
            a = analyse(im)
            tiles.append({'thumb': thumb_img(im), 'name': f.name, 'blend': a['blend'],
                          'resnet': a['resnet'], 'md': a['md'], 'clip': a['clip'],
                          'boxes': len(a['dets'])})
        bar.empty()
        st.session_state.tiles = sorted(tiles, key=lambda t: -t['blend'])

    if 'tiles' in st.session_state:
        tiles = st.session_state.tiles

        bthr = st.slider("Decision threshold", min_value=0.05, max_value=0.75,
                         value=DEFAULT_THRESHOLD, step=0.005, format="%.3f", key='bthr')

        for t in tiles:
            t['verdict'] = "Animal" if t['blend'] > bthr else "Empty"
            t['assessment'] = band(t['blend'], bthr)[0]

        n_a = sum(t['verdict'] == "Animal" for t in tiles)
        n_e = len(tiles) - n_a
        n_u = sum(t['assessment'].startswith("Uncertain") for t in tiles)

        c1, c2, c3 = st.columns(3)
        c1.metric("Frames", len(tiles))
        c2.metric("Animal present", n_a)
        c3.metric("Need review", n_u)
        st.markdown(f"<p class='sub'>Discarding the {n_e} empty frames would cut manual "
                    f"review by {n_e/len(tiles):.0%}. Sorted by animal score, highest "
                    f"first.</p>", unsafe_allow_html=True)

        choice = st.radio("Show", [f"All ({len(tiles)})", f"Animal ({n_a})",
                                   f"Empty ({n_e})", f"Uncertain ({n_u})"],
                          horizontal=True, label_visibility="collapsed")
        if choice.startswith("Animal"):
            shown = [t for t in tiles if t['verdict'] == "Animal"]
        elif choice.startswith("Empty"):
            shown = [t for t in tiles if t['verdict'] == "Empty"]
        elif choice.startswith("Uncertain"):
            shown = [t for t in tiles if t['assessment'].startswith("Uncertain")]
        else:
            shown = tiles

        if not shown:
            st.markdown("<p class='sub'>Nothing in this category at the current "
                        "threshold.</p>", unsafe_allow_html=True)

        PER_ROW = 4
        for start in range(0, len(shown), PER_ROW):
            cols = st.columns(PER_ROW)
            for c, t in zip(cols, shown[start:start + PER_ROW]):
                with c:
                    st.image(t['thumb'], width='stretch')
                    col = band(t['blend'], bthr)[1]
                    md_txt = f"{t['md']:.0%}" if t['md'] is not None else "n/a"
                    st.markdown(
                        f"<div class='tile'>"
                        f"<b style='color:{col};font-size:1.15rem;'>{t['blend']:.0%}</b>"
                        f"<span style='color:{MUTED};'> combined</span><br>"
                        f"<span style='color:{col};'>{t['assessment']}</span><br>"
                        f"<span style='color:{MUTED};'>ResNet {t['resnet']:.0%} &middot; "
                        f"MegaDet {md_txt} &middot; CLIP {t['clip']:.0%}</span></div>",
                        unsafe_allow_html=True)

        df = pd.DataFrame([{'file': t['name'], 'verdict': t['verdict'],
                            'assessment': t['assessment'],
                            'animal score %': round(t['blend'] * 100, 1),
                            'ResNet %': round(t['resnet'] * 100, 1),
                            'MegaDetector %': round(t['md'] * 100, 1) if t['md'] is not None else None,
                            'CLIP %': round(t['clip'] * 100, 1),
                            'boxes': t['boxes']} for t in tiles])
        with st.expander("Full results table"):
            st.dataframe(df, width='stretch', hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False),
                           "camera_trap_results.csv", "text/csv")

footer()