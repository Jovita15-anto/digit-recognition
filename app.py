"""Digit Recognition app: upload, photograph or pick an image and read every digit in it."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from digit_model import annotate, classify_whole_image, recognize

ROOT = Path(__file__).parent
SAMPLES = {
    "2025": "samples/number_2025.png",
    "7305": "samples/number_7305.png",
    "Single 2": "two.jpeg",
    "Single 0": "zero.jpeg",
}
CONFIDENT = 0.60   # below this a digit is flagged as uncertain
MAX_SIDE = 1600    # very large photos are shrunk before processing
MULTI, SINGLE = "Multiple digits", "Single digit"

st.set_page_config(page_title="Digit Recognition", page_icon="🔢", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family:'DM Sans',system-ui,sans-serif;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1080px; padding-top:1.4rem; padding-bottom:3rem;}
.nav {display:flex; justify-content:space-between; align-items:center; padding:6px 0 24px;}
.brand {display:flex; align-items:center; gap:10px; font-weight:700; font-size:19px;}
.logo {width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,#3547E8,#6a7bff); color:#fff; display:grid; place-items:center; font-size:17px;}
.pill {font-size:13px; color:#5b6485; background:#fff; border:1px solid #d9deec; border-radius:999px; padding:5px 13px;}
.hero h1 {font-size:clamp(30px,5vw,48px); line-height:1.08; letter-spacing:-.03em; margin:0 0 10px; font-weight:700;}
.hero p {color:#5b6485; font-size:18px; max-width:620px; margin:0 0 22px;}
.label {font-size:13px; font-weight:600; color:#5b6485; margin:0 0 8px;}
.result {background:#fff; border:1px solid #d9deec; border-radius:14px; padding:20px 22px;}
.number {font-size:clamp(44px,8vw,80px); font-weight:700; letter-spacing:.12em; line-height:1.1; color:#1B2340; word-break:break-all;}
.meta {font-size:14px; color:#7a83a6; margin-top:6px;}
.note {font-size:14px; border-radius:10px; padding:10px 14px; margin-top:14px; background:#fff7ea; color:#9a4a00; border:1px solid #f3d9ad;}
.dcard {text-align:center; margin-top:4px;}
.dd {font-size:30px; font-weight:700; line-height:1.1;}
.cf {font-size:13px; font-weight:600; color:#1a7f4b;}
.cf.low {color:#c25a00;}
.stButton > button {border-radius:10px; font-weight:600;}
.foot {color:#7a83a6; font-size:13px; text-align:center; margin-top:42px;}
</style>
<div class="nav">
  <div class="brand"><div class="logo">🔢</div>Digit Recognition</div>
  <span class="pill">PyTorch CNN · 0–9</span>
</div>
<div class="hero">
  <h1>Read any digits from an image.</h1>
  <p>Upload a photo, take a picture or try a sample. Every digit is found, cropped and classified by a convolutional neural network.</p>
</div>
""",
    unsafe_allow_html=True,
)


def decode(data):
    """Bytes of an image file -> BGR array (or None); huge images are scaled down."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None and max(img.shape[:2]) > MAX_SIDE:
        s = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


c1, c2 = st.columns([3, 2])
source = c1.radio("Image source", ["Upload", "Camera", "Sample"], horizontal=True)
mode = c2.radio("Mode", [MULTI, SINGLE], horizontal=True,
                help="Multiple: finds every digit in the image. Single: treats the whole image as one digit.")

image = None
if source == "Upload":
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if f:
        image = decode(f.getvalue())
elif source == "Camera":
    if st.toggle("Turn on camera"):
        shot = st.camera_input("Photo of the digits", label_visibility="collapsed")
        if shot:
            image = decode(shot.getvalue())
else:
    name = st.radio("Sample", list(SAMPLES), horizontal=True, label_visibility="collapsed")
    image = cv2.imread(str(ROOT / SAMPLES[name]))

if image is None:
    if source == "Upload":
        st.info("Upload a clear image of one or more digits to get started. Dark digits on a light background work best.")
else:
    result = classify_whole_image(image) if mode == SINGLE else recognize(image)
    digits = result["digits"]

    if not digits:
        st.warning("No digits found. Try a sharper image where the digits contrast clearly with the background.")
    else:
        confs = [d["confidence"] for d in digits]
        uncertain = [i for i, c in enumerate(confs) if c < CONFIDENT]

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown('<div class="label">Detected digits</div>', unsafe_allow_html=True)
            st.image(annotate(image, digits))
        with right:
            st.markdown('<div class="label">Recognised number</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result"><div class="number">{result["number"]}</div>'
                f'<div class="meta">{len(digits)} digit{"s" if len(digits) != 1 else ""} · '
                f'average confidence {np.mean(confs):.0%}</div></div>',
                unsafe_allow_html=True,
            )
            if uncertain:
                pos = ", ".join(str(i + 1) for i in uncertain)
                st.markdown(
                    f'<div class="note">Low confidence for digit {pos}. '
                    "Small 8×8 models often confuse 6, 8 and 9, so double-check those.</div>",
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="label" style="margin-top:26px">Digit by digit · what the CNN sees (8×8)</div>', unsafe_allow_html=True)
        for start in range(0, len(digits), 8):
            chunk = digits[start:start + 8]
            cols = st.columns(8)
            for col, d in zip(cols, chunk):
                big = cv2.resize(d["tile"], (96, 96), interpolation=cv2.INTER_NEAREST)
                col.image(big, width=80)
                low = "low" if d["confidence"] < CONFIDENT else ""
                col.markdown(
                    f'<div class="dcard"><div class="dd">{d["digit"]}</div>'
                    f'<div class="cf {low}">{d["confidence"]:.0%}</div></div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Inspect probabilities"):
            idx = st.selectbox(
                "Digit", range(len(digits)),
                format_func=lambda i: f"Digit {i + 1} (predicted {digits[i]['digit']})",
            )
            probs = digits[idx]["probs"]
            top = np.argsort(probs)[::-1][:3]
            st.write("Top 3: " + ", ".join(f"**{int(k)}** ({probs[k]:.1%})" for k in top))
            st.bar_chart(pd.DataFrame({"probability": probs}, index=[str(i) for i in range(10)]))

with st.expander("How it works"):
    st.markdown(
        """
1. **Grayscale** the image and make the ink bright and the paper dark (automatic, so light-on-dark also works).
2. **Find each digit** with contour detection and crop it into a centred square (multiple-digit mode).
3. **Resize to 8 × 8** pixels, the input size the network was trained on.
4. A **CNN** (2 convolution + max-pool blocks, then two linear layers) outputs 10 scores.
5. **Softmax** turns the scores into probabilities and **argmax** picks the digit.

**Limits:** the network is intentionally small and works on single printed or clearly written digits (0–9). Messy handwriting, touching digits, or 6, 8 and 9 in unusual styles can be misread, which is why low-confidence digits are flagged.
        """
    )

st.markdown('<div class="foot">Built with PyTorch, OpenCV and Streamlit</div>', unsafe_allow_html=True)
