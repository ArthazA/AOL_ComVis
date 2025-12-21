import streamlit as st
import numpy as np
import cv2
import joblib
import os
from skimage.feature import hog, graycomatrix, graycoprops


bundle = joblib.load("volcano_detector.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
CHIP_SIZE = bundle["chip_size"]


def vread_bytes(sdt_bytes, spr_bytes):
    lines = spr_bytes.decode("utf-8").splitlines()
    idx = 0

    ndim = int(lines[idx].strip()); idx += 1
    if ndim != 2:
        raise ValueError("Only 2D images supported")

    nc = int(lines[idx].strip()); idx += 1
    idx += 2 
    nr = int(lines[idx].strip()); idx += 1
    idx += 2 

    dtype_code = int(lines[idx].strip())
    dtype_map = {
        0: np.uint8,
        2: np.int32,
        3: np.float32,
        5: np.float64
    }

    if dtype_code not in dtype_map:
        raise ValueError(f"Unknown dtype code {dtype_code}")

    dtype = dtype_map[dtype_code]

    data = np.frombuffer(sdt_bytes, dtype=dtype)
    data = data[:nr * nc]

    return data.reshape((nr, nc))

def load_lxyv(path):
    points = []
    with open(path, "r") as f:
        for line in f:
            _, x, y, value = line.strip().split()
            points.append((int(float(x)), int(float(y)), float(value)))
    return points

def preprocess(img):
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
    return clahe.apply(img.astype(np.uint8))

def extract_chip(img, cx, cy, size):
    h, w = img.shape
    half = size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    chip = img[y1:y2, x1:x2]

    return cv2.copyMakeBorder(
        chip,
        top=max(0, size - chip.shape[0]),
        bottom=0,
        left=max(0, size - chip.shape[1]),
        right=0,
        borderType=cv2.BORDER_REFLECT
    )


def extract_features(chip):
    chip = cv2.resize(chip, (32, 32))

    hog_feat = hog(
        chip,
        orientations=6,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2",
        visualize=False
    )

    glcm = graycomatrix(
        chip,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    glcm_feats = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        glcm_feats.extend(graycoprops(glcm, prop).flatten())

    return np.hstack([hog_feat, glcm_feats])



# UI

st.title("🌋 Venus Volcano Detector (FOA-based)")

sdt_file = st.file_uploader("Upload .sdt file", type=["sdt"])
spr_file = st.file_uploader("Upload matching .spr file", type=["spr"])

if sdt_file and spr_file:

    img_id = sdt_file.name.replace(".sdt", "")

    img = vread_bytes(sdt_file.getvalue(), spr_file.getvalue())
    img = preprocess(img)

    foa_path = f"./package/FOA/exp_C/exp_C1/tst/{img_id}.lxyv" # img79 until img134
    if not os.path.exists(foa_path):
        st.error(f"No FOA test file found for {img_id}")
        st.stop()

    foa_points = load_lxyv(foa_path)

    heatmap = np.zeros(img.shape, dtype=np.float32)
    detections = []

    for x, y, _ in foa_points:
        chip = extract_chip(img, x, y, CHIP_SIZE)
        feat = extract_features(chip)
        feat = scaler.transform([feat])

        prob = model.predict_proba(feat)[0, 1]
        detections.append((x, y, prob))
        heatmap[y, x] = max(heatmap[y, x], prob)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15)
    heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    img_vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    half = CHIP_SIZE // 2
    for x, y, p in detections:
        if p < 0.5:
            continue

        if p >= 0.85:
            color = (0, 255, 0) # Shrek
        elif p >= 0.7:
            color = (0, 255, 255) # Piss Yella
        else:
            color = (0, 0, 255) # . Red

        cv2.rectangle(
            img_vis,
            (x - half, y - half),
            (x + half, y + half),
            color,
            2
        )
        cv2.putText(
            img_vis,
            f"{p:.2f}",
            (x - half, y - half - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    st.subheader("Detected Volcanoes")
    st.image(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

    overlay = cv2.addWeighted(
        cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    st.subheader("Probability Heatmap")
    st.image(overlay, use_column_width=True)