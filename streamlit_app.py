import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog, local_binary_pattern
from PIL import Image

# Load model + scaler
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# vread() same as before
def vread(basepath):
    spr_path = basepath + ".spr"
    sdt_path = basepath + ".sdt"

    with open(spr_path, "r") as f:
        ndim = int(f.readline().strip())
        nc = int(f.readline().strip())
        _ = f.readline()
        _ = f.readline()
        nr = int(f.readline().strip())
        _ = f.readline()
        _ = f.readline()
        dtype_code = int(f.readline().strip())

    dtype = {0: np.uint8, 2: np.int32, 3: np.float32, 5: np.float64}[dtype_code]

    with open(sdt_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)

    return data.reshape((nr, nc))

def extract_features(img):
    img = cv2.resize(img, (64, 64))

    fd, _ = hog(img, orientations=8, pixels_per_cell=(8,8),
                cells_per_block=(2,2), visualize=True, block_norm='L2')

    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)

    return np.hstack([fd, hist])


st.title("Venus Volcano Detection (SVM + HOG + LBP)")

uploaded = st.file_uploader("Upload SAR .sdt file", type=["sdt"])

if uploaded:
    basepath = uploaded.name[:-4]
    
    with open(basepath + ".sdt", "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Upload the matching .spr file")
    uploaded_spr = st.file_uploader("Upload matching .spr", type=["spr"])

    if uploaded_spr:
        with open(basepath + ".spr", "wb") as f:
            f.write(uploaded_spr.getbuffer())

        img = vread(basepath)
        img_disp = img.copy()

        win = 64
        step = 32

        boxes = []
        scores = []

        heatmap = np.zeros((
            (img.shape[0] - win) // step + 1,
            (img.shape[1] - win) // step + 1
        ), dtype=np.float32)

        heat_y = 0
        for y in range(0, img.shape[0] - win, step):
            heat_x = 0
            for x in range(0, img.shape[1] - win, step):

                crop = img[y:y+win, x:x+win]
                feat = extract_features(crop)
                feat = scaler.transform([feat])

                prob = svm.predict_proba(feat)[0][1]
                heatmap[heat_y, heat_x] = prob

                # detection threshold
                if prob > 0.5:
                    boxes.append((x, y, prob))

                heat_x += 1
            heat_y += 1

        heatmap_norm = (heatmap * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(
            heatmap_norm,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    

        # for (x, y, p) in boxes:
        #     color = (0, int(255*p), 0)
        #     cv2.rectangle(img_disp, (x,y), (x+win, y+win), color, 2)

        # st.image(img_disp, caption="Detected volcanoes", use_column_width=True)


        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_disp_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

        for (x, y, p) in boxes:
            if p > 0.85:
                color = (0, 255, 0)
                label = "High" # Green
            elif p > 0.70:
                color = (0, 255, 255)
                label = "Med" # Yella
            else:
                color = (0, 0, 255)
                label = "Low" # Red

            cv2.rectangle(img_disp_color, (x, y), (x+win, y+win), color, 2)
            
            text = f"{p:.2f}"
            cv2.putText(img_disp_color, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        img_rgb = cv2.cvtColor(img_disp_color, cv2.COLOR_BGR2RGB)
        
        st.image(img_rgb, caption="Detected volcanoes (Green = High, Yellow = Med, Red = Low)", use_column_width=True)
        

        overlay = cv2.addWeighted(
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR),
            0.6,
            heatmap_color,
            0.4,
            0
        )

        st.subheader("Volcano Confidence Heatmap")
        st.image(overlay, use_column_width=True)
