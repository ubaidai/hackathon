import joblib
import streamlit as st
from PIL import Image
import numpy as np

# ---- Load model once ----
lr = joblib.load("logistic_regression.pkl")

# ---- Page configuration ----
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="centered",
)

# ---- Title & Subtitle ----
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🖐 Hand Gesture Recognition</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6A5ACD;'>Upload an image and get instant predictions</h4>", unsafe_allow_html=True)
st.write("---")

# ---- Upload Section ----
upload_file = st.file_uploader("Upload an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

IMG_SIZE = (64, 64)
CLASS_NAMES = ['0', '1', '10', '11', '12', '13', '14', '15', '16',
               '17', '18', '19', '2', '3', '4', '5', '6', '7', '8', '9']

def preprocess_like_training(file_obj):
    img = Image.open(file_obj).convert("L")   # grayscale
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    flat = arr.reshape(1, -1)
    pred_idx = lr.predict(flat)[0]
    label_name = CLASS_NAMES[pred_idx]
    return flat, arr, label_name, pred_idx

# ---- Prediction ----
if upload_file is not None:
    st.subheader("Uploaded Image")
    st.image(upload_file, width=200, caption="Original Image")

    flat, arr, label_name, pred_idx = preprocess_like_training(upload_file)

    st.success(f"**Predicted Class:** {label_name} (Index: {pred_idx})")

    with st.expander("See Preprocessed Image (64x64 Grayscale)"):
        st.image(arr, width=150, caption="Preprocessed Image")

st.write("---")

# ---- Model Evaluation Section ----
st.markdown("<h2 style='text-align: center; color: #4B0082;'>📊 Model Evaluation</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample Dataset")
    sampledataset_img = Image.open("sample_dataset.png")
    st.image(sampledataset_img, caption="Sample Dataset", use_column_width=True)

with col2:
    st.subheader("Confusion Matrix")
    cm_img = Image.open("confusion_matrix.png")
    st.image(cm_img, caption="Confusion Matrix", use_column_width=True)

st.subheader("Per Class Accuracy")
per_class_img = Image.open("perr_class_accuracy.png")
st.image(per_class_img, caption="Per Class Accuracy", use_column_width=True)

st.markdown("<h4 style='text-align: center; color: #6A5ACD;'>Built with ❤️ using Python, Streamlit & scikit-learn</h4>", unsafe_allow_html=True)

