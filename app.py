import streamlit as st
import numpy as np
import joblib
import requests
from io import BytesIO
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Face Emotion Detection", layout="wide", page_icon="😊")

# =========================
# LOAD MODEL FROM GOOGLE DRIVE
# =========================
def load_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    res = requests.get(url)
    res.raise_for_status()  # ตรวจสอบว่าโหลดสำเร็จ
    return joblib.load(BytesIO(res.content))

# File ID ของ Google Drive (ของคุณ)
MODEL_ML_ID = "1kKjdFPWTg4I01EGb0vxQLIA3smpHkebn"
MODEL_NN_ID = "1RfiC2ZGh_idd4cwi2nsNw-2QKW65HZRQ"

# โหลดโมเดล
with st.spinner("Loading ML model..."):
    model_ml = load_model_from_drive(MODEL_ML_ID)
with st.spinner("Loading NN model..."):
    model_nn = load_model_from_drive(MODEL_NN_ID)

# =========================
# HEADER
# =========================
st.title("😊 Face Emotion Detection System 😎")
st.markdown("""
วิเคราะห์อารมณ์จากภาพใบหน้าโดยใช้ Machine Learning และ Neural Network 🧠  
ระบบนี้ออกแบบเพื่อเรียนรู้ตั้งแต่ Data Preparation จนถึงการ Deploy 🚀
""")

# =========================
# SIDEBAR MENU
# =========================
page = st.sidebar.radio("Menu", [
    "Overview",
    "Dataset",
    "Data Preparation",
    "Machine Learning",
    "Neural Network",
    "Test ML",
    "Test Neural Network"
])

# =========================
# TEST ML
# =========================
if page == "Test ML":
    st.header("🖼️ Test Machine Learning Model")
    uploaded = st.file_uploader("Upload Image", type=["jpg","png"])
    if uploaded:
        # แยกภาพสำหรับโมเดล และสำหรับโชว์
        img_for_show = Image.open(uploaded)  # ต้นฉบับสำหรับโชว์
        img_for_model = img_for_show.convert('L').resize((48,48))  # สำหรับโมเดล
        img_arr = np.array(img_for_model).flatten()

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_for_show, caption="Input Image", use_column_width=True)
        with col2:
            pred = model_ml.predict([img_arr])
            st.success(f"Prediction: {pred[0]} 🎉")

# =========================
# TEST NN
# =========================
elif page == "Test Neural Network":
    st.header("🖼️ Test Neural Network Model")
    uploaded = st.file_uploader("Upload Image", type=["jpg","png"])
    if uploaded:
        # แยกภาพสำหรับโมเดล และสำหรับโชว์
        img_for_show = Image.open(uploaded)
        img_for_model = img_for_show.convert('L').resize((48,48))
        img_arr = np.array(img_for_model).flatten()

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_for_show, caption="Input Image", use_column_width=True)
        with col2:
            pred = model_nn.predict([img_arr])
            st.success(f"Prediction: {pred[0]} 🎉")