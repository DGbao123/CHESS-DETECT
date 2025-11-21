import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import time
# import firebase_admin   # <-- removed
# from firebase_admin import credentials,db  # <-- removed
import av
import threading
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh
import time
from PIL import Image
import numpy as np
from datetime import datetime
from streamlit_elements import elements, mui, html
import requests
import json

# -------------------------------
# CẤU HÌNH DATABASE (THAY BẰNG PROJECT CỦA BẠN)
# -------------------------------
DATABASE_URL = "https://check-detect-80389-default-rtdb.firebaseio.com/"  # nhớ có / ở cuối

# -------------------------------
# HÀM REST API TIỆN ÍCH
# -------------------------------
def _url(path: str):
    # đảm bảo không có slash dư
    path = path.lstrip('/')
    return f"{DATABASE_URL}{path}.json"

def read_data(path: str):
    try:
        res = requests.get(_url(path), timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu từ Firebase: {e}")
        return None

def write_data(path: str, data):
    """Ghi (PUT) đè dữ liệu tại path"""
    try:
        res = requests.put(_url(path), json=data, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Lỗi khi ghi dữ liệu vào Firebase: {e}")
        return None

def push_data(path: str, data):
    """Push (POST) tạo key mới"""
    try:
        res = requests.post(_url(path), json=data, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f" Lỗi khi push dữ liệu vào Firebase: {e}")
        return None

def patch_data(path: str, data):
    """Cập nhật 1 phần (PATCH)"""
    try:
        res = requests.patch(_url(path), json=data, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f" Lỗi khi patch dữ liệu vào Firebase: {e}")
        return None

# -------------------------------
# XỬ LÝ MODEL
# -------------------------------
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

st.set_page_config(
    page_title="CHESS ♔",
    page_icon="♘",
    initial_sidebar_state="expanded",
    layout="wide")
st.title("CHESS DETECT ♘")
st.write("Ứng dụng này giúp bạn ghi lại các nước cờ một cách tự động.")

model_path = 'best.pt'
try:
    model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Lỗi khi tải model: {e}")
    st.stop()

# ===========================
# WEBCAM DETECTION (giữ nguyên)
# ===========================
def func_detect_webcam():
    detections_container = {"detections": []}
    lock = threading.Lock()

    class YoloVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.lock = lock
            self.container = detections_container

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img, stream=True, verbose=False)
            detections_list = []
            annotated_frame = img.copy()
            for r in results:
                annotated_frame = r.plot()
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    detections_list.append({
                        "Vật thể": label,
                        "Độ tự tin": conf,
                        "Tọa độ": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                    })
            with self.lock:
                self.container["detections"] = detections_list
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    st.subheader("📹 Video Webcam")
    ctx = webrtc_streamer(
        key="yolo_webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YoloVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.subheader("📊 Danh sách vật thể phát hiện (Realtime)")
    st_autorefresh(interval=500, key="data_refresh")
    placeholder = st.empty()

    if ctx.video_processor:
        with lock:
            detections = ctx.video_processor.container["detections"]
        if detections:
            df = pd.DataFrame(detections)
            placeholder.dataframe(df.style.format({"Độ tự tin": "{:.2%}"}), use_container_width=True)
        else:
            placeholder.write("⏳ Chưa phát hiện vật thể nào...")
    else:
        placeholder.write("⏸ Webcam chưa bật hoặc đang tạm dừng.")

# ===========================
# IMAGE DETECTION
# ===========================
def func_detect_imgs():
    uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png)", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)
        results = model(img)
        result_img = results[0].plot()
        st.subheader("Ảnh sau khi detect quân cờ")
        st.image(result_img, use_column_width=True)

        height, width, _ = img.shape
        boxes = results[0].boxes
        yolo_results = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label_id = int(box.cls[0])
            label = model.names[label_id]
            yolo_results.append((label, x1, y1, x2, y2))

        cell_width = width / 8
        cell_height = height / 8

        def get_square_name(center_x, center_y):
            col = int(center_x / cell_width)
            row = int(center_y / cell_height)
            col = min(max(col, 0), 7)
            row = min(max(row, 0), 7)
            square = chr(ord('a') + col) + str(8 - row)
            return square

        piece_positions = {}
        for label, x1, y1, x2, y2 in yolo_results:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            square = get_square_name(center_x, center_y)
            piece_positions[square] = label

        st.subheader("Vị trí quân cờ trên bàn cờ")
        for square, piece in piece_positions.items():
            st.write(f"{piece} ở ô {square}")
        return piece_positions

# ===========================
# Realtime DB functions (thay firebase_admin)
# ===========================
def take_data_match(email):
    """Lấy danh sách match cho user (email trước @)"""
    key = email.split('@')[0].replace('.', '_')
    data = read_data(f"match_data/{key}")
    st.session_state.data_match = data

def add_match(data_match, email):
    """Thêm 1 ván cờ mới cho user"""
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email_key = email.split('@')[0].replace('.', '_')

    existing = read_data(f"match_data/{email_key}")
    if existing is None:
        # tạo mới
        payload = {
            "quantity": 1,
            f"match-1": {
                "time": now_time,
                "data": data_match
            }
        }
        write_data(f"match_data/{email_key}", payload)
    else:
        qty = existing.get("quantity", 0)
        new_id = f"match-{qty + 1}"
        # thêm match mới (PATCH để ko ghi đè)
        patch_data(f"match_data/{email_key}/{new_id}", {"time": now_time, "data": data_match})
        # cập nhật quantity
        patch_data(f"match_data/{email_key}", {"quantity": qty + 1})
    take_data_match(email)

def add_user(information):
    """Thêm user (ghi đè key là phần trước @ để dễ truy xuất)"""
    key = information['email'].split('@')[0].replace('.', '_')
    user_payload = {
        "email": information['email'],
        "name": information['name'],
        "age": information['age'],
        "gender": information['gender'],
        "password": information['password']
    }
    write_data(f"users/{key}", user_payload)

def take_data_user(email_user, password_user):
    key = email_user.split('@')[0].replace('.', '_')
    data = read_data(f"users/{key}")
    if data is None:
        st.info('Tài khoản không tồn tại!')
        return False
    if password_user == data.get('password'):
        st.success('Đăng nhập thành công!')
        if 'inforlogin' not in st.session_state:
            # copy inforlogin and remove password
            safe = dict(data)
            safe.pop('password', None)
            st.session_state.inforlogin = safe
            take_data_match(data['email'])
        return True
    else:
        st.info('Email hoặc mật khẩu sai')
        return False

# ===========================
# UI chính (giữ logic của bạn)
# ===========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🏠 Trang chủ"):
            st.session_state.page = "home"
            st.rerun()
        st.title("🏠 Trang chủ")
        
    with col2:
        if st.button("♟️ Ván cờ"):
            if "inforlogin" in st.session_state:
                st.session_state.page = "games"
                st.rerun()
            else:
                p=st.error('Vui lòng đăng nhập tài khoản')
                time.sleep(3)
                p.empty()
        
    with col3:
        if st.button("👤 Hồ sơ"):
            st.session_state.page = "profile"
            st.rerun()
    with col4:
        if st.button("⚙️ Cài đặt"):
            st.session_state.page = "settings"
            st.rerun()
    if st.checkbox('Mở detect'):
        if 'inforlogin' in st.session_state:
            detect_section = st.radio('',['Webcam', 'Images'], horizontal=True)
            if detect_section == 'Webcam':
                func_detect_webcam()
            elif detect_section == 'Images':
                re = func_detect_imgs()
                if re:
                    add_match(re, st.session_state.inforlogin['email'])
        else:
            st.info('Vui lòng đăng nhập!')

elif st.session_state.page == "games":
    col_GoBack_DataOld, col_title_DataOld = st.columns(2)
    with col_GoBack_DataOld:
        if st.button("⬅️ Quay lại"):
            st.session_state.page = "home"
            st.rerun()
    with col_title_DataOld:
        st.title('Ván cờ')
    # hiển thị ván cờ đã lưu
    data = st.session_state.get('data_match', None)
    if data:
        for name in data:
            if name != "quantity":
                time_ = data[name]["time"]
                dict_data = data[name]["data"]
                st.write(f"### 🕹️ {name}")
                st.write(f"**Ngày tạo:** {time_}")
                with st.expander("📄 Xem dữ liệu ván cờ"):
                    st.json(dict_data)
                st.divider()
    else:
        st.info("Không có dữ liệu ván cờ.")

elif st.session_state.page == "profile":
    if st.button("⬅️ Quay lại"):
        st.session_state.page = "home"
        st.session_state.login_register = "login"
        st.rerun()
    if "inforlogin" in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with elements("info_page"):
                mui.Typography("INFORMATIONS", variant="h4", style={"marginBottom": "20px", 
                        'backgroundColor':"#352f57", "borderRadius": "10px", 'text-align':'center', "border": "2px solid #ffffff"})
                for key, value in st.session_state.inforlogin.items():
                    with mui.Card(style={"padding": "15px", "marginBottom": "10px", 
                                         "borderRadius": "10px", "border": "2px solid #ffffff"}):
                        mui.Typography(f"{key.upper()}: {value}", variant="body1")
                if st.button('Đăng Xuất'): 
                    del st.session_state.inforlogin 
                    st.rerun()
    else:
        if "login_register" not in st.session_state:
            st.session_state.login_register = "login"
        col1, col2, col3 = st.columns([1, 2, 1])
        if st.session_state.login_register == "login":
            with col2:
                with st.container():
                    st.subheader("LOGIN")
                    email_login = st.text_input('Email')
                    password_login = st.text_input('Mật Khẩu')
                    co1, co2 = st.columns(2)
                    with co1:
                        button_login = st.button('Login')
                    with co2:
                        if st.button('-Đăng ký tài khoản-'):
                            st.session_state.login_register = "register"
                            st.rerun()
            if button_login:
                if take_data_user(email_login,password_login):
                    st.session_state.page = "home"
                    st.rerun()

        if st.session_state.login_register == 'register':
            with col2:
                with st.container():
                    information = {"email":None, "name":None, "age":None, "gender":None, "password":None}
                    st.subheader('REGISTER')
                    information["email"] = st.text_input('Email')
                    information["name"] = st.text_input('Họ & tên')
                    information["age"] = st.number_input('Tuổi')
                    information["gender"] = st.radio("Giới tính: ", ['', "Nam", "Nữ"], horizontal=True)
                    information["password"] = st.text_input('Mật Khẩu', type="password")
                    confirm_password = st.text_input('Xác nhận mật khẩu', type="password")
                    check_information=""
                    co1, co2 = st.columns(2)
                    with co1:
                        if st.button('Register'):
                            for i in information:
                                if information[i]=='' or information[i]==0.0 or information[i] is None:
                                    check_information+=i+' ,'
                            if check_information!='':
                                placeholder = st.empty()
                                with placeholder.container():
                                    st.error("Hãy nhập thêm các thông tin: "+ check_information[:-2])
                            else:
                                c1, c2 = True, True
                                check_email=information["email"]
                                if '@' not in check_email or '.' not in check_email:
                                    st.error('Email không hợp lệ!')
                                    c1 = False
                                if information["password"]!=confirm_password:
                                    st.error('Xác nhận mật khẩu chưa đúng!')
                                    c2 = False
                                if c1 and c2:
                                    placeholder = st.empty()
                                    with placeholder.container():
                                        st.success("Tạm thời chắc được rồi đó")
                                        # chuẩn hóa email lưu vào DB (giữ nguyên email thật cho hiển thị)
                                        info_to_save = dict(information)
                                        add_user(info_to_save)
                                    time.sleep(1)
                                    st.session_state.login_register = "login"
                                    st.rerun()
                    with co2:
                        if st.button('-Đăng nhập tài khoản-'):
                            st.session_state.login_register = "login"
                            st.rerun()

elif st.session_state.page == 'settings':
    if st.button("⬅️ Quay lại"):
        st.session_state.page = "home"
        st.rerun()
