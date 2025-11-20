import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials,db
import av
import threading
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh
import time
from PIL import Image
import numpy as np
from datetime import datetime
from streamlit_elements import elements, mui, html
#connect firebase google
@st.cache_resource
def init_firebase():
    SERVICE_ACCOUNT_PATH = r"D:\MINDX\YOLO_projects\check-detect-80389-firebase-adminsdk-fbsvc-3786272c2d.json"
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://check-detect-80389-default-rtdb.firebaseio.com/'
    })
    return True
if 'fire_base_create' not in st.session_state:
    st.session_state.fire_base_create = True
# ---- Gọi hàm ----
if st.session_state.fire_base_create:
    try:
        init_firebase()
        create_firebase = st.success("Firebase Admin SDK đã khởi tạo thành công ✅")
        time.sleep(3)
        create_firebase.empty()
        st.session_state.fire_base_create = False
    except Exception as e:
        st.error(f"Lỗi khởi tạo Firebase: {e}")
# Sử dụng cache của Streamlit để tải model chỉ một lần
@st.cache_resource
def load_yolo_model(model_path):
    """
    Tải model YOLOv8 từ đường dẫn.
    """
    model = YOLO(model_path)
    return model

# ---- Cấu hình chính của App ----
st.set_page_config(
    page_title="CHESS ♔",
    page_icon="♘",
    initial_sidebar_state="expanded",
    layout="wide")
st.title("CHESS DETECT ♘")
st.write("Ứng dụng này giúp bạn ghi lại các nước cờ một cách tự động.")
# ---- Lựa chọn Model ----
model_path = 'runs/detect/chess_model5/weights/best.pt'  # Mặc định dùng model 'n'
# # (Tùy chọn): Bạn có thể cho phép người dùng upload model
# # uploaded_model = st.file_uploader("Hoặc tải lên file model (.pt) của bạn", type="pt")
# # if uploaded_model:
# #     # Lưu file tạm
# #     tfile = tempfile.NamedTemporaryFile(delete=False) 
# #     tfile.write(uploaded_model.read())
# #     model_path = tfile.name

# Tải model
try:
    model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Lỗi khi tải model: {e}")
    st.stop()

# logic chạy webcam
def func_detect_webcam():
    # ----- Biến toàn cục để chia sẻ giữa các thread -----
    # ---- Biến toàn cục (để chia sẻ dữ liệu giữa các thread) ----
    detections_container = {"detections": []}
    lock = threading.Lock()
    # ===============================
    # ---- 4. Class xử lý video ----
    # ===============================
    class YoloVideoProcessor(VideoProcessorBase):
        global detections_container
        def __init__(self):
            # Debug: xem class có được gọi không
            # print("✅ Khởi tạo YOLO Video Processor!")
            self.model = model
            self.lock = lock
            self.container = detections_container

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Chạy detect
            results = self.model(img, stream=True, verbose=False) 
            detections_list = []
            annotated_frame = img.copy() # Phải copy
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
    # ===============================
    # ---- 5. Streamlit WebRTC ----
    # ===============================
    st.subheader("📹 Video Webcam")
    ctx = webrtc_streamer(
        key="yolo_webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YoloVideoProcessor,  # khởi tạo class
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ===============================
    # ---- 6. Bảng dữ liệu realtime ----
    # ===============================
    st.subheader("📊 Danh sách vật thể phát hiện (Realtime)")
    st_autorefresh(interval=500, key="data_refresh")
    placeholder = st.empty()

    # Cập nhật dataframe từ detections_container
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

def func_detect_imgs():
    uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png)", type=["jpg", "png"])
    if uploaded_file is not None:
        # Chuyển sang PIL Image và hiển thị ngay ảnh gốc
        image = Image.open(uploaded_file).convert("RGB")
        # st.subheader("Ảnh gốc")
        # st.image(image, use_column_width=True)

        # Chuyển sang numpy để YOLO detect
        img = np.array(image)

        # Detect
        results = model(img)
        result_img = results[0].plot()
        st.subheader("Ảnh sau khi detect quân cờ")
        st.image(result_img, use_column_width=True)
        results = model(img)
        height, width, _ = img.shape

        # Trích xuất bounding boxes
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
        # for square, piece in piece_positions.items():
        #     st.write(f"{piece} ở ô {square}")
        return piece_positions


#display các ván cờ đã ghi lạ
def display_match_old():
    data = st.session_state.data_match
    if data != None:
        for name in data:
            if name != "quantity":
                time = data[name]["time"]
                dict_data = data[name]["data"]

                st.write(f"### 🕹️ {name}")
                st.write(f"**Ngày tạo:** {time}")

                with st.expander("📄 Xem dữ liệu ván cờ"):
                    st.json(dict_data)

                st.divider()

#take data old match
def take_data_match(email):
    email = email[:email.index('@')]
    ref = db.reference(f'/match_data/{email}')
    data = ref.get()
    st.session_state.data_match = data

# func thêm ván cờ cũ vào data realtine
def add_match(data_match, email):
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email_ = email[:email.index('@')]
    d = st.session_state.data_match
    if d == None:
        ref1 = db.reference('/match_data')
        ref1.set({
            email_: {
                'quantity': 1,
                'match-'+str(1):{
                'time': now_time,
                'data': data_match
                }
            }
        })
    else:
        match_id = f"match-{d['quantity'] + 1}"
        ref = db.reference(f"/match_data/{email_}/{match_id}")
        ref.set({
            "time": now_time,
            "data": data_match
        })
        ref1 = db.reference(f"/match_data/{email_}")
        ref1.update({
            'quantity':d['quantity']+1
        })
    take_data_match(email)

# hàm thêm người dùng vào dữ liệu đám may firebase
def add_user(information):
    ref = db.reference('/users')
    indexa = information['email'].index('@')
    ref.push({
        information['email'][:indexa]: {
        "email": information['email'],
        "name": information['name'],
        "age": information['age'],
        'gender':information['gender'],
        "password" : information['password']
        }
    })

# --- lấy thông tin người dùng ---
def take_data_user(email_user,password_user):
    # Lấy phần trước @ và xử lý ký tự cấm
    key = email_user.split('@')[0].replace('.', '_')
    # Đọc dữ liệu
    data = db.reference(f"/users/{key}").get()
    if data == None:
        st.info('Tài khoản không tồn tại!')
    elif password_user == data['password']:
        st.success('Đăng nhập thành công!')
        if 'inforlogin' not in st.session_state:
            del data['password']
            st.session_state.inforlogin = data
            take_data_match(data['email'])
        return True
    else:
        st.info('Email hoặc mật khẩu sai')

# use st.session_state để tạo trạng thái trang
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- Trang chủ ---
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
# ----- Trang 2 -----
elif st.session_state.page == "games":
    col_GoBack_DataOld, col_title_DataOld = st.columns(2)
    with col_GoBack_DataOld:
        if st.button("⬅️ Quay lại"):
            st.session_state.page = "home"
            st.rerun()
    with col_title_DataOld:
        st.title('Ván cờ')
    display_match_old()

# ----- Trang 3 -----
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
                    
                # Hiển thị thông tin mỗi cái 1 card
                for key, value in st.session_state.inforlogin.items():
                    with mui.Card(style={"padding": "15px", "marginBottom": "10px", 
                                         "borderRadius": "10px", "border": "2px solid #ffffff"}):
                        mui.Typography(f"{key.upper()}: {value}", variant="body1")
                if st.button('Đăng Xuất'): 
                    del st.session_state.inforlogin 
                    st.rerun()
    else:
        # --- tạo bộ nhớ trạng thái cho login-register ---
        if "login_register" not in st.session_state:
            st.session_state.login_register = "login"
        
        col1, col2, col3 = st.columns([1, 2, 1])  # giữa rộng hơn
        
        if st.session_state.login_register == "login":
        
            with col2:  # container nằm giữa
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
                            # --- check thông tin xem hợp lệ chưa
                            # -- check xem còn thông tin nào chưa nhập
                            for i in information:
                                if information[i]=='' or information[i]==0.0:
                                    check_information+=i+' ,'
                            if check_information!='':
                                placeholder = st.empty()
                                with placeholder.container():
                                    st.error("Hãy nhập thêm các thông tin: "+ check_information[:-2])
                            else:
                                c1, c2 = True, True
                                check_email=information["email"]
                                information['email']=information['email'].split('@')[0].replace('.', '_')+'@gmail.com'
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
                                        add_user(information)
                                    time.sleep(3)
                                    st.session_state.login_register = "login"
                                    st.rerun()
                    with co2:
                        if st.button('-Đăng nhập tài khoản-'):
                            st.session_state.login_register = "login"
                            st.rerun()

# ----- Trang 4 -----                    
elif st.session_state.page == 'settings':
    if st.button("⬅️ Quay lại"):
        st.session_state.page = "home"
        st.rerun()

                






    






