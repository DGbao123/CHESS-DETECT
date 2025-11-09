import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import time
import firebase_admin
from firebase_admin import credentials,db
#connect firebase google
if "fire_base" not in st.session_state:
    st.session_state.fire_base=True
if st.session_state.fire_base:
    # 1. SỬA DD: THÀNH D: (hoặc đường dẫn chính xác của bạn)
    SERVICE_ACCOUNT_PATH = r"D:\MINDX\YOLO_projects\check-detect-80389-firebase-adminsdk-fbsvc-3786272c2d.json"
        
    try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://check-detect-80389-default-rtdb.firebaseio.com/'
            })
            st.write("Firebase Admin SDK đã khởi tạo thành công.")
            st.session_state.fire_base=False
            st.rerun()
    except Exception as e:
            st.write(f"Lỗi khởi tạo Firebase: {e}")
            # Lỗi sẽ xuất hiện ở đây nếu đường dẫn sai
            st.session_state.fire_base=True
            st.rerun()
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
model_path = 'yolov8n.pt'  # Mặc định dùng model 'n'
# (Tùy chọn): Bạn có thể cho phép người dùng upload model
# uploaded_model = st.file_uploader("Hoặc tải lên file model (.pt) của bạn", type="pt")
# if uploaded_model:
#     # Lưu file tạm
#     tfile = tempfile.NamedTemporaryFile(delete=False) 
#     tfile.write(uploaded_model.read())
#     model_path = tfile.name

# Tải model
try:
    model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Lỗi khi tải model: {e}")
    st.stop()
# ---- Logic chạy Webcam ----
def func_run_webcam(run_webcam):
    # Vùng chứa ảnh (placeholder)
    st_frame = st.empty()

    # Mở webcam
    cap = cv2.VideoCapture(0) # 0 là webcam mặc định

    while run_webcam and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Không thể nhận khung hình từ webcam. Vui lòng kiểm tra.")
            break

        # Chạy detect
        # verbose=False để tắt log
        results = model(frame, stream=True, verbose=False) 

        # Lấy khung hình đã vẽ
        annotated_frame = frame
        for r in results:
            annotated_frame = r.plot() # r.plot() trả về ảnh (numpy array) đã vẽ

        # Hiển thị ảnh
        # Cần chuyển từ BGR (OpenCV) sang RGB (Streamlit)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, use_column_width=True)

    else:
        # Khi tắt checkbox, giải phóng webcam
        cap.release()
        st.write("Đã tắt webcam.")
#các ván cờ đã ghi lại
def data_old():
    df = pd.DataFrame({
        "NAME":[],
        "DAY": [],
        "LINK": []
    })
    path_data_old = 'D:\MINDX\YOLO_projects\data_old'
    list_data_old = os.listdir(path_data_old)
    for i in list_data_old:
        with open(f'{path_data_old}\{i}') as f:
            df.loc[len(df)]=[i, f.readline()[:-1], f'{path_data_old}\{i}']
    st.write("♟️ **Các ván cờ đã ghi lại:**")

    # Hiển thị bảng có thể cuộn, sort, và click được link
    st.dataframe(
        df,
        column_config={
            "NAME": "Tên ván cờ",
            "DAY": "Ngày tạo",
            "LINK": st.column_config.LinkColumn(
                "Mở file", display_text="🗂️ Mở", help="Bấm để mở file", max_chars=50
            ),
        },
        use_container_width=True,
        hide_index=True
    )
    
def add_user():
    ref = db.reference('/users')
    ref.set({
        'alice': {
        'name': 'Alice',
        'age': 30
        }
    })
# --- Trạng thái trang ---
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
        # Checkbox để bật/tắt webcam
        run_webcam = st.checkbox('Bật Webcam')
    with col2:
        if st.button("♟️ Ván cờ"):
            st.session_state.page = "games"
            st.rerun()
        
    with col3:
        if st.button("👤 Hồ sơ"):
            st.session_state.page = "profile"
            st.rerun()
    with col4:
        if st.button("⚙️ Cài đặt"):
            st.session_state.page = "settings"
            st.rerun()
    if run_webcam:
        func_run_webcam(run_webcam)

# ----- Trang 2 -----
elif st.session_state.page == "games":
    col_GoBack_DataOld, col_title_DataOld = st.columns(2)
    with col_GoBack_DataOld:
        if st.button("⬅️ Quay lại"):
            st.session_state.page = "home"
            st.rerun()
    with col_title_DataOld:
        st.title('Ván cờ')
    data_old()

# ----- Trang 3 -----
elif st.session_state.page == "profile":
    if st.button("⬅️ Quay lại"):
        st.session_state.page = "home"
        st.rerun()
    col1, col2, col3 = st.columns([1, 2, 1])  # giữa rộng hơn
    # --- tạo bộ nhớ trạng thái cho login-register ---
    if "login_register" not in st.session_state:
        st.session_state.login_register = "login"
    if st.session_state.login_register == "login":
    
        with col2:  # container nằm giữa
            with st.container():
                st.subheader("LOGIN")
                user = st.text_input('Email hoặc số điện thoại')
                password = st.text_input('Mật Khẩu')
                co1, co2 = st.columns(2)
                with co1:
                    st.button('Login')
                with co2:
                    if st.button('-Đăng ký tài khoản-'):
                        st.session_state.login_register = "register"
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
                                time.sleep(3)
                                st.session_state.page = "home"
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

# Gọi hàm test
if st.button("Thêm user"):
    add_user()               
                






    






