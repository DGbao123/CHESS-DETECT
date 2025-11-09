# Ứng dụng Detect Vật thể Real-time với YOLOv8 và Streamlit

Đây là một ứng dụng web đơn giản, được build bằng Streamlit, cho phép người dùng chạy model YOLOv8 để phát hiện vật thể trong thời gian thực thông qua webcam.

## Công nghệ sử dụng

Python 3.10+ (Bắt buộc 64-bit)

YOLOv8 (ultralytics) cho model AI detect.

Streamlit để xây dựng giao diện web.

OpenCV để xử lý hình ảnh từ webcam.

## Hướng dẫn Cài đặt (Quan trọng)

Môi trường cài đặt cho các thư viện AI/Data Science trên Windows rất phức tạp. Vui lòng làm theo CHÍNH XÁC các bước sau để tránh các lỗi build (như Failed building wheel for pyarrow hay cmake.EXE failed).

# 1. Yêu cầu Tiên quyết (Cài 1 lần)

Đây là các công cụ nền tảng bạn BẮT BUỘC phải có trên Windows:

Python 64-bit:

Nguyên nhân của 90% lỗi build là do bạn đang dùng Python 32-bit. Các thư viện AI/Data đã ngừng hỗ trợ 32-bit.

Truy cập python.org và tải bản "Windows installer (64-bit)".

Khi cài đặt, nhớ tick vào ô "Add Python to PATH".

Microsoft C++ Build Tools:

Cần thiết để biên dịch các thư viện (như numpy, pyarrow).

Truy cập Visual Studio Downloads.

Cuộn xuống "Tools for Visual Studio" (Công cụ cho Visual Studio) và tải về "Build Tools for Visual Studio".

Khi chạy file cài đặt, trong tab "Workloads", hãy tick chọn "Desktop development with C++" (Phát triển ứng dụng máy tính với C++).

2. Cài đặt Dự án

Mở PowerShell và di chuyển đến thư mục bạn muốn:

# Ví dụ:
cd D:\Learning


Tạo thư mục dự án và thư mục môi trường ảo (venv):

mkdir Yolo_Project
cd Yolo_Project
python -m venv venv


Kích hoạt môi trường ảo:

.\venv\Scripts\activate


(Lỗi thường gặp) Nếu bạn nhận được thông báo lỗi màu đỏ (UnauthorizedAccess), hãy chạy lệnh sau để cho phép PowerShell chạy script, sau đó kích hoạt lại:

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser


3. Cài đặt Các thư viện (Công thức Chống lỗi)

Đây là "công thức vàng" các phiên bản đã được kiểm chứng để hoạt động trơn tru với nhau, tránh mọi lỗi xung đột (protobuf, TypeError: 'closed', AttributeError: '_typing_check').

Hãy chạy lần lượt từng lệnh sau:

# Bước 1: Nâng cấp pip để nó ưu tiên file cài đặt sẵn (.whl)
pip install --upgrade pip wheel

# Bước 2: Cài 2 thư viện chính
pip install ultralytics streamlit

# Bước 3: Sửa lỗi xung đột Protobuf (do ultralytics và streamlit gây ra)
pip install protobuf==3.20.3

# Bước 4: Sửa lỗi xung đột của Streamlit (altair, typing-extensions, referencing)
pip install altair==5.2.0 typing-extensions==4.10.0 referencing==0.32.0


4. Cài đặt bằng file requirements.txt (Cách 2)

Nếu bạn đã làm các bước 1 và 2, bạn có thể thử cài đặt trực tiếp từ file requirements.txt:

pip install -r requirements.txt


## Cách Chạy ứng dụng

Sau khi cài đặt thành công, hãy đảm bảo bạn đã có file app.py trong thư mục Yolo_Project.

Kích hoạt môi trường ảo (nếu bạn chưa làm):

```
.\venv\Scripts\activate
```

Chạy Streamlit:

streamlit run app.py


Streamlit sẽ tự động mở trình duyệt. Bạn sẽ thấy 2 đường link:

Local URL: Dùng để mở trên máy của bạn.

Network URL: Dùng để các thiết bị khác (như điện thoại) trong cùng mạng WiFi truy cập.