import streamlit as st
import pandas as pd
import numpy as np

# Thiết kế giao diện
st.set_page_config(page_title="Tool Phân Tích Oẳn Tù Tì KU", layout="wide")
st.title("📊 Hệ Thống Dự Đoán Oẳn Tù Tì KU")

# 1. Nhập dữ liệu kỳ vừa ra (Do nhà cái không có API công khai nên anh phải nhập tay hoặc dùng tool crawl)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Nhập kết quả kỳ gần nhất")
    cai_result = st.selectbox("Nhà Cái ra:", ["Búa", "Kéo", "Bao"])
    con_result = st.selectbox("Nhà Con ra:", ["Búa", "Kéo", "Bao"])
    if st.button("Cập nhật dữ liệu"):
        # Logic lưu vào file csv trên github ở đây
        st.success(f"Đã ghi nhận kỳ: {cai_result} vs {con_result}")

# 2. Thuật toán phân tích (Điểm yếu RNG)
def predict_next(history):
    # Sử dụng logic: Nếu ra 'Búa' thì ván sau xác suất ra 'Bao' là bao nhiêu %
    # Đây là nơi cài đặt thuật toán soi cầu
    return "Bao", "85%"

# 3. Hiển thị bảng dự đoán
st.divider()
st.subheader("🔮 Dự đoán kỳ tiếp theo")
c1, c2, c3 = st.columns(3)
p_shape, p_percent = predict_next(None)

c1.metric("Cửa nên đánh", "Nhà Cái")
c2.metric("Hình dạng", p_shape)
c3.metric("Độ tin cậy", p_percent)

# 4. Biểu đồ thống kê cầu (Dựa trên ảnh anh gửi có dãy xanh đỏ)
st.subheader("📈 Thống kê chu kỳ (Cầu)")
# Vẽ biểu đồ bằng Plotly hoặc đơn giản là bảng màu
