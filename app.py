import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Cấu hình giao diện chuyên nghiệp
st.set_page_config(page_title="AI 5D BET PREDICTOR", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI 5D BET - TRỢ LÝ ĐỐI SOÁT & DỰ ĐOÁN")

# Khởi tạo bộ nhớ (Session State) để lưu lịch sử thắng thua
if 'history' not in st.session_state:
    st.session_state.history = []

# --- KHU VỰC NHẬP DỮ LIỆU ---
with st.sidebar:
    st.header("🎮 Nhập Kết Quả Kỳ Hiện Tại")
    ky_so = st.number_input("Số kỳ (Ví dụ: 123)", min_value=1, step=1)
    ket_qua = st.text_input("Nhập 5 số mở thưởng (VD: 56789)", max_length=5)
    
    if st.button("🚀 Phân Tích & Đối Soát"):
        if len(ket_qua) == 5 and ket_qua.isdigit():
            nums = [int(d) for d in ket_qua]
            tong = sum(nums)
            status = "TÀI" if tong >= 23 else "XỈU"
            
            # Lưu vào lịch sử
            st.session_state.history.append({
                "Kỳ": ky_so,
                "Kết quả": ket_qua,
                "Tổng": tong,
                "Hệ thống": status
            })
            st.success(f"Đã lưu kỳ {ky_so}!")
        else:
            st.error("Vui lòng nhập đúng 5 chữ số!")

# --- KHU VỰC THỐNG KÊ & AI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Nhật Ký Đối Soát Tự Động")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df.tail(10)) # Hiển thị 10 kỳ gần nhất
        
        # Biểu đồ xu hướng tổng điểm
        fig = px.line(df, x="Kỳ", y="Tổng", title="Biểu đồ xu hướng Tổng 5 banh", markers=True)
        fig.add_hline(y=22.5, line_dash="dash", line_color="red", annotation_text="Ranh giới Tài/Xỉu")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu. Hãy nhập kỳ đầu tiên ở bên trái.")

with col2:
    st.subheader("🔮 AI Dự Đoán Kỳ Tiếp")
    if len(st.session_state.history) >= 3:
        df = pd.DataFrame(st.session_state.history)
        last_3 = df['Hệ thống'].tail(3).tolist()
        
        # Thuật toán AI đơn giản: Nhận diện cầu bệt hoặc cầu nghiêng
        tai_count = df['Hệ thống'].tail(10).tolist().count("TÀI")
        xiu_count = 10 - tai_count
        
        st.metric("Tỉ lệ Tài/Xỉu (10 kỳ)", f"{tai_count*10}% / {xiu_count*10}%")
        
        # Logic dự đoán
        if last_3 == ["TÀI", "TÀI", "TÀI"]:
            prediction = "XỈU"
            reason = "Cầu bệt dài, AI dự báo bẻ cầu (Hồi đầu)"
        elif last_3 == ["XỈU", "XỈU", "XỈU"]:
            prediction = "TÀI"
            reason = "Cầu bệt dài, AI dự báo bẻ cầu"
        else:
            prediction = "TÀI" if xiu_count > tai_count else "XỈU"
            reason = "Đánh theo thuật toán bù trừ xác suất"

        st.markdown(f"### Lệnh nên đánh: **{prediction}**")
        st.caption(f"Lý do: {reason}")
        
        # Quản lý vốn
        st.warning("💰 Mức vào tiền gợi ý: 1% - 2% vốn")
    else:
        st.write("Cần tối thiểu 3 kỳ để AI bắt đầu soi cầu.")

# --- NÂNG CẤP ĐÁNG GIÁ ---
st.divider()
st.subheader("🚀 Các nâng cấp AI anh nhận được:")
c1, c2, c3 = st.columns(3)
c1.write("✅ **Tự động tính Tổng:** Anh không cần bấm máy tính, nhập 5 số là xong.")
c2.write("✅ **Báo lỗi nhập:** Nếu nhập thiếu số hoặc sai định dạng, tool sẽ chặn ngay.")
c3.write("✅ **Nhận diện Cầu Bệt:** AI tự động cảnh báo khi một bên ra quá nhiều.")
