import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="5D Bet Master Tool", layout="wide")
st.title("🏆 Hệ Thống Đối Soát & Soi Cầu 5D Bet")

# --- PHẦN 1: NHẬP DỮ LIỆU & ĐỐI SOÁT ---
st.header("1. Đối soát tự động")
raw_data = st.text_area("Dán danh sách kết quả (Ví dụ: 56789, 12345, 09876)", height=150)

if raw_data:
    list_kỳ = [s.strip() for s in raw_data.split(",") if len(s.strip()) == 5]
    results = []
    
    for ky in list_kỳ:
        tong = sum(int(d) for d in ky)
        ket_qua = "TÀI" if tong >= 23 else "XỈU"
        results.append({"Kỳ": ky, "Tổng": tong, "Kết quả": ket_qua})
    
    df = pd.DataFrame(results)
    
    # Hiển thị bảng đối soát
    st.dataframe(df.style.applymap(lambda x: 'color: red' if x == 'TÀI' else 'color: blue', subset=['Kết quả']), use_container_width=True)

    # --- PHẦN 2: THỐNG KÊ CHIẾN THUẬT ---
    st.header("2. Phân tích xác suất")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bổ Tổng số")
        fig = px.histogram(df, x="Tổng", nbins=46, range_x=[0, 45], title="Biểu đồ hội tụ tổng 5 banh")
        st.plotly_chart(fig)
        
    with col2:
        count_tx = df['Kết quả'].value_counts()
        st.subheader("Tỉ lệ Tài/Xỉu thực tế")
        st.write(count_tx)
        
        # Cảnh báo cầu bệt
        if len(df) >= 3:
            last_3 = df['Kết quả'].tail(3).tolist()
            if all(x == last_3[0] for x in last_3):
                st.warning(f"⚠️ CẢNH BÁO: Đang có cầu BỆT {last_3[0]} (3 kỳ liên tiếp)!")

# --- PHẦN 3: NÂNG CẤP ĐÁNG GIÁ: QUẢN LÝ VỐN ---
st.sidebar.header("💰 Quản lý vốn (Risk Management)")
von_bandau = st.sidebar.number_input("Vốn ban đầu:", value=1000)
muc_tie_u = st.sidebar.number_input("Mục tiêu chốt lãi (Target):", value=1500)
cat_lo = st.sidebar.number_input("Cắt lỗ (Stop Loss):", value=700)

current_money = st.sidebar.number_input("Số dư hiện tại:", value=1000)
if current_money >= muc_tie_u:
    st.sidebar.success("🎉 Đạt mục tiêu rồi! Rút tiền nghỉ thôi anh.")
elif current_money <= cat_lo:
    st.sidebar.error("❌ Chạm ngưỡng cắt lỗ. Nghỉ để bảo toàn vốn!")
