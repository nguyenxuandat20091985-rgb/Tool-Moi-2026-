import streamlit as st
import pandas as pd

st.set_page_config(page_title="Tool 5D Bet Pro v2.0", layout="wide")

# Khởi tạo bộ nhớ tạm để lưu lịch sử thắng thua
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🚀 Tool 5D Bet Pro - Hệ Thống Theo Dõi & Dự Đoán")

# --- PHẦN 1: NHẬP DỮ LIỆU & SOI CẦU ---
st.subheader("📊 Phân tích Kỳ Hiện Tại")
with st.expander("Hướng dẫn nhập liệu", expanded=True):
    data_input = st.text_input("Nhập 5 số kỳ vừa ra (Ví dụ: 12345):", "")
    bet_amount = st.number_input("Số vốn hiện có ($):", value=100.0)

def predict_logic(last_digit):
    # Thuật toán dự đoán Tài/Xỉu dựa trên số cuối (Hàng đơn vị)
    # Nếu số cuối là 0,1,2,3,4 -> Xỉu | 5,6,7,8,9 -> Tài
    pred = "TÀI" if last_digit <= 4 else "XỈU" # Đánh nghịch đảo (Counter-Trend)
    return pred

if st.button("🔥 CHỐT LỆNH DỰ ĐOÁN"):
    if len(data_input) == 5:
        last_num = int(data_input[-1])
        prediction = predict_logic(last_num)
        
        # Lưu vào lịch sử để theo dõi ván sau
        st.session_state.last_pred = prediction
        st.session_state.last_result_num = data_input
        
        st.info(f"Kỳ trước ra: {data_input} (Số cuối: {last_num})")
        st.success(f"🎯 DỰ ĐOÁN KỲ TIẾP THEO: **{prediction}**")
        
        # Gợi ý đi tiền
        st.warning(f"💰 Gợi ý vào lệnh: {bet_amount * 0.01:.2f} $ (1% vốn)")
    else:
        st.error("Anh nhập đủ 5 số của kỳ vừa rồi nhé!")

# --- PHẦN 2: ĐỐI CHIẾU THẮNG THUA (Điểm nâng cấp) ---
st.divider()
st.subheader("📝 Nhật Ký Đối Chiếu Thắng/Thua")

col1, col2 = st.columns(2)
with col1:
    actual_result = st.radio("Kết quả ván vừa rồi thực tế ra gì?", ["Chưa có", "TÀI", "XỈU"])

if st.button("Xác nhận Thắng/Thua"):
    if "last_pred" in st.session_state and actual_result != "Chưa có":
        status = "✅ THẮNG" if st.session_state.last_pred == actual_result else "❌ THUA"
        st.session_state.history.insert(0, {
            "Dự đoán": st.session_state.last_pred,
            "Thực tế": actual_result,
            "Trạng thái": status
        })
        st.success("Đã cập nhật lịch sử!")

# Hiển thị bảng lịch sử
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.table(df_history.head(10)) # Hiển thị 10 ván gần nhất
    
    # Tính tỉ lệ thắng
    wins = len(df_history[df_history["Trạng thái"] == "✅ THẮNG"])
    win_rate = (wins / len(df_history)) * 100
    st.metric("Tỉ lệ thắng hiện tại", f"{win_rate:.1f}%")
    
    if win_rate < 40:
        st.error("⚠️ CẢNH BÁO: Thuật toán đang bị 'nhà cái' soi, hãy nghỉ 30 phút!")
