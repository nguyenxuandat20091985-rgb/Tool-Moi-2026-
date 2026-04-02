import streamlit as st
import pandas as pd
import numpy as np

# Cấu hình giao diện
st.set_page_config(page_title="AI 5D Bet Pro", layout="wide")
st.title("🤖 AI Phân Tích 5D Bet - Tổng 5 Banh")

# Khởi tạo bộ nhớ lưu trữ phiên chơi (Session State)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'balance' not in st.session_state:
    st.session_state.balance = 0

# --- KHU VỰC NHẬP DỮ LIỆU ---
st.sidebar.header("📥 Nhập dữ liệu kỳ mới")
ky_id = st.sidebar.text_input("Mã kỳ (Ví dụ: 1491)", "")
result_raw = st.sidebar.text_input("Kết quả 5 số (Ví dụ: 56789)", "")

if st.sidebar.button("Cập nhật & Dự đoán"):
    if len(result_raw) == 5 and result_raw.isdigit():
        nums = [int(d) for d in result_raw]
        tong = sum(nums)
        loai = "Tài" if tong >= 23 else "Xỉu"
        
        # Lưu vào lịch sử
        entry = {"Kỳ": ky_id, "Số": result_raw, "Tổng": tong, "Kết quả": loai}
        st.session_state.history.append(entry)
        st.success(f"Đã cập nhật kỳ {ky_id}: {tong} ({loai})")
    else:
        st.error("Vui lòng nhập đúng 5 con số!")

# --- THUẬT TOÁN AI DỰ ĐOÁN ---
def ai_predict(history):
    if len(history) < 3:
        return "Cần thêm dữ liệu...", "---"
    
    # Lấy danh sách tổng điểm gần nhất
    tongs = [h['Tổng'] for h in history]
    
    # 1. Phân tích xu hướng (Trend)
    diff = np.diff(tongs)
    avg_diff = np.mean(diff)
    
    # 2. Dự đoán dựa trên xác suất hồi tụ (Mức trung bình 22.5)
    last_tong = tongs[-1]
    if last_tong < 15: # Quá xỉu, xu hướng sẽ bật lên Tài
        pred = "Tài (Xác suất cao)"
    elif last_tong > 35: # Quá tài, xu hướng sẽ sập về Xỉu
        pred = "Xỉu (Xác suất cao)"
    else:
        # Nếu đang ở giữa, dùng xu hướng tăng/giảm của 3 kỳ gần nhất
        if avg_diff > 0:
            pred = "Tài (Theo trend tăng)"
        else:
            pred = "Xỉu (Theo trend giảm)"
            
    return pred, f"{abs(avg_diff):.1f}"

# --- HIỂN THỊ KẾT QUẢ ---
col1, col2, col3 = st.columns(3)
prediction, confidence = ai_predict(st.session_state.history)

with col1:
    st.metric("DỰ ĐOÁN KỲ TIẾP", prediction)
with col2:
    st.metric("BIẾN ĐỘNG (Lực cầu)", confidence)
with col3:
    st.metric("SỐ KỲ ĐÃ PHÂN TÍCH", len(st.session_state.history))

# --- BẢNG ĐỐI SOÁT THẮNG THUA ---
st.divider()
st.subheader("📊 Nhật ký Đối soát & Theo dõi")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    # Đảo ngược để xem kỳ mới nhất lên đầu
    st.table(df.iloc[::-1])
    
    if st.button("Xóa lịch sử để chơi phiên mới"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Chưa có dữ liệu. Anh vui lòng nhập kết quả ở bên trái để AI bắt đầu học.")

# --- NÂNG CẤP ĐÁNG GIÁ: CẢNH BÁO CẦU ---
if len(st.session_state.history) >= 4:
    last_4 = [h['Kết quả'] for h in st.session_state.history[-4:]]
    if len(set(last_4)) == 1:
        st.warning(f"⚠️ CẢNH BÁO: Phát hiện cầu bệt {last_4[0]} 4 kỳ liên tiếp. Anh nên bám cầu, đừng bẻ!")
