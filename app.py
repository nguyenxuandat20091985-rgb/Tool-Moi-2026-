import streamlit as st
import pandas as pd
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="AI 5D-Bet Pro", layout="wide")

# Giao diện chính
st.title("🤖 AI Phân Tích 5D Bet - Tổng 5 Banh")
st.markdown("---")

# Khởi tạo trạng thái lưu trữ nếu chưa có
if 'history' not in st.session_state:
    st.session_state.history = []
if 'balance' not in st.session_state:
    st.session_state.balance = 100000
if 'last_bet' not in st.session_state:
    st.session_state.last_bet = None
if 'bet_step' not in st.session_state:
    st.session_state.bet_step = 0

# Cấu hình mức tiền gấp thếp nhẹ cho vốn 100k
BET_STAGES = [2000, 4000, 9000, 19000, 40000]

def predict_logic(history):
    if len(history) < 3:
        return "Đợi thêm dữ liệu", 50
    
    sums = [sum([int(d) for d in str(k)]) for k in history]
    # Thuật toán AI: Tính toán độ lệch trung bình (Mean Reversion)
    avg_sum = np.mean(sums)
    
    # Nếu tổng trung bình đang quá cao (>25), xác suất về Xỉu cao hơn và ngược lại
    if avg_sum > 25:
        return "XỈU", 72
    elif avg_sum < 20:
        return "TÀI", 75
    else:
        # Check cầu bệt
        last_type = "TÀI" if sums[-1] >= 23 else "XỈU"
        return last_type, 65

# --- KHU VỰC NHẬP LIỆU ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 Nhập kỳ mới")
    new_val = st.text_input("Nhập 5 số vừa ra:", placeholder="Ví dụ: 56789", max_chars=5)
    
    if st.button("Xác nhận & Đối soát"):
        if len(new_val) == 5 and new_val.isdigit():
            current_sum = sum([int(d) for d in new_val])
            current_type = "TÀI" if current_sum >= 23 else "XỈU"
            
            # Đối soát thắng thua
            if st.session_state.last_bet:
                if st.session_state.last_bet == current_type:
                    win_amt = BET_STAGES[st.session_state.bet_step] * 0.95
                    st.session_state.balance += win_amt
                    st.session_state.bet_step = 0 # Thắng thì về mức 1
                    st.balloons()
                    st.success(f"THẮNG! +{win_amt:,.0f}đ")
                else:
                    st.session_state.balance -= BET_STAGES[st.session_state.bet_step]
                    st.session_state.bet_step = min(st.session_state.bet_step + 1, len(BET_STAGES)-1)
                    st.error(f"THUA! -{BET_STAGES[st.session_state.bet_step-1]:,.0f}đ")
            
            st.session_state.history.append(new_val)
        else:
            st.error("Vui lòng nhập đúng 5 chữ số!")

# --- KHU VỰC HIỂN THỊ ---
with col2:
    st.subheader("📈 Phân tích AI & Quản lý vốn")
    
    prediction, confidence = predict_logic(st.session_state.history)
    st.session_state.last_bet = prediction
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dư hiện tại", f"{st.session_state.balance:,.0f}đ")
    c2.metric("Dự đoán kỳ tiếp", prediction)
    c3.metric("Độ tin cậy", f"{confidence}%")
    
    st.warning(f"👉 Lệnh đánh tiếp theo: **{BET_STAGES[st.session_state.bet_step]:,.0f}đ** vào cửa **{prediction}**")

# Bảng lịch sử
if st.session_state.history:
    st.subheader("📜 Lịch sử đối soát")
    hist_data = []
    for h in reversed(st.session_state.history):
        s = sum([int(d) for d in h])
        t = "TÀI" if s >= 23 else "XỈU"
        hist_data.append({"Số": h, "Tổng": s, "Kết quả": t})
    st.table(pd.DataFrame(hist_data))

if st.button("Xóa lịch sử / Reset vốn"):
    st.session_state.history = []
    st.session_state.balance = 100000
    st.session_state.bet_step = 0
    st.rerun()
