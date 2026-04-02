import streamlit as st
import pandas as pd

# Cấu hình giao diện
st.set_page_config(page_title="AI 5D Bet Pro", layout="wide")
st.title("🚀 AI 5D BET PRO - TỔNG 5 BANH")
st.markdown("---")

# 1. Khởi tạo trạng thái hệ thống (Vốn & Quản lý cược)
if 'balance' not in st.session_state:
    st.session_state.balance = 100000 # Vốn 100k
if 'bet_step' not in st.session_state:
    st.session_state.bet_step = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# Danh sách gấp thếp nhẹ (1k, 2k, 3k, 5k, 8k, 13k...) - Dãy Fibonacci
bet_amounts = [1000, 2000, 3000, 5000, 8000, 13000, 21000]

def get_prediction(history_sums):
    """AI dự đoán dựa trên xu hướng tổng số"""
    if len(history_sums) < 3:
        return "CHỜ DỮ LIỆU"
    
    avg_sum = sum(history_sums) / len(history_sums)
    last_sum = history_sums[-1]
    
    # Logic AI: Dự đoán dựa trên sự hồi quy về mức trung bình (Regression to Mean)
    # Nếu tổng vừa rồi quá cao (>30), xác suất kỳ tới về Xỉu (<22) tăng lên.
    if last_sum > 30:
        return "XỈU"
    elif last_sum < 15:
        return "TÀI"
    else:
        # Nếu đang ở giữa, đánh theo cầu bệt (Trend Following)
        return "TÀI" if last_sum >= 23 else "XỈU"

# 2. Giao diện nhập liệu
st.sidebar.header("🕹️ BÀN ĐIỀU KHIỂN")
st.sidebar.write(f"💰 Vốn hiện tại: **{st.session_state.balance:,} VNĐ**")
if st.sidebar.button("Reset Vốn (100k)"):
    st.session_state.balance = 100000
    st.session_state.bet_step = 0
    st.rerun()

current_input = st.text_input("Nhập kết quả kỳ vừa ra (5 số viết liền, VD: 56789):", "")

if st.button("CẬP NHẬT & ĐỐI SOÁT"):
    if len(current_input) == 5 and current_input.isdigit():
        digits = [int(d) for d in current_input]
        total = sum(digits)
        result_type = "TÀI" if total >= 23 else "XỈU"
        
        # Đối soát thắng thua
        if len(st.session_state.history) > 0:
            last_pred = st.session_state.history[-1]['prediction']
            bet_value = bet_amounts[st.session_state.bet_step]
            
            if last_pred == result_type:
                st.balloons()
                st.success(f"THẮNG! Kỳ trước ra {total} ({result_type})")
                st.session_state.balance += (bet_value * 0.95) # Trừ phế nhà cái
                st.session_state.bet_step = 0 # Thắng thì về mức cược ban đầu
            else:
                st.error(f"THUA! Kỳ trước ra {total} ({result_type})")
                st.session_state.balance -= bet_value
                st.session_state.bet_step = min(st.session_state.bet_step + 1, len(bet_amounts)-1)

        # Lưu lịch sử
        st.session_state.history.append({
            "input": current_input,
            "total": total,
            "type": result_type,
            "prediction": "" # Sẽ cập nhật ở bước sau
        })
    else:
        st.warning("Vui lòng nhập đúng 5 chữ số!")

# 3. Phân tích và đưa ra lệnh đánh
if len(st.session_state.history) >= 1:
    history_totals = [h['total'] for h in st.session_state.history]
    next_pred = get_prediction(history_totals)
    st.session_state.history[-1]['prediction'] = next_pred
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("LỆNH TIẾP THEO", next_pred)
    with col2:
        current_bet = bet_amounts[st.session_state.bet_step]
        st.metric("TIỀN CƯỢC", f"{current_bet:,} VNĐ")
    with col3:
        status = "NÊN ĐÁNH" if st.session_state.bet_step < 4 else "CẢNH BÁO: CẦU BIẾN"
        st.metric("TRẠNG THÁI", status)

# 4. Hiển thị bảng lịch sử để theo dõi
if st.session_state.history:
    st.subheader("📋 Nhật ký đối soát")
    df = pd.DataFrame(st.session_state.history).tail(10)
    st.table(df)

st.markdown("---")
st.caption("Lưu ý: Tool dựa trên xác suất. Hãy dừng lại khi thắng 20% vốn mỗi ngày để tránh bị nhà cái quét ID.")
