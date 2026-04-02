import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI 5D BET PRO - SMART PREDICTION", layout="wide")

# --- GIAO DIỆN DARK MODE / CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4a4a4a; }
    [data-testid="stTable"] { background-color: #1e2130; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- KHỞI TẠO STATE ---
if 'balance' not in st.session_state:
    st.session_state.balance = 100000  # Vốn ban đầu 100k
if 'bet_step' not in st.session_state:
    st.session_state.bet_step = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'wins' not in st.session_state:
    st.session_state.wins = 0
if 'losses' not in st.session_state:
    st.session_state.losses = 0

# --- QUẢN LÝ VỐN GẤP THẾP NHẸ (100K) ---
# Chuỗi cược tối ưu cho 100k: 1k -> 2k -> 4k -> 9k -> 21k (Tổng ~37k cho 1 chuỗi)
BET_LEVELS = [1000, 2000, 4000, 9000, 21000]

# --- HÀM LOGIC ---
def calculate_result(numbers_str):
    """Tính tổng 5 số và phân loại Tài/Xỉu (0-22 Xỉu, 23-45 Tài)"""
    digits = [int(d) for d in list(numbers_str)]
    total = sum(digits)
    res_type = "TAI" if total >= 23 else "XIU"
    return total, res_type

def ai_smart_predict(history):
    """AI Nâng cấp: Nhận diện mẫu hình cầu"""
    if len(history) < 4:
        return np.random.choice(["TAI", "XIU"]) # Thiếu dữ liệu thì random theo xác suất 50/50

    results = [h['Kết quả'] for h in history]
    
    # 1. Kiểm tra cầu bệt (Dây liên tiếp)
    if results[-1] == results[-2] == results[-3]:
        return results[-1] # Đánh theo bệt (Bệt thường ra tiếp)

    # 2. Kiểm tra cầu 1-1 (Gãy liên tục)
    if results[-1] != results[-2] and results[-2] != results[-3]:
        return "TAI" if results[-1] == "XIU" else "XIU" # Đánh tiếp nhịp 1-1

    # 3. Thuật toán xác suất Bayes (Dựa trên tần suất)
    tai_count = results.count("TAI")
    xiu_count = results.count("XIU")
    if tai_count > xiu_count:
        return "XIU" # Cầu nghiêng, đánh nghịch đảo để chờ hồi
    
    return "TAI" if results[-1] == "XIU" else "XIU"

# --- GIAO DIỆN CHÍNH ---
st.title("🤖 AI 5D BET PRO - SMART TRADING SYSTEM")
st.markdown("---")

# Cột thông tin tổng quan
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("💰 VỐN HIỆN TẠI", f"{st.session_state.balance:,} VNĐ")
with m2:
    win_rate = (st.session_state.wins / len(st.session_state.history) * 100) if st.session_state.history else 0
    st.metric("🎯 TỈ LỆ THẮNG AI", f"{win_rate:.1f}%")
with m3:
    st.metric("🔥 LỆNH TIẾP THEO", f"{BET_LEVELS[st.session_state.bet_step]:,} VNĐ")
with m4:
    profit = st.session_state.balance - 100000
    st.metric("📈 LỢI NHUẬN RÒNG", f"{profit:,} VNĐ", delta=profit)

# Khu vực nhập liệu và dự đoán
col_input, col_pred = st.columns([1, 1])

with col_input:
    st.subheader("📥 Cập nhật kết quả")
    new_data = st.text_input("Nhập 5 số kỳ vừa ra (Ví dụ: 01458):", max_chars=5)
    
    if st.button("XỬ LÝ DỮ LIỆU"):
        if len(new_data) == 5 and new_data.isdigit():
            total, res_type = calculate_result(new_data)
            
            # ĐỐI SOÁT TỰ ĐỘNG
            if st.session_state.last_prediction:
                current_bet = BET_LEVELS[st.session_state.bet_step]
                if st.session_state.last_prediction == res_type:
                    st.success(f"✅ THẮNG KỲ TRƯỚC! +{current_bet:,}")
                    st.session_state.balance += (current_bet * 0.95) # Trừ phí sàn nhẹ
                    st.session_state.wins += 1
                    st.session_state.bet_step = 0 # Thắng thì về mức 1
                else:
                    st.error(f"❌ THUA KỲ TRƯỚC! -{current_bet:,}")
                    st.session_state.balance -= current_bet
                    st.session_state.losses += 1
                    # Gấp thếp lên mức tiếp theo, nếu hết chuỗi thì reset
                    st.session_state.bet_step = (st.session_state.bet_step + 1) if st.session_state.bet_step < 4 else 0
            
            # Lưu lịch sử
            st.session_state.history.append({
                "Kỳ": len(st.session_state.history) + 1,
                "Thời gian": datetime.now().strftime("%H:%M:%S"),
                "Số": new_data,
                "Tổng": total,
                "Kết quả": res_type,
                "Dự đoán": st.session_state.last_prediction
            })
            
            # AI DỰ ĐOÁN KỲ MỚI
            st.session_state.last_prediction = ai_smart_predict(st.session_state.history)
            st.rerun()
        else:
            st.warning("Vui lòng nhập đúng 5 chữ số từ nhà cái!")

with col_pred:
    st.subheader("🔮 AI Prediction")
    if st.session_state.last_prediction:
        color = "#ff4b4b" if st.session_state.last_prediction == "TAI" else "#00c853"
        st.markdown(f"""
            <div style="background-color:{color}; padding: 30px; border-radius: 15px; text-align: center;">
                <h1 style="color: white; margin:0;">{st.session_state.last_prediction}</h1>
                <p style="color: white; font-weight: bold;">MỨC CƯỢC: {BET_LEVELS[st.session_state.bet_step]:,} VNĐ</p>
            </div>
            """, unsafe_allow_html=True)
        st.info(f"Lưu ý: Đánh kỳ tiếp theo ngay khi nhà cái mở thưởng.")
    else:
        st.write("Hãy nhập kỳ đầu tiên để AI bắt đầu học luồng cầu.")

# --- BẢNG ĐỐI SOÁT CHI TIẾT ---
st.markdown("---")
if st.session_state.history:
    st.subheader("📋 Nhật ký đối soát thắng thua")
    df = pd.DataFrame(st.session_state.history).iloc[::-1] # Mới nhất lên đầu
    st.dataframe(df, use_container_width=True)

# Nút Reset khẩn cấp
if st.sidebar.button("RESET TOÀN BỘ DỮ LIỆU"):
    st.session_state.clear()
    st.rerun()
