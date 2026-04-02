import streamlit as st
import pandas as pd
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI 5D BET PRO - SMART BOT", layout="wide", page_icon="🤖")

# --- GIAO DIỆN CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4f5b66; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .stTextInput>div>div>input { background-color: #262730; color: #00FF00; font-size: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- KHỞI TẠO DỮ LIỆU ---
if 'balance' not in st.session_state:
    st.session_state.balance = 100000 # Vốn 100k
if 'bet_step' not in st.session_state:
    st.session_state.bet_step = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'total_wins' not in st.session_state:
    st.session_state.total_wins = 0

# Cấu hình gấp thếp nhẹ cho vốn 100k: 1k -> 2k -> 4k -> 8k -> 17k -> 35k
bet_levels = [1000, 2000, 4000, 8000, 17000, 35000]

# --- HÀM LOGIC AI ---
def get_result_type(numbers_str):
    digits = [int(d) for d in list(numbers_str.strip())]
    total = sum(digits)
    res_type = "TÀI" if total >= 23 else "XỈU"
    return total, res_type

def ai_smart_predict(history_types):
    if len(history_types) < 5:
        return "TÀI" # Dữ liệu mồi
    
    # Phân tích chu kỳ gần nhất
    last_5 = history_types[-5:]
    
    # 1. Phát hiện cầu bệt
    if last_5.count(last_5[-1]) >= 4:
        return last_5[-1] # Đu theo bệt
        
    # 2. Phát hiện cầu 1-1
    if last_5[-1] != last_5[-2] and last_5[-2] != last_5[-3]:
        return "XỈU" if last_5[-1] == "TÀI" else "TÀI" # Đánh nghịch đảo
        
    # 3. Thuật toán xác suất ngẫu nhiên có trọng số (AI học nhịp)
    weights = {"TÀI": history_types.count("TÀI"), "XỈU": history_types.count("XỈU")}
    return "TÀI" if weights["TÀI"] < weights["XỈU"] else "XỈU"

# --- GIAO DIỆN CHÍNH ---
st.title("🤖 AI 5D BET SMART PREDICTOR V2.0")
st.caption("Hệ thống tự động học nhịp nhà cái và quản lý vốn thông minh")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vốn Hiện Tại", f"{st.session_state.balance:,} đ")
with col2:
    current_bet = bet_levels[st.session_state.bet_step] if st.session_state.bet_step < len(bet_levels) else bet_levels[0]
    st.metric("Lệnh Vào Tiếp Theo", f"{current_bet:,} đ", delta=f"Cấp độ {st.session_state.bet_step + 1}")
with col3:
    win_rate = (st.session_state.total_wins / len(st.session_state.history) * 100) if st.session_state.history else 0
    st.metric("Tỉ Lệ Thắng AI", f"{win_rate:.1f}%")

# --- NHẬP LIỆU & ĐỐI SOÁT ---
st.divider()
input_val = st.text_input("👉 NHẬP 5 SỐ VỪA RA (VD: 12345):", max_chars=5)

if st.button("🔥 PHÂN TÍCH & DỰ ĐOÁN"):
    if len(input_val) == 5 and input_val.isdigit():
        total, res_type = get_result_type(input_val)
        
        # ĐỐI SOÁT TỰ ĐỘNG
        status = "Bắt đầu"
        if st.session_state.last_prediction:
            if st.session_state.last_prediction == res_type:
                status = "THẮNG ✅"
                st.session_state.balance += (current_bet * 0.95) # Trừ phế sàn nhẹ
                st.session_state.bet_step = 0
                st.session_state.total_wins += 1
            else:
                status = "THUA ❌"
                st.session_state.balance -= current_bet
                st.session_state.bet_step += 1
                if st.session_state.bet_step >= len(bet_levels):
                    st.session_state.bet_step = 0 # Reset nếu cháy chuỗi gấp
        
        # LƯU LỊCH SỬ
        st.session_state.history.append({
            "Kỳ": len(st.session_state.history) + 1,
            "Kết Quả": input_val,
            "Tổng": total,
            "Hệ Thống Ra": res_type,
            "AI Dự Đoán": st.session_state.last_prediction if st.session_state.last_prediction else "N/A",
            "Trạng Thái": status
        })
        
        # AI HỌC VÀ DỰ ĐOÁN CHO KỲ TIẾP THEO
        all_res_types = [h["Hệ Thống Ra"] for h in st.session_state.history]
        st.session_state.last_prediction = ai_smart_predict(all_res_types)
        
        st.rerun()
    else:
        st.error("Lỗi: Nhập đúng 5 con số kỳ vừa ra!")

# --- HIỂN THỊ KẾT QUẢ DỰ ĐOÁN ---
if st.session_state.last_prediction:
    st.markdown(f"""
        <div style="background-color: #2e3141; padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #00FF00;">
            <h2 style="color: white; margin: 0;">DỰ ĐOÁN KỲ TIẾP THEO</h2>
            <h1 style="color: #00FF00; font-size: 60px; margin: 10px 0;">{st.session_state.last_prediction}</h1>
            <p style="color: #888;">Hãy vào tiền mức: {current_bet:,} VNĐ</p>
        </div>
    """, unsafe_allow_html=True)

# --- BẢNG LỊCH SỬ ĐỐI SOÁT ---
if st.session_state.history:
    st.subheader("📋 BÁO CÁO CHI TIẾT THẮNG/THUA")
    df = pd.DataFrame(st.session_state.history).iloc[::-1]
    
    # Định dạng màu sắc cho bảng
    def color_status(val):
        color = 'green' if 'THẮNG' in val else 'red' if 'THUA' in val else 'white'
        return f'color: {color}'
    
    st.table(df.head(15))

if st.button("Xóa dữ liệu & Reset Vốn"):
    st.session_state.balance = 100000
    st.session_state.history = []
    st.session_state.bet_step = 0
    st.session_state.last_prediction = None
    st.session_state.total_wins = 0
    st.rerun()
