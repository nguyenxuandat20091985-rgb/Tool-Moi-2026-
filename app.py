import streamlit as st
import re
import json
import os
from collections import Counter
from datetime import datetime

# ================= CẤU HÌNH ĐƠN GIẢN =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_data.json"

# ================= XỬ LÝ DỮ LIỆU =================
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_data(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data[-1000:], f)

# Khởi tạo
if 'history' not in st.session_state:
    st.session_state.history = load_data()
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None

# ================= THUẬT TOÁN BẮT CẦU ĐƠN GIẢN =================
def catch_bet(history):
    """Bắt cầu bệt - số về nhiều nhất"""
    if len(history) < 5:
        return []
    
    all_digits = ''.join(history[-10:])
    counter = Counter(all_digits)
    
    # Lấy top 5 số về nhiều nhất
    top = [d for d, _ in counter.most_common(5)]
    return top

def catch_lo_roi(history):
    """Bắt lô rơi - số về liên tiếp"""
    if len(history) < 3:
        return []
    
    last = history[-1]
    prev = history[-2]
    
    # Số xuất hiện ở cả 2 kỳ
    lo_roi = []
    for d in last:
        if d in prev and d not in lo_roi:
            lo_roi.append(d)
    
    return lo_roi

def catch_dao(history):
    """Bắt cầu đảo"""
    if len(history) < 3:
        return []
    
    last = history[-1]
    prev = history[-2]
    
    # Đảo đầu cuối
    if last[0] == prev[4] and last[4] == prev[0]:
        return [last[0], last[4]]
    
    return []

def predict_next(history):
    """Dự đoán số tiếp theo"""
    if len(history) < 5:
        return "123", "4567"  # Default
    
    # Bắt các loại cầu
    bet_numbers = catch_bet(history)
    lo_roi_numbers = catch_lo_roi(history)
    dao_numbers = catch_dao(history)
    
    # Kết hợp các số
    main_candidates = []
    
    # Ưu tiên số bệt
    main_candidates.extend(bet_numbers[:3])
    
    # Thêm lô rơi
    main_candidates.extend(lo_roi_numbers)
    
    # Thêm số đảo
    main_candidates.extend(dao_numbers)
    
    # Loại bỏ trùng
    main_candidates = list(dict.fromkeys(main_candidates))
    
    # Lấy 3 số cho main
    main = ''.join(main_candidates[:3])
    while len(main) < 3:
        main += main_candidates[0] if main_candidates else '0'
    
    # Lấy 4 số cho support
    support_candidates = bet_numbers[3:7] if len(bet_numbers) > 3 else []
    support = ''.join(support_candidates[:4])
    while len(support) < 4:
        support += '0'
    
    return main[:3], support[:4]

# ================= GIAO DIỆN =================
st.set_page_config(page_title="BẮT CẦU 5D", layout="wide")

st.markdown("""
<style>
    .main { background: #0a0f1e; }
    .pred-box {
        background: linear-gradient(145deg, #1a1f35, #0d1225);
        border: 2px solid #4a6fa5;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .number-big {
        font-size: 100px;
        font-weight: 900;
        color: #ffd700;
        text-align: center;
        text-shadow: 0 0 20px #ffd700;
        letter-spacing: 15px;
    }
    .number-small {
        font-size: 60px;
        font-weight: 700;
        color: #4a9eff;
        text-align: center;
        letter-spacing: 10px;
    }
    .stats {
        background: #1e2438;
        padding: 15px;
        border-radius: 12px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 BẮT CẦU 5D - ĐƠN GIẢN MÀ HIỆU QUẢ")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area("📥 NHẬP KẾT QUẢ:", height=100,
                              placeholder="Ví dụ: 12345 67890 54321")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze = st.button("🔍 PHÂN TÍCH", use_container_width=True)
    with col_btn2:
        reset = st.button("🔄 RESET", use_container_width=True)

with col2:
    st.metric("📊 TỔNG SỐ KỲ", len(st.session_state.history))
    
    # Hiển thị 5 số gần nhất
    if st.session_state.history:
        st.write("**5 KỲ GẦN NHẤT:**")
        for i, num in enumerate(st.session_state.history[-5:]):
            st.code(f"Kỳ {i+1}: {num}")

# Xử lý reset
if reset:
    st.session_state.history = []
    st.session_state.last_pred = None
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.success("✅ Đã reset dữ liệu")
    st.rerun()

# Xử lý phân tích
if analyze and input_text:
    # Lọc số 5 chữ số
    numbers = re.findall(r'\b\d{5}\b', input_text)
    
    if numbers:
        # Thêm vào lịch sử
        for num in numbers:
            if num not in st.session_state.history:
                st.session_state.history.append(num)
        
        save_data(st.session_state.history)
        
        # Dự đoán số tiếp theo
        main, support = predict_next(st.session_state.history)
        
        st.session_state.last_pred = {
            'main': main,
            'support': support,
            'time': datetime.now().strftime("%H:%M:%S")
        }
        
        st.rerun()

# Hiển thị kết quả dự đoán
if st.session_state.last_pred:
    pred = st.session_state.last_pred
    
    st.markdown("---")
    st.markdown("<div class='pred-box'>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 DỰ ĐOÁN KỲ TIẾP THEO")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("**🔴 3 SỐ CHÍNH (ĐÁNH CHÍNH)**")
        st.markdown(f"<div class='number-big'>{pred['main']}</div>", unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("**🔵 4 SỐ LÓT (GIỮ VỐN)**")
        st.markdown(f"<div class='number-small'>{pred['support']}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='text-align:right; color:#666;'>⏰ {pred['time']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Hướng dẫn
    st.info("""
    **📌 CÁCH DÙNG:**
    1. Nhập kết quả thật vào ô trên
    2. Nhấn PHÂN TÍCH để AI dự đoán
    3. Khi có kết quả mới, nhập tiếp để AI học
    """)

# Phân tích thống kê đơn giản
if st.session_state.history:
    with st.expander("📊 THỐNG KÊ CƠ BẢN"):
        all_digits = ''.join(st.session_state.history[-50:])
        
        if all_digits:
            # Tần suất các số
            freq = Counter(all_digits)
            freq_data = []
            for i in range(10):
                freq_data.append({
                    'Số': str(i),
                    'Lần': freq.get(str(i), 0)
                })
            
            st.subheader("📈 TẦN SUẤT 10 SỐ (50 KỲ GẦN)")
            st.dataframe(freq_data, use_container_width=True)
            
            # Top 3 số nóng nhất
            top3 = [d for d, _ in freq.most_common(3)]
            st.success(f"🔥 SỐ NÓNG NHẤT: {', '.join(top3)}")

# Hiển thị lịch sử
if st.session_state.history:
    with st.expander("📜 LỊCH SỬ KẾT QUẢ"):
        # Tạo bảng lịch sử
        history_table = []
        for i, num in enumerate(st.session_state.history[-20:], 1):
            history_table.append({
                'Kỳ': f"Kỳ {i}",
                'Số': num,
                'Tổng': sum(int(d) for d in num),
                'Chẵn': sum(1 for d in num if int(d) % 2 == 0)
            })
        
        st.dataframe(history_table, use_container_width=True)