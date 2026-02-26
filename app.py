import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

# ================= CẤU HÌNH =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_data.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except:
                return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f)

# Khởi tạo
if "history" not in st.session_state:
    st.session_state.history = load_db()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = {"total": 0, "correct": 0}

# ================= THUẬT TOÁN DỰ ĐOÁN THỰC CHIẾN =================

def detect_cau_bac_nho(history):
    """
    Thuật toán bạc nhớ - dựa trên lịch sử lặp lại
    """
    if len(history) < 10:
        return []
    
    last = history[-1]
    predictions = []
    
    # Tìm các lần xuất hiện của số cuối cùng trong quá khứ
    for i in range(len(history) - 1):
        if history[i] == last and i + 1 < len(history):
            # Số thường về sau số này
            next_num = history[i + 1]
            predictions.append(next_num)
    
    if predictions:
        # Đếm tần suất
        counter = Counter(predictions)
        # Lấy top 3 số hay về nhất
        top = [num for num, _ in counter.most_common(3)]
        return top
    
    return []

def detect_cau_bet(history):
    """
    Phát hiện cầu bệt - số đang về liên tục
    """
    if len(history) < 5:
        return []
    
    # Lấy 5 số gần nhất
    recent = history[-5:]
    
    # Tìm số xuất hiện nhiều nhất
    all_digits = ''.join(recent)
    counter = Counter(all_digits)
    
    # Số có tần suất > 3 trong 5 kỳ
    bet_numbers = [d for d, count in counter.items() if count >= 3]
    
    return bet_numbers

def detect_cau_dao(history):
    """
    Phát hiện cầu đảo - số đảo chiều
    """
    if len(history) < 3:
        return []
    
    last = history[-1]
    prev = history[-2]
    
    # Kiểm tra đảo đầu cuối
    if last[0] == prev[4] and last[4] == prev[0]:
        return [last[0], last[4]]
    
    # Kiểm tra đảo toàn bộ
    if last[::-1] == prev:
        return list(last)
    
    return []

def detect_cau_tong(history):
    """
    Phân tích tổng các số
    """
    if len(history) < 10:
        return []
    
    tongs = []
    for num in history[-10:]:
        tong = sum(int(d) for d in num)
        tongs.append(tong % 10)  # Lấy hàng đơn vị
    
    counter = Counter(tongs)
    hot_tong = [str(t) for t, _ in counter.most_common(3)]
    
    return hot_tong

def predict_numbers(history):
    """
    Tổng hợp các thuật toán để dự đoán
    """
    if len(history) < 5:
        return "123", "4567", "CHỜ DỮ LIỆU"
    
    # Thu thập các dự đoán từ các thuật toán
    predictions = []
    
    # 1. Bạc nhớ
    predictions.extend(detect_cau_bac_nho(history))
    
    # 2. Cầu bệt
    predictions.extend(detect_cau_bet(history))
    
    # 3. Cầu đảo
    predictions.extend(detect_cau_dao(history))
    
    # 4. Cầu tổng
    predictions.extend(detect_cau_tong(history))
    
    # Lấy số từ lịch sử gần nhất
    if history:
        predictions.extend(list(history[-1]))
    
    # Đếm tần suất và lấy top
    if predictions:
        counter = Counter(predictions)
        top_numbers = [num for num, _ in counter.most_common(7)]
        
        # Đảm bảo đủ 7 số
        while len(top_numbers) < 7:
            top_numbers.append(str(np.random.randint(0, 10)))
        
        main = ''.join(top_numbers[:3])
        support = ''.join(top_numbers[3:7])
        
        # Xác định trạng thái cầu
        if len(detect_cau_bet(history)) >= 2:
            status = "CẦU BỆT RÕ - ĐÁNH MẠNH"
        elif detect_cau_dao(history):
            status = "CẦU ĐẢO - THEO DÕI"
        elif len(history) > 10 and history[-1] == history[-2]:
            status = "BỆT 2 KỲ - ĐÁNH"
        else:
            status = "CHỜ CẦU RÕ"
        
        return main, support, status
    
    return "123", "4567", "KHÔNG RÕ CẦU"

# ================= GIAO DIỆN =================
st.set_page_config(page_title="TITAN BẠC NHỚ 5D", layout="wide")

st.markdown("""
<style>
    .stApp { background: #010409; color: #e6edf3; }
    .pred-card {
        background: #0d1117;
        border: 2px solid #58a6ff;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }
    .main-number {
        font-size: 90px;
        font-weight: 900;
        color: #ff5858;
        text-align: center;
        letter-spacing: 15px;
        text-shadow: 0 0 25px rgba(255,88,88,0.5);
    }
    .support-number {
        font-size: 60px;
        font-weight: 700;
        color: #58a6ff;
        text-align: center;
        letter-spacing: 10px;
        text-shadow: 0 0 15px rgba(88,166,255,0.3);
    }
    .status-bar {
        background: #1f6feb;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .history-box {
        background: #161b22;
        padding: 10px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🎯 TITAN BẠC NHỚ 5D</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Kết hợp 4 thuật toán: Bạc nhớ + Bệt + Đảo + Tổng</p>", unsafe_allow_html=True)

# Layout
col_input, col_info = st.columns([2, 1])

with col_input:
    raw_input = st.text_area("📥 NHẬP KẾT QUẢ MỚI:", height=100,
                            placeholder="Dán số 5D mới nhất vào đây (VD: 12345 67890)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        analyze_btn = st.button("🔍 PHÂN TÍCH", use_container_width=True)
    with c2:
        reset_btn = st.button("🔄 RESET", use_container_width=True)
    with c3:
        if st.session_state.last_prediction:
            if st.button("✅ ĐÚNG", use_container_width=True):
                st.session_state.accuracy["total"] += 1
                st.session_state.accuracy["correct"] += 1
                st.rerun()

with col_info:
    st.metric("📊 TỔNG KỲ", len(st.session_state.history))
    
    if st.session_state.accuracy["total"] > 0:
        acc = (st.session_state.accuracy["correct"] / st.session_state.accuracy["total"]) * 100
        st.metric("🎯 ĐỘ CHÍNH XÁC", f"{acc:.1f}%")
    
    # Hiển thị 5 số gần nhất
    if st.session_state.history:
        st.write("**5 KỲ GẦN NHẤT:**")
        recent_html = "<div class='history-box'>"
        for num in st.session_state.history[-5:]:
            recent_html += f"{num} "
        recent_html += "</div>"
        st.markdown(recent_html, unsafe_allow_html=True)

# Xử lý reset
if reset_btn:
    st.session_state.history = []
    st.session_state.last_prediction = None
    st.session_state.accuracy = {"total": 0, "correct": 0}
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.success("✅ Đã reset dữ liệu")
    st.rerun()

# Xử lý phân tích
if analyze_btn and raw_input:
    # Lọc số 5 chữ số
    numbers = re.findall(r'\b\d{5}\b', raw_input)
    
    if numbers:
        # Thêm số mới vào lịch sử
        for num in numbers:
            if num not in st.session_state.history:
                st.session_state.history.append(num)
        
        save_db(st.session_state.history)
        
        # Dự đoán số tiếp theo
        main, support, status = predict_numbers(st.session_state.history)
        
        # Kiểm tra nếu có dự đoán trước đó
        if st.session_state.last_prediction and numbers:
            prev = st.session_state.last_prediction
            actual = numbers[0]
            
            # Đếm số đúng
            correct_count = 0
            for i in range(3):
                if i < len(prev['main']) and i < len(actual) and prev['main'][i] == actual[i]:
                    correct_count += 1
            
            # Cập nhật accuracy
            st.session_state.accuracy["total"] += 1
            if correct_count >= 2:
                st.session_state.accuracy["correct"] += 1
        
        # Lưu dự đoán mới
        st.session_state.last_prediction = {
            'main': main,
            'support': support,
            'status': status,
            'time': datetime.now().strftime("%H:%M:%S")
        }
        
        st.rerun()

# Hiển thị kết quả dự đoán
if st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    
    st.markdown(f"<div class='status-bar'>{pred['status']}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([1.5, 1])
    
    with col_m1:
        st.markdown("<p style='text-align:center; font-weight:bold;'>🎯 3 SỐ CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-number'>{pred['main']}</div>", unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("<p style='text-align:center; font-weight:bold;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='support-number'>{pred['support']}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='text-align:right; color:#666;'>⏰ {pred['time']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Hướng dẫn
    with st.expander("📘 HƯỚNG DẪN SỬ DỤNG"):
        st.write("""
        **Cách dùng đúng:**
        1. Nhập kết quả thật vào ô trên
        2. Nhấn PHÂN TÍCH để AI dự đoán kỳ tiếp theo
        3. Khi có kết quả thật, nhập tiếp và nhấn PHÂN TÍCH
        4. Nhấn nút ĐÚNG nếu dự đoán chính xác
        
        **Thuật toán sử dụng:**
        - 🎯 Bạc nhớ: Học từ lịch sử lặp lại
        - 📈 Cầu bệt: Bắt số đang về nhiều
        - 🔄 Cầu đảo: Phát hiện đảo chiều
        - 📊 Cầu tổng: Phân tích tổng các số
        """)

# Phân tích chuyên sâu
if st.session_state.history:
    with st.expander("📊 PHÂN TÍCH CHUYÊN SÂU"):
        tab1, tab2, tab3 = st.tabs(["Tần suất", "Bạc nhớ", "Lịch sử"])
        
        with tab1:
            all_digits = ''.join(st.session_state.history[-50:])
            if all_digits:
                freq = Counter(all_digits)
                df = pd.DataFrame({
                    'Số': list(range(10)),
                    'Lần': [freq.get(str(i), 0) for i in range(10)]
                })
                st.bar_chart(df.set_index('Số'))
        
        with tab2:
            if len(st.session_state.history) > 10:
                st.write("**Phân tích bạc nhớ 10 kỳ gần:**")
                for i in range(min(10, len(st.session_state.history))):
                    idx = -i-1
                    if idx < -1:
                        st.write(f"{st.session_state.history[idx]} → {st.session_state.history[idx+1]}")
        
        with tab3:
            history_df = pd.DataFrame({
                'Kỳ': range(1, len(st.session_state.history[-20:]) + 1),
                'Số': st.session_state.history[-20:]
            })
            st.dataframe(history_df, use_container_width=True)