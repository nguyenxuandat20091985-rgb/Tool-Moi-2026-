import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH SIÊU TRÍ TUỆ =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_elite_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG LƯU TRỮ VĨNH VIỄN =================
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f) # Lưu tối đa 3000 kỳ để soi cầu dài hạn

if "history" not in st.session_state:
    st.session_state.history = load_data()

# ================= THUẬT TOÁN NHẬN DIỆN CẦU BỆT & ĐẢO =================
def detect_patterns(history):
    if len(history) < 10: return "Dữ liệu mỏng"
    
    # Chuyển thành ma trận số đơn
    matrix = np.array([[int(d) for d in str(s)] for s in history[-20:]])
    
    # 1. Kiểm tra bệt (Streak)
    last_row = matrix[-1]
    streaks = []
    for i in range(10):
        count = 0
        for row in reversed(matrix):
            if i in row: count += 1
            else: break
        if count >= 3: streaks.append(f"Số {i} bệt {count} kỳ")
    
    # 2. Kiểm tra cầu đảo (Vị trí)
    is_reversing = np.array_equal(matrix[-1], matrix[-2][::-1])
    
    return {
        "streaks": streaks,
        "is_reversing": is_reversing,
        "avg_val": np.mean(matrix)
    }

# ================= GIAO DIỆN TITAN ELITE =================
st.set_page_config(page_title="TITAN v24.0 ELITE", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #ffffff; }
    .prediction-panel {
        background: linear-gradient(180deg, #10141b 0%, #07090d 100%);
        border: 1px solid #1f2937; border-radius: 20px; padding: 40px;
        box-shadow: 0 10px 50px rgba(0,0,0,0.7);
    }
    .main-number-box {
        font-size: 110px; font-weight: 800; color: #00ff88;
        text-align: center; text-shadow: 0 0 40px rgba(0,255,136,0.5);
        margin: 20px 0;
    }
    .decision-label {
        font-size: 24px; font-weight: bold; text-align: center;
        padding: 10px; border-radius: 10px; margin-bottom: 20px;
    }
    .status-ok { background: #064e3b; color: #34d399; }
    .status-stop { background: #7f1d1d; color: #f87171; border: 1px solid #f87171; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#00ff88;'>🧬 TITAN v24.0 ELITE OMNI</h1>", unsafe_allow_html=True)

# Container nhập liệu mượt mà
with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        raw_input = st.text_area("📡 NẠP DỮ LIỆU (Tự động lưu trữ):", height=80, placeholder="Dán dãy số 5D...")
    with c2:
        st.write("###")
        if st.button("🚀 GIẢI MÃ SIÊU CẤP"):
            new_nums = re.findall(r"\d{5}", raw_input)
            if new_nums:
                # Chỉ thêm những số chưa có để tránh trùng
                st.session_state.history = list(dict.fromkeys(st.session_state.history + new_nums))
                save_data(st.session_state.history)
                
                # Gọi trí tuệ Gemini kết hợp Logic bẻ cầu
                patterns = detect_patterns(st.session_state.history)
                prompt = f"""
                Bạn là TITAN v24.0 - Siêu trí tuệ phân tích Lotobet.
                Dữ liệu lịch sử: {st.session_state.history[-100:]}
                Phân tích kỹ thuật: {patterns}
                
                Nhiệm vụ:
                1. Xác định kỳ này nhà cái có đang "thả cầu" hay "siết cầu".
                2. Nếu bệt quá dài, hãy dự đoán điểm gãy.
                3. Đưa ra 3 số CHỦ LỰC (3D) chính xác nhất.
                4. Quyết định: NÊN ĐÁNH hay DỪNG (Rất quan trọng).
                
                TRẢ VỀ JSON:
                {{
                    "decision": "ĐÁNH" hoặc "DỪNG",
                    "main_3": "3 số",
                    "support_4": "4 số",
                    "logic": "Giải thích sâu về nhịp cầu",
                    "confidence": %
                }}
                """
                try:
                    response = neural_engine.generate_content(prompt)
                    res_json = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                    st.session_state.result = res_json
                except:
                    st.error("Neural Link gián đoạn. Đang dùng thuật toán dự phòng...")
            st.rerun()

# ================= HIỂN THỊ KẾT QUẢ TINH HOA =================
if "result" in st.session_state:
    res = st.session_state.result
    
    st.markdown("<div class='prediction-panel'>", unsafe_allow_html=True)
    
    # Trạng thái Nên đánh hay Dừng
    status_class = "status-ok" if res['decision'] == "ĐÁNH" else "status-stop"
    st.markdown(f"<div class='decision-label {status_class}'>LỜI KHUYÊN AI: {res['decision']}</div>", unsafe_allow_html=True)
    
    st.write(f"💡 **PHÂN TÍCH CHIẾN THUẬT:** {res['logic']}")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"<div class='main-number-box'>{res['main_3']}</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#888;'>🔥 3 SỐ VÀNG (CHỦ LỰC)</p>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"<h1 style='text-align:center; color:#00d1ff; margin-top:40px;'>{res['support_4']}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#888;'>🛡️ DÀN LÓT AN TOÀN</p>", unsafe_allow_html=True)

    st.divider()
    
    full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 DÀN 7 SỐ TỔNG HỢP:", full_dan)
    
    st.progress(res['confidence'] / 100)
    st.markdown(f"<p style='text-align:right;'>Độ tin cậy hệ thống: {res['confidence']}%</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê trực quan
if st.session_state.history:
    with st.expander("📊 BẢN ĐỒ NHỊP CẦU (REAL-TIME)"):
        p = detect_patterns(st.session_state.history)
        st.write(f"🚩 **Cầu bệt đang chạy:** {p['streaks']}")
        st.write(f"🔄 **Cầu đảo vị trí:** {'CÓ DẤU HIỆU' if p['is_reversing'] else 'KHÔNG'}")
        
        # Biểu đồ tần suất 20 kỳ
        flat_data = "".join(st.session_state.history[-20:])
        df_chart = pd.DataFrame.from_dict(Counter(flat_data), orient='index').sort_index()
        st.bar_chart(df_chart)

if st.sidebar.button("🗑️ XÓA TOÀN BỘ DỮ LIỆU"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
