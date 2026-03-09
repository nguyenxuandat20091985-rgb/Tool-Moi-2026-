import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter
from datetime import datetime

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_permanent_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= GIAO DIỆN HIỆN ĐẠI =================
st.set_page_config(page_title="TITAN v24.2 - 3 SỐ 5 TINH", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .main-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 15px; padding: 25px; text-align: center;
    }
    .number-display {
        font-size: 80px; font-weight: 800; color: #ff5858;
        letter-spacing: 15px; text-shadow: 0px 0px 10px rgba(255,88,88,0.5);
    }
    .logic-box { background: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #58a6ff; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🎯 TITAN v24.2 - CHIẾN THUẬT 3 SỐ 5 TINH</h1>", unsafe_allow_html=True)

# ================= NHẬP LIỆU & XỬ LÝ =================
col_in, col_st = st.columns([2, 1])
with col_in:
    raw_input = st.text_area("📡 Dán dữ liệu 5D (Mới nhất ở trên):", height=120, placeholder="12864\n80673...")
with col_st:
    st.write(f"📊 Kho dữ liệu: **{len(st.session_state.history)} kỳ**")
    if st.button("🚀 GIẢI MÃ BỘ 3", use_container_width=True):
        clean = re.findall(r"\d{5}", raw_input)
        if clean:
            # Ưu tiên dữ liệu mới nhất
            st.session_state.history = clean + st.session_state.history
            st.session_state.history = list(dict.fromkeys(st.session_state.history))[:3000]
            save_db(st.session_state.history)
            
            # Phân tích bộ 3 (Combo 3 số có xác suất xuất hiện cùng nhau cao nhất)
            prompt = f"""
            Hệ thống: TITAN v24.2. Mục tiêu: Tìm bộ 3 số (3 số 5 tinh).
            Dữ liệu 50 kỳ gần nhất: {st.session_state.history[:50]}
            Quy tắc: Kết quả 5 số phải chứa đủ 3 số được chọn.
            Yêu cầu:
            1. Tìm 3 số (0-9) có tần suất xuất hiện cùng nhau (co-occurrence) cao nhất.
            2. Kiểm tra nhịp rơi của từng hàng để tránh các số đang "gan".
            3. Trả về JSON: {{"combo_3": "123", "backup_combo": "456", "decision": "ĐÁNH/DỪNG", "logic": "...", "conf": 95}}
            """
            try:
                response = neural_engine.generate_content(prompt)
                st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            except:
                # Logic dự phòng nếu AI lỗi: Lấy 3 số xuất hiện nhiều nhất trong 20 kỳ
                all_n = "".join(st.session_state.history[:20])
                top_3 = "".join([x[0] for x in Counter(all_n).most_common(3)])
                st.session_state.last_prediction = {"combo_3": top_3, "backup_combo": "---", "decision": "ĐÁNH", "logic": "Thống kê tần suất cơ bản.", "conf": 60}
            st.rerun()

if st.button("🗑️ RESET TOÀN BỘ"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3>🔥 BỘ 3 CHỦ LỰC (3 SỐ 5 TINH)</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='number-display'>{res['combo_3']}</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("TRẠNG THÁI", res['decision'])
    col2.metric("ĐỘ TIN CẬY", f"{res['conf']}%")
    col3.metric("BỘ DỰ PHÒNG", res['backup_combo'])
    
    st.markdown(f"<div class='logic-box'><b>💡 PHÂN TÍCH NHỊP RƠI:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.info(f"📋 **Hướng dẫn vào tiền:** Đặt cược bộ 3 số **{res['combo_3']}** cho tất cả các hàng. Chỉ cần kỳ tới ra đủ 3 số này là thắng.")
    st.markdown("</div>", unsafe_allow_html=True)

# Biểu đồ soi nhịp gan
if st.session_state.history:
    with st.expander("📊 Soi nhịp rơi 10 số (Dòng chảy dữ liệu)"):
        all_digits = "".join(st.session_state.history[:30])
        count_data = Counter(all_digits)
        df = pd.DataFrame.from_dict(count_data, orient='index').sort_index()
        st.bar_chart(df)
