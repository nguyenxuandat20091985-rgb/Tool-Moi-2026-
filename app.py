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

# ================= THIẾT KẾ GIAO DIỆN v22.0 STYLE =================
st.set_page_config(page_title="TITAN v24.2 OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 20px; margin-top: 20px;
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px; border-right: 2px solid #30363d;
    }
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px; padding-left: 20px;
    }
    .status-bar { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 15px; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🎯 TITAN v24.2 - ĐẶC CHẾ 3 SỐ 5 TINH</h2>", unsafe_allow_html=True) 

# ================= PHẦN 1: NHẬP LIỆU =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Dán dữ liệu (Dòng mới nhất ở trên cùng):", height=120, placeholder="80586\n64549\n03886...")
    with col_st:
        st.write(f"📊 Kho dữ liệu: **{len(st.session_state.history)} kỳ**")
        st.write(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 GIẢI MÃ")
        btn_reset = c2.button("🗑️ RESET") 

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun() 

if btn_save:
    # Lọc dữ liệu chuẩn 5 chữ số
    clean = [line.strip() for line in raw_input.split('\n') if re.match(r"^\d{5}$", line.strip())]
    if clean:
        # Giữ tính nhất quán: Dữ liệu mới nhất luôn được ưu tiên
        st.session_state.history = clean + [x for x in st.session_state.history if x not in clean]
        save_db(st.session_state.history)
        
        # LOGIC PHÂN TÍCH CHO 3 SỐ 5 TINH
        prompt = f"""
        Hệ thống: TITAN v24.2 ELITE. Dữ liệu thực tế: {st.session_state.history[:50]}
        Quy tắc: Thắng khi kết quả mở thưởng chứa ĐỦ 3 con số đã chọn (không phân biệt thứ tự).
        Nhiệm vụ:
        1. Phân tích xác suất xuất hiện cùng lúc của các bộ 3 số trong 50 kỳ qua.
        2. Tìm các số đang 'về cùng nhau' (Combo).
        3. Chốt 3 số 'CHỦ LỰC' có khả năng nổ cùng lúc cao nhất.
        4. Chốt 4 số 'LÓT' để ghép dàn 7 số giữ vốn.
        Trả về JSON: {{"main_3": "abc", "support_4": "defg", "decision": "ĐÁNH/DỪNG", "logic": "...", "color": "Green/Red", "conf": 98}}
        """
        try:
            response = neural_engine.generate_content(prompt)
            st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        except:
            # Dự phòng bằng thuật toán thống kê Combo nếu AI lỗi
            all_str = "".join(st.session_state.history[:20])
            top_7 = [x[0] for x in Counter(all_str).most_common(7)]
            st.session_state.last_prediction = {
                "main_3": "".join(top_7[:3]), 
                "support_4": "".join(top_7[3:]), 
                "decision": "CÂN NHẮC", 
                "logic": "Dựa trên tần suất xuất hiện dày đặc nhất trong 20 kỳ.", 
                "color": "Red", 
                "conf": 70
            }
        st.rerun() 

# ================= PHẦN 2: KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    bg_color = "#238636" if res['color'].lower() == "green" else "#da3633"
    st.markdown(f"<div class='status-bar' style='background: {bg_color};'>📢 KHUYẾN NGHỊ: {res['decision']} ({res['conf']}%)</div>", unsafe_allow_html=True) 

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(f"<p style='color:#8b949e; margin-bottom:5px; text-align:center;'>🔥 BỘ 3 SỐ CHỦ LỰC (Dùng cho 3 số 5 tinh)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<p style='color:#8b949e; margin-bottom:5px; text-align:center;'>🛡️ 4 SỐ LÓT (Phòng thủ)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"💡 **PHÂN TÍCH NHỊP CẦU:** {res['logic']}")
    
    # Khu vực Copy dàn 7 số cho Kubet (Kết hợp Chính + Lót)
    full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 DÀN 7 SỐ (Ghép sẵn cho Kubet):", full_dan)
    st.caption("Mẹo: Với quy tắc 3 số 5 tinh, hãy ưu tiên vào tiền mạnh ở bộ 3 số chính.")
    st.markdown("</div>", unsafe_allow_html=True) 

# Thống kê nhịp rơi
if st.session_state.history:
    with st.expander("📊 Biểu đồ nhịp rơi 50 kỳ gần nhất"):
        all_digits = "".join(st.session_state.history[:50])
        counts = Counter(all_digits)
        df_chart = pd.DataFrame.from_dict(counts, orient='index').sort_index()
        st.bar_chart(df_chart)
