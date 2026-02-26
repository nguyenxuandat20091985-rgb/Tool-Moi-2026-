import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG ULTIMATE =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_ultimate_v25.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, dict) else {"history": [], "last_res": None}
            except: return {"history": [], "last_res": None}
    return {"history": [], "last_res": None}

def save_db(history, last_res):
    with open(DB_FILE, "w") as f:
        json.dump({"history": history[-3000:], "last_res": last_res}, f)

# Khởi tạo bộ nhớ vĩnh viễn
db = load_db()
if "history" not in st.session_state:
    st.session_state.history = db.get("history", [])
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = db.get("last_res", None)

# ================= GIAO DIỆN v22 STYLE NÂNG CẤP =================
st.set_page_config(page_title="TITAN v25.0 ULTIMATE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .main-box {
        font-size: 65px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 8px; border-right: 2px solid #30363d;
    }
    .lot-box {
        font-size: 45px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }
    .status-bar { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 18px; margin-bottom: 15px; }
    .warning-panel { background: #331010; border: 1px solid #f85149; color: #ff7b72; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v25.0 ULTIMATE - CHIẾN THẮNG KUBET</h1>", unsafe_allow_html=True)

# ================= KHU VỰC NHẬP LIỆU =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 NẠP DỮ LIỆU KỲ MỚI (Lọc sạch bẩn & trùng):", height=120, placeholder="Dán dãy số 5D tại đây...")
    with col_st:
        st.write(f"📊 Tổng dữ liệu học được: **{len(st.session_state.history)} kỳ**")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 GIẢI MÃ TINH HOA", use_container_width=True)
        btn_reset = c2.button("🗑️ RESET BỘ NHỚ", use_container_width=True)

if btn_reset:
    st.session_state.history = []
    st.session_state.last_prediction = None
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

if btn_save:
    # Bước 1: Loại bỏ 5 số không trúng, chỉ lấy đúng định dạng 5 số cho kỳ tiếp theo
    clean = re.findall(r"\b\d{5}\b", raw_input)
    if clean:
        # Loại trùng lặp và gộp vào lịch sử
        new_history = list(dict.fromkeys(st.session_state.history + clean))
        st.session_state.history = new_history
        
        # Bước 2: Gemini phân tích đa tầng
        prompt = f"""
        Hệ thống: TITAN v25.0 ULTIMATE. 
        Dữ liệu lịch sử: {st.session_state.history[-150:]}
        Yêu cầu khắt khe:
        1. Nhận diện nhịp Bệt (Streak) và Cầu Đảo của nhà cái. Cảnh báo nếu cầu bệt nguy hiểm.
        2. Dự đoán 2 dàn số CHỦ LỰC (mỗi dàn 3 số). Ví dụ: 456 và 789.
        3. Dự đoán 4 số LÓT an toàn để giữ vốn.
        4. Phân tích rõ nhịp 'NÊN ĐÁNH' hoặc 'DỪNG' dựa trên độ nhạy bén ma trận.
        
        Trả về JSON chuẩn: 
        {{
            "core_1": "abc", 
            "core_2": "xyz", 
            "support_4": "defg", 
            "decision": "ĐÁNH/DỪNG/CẢNH BÁO BỆT", 
            "logic": "Phân tích kỹ nhịp cầu đảo...", 
            "color": "Green/Red/Yellow", 
            "conf": 99
        }}
        """
        try:
            response = neural_engine.generate_content(prompt)
            res_data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.last_prediction = res_data
            save_db(st.session_state.history, res_data)
        except Exception as e:
            st.error(f"Lỗi AI: {e}")
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ DỰ ĐOÁN =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    
    # Thanh trạng thái nhạy bén
    color_map = {"green": "#238636", "red": "#da3633", "yellow": "#d29922"}
    bg_color = color_map.get(res['color'].lower(), "#30363d")
    
    st.markdown(f"<div class='status-bar' style='background: {bg_color};'>📢 QUYẾT ĐỊNH: {res['decision']} | ĐỘ TIN CẬY: {res['conf']}%</div>", unsafe_allow_html=True)

    if res['color'].lower() == 'red':
        st.markdown("<div class='warning-panel'>⚠️ CẢNH BÁO: CẦU BỆT SÂU HOẶC ĐẢO CẦU LIÊN TỤC - HẠN CHẾ VÀO TIỀN</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🔥 CHỦ LỰC 1</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-box'>{res['core_1']}</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🔥 CHỦ LỰC 2</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-box' style='color:#f2cc60;'>{res['core_2']}</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box' style='margin-top:15px;'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"💡 **PHÂN TÍCH TỪ AI:** {res['logic']}")
    
    # Dàn 7-8 số tổng hợp
    full_dan = "".join(sorted(set(res['core_1'] + res['core_2'] + res['support_4'])))
    st.text_input("📋 SAO CHÉP DÀN TỔNG HỢP:", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê nhịp rơi
if st.session_state.history:
    with st.expander("📊 Phân tích ma trận tần suất (50 kỳ gần nhất)"):
        all_d = "".join(st.session_state.history[-50:])
        st.bar_chart(pd.Series(Counter(all_d)).sort_index())
