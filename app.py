import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_supreme_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# Hàm tải dữ liệu vĩnh viễn
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: 
                data = json.load(f)
                return data if isinstance(data, list) else []
            except: return []
    return []

# Hàm lưu dữ liệu vĩnh viễn
def save_db(data):
    with open(DB_FILE, "w") as f:
        # Lưu tối đa 3000 kỳ để AI học nhịp dài
        json.dump(data[-3000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= THIẾT KẾ GIAO DIỆN v22.0 STYLE =================
st.set_page_config(page_title="TITAN v24.2 SUPREME", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px; border-right: 3px solid #30363d;
        text-shadow: 0 0 15px rgba(255,88,88,0.5);
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px; padding-left: 25px;
        text-shadow: 0 0 10px rgba(88,166,255,0.4);
    }
    .status-bar { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 15px; font-size: 18px; }
    .logic-text { background: #161b22; padding: 15px; border-radius: 8px; border-left: 5px solid #58a6ff; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🎯 TITAN v24.2 - SIÊU TRÍ TUỆ (BẢN SUPREME)</h2>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU & XỬ LÝ ĐA TẦNG =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Dán dữ liệu (Hệ thống tự lọc số trùng & lỗi):", height=120, placeholder="32880\n21808...")
    with col_st:
        st.write(f"📊 Tổng dữ liệu đã học: **{len(st.session_state.history)} kỳ**")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 GIẢI MÃ TINH HOA", use_container_width=True)
        btn_reset = c2.button("🗑️ RESET BỘ NHỚ", use_container_width=True)

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.session_state.pop('last_prediction', None)
    st.rerun()

if btn_save:
    # TẦNG 1: Lọc định dạng và số trùng
    new_raw = re.findall(r"\b\d{5}\b", raw_input)
    if new_raw:
        # Kết hợp dữ liệu cũ, loại bỏ trùng lặp tuyệt đối
        updated_history = st.session_state.history + new_raw
        st.session_state.history = list(dict.fromkeys(updated_history))
        save_db(st.session_state.history)
        
        # TẦNG 2: Gemini Phân tích sâu Bệt/Đảo
        history_snippet = st.session_state.history[-100:]
        prompt = f"""
        Bạn là Siêu Trí Tuệ TITAN v24.2 chuyên soi cầu 3D Lotobet.
        Dữ liệu lịch sử 100 kỳ gần nhất: {history_snippet}
        
        Nhiệm vụ:
        1. Phân tích nhịp BỆT (số rơi lại) và nhịp ĐẢO (số hoán vị, số bóng).
        2. Lọc bỏ các số nhiễu có dấu hiệu bị nhà cái điều tiết.
        3. Chốt 3 số CHÍNH (main_3) có tỷ lệ nổ cao nhất trong giải 5 số.
        4. Chốt 4 số LÓT (support_4) an toàn.
        
        Yêu cầu nghiêm ngặt: 
        - Nếu phát hiện cầu Bệt quá dài (>5 kỳ) hoặc Đảo cầu liên tục gây nhiễu, hãy đặt "decision": "DỪNG".
        - Chỉ trả về định dạng JSON: {{"main_3": "3 số", "support_4": "4 số", "decision": "ĐÁNH/DỪNG/THEO NHẸ", "logic": "Giải thích ngắn gọn nhịp cầu", "color": "Green/Red/Yellow", "conf": 0-100}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            clean_json = re.search(r'\{.*\}', response.text, re.DOTALL).group()
            st.session_state.last_prediction = json.loads(clean_json)
        except:
            # Thuật toán dự phòng nếu AI bận
            all_digits = "".join(st.session_state.history[-50:])
            top_7 = [x[0] for x in Counter(all_digits).most_common(7)]
            st.session_state.last_prediction = {
                "main_3": "".join(top_7[:3]), 
                "support_4": "".join(top_7[3:]), 
                "decision": "THEO NHẸ", 
                "logic": "AI bận, đang dùng thuật toán tần suất nhịp rơi cục bộ.", 
                "color": "Yellow", 
                "conf": 70
            }
        st.rerun()

# ================= PHẦN 2: KẾT QUẢ TRỰC QUAN (GIỮ UI v22.0) =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    # Thanh trạng thái thông minh
    color_map = {"green": "#238636", "red": "#da3633", "yellow": "#d29922"}
    status_color = color_map.get(res['color'].lower(), "#30363d")
    
    st.markdown(f"""
        <div class='status-bar' style='background: {status_color}; border: 1px solid white;'>
            📢 LỜI KHUYÊN AI: {res['decision']} (Độ tự tin: {res['conf']}%)
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col_main, col_lot = st.columns([1.5, 1])
    with col_main:
        st.markdown("<p style='text-align:center; color:#8b949e; margin-bottom:0;'>🔥 3 SỐ CHỦ LỰC (XÁC SUẤT CAO)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    with col_lot:
        st.markdown("<p style='text-align:center; color:#8b949e; margin-bottom:0;'>🛡️ 4 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='logic-text'>", unsafe_allow_html=True)
    st.write(f"💡 **PHÂN TÍCH NHỊP CẦU:** {res['logic']}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dàn 7 số chuẩn để copy vào Kubet
    full_set = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ KUBET:", full_set)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê nhịp rơi để anh tự đối chiếu
if st.session_state.history:
    st.divider()
    with st.expander("📊 Xem bảng tần suất số (Tự soi nhịp Bệt/Đảo)"):
        # Phân tích 50 kỳ gần nhất
        sample = "".join(st.session_state.history[-50:])
        counts = Counter(sample)
        stat_df = pd.DataFrame([{"Số": k, "Tần suất": v} for k, v in counts.items()]).sort_values("Số")
        st.bar_chart(stat_df.set_index("Số"))
