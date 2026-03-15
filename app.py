import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG TITAN V25 =================
# Anh thay API Key của anh vào đây
API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_permanent_v25.json"

# BẢNG CẶP SỐ HAY ĐI CÙNG (Dữ liệu cơ sở anh cung cấp)
PAIR_DATA = {
    "178": ["0", "2", "4"], "034": ["5", "8", "9"], "458": ["0", "2", "6"],
    "578": ["1", "3", "4"], "019": ["2", "4", "7"], "679": ["0", "3", "5"],
    "235": ["4", "6", "8"], "456": ["1", "7", "9"], "124": ["0", "5", "8"],
    "245": ["3", "7", "0"], "247": ["1", "8", "6"], "248": ["0", "3", "9"],
    "246": ["5", "7", "1"], "340": ["2", "6", "8"], "349": ["1", "5", "7"],
    "348": ["0", "2", "6"], "015": ["3", "7", "9"], "236": ["0", "4", "8"],
    "028": ["1", "5", "9"], "026": ["3", "7", "4"], "047": ["2", "5", "8"],
    "046": ["1", "3", "9"], "056": ["2", "4", "7"], "136": ["0", "5", "8"],
    "138": ["2", "4", "7"], "378": ["0", "1", "6"]
}

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
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= GIAO DIỆN CYBER V25 =================
st.set_page_config(page_title="TITAN v25 OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #05050a; color: #ffffff; }
    .main-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 2px solid #30363d; border-radius: 20px;
        padding: 25px; margin-top: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .num-display {
        font-size: 85px; font-weight: 900; color: #00ff88;
        text-align: center; letter-spacing: 15px;
        text-shadow: 0 0 20px rgba(0,255,136,0.4);
    }
    .support-display {
        font-size: 45px; font-weight: 700; color: #00d4ff;
        text-align: center; letter-spacing: 10px; opacity: 0.8;
    }
    .status-bar { 
        padding: 15px; border-radius: 50px; text-align: center; 
        font-weight: 900; text-transform: uppercase; margin-bottom: 20px;
        letter-spacing: 2px;
    }
    .btn-decrypt {
        background: linear-gradient(90deg, #00ff88, #00d4ff) !important;
        color: black !important; font-weight: 900 !important;
        border-radius: 50px !important; border: none !important; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #00ff88;'>⚡ TITAN v25 OMNI ELITE</h1>", unsafe_allow_html=True)

# ================= KHU VỰC ĐIỀU KHIỂN =================
col_in, col_stats = st.columns([2, 1])

with col_in:
    raw_input = st.text_area("📡 DÁN DỮ LIỆU KỲ QUAY (5 SỐ):", height=120, placeholder="32457\n83465\n...")

with col_stats:
    st.markdown(f"📊 Hệ thống đã lưu: **{len(st.session_state.history)} kỳ**")
    c1, c2 = st.columns(2)
    btn_save = c1.button("🚀 GIẢI MÃ AI", use_container_width=True)
    btn_reset = c2.button("🗑️ RESET DB", use_container_width=True)

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

# ================= THUẬT TOÁN XỬ LÝ =================
if btn_save:
    clean = re.findall(r"\d{5}", raw_input)
    if clean:
        # Thêm dữ liệu mới, loại bỏ trùng lặp và lưu
        st.session_state.history.extend(clean)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)

        # PHÂN TÍCH THUẬT TOÁN NỘI BỘ (Nếu Gemini lỗi hoặc bận)
        all_numbers = "".join(st.session_state.history[-50:])
        counts = Counter(all_numbers)
        # Lấy 3 số hay nổ nhất
        top_3_stat = "".join([x[0] for x in counts.most_common(3)])
        # Lấy 4 số tiếp theo làm lót
        support_4_stat = "".join([x[0] for x in counts.most_common(7)[3:]])

        # GỬI GEMINI PHÂN TÍCH SÂU (Kết hợp bảng cặp số)
        prompt = f"""
        Hệ thống: TITAN v25 OMNI. Dữ liệu: {st.session_state.history[-50:]}
        Nhiệm vụ: 
        1. Dựa vào các cặp số hay đi kèm: {json.dumps(PAIR_DATA)}
        2. Phân tích nhịp rơi của kỳ cuối: {st.session_state.history[-1]}
        3. LOẠI BỎ 7 SỐ, CHỈ CHỌN 3 SỐ KHẢ NĂNG VỀ CAO NHẤT.
        4. Trả về đúng định dạng JSON:
        {{"main_3": "3 số chính", "support_4": "4 số lót", "decision": "ĐÁNH/DỪNG", "logic": "vì sao chọn...", "conf": 98}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        except:
            st.session_state.last_prediction = {
                "main_3": top_3_stat,
                "support_4": support_4_stat,
                "decision": "ĐÁNH",
                "logic": "Sử dụng thuật toán nén tần suất dự phòng.",
                "conf": 75
            }
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    # Thanh trạng thái
    status_color = "#00ff88" if res['decision'] == "ĐÁNH" else "#ff4444"
    st.markdown(f"""
        <div class='status-bar' style='background: {status_color}; color: black;'>
            📢 KHUYÊN DÙNG: {res['decision']} | ĐỘ TIN CẬY: {res['conf']}%
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<p style='text-align:center; color:#888;'>🔥 3 SỐ CHỦ LỰC (LOẠI 7 CHỌN 3)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-display'>{res['main_3']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<p style='text-align:center; color:#888;'>🛡️ 4 SỐ LÓT AN TOÀN</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='support-display'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **PHÂN TÍCH AI:** {res['logic']}")
    
    # Dàn copy nhanh
    full_set = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (CLICK ĐỂ COPY):", full_set)
    
    st.markdown("</div>", unsafe_allow_html=True)

# BIỂU ĐỒ TẦN SUẤT
if st.session_state.history:
    with st.expander("📊 PHÂN TÍCH NHỊP RƠI 0-9 (50 KỲ GẦN NHẤT)"):
        all_d = "".join(st.session_state.history[-50:])
        df_stats = pd.Series(Counter(all_d)).sort_index()
        st.bar_chart(df_stats)
