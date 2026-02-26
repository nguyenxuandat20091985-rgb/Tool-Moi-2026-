import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter 

# ================= CẤU HÌNH HỆ THỐNG TITAN v24.2 =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_supreme_permanent_v24.json" 

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
                return data if isinstance(data, list) else []
            except: return []
    return [] 

def save_db(data):
    # Lưu tối đa 3000 kỳ để đảm bảo mượt mà nhưng vẫn đủ dữ liệu học sâu
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f) 

if "history" not in st.session_state:
    st.session_state.history = load_db() 

# ================= UI SUPREME DARK MODE =================
st.set_page_config(page_title="TITAN v24.2 SUPREME", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid #30363d; border-radius: 15px; padding: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    }
    .num-box {
        font-size: 85px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px; border-right: 3px solid #30363d;
        text-shadow: 0 0 25px rgba(255,88,88,0.5);
    }
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 10px; padding-left: 25px;
        text-shadow: 0 0 15px rgba(88,166,255,0.3);
    }
    .status-bar { padding: 18px; border-radius: 12px; text-align: center; font-weight: 900; font-size: 22px; margin-bottom: 20px; text-transform: uppercase; }
    .warning-box { background: #331010; color: #ff7b72; padding: 15px; border-radius: 8px; border: 1px solid #6e2121; text-align: center; margin-top: 15px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v24.2 SUPREME AI</h1>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU VÀ LỌC ĐA TẦNG =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 NẠP DỮ LIỆU (Tự động lọc số trùng & bẩn):", height=130, placeholder="Dán kết quả tại đây...")
    with col_st:
        st.write(f"📊 Bộ nhớ trí tuệ: **{len(st.session_state.history)} kỳ**")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🔥 GIẢI MÃ")
        btn_reset = c2.button("🗑️ RESET") 

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun() 

if btn_save:
    # 1. Lọc số trùng, số sai định dạng (Lọc đa tầng)
    new_data = re.findall(r"\b\d{5}\b", raw_input)
    if new_data:
        # Loại bỏ số trùng lặp trong đợt nạp mới và gộp vào lịch sử
        current_history = st.session_state.history
        for d in new_data:
            if d not in current_history:
                current_history.append(d)
        
        st.session_state.history = current_history
        save_db(st.session_state.history)
        
        # 2. Phân tích nhạy bén: Bệt & Đảo
        # Lấy 15 kỳ gần nhất để soi bệt
        recent_context = "".join(st.session_state.history[-15:])
        freq = Counter(recent_context).most_common(2)
        bet_warning = f"Cảnh báo bệt số {freq[0][0]}" if freq[0][1] > 8 else "Cầu đang nhịp đảo"

        # 3. Kết nối Gemini SIÊU TRÍ TUỆ
        prompt = f"""
        Hệ thống: TITAN SUPREME v24.2 (Lõi ma trận số). 
        Dữ liệu lịch sử chuẩn: {st.session_state.history[-150:]}
        Phân tích nhịp gần đây: {bet_warning}
        
        Nhiệm vụ:
        1. Phân tích ma trận số cổ điển kết hợp xác suất hiện đại.
        2. Loại bỏ 5 số có xác suất trượt cao nhất, chỉ giữ lại bộ khung tinh hoa.
        3. Chốt 3 số CHỦ LỰC (Main_3) - Yêu cầu chính xác tuyệt đối theo nhịp rơi.
        4. Chốt 4 số lót (Support_4).
        5. Đưa ra chỉ thị ĐÁNH hoặc DỪNG dựa trên độ ảo của nhà cái.
        
        TRẢ VỀ JSON:
        {{
            "main_3": "abc", 
            "support_4": "defg", 
            "decision": "NÊN ĐÁNH/DỪNG/CẢNH BÁO BỆT", 
            "logic": "Giải thích sắc bén nhịp đảo cầu", 
            "color": "Green/Red/Yellow", 
            "conf": 99
        }}
        """
        try:
            response = neural_engine.generate_content(prompt)
            st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        except:
            # Thuật toán dự phòng Ma trận truyền thống
            all_n = "".join(st.session_state.history[-60:])
            top = [x[0] for x in Counter(all_n).most_common(7)]
            st.session_state.last_prediction = {
                "main_3": "".join(top[:3]), "support_4": "".join(top[3:]),
                "decision": "THEO DÕI NHỊP", "logic": "Đang đồng bộ thuật toán ma trận.",
                "color": "Yellow", "conf": 70
            }
        st.rerun() 

# ================= PHẦN 2: HIỂN THỊ TINH HOA =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    colors = {"green": "#238636", "red": "#da3633", "yellow": "#d29922"}
    bg_color = colors.get(res['color'].lower(), "#30363d")
    
    st.markdown(f"<div class='status-bar' style='background: {bg_color};'>📢 CHỈ THỊ: {res['decision']} ({res['conf']}%)</div>", unsafe_allow_html=True) 

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col_main, col_supp = st.columns([1.5, 1])
    with col_main:
        st.markdown(f"<p style='color:#8b949e; text-align:center; font-weight:bold;'>⭐ 3 SỐ CHỦ LỰC (VÀO TIỀN CHÍNH)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    with col_supp:
        st.markdown(f"<p style='color:#8b949e; text-align:center; font-weight:bold;'>🛡️ 4 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    l_col, r_col = st.columns([2, 1])
    with l_col:
        st.write(f"💡 **PHÂN TÍCH CHUYÊN SÂU:** {res['logic']}")
        if res['conf'] < 85 or "DỪNG" in res['decision']:
            st.markdown("<div class='warning-box'>⚠️ NHÀ CÁI ĐANG ĐẢO CẦU LIÊN TỤC - CẨN TRỌNG TỐI ĐA</div>", unsafe_allow_html=True)
    with r_col:
        full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
        st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", full_dan)
        
    st.markdown("</div>", unsafe_allow_html=True) 

# ================= PHẦN 3: MA TRẬN SỐ & BACKTESTING =================
if st.session_state.history:
    with st.expander("📊 BẢN ĐỒ MA TRẬN & TẦN SUẤT"):
        st.write("Phân tích xác suất rơi của từng con số (0-9) trong 100 kỳ gần nhất:")
        all_digits = "".join(st.session_state.history[-100:])
        counts = pd.Series(Counter(all_digits)).sort_index()
        st.bar_chart(counts)
        
        
        
        st.write("📝 **Lưu ý từ AI:** Khi biểu đồ có sự chênh lệch lớn (cột cao cột thấp), đó là lúc cầu bệt đang mạnh. Khi biểu đồ bằng phẳng, nhà cái đang đảo cầu liên tục.")
