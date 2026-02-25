import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH SIÊU TRÍ TUỆ v24.0 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_elite_memory.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        # Sử dụng model mạnh nhất để phân tích sâu
        return genai.GenerativeModel('gemini-1.5-pro') 
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG BẢO LƯU VĨNH VIỄN =================
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_data(data):
    # Giữ tối đa 3000 kỳ để AI học hỏi sâu chu kỳ dài
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_data()

# ================= THUẬT TOÁN NHẬN BIẾT CẦU (SMART-LOGIC) =================
def detect_bridge_type(data):
    if len(data) < 15: return "Dữ liệu mỏng", "Chưa rõ"
    
    last_5 = data[-5:]
    all_digits = "".join(last_5)
    counts = Counter(all_digits)
    
    # 1. Nhận biết cầu bệt (Streak)
    most_common = counts.most_common(1)[0]
    if most_common[1] >= 4: # Một số xuất hiện > 4 lần trong 5 kỳ
        return "⚠️ CẦU BỆT ĐANG CHẠY", "NÊN BÁM HOẶC DỪNG"
    
    # 2. Nhận biết cầu đảo (Zigzag)
    # So sánh 2 kỳ cuối xem có hoán vị số không
    if len(data) >= 2:
        s1, s2 = set(data[-1]), set(data[-2])
        if len(s1.intersection(s2)) >= 3:
            return "🔄 CẦU ĐẢO/NHẢY", "ĐÁNH NHẸ"
            
    return "✅ CẦU ỔN ĐỊNH", "NÊN ĐÁNH"

# ================= GIAO DIỆN ELITE DESIGN =================
st.set_page_config(page_title="TITAN v24.0 ELITE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #000000; color: #ffffff; }
    .status-panel { background: #111; padding: 20px; border-radius: 15px; border: 1px solid #222; margin-bottom: 20px; }
    .decision-box { padding: 25px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0; }
    .should-bet { background: #003300; border: 2px solid #00ff00; color: #00ff00; }
    .should-stop { background: #330000; border: 2px solid #ff0000; color: #ff0000; }
    .main-num { font-size: 110px; color: #00d4ff; font-weight: 900; text-shadow: 0 0 40px #00d4ff; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #00d4ff;'>🧬 TITAN v24.0 ELITE OMNI</h1>", unsafe_allow_html=True)

# Hiển thị trạng thái bộ nhớ
st.sidebar.title("🧠 TRÍ TUỆ NHÂN TẠO")
st.sidebar.info(f"Dữ liệu đã học: {len(st.session_state.history)} kỳ")
if st.sidebar.button("🗑️ XÓA DỮ LIỆU CŨ"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

# Khu vực nhập liệu (Zero-Lag)
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Hệ thống tự động lọc bẩn và lưu trữ):", height=100)

if st.button("🚀 KÍCH HOẠT SIÊU TRÍ TUỆ"):
    clean_data = re.findall(r"\d{5}", raw_input)
    if clean_data:
        # Chỉ thêm những kỳ chưa có trong lịch sử (Tránh trùng)
        for d in clean_data:
            if d not in st.session_state.history:
                st.session_state.history.append(d)
        save_data(st.session_state.history)
        
        # Phân tích trạng thái cầu
        bridge_status, advice = detect_bridge_type(st.session_state.history)
        
        # PROMPT TINH HOA v24.0
        prompt = f"""
        Hệ thống: TITAN v24.0 ELITE. Chuyên gia tối thượng 5D/Lotobet.
        Lịch sử: {st.session_state.history[-150:]}
        Trạng thái cầu hiện tại: {bridge_status}
        
        Nhiệm vụ:
        1. Sử dụng thuật toán bẻ cầu nhà cái dựa trên bóng số, nhịp Fibonacci và độ lệch ma trận.
        2. Nếu phát hiện CẦU BỆT, tuyệt đối không dự đoán số ngược cầu.
        3. Chọn ra 3 số VÀNG (Main_3) có độ tin cậy tuyệt đối.
        4. Quyết định: Đánh (Bet) hay Dừng (Wait) dựa trên rủi ro.

        TRẢ VỀ JSON:
        {{
            "decision": "BET" hoặc "STOP",
            "main_3": "ABC",
            "support_4": "DEFG",
            "reason": "Giải thích logic sâu sắc",
            "risk_level": "Low/High",
            "confidence": 99
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.elite_res = data
        except:
            st.error("AI đang học hỏi thêm, vui lòng bấm lại!")
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ SINH TỬ =================
if "elite_res" in st.session_state:
    res = st.session_state.elite_res
    
    # 1. Hiển thị Quyết định
    if res['decision'] == "BET" and res['confidence'] > 85:
        st.markdown(f"<div class='decision-box should-bet'>🔥 TRẠNG THÁI: NÊN ĐÁNH (Độ tin cậy: {res['confidence']}%)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='decision-box should-stop'>⚠️ TRẠNG THÁI: DỪNG LẠI - CẦU ĐANG BIẾN ĐỘNG ẢO</div>", unsafe_allow_html=True)

    # 2. Hiển thị Số dự đoán
    st.markdown("<div style='background: #111; padding: 30px; border-radius: 20px; border: 1px solid #333;'>", unsafe_allow_html=True)
    st.write(f"🧬 **LÝ DO TỪ AI:** {res['reason']}")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"<div class='main-num'>{res['main_3']}</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#00d4ff;'>🎯 3 SỐ CHỦ LỰC TỐI THƯỢNG</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h1 style='text-align:center; color:#888; margin-top:30px;'>{res['support_4']}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>🛡️ DÀN LÓT AN TOÀN</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Phân tích cầu chuyên sâu
    b_status, b_advice = detect_bridge_type(st.session_state.history)
    st.subheader("📊 PHÂN TÍCH NHỊP CẦU")
    c1, c2 = st.columns(2)
    c1.metric("Loại cầu", b_status)
    c2.metric("Lời khuyên nhịp", b_advice)

st.markdown("<br><p style='text-align:center; color:#444;'>TITAN v24.0 Elite - Trí tuệ nhân tạo độc quyền cho người chơi chuyên nghiệp</p>", unsafe_allow_html=True)
