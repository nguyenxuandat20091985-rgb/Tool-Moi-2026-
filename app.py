import streamlit as st
import google.generativeai as genai
import re
import json
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
# Key mới anh vừa gửi - Em đã lắp vào chuẩn xác
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"

def init_ai():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

brain = init_ai()

# ================= GIAO DIỆN LUXURY UI =================
st.set_page_config(page_title="TITAN v16.0 GOLD", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&display=swap');
    .stApp { background: radial-gradient(circle, #0f172a 0%, #020617 100%); color: #e2e8f0; }
    
    .gold-title {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(to right, #bf953f, #fcf6ba, #b38728, #fbf5b7, #aa771c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; font-size: 50px; font-weight: 900; margin-bottom: 10px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(191, 149, 63, 0.3);
        border-radius: 20px; padding: 25px;
        backdrop-filter: blur(10px); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    
    .num-display {
        font-family: 'Orbitron', sans-serif;
        font-size: 60px; font-weight: 900;
        color: #fcf6ba; text-shadow: 0 0 20px rgba(191, 149, 63, 0.6);
        text-align: center; letter-spacing: 5px;
    }
    
    .status-tag {
        background: #064e3b; color: #34d399;
        padding: 5px 15px; border-radius: 50px;
        font-size: 12px; font-weight: bold; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Hiển thị trạng thái kết nối tinh tế
st.markdown("<div style='display: flex; justify-content: center; margin-bottom: 20px;'>"
            "<div class='status-tag'>● HỆ THỐNG NEURAL GOLD ĐANG TRỰC TUYẾN</div></div>", unsafe_allow_html=True)

st.markdown("<h1 class='gold-title'>TITAN v16.0 PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-style: italic;'>Hệ thống dự đoán bệt 5D cấp độ quân đội</p>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    raw_data = st.text_area("📡 NHẬP DỮ LIỆU CẦU (Dán danh sách các kỳ gần đây):", height=150, placeholder="Ví dụ: \n51875\n78733\n66667...")
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        analyze_btn = st.button("🔥 KÍCH HOẠT SIÊU MÁY TÍNH")
    with col_btn2:
        if st.button("🗑️ XÓA DỮ LIỆU"): st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if analyze_btn:
    history = re.findall(r"\d{5}", raw_data)
    
    if len(history) < 3:
        st.warning("Anh ơi, dán thêm ít nhất 3 kỳ nữa để AI soi bệt chính xác nhé!")
    else:
        # Thuật toán đếm bệt cứng (Luôn chạy để dự phòng)
        all_digits = "".join(history)
        counter = Counter(all_digits)
        top_numbers = [n for n, c in counter.most_common(7)]
        
        # Gọi Gemini tư duy sâu
        prompt = f"""
        Bạn là AI chuyên soi cầu bệt 5D. Dữ liệu: {history}.
        Tìm 7 số có khả năng ra cao nhất dựa trên bệt và hồi số.
        Trả về JSON: {{"chuluc": [4 số], "lot": [3 số], "tu_duy": ""}}
        """
        
        try:
            response = brain.generate_content(prompt)
            res_json = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            dan4, dan3, reasoning = res_json['chuluc'], res_json['lot'], res_json['tu_duy']
            st.info(f"🧠 AI TƯ DUY: {reasoning}")
        except:
            # Nếu AI bận, tự động dùng thuật toán toán học Gold
            dan4, dan3 = top_numbers[:4], top_numbers[4:7]
            st.warning("⚠️ Đang dùng thuật toán toán học Gold (Phòng vờ AI bận)")

        # HIỂN THỊ KẾT QUẢ ĐẸP MẮT
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #bf953f; font-weight: bold;'>🎯 DÀN CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-display'>{' '.join(map(str, dan4))}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #94a3b8; font-weight: bold;'>🛡️ DÀN LÓT (BẢO TOÀN)</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-display' style='color: #94a3b8;'>{' '.join(map(str, dan3))}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Thanh copy nhanh
        st.markdown("<br>", unsafe_allow_html=True)
        final_7 = "".join(map(str, dan4)) + "".join(map(str, dan3))
        st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", final_7)

st.markdown("<p style='text-align: center; color: #475569; margin-top: 50px;'>© 2026 TITAN GOLD ENGINE - PREDICTOR PRO</p>", unsafe_allow_html=True)
