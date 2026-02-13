import streamlit as st
import google.generativeai as genai
import re
import json
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
# Key mới tinh anh vừa gửi
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"

def init_brain():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

brain = init_brain()

# ================= GIAO DIỆN DARK MODE LUXURY =================
st.set_page_config(page_title="TITAN v16.0", layout="centered")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle, #0a192f 0%, #02060c 100%); color: #e6f1ff; }
    .status-badge { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; border: 1px solid #00ffcc; color: #00ffcc; display: inline-block; margin-bottom: 10px; }
    .result-container { background: rgba(16, 33, 65, 0.7); border: 1px solid #1e3a8a; border-radius: 12px; padding: 15px; margin-top: 15px; }
    .num-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; }
    .num-card { background: #112240; border-bottom: 3px solid #64ffda; padding: 10px; border-radius: 8px; text-align: center; }
    .num-val { font-size: 32px; font-weight: 800; color: #64ffda; }
    .num-label { font-size: 10px; color: #8892b0; text-transform: uppercase; }
    .copy-box { background: #02060c; border: 1px dashed #64ffda; padding: 8px; color: #64ffda; text-align: center; font-family: monospace; border-radius: 5px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# Header nhỏ gọn
st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
if brain:
    st.markdown("<span class='status-badge'>● AI ACTIVE</span>", unsafe_allow_html=True)
else:
    st.markdown("<span class='status-badge' style='color:red; border-color:red;'>● AI ERROR</span>", unsafe_allow_html=True)

st.markdown("<h2 style='margin:0; color:#64ffda;'>🧠 TITAN v16.0</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size:12px; color:#8892b0;'>NEURAL ENGINE: SOI CẦU BỆT CHUYÊN SÂU</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
input_data = st.text_area("📡 Dán kỳ vừa về:", height=90, placeholder="Ví dụ: 78733\n66667...")

if st.button("🔥 PHÂN TÍCH NGAY"):
    history = re.findall(r"\d{5}", input_data)
    
    if len(history) < 3:
        st.warning("Dán thêm 3-5 kỳ đi anh!")
    else:
        with st.spinner('AI đang tính luồng bệt...'):
            # Prompt tối ưu nhất để tránh lỗi JSON
            prompt = f"""
            Lịch sử 5D: {history}. 
            Phân tích:
            1. Tìm các số đang bệt (lặp lại nhiều).
            2. Chốt dàn 7 số (4 chính, 3 lót).
            Trả về JSON duy nhất format: {{"main": [4 số], "sub": [3 số], "logic": "viết ngắn gọn 1 câu"}}
            """
            
            try:
                response = brain.generate_content(prompt)
                res_text = response.text
                data = json.loads(re.search(r'\{.*\}', res_text, re.DOTALL).group())
                
                # Hiển thị UI kết quả
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:13px;'><b>💡 Chiến thuật:</b> {data['logic']}</p>", unsafe_allow_html=True)
                
                # Dàn chính (4 số)
                st.markdown("<p style='font-size:12px; margin-bottom:5px;'>🎯 DÀN CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
                cols = st.columns(4)
                for i, n in enumerate(data['main']):
                    cols[i].markdown(f"<div class='num-card'><div class='num-label'>TOP {i+1}</div><div class='num-val'>{n}</div></div>", unsafe_allow_html=True)
                
                # Dàn lót (3 số)
                st.markdown("<p style='font-size:12px; margin-top:15px; margin-bottom:5px;'>🛡️ DÀN LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
                cols2 = st.columns(3)
                for i, n in enumerate(data['sub']):
                    cols2[i].markdown(f"<div class='num-card' style='border-color:#ffcc00;'><div class='num-label'>LÓT {i+1}</div><div class='num-val' style='color:#ffcc00;'>{n}</div></div>", unsafe_allow_html=True)
                
                # Copy nhanh
                full_7 = "".join(map(str, data['main'])) + "".join(map(str, data['sub']))
                st.markdown(f"<div class='copy-box'>DÀN 7 SỐ: {full_7}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                # Thuật toán dự phòng nếu Key bị lag
                all_nums = "".join(history)
                counts = Counter(all_nums)
                fallback = [n for n, c in counts.most_common(7)]
                st.error("AI ĐANG BẬN - DÙNG TẦN SUẤT THỰC TẾ")
                st.write(f"Dàn dự phòng: {' - '.join(fallback)}")

st.markdown("<br><p style='text-align:center; font-size:10px; color:#444;'>Hệ thống bảo mật bởi Neural Shield</p>", unsafe_allow_html=True)
