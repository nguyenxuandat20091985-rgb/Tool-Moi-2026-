import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v20.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG GHI NHỚ VĨNH VIỄN =================
def get_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return []

def update_memory(new_data):
    current = get_memory()
    current.extend(new_data)
    # Giữ lại 1000 kỳ gần nhất để AI không bị loạn
    with open(DB_FILE, "w") as f: json.dump(current[-1000:], f)
    return current[-1000:]

# ================= UI DESIGN (Tối giản - Chính xác) =================
st.set_page_config(page_title="TITAN v20.0 PRO", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-ok { color: #238636; font-weight: bold; font-size: 14px; }
    .prediction-box {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 20px; margin-top: 15px;
    }
    .num-highlight { 
        font-size: 55px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 5px; text-shadow: 0 0 20px #58a6ff;
    }
    .logic-text { font-size: 13px; color: #8b949e; font-style: italic; border-left: 3px solid #58a6ff; padding-left: 10px; }
    </style>
""", unsafe_allow_html=True)

# Hiển thị trạng thái
st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v20.0 OMNI</h2>", unsafe_allow_html=True)
if neural_engine:
    st.markdown("<p style='text-align: center;' class='status-ok'>● KẾT NỐI NEURAL-LINK THÀNH CÔNG</p>", unsafe_allow_html=True)
else:
    st.error("LỖI KẾT NỐI API - KIỂM TRA LẠI KEY")

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU THỰC CHIẾN (Copy dãy số 5D):", height=120)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 GIẢI MÃ THUẬT TOÁN"):
        valid_nums = re.findall(r"\d{5}", raw_input)
        if valid_nums:
            history = update_memory(valid_nums)
            
            # PROMPT ÉP AI SUY LUẬN ĐA TẦNG
            prompt = f"""
            Bạn là AI chuyên giải mã thuật toán 5D. 
            Dữ liệu lịch sử (1000 kỳ): {history[-100:]}.
            Yêu cầu phân tích:
            1. Tìm các số đang chạy theo cầu bệt (Streak).
            2. Tìm các số đang chạy theo nhịp đảo 1-1 hoặc 2-2.
            3. Tính toán 7 số có xác suất nổ cao nhất trong 3 kỳ tới.
            TRẢ VỀ JSON: {{"dan4": [4 số], "dan3": [3 số], "logic": "giải thích thuật toán nhà cái đang dùng"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown(f"<p class='logic-text'><b>Phân tích cầu:</b> {data['logic']}</p>", unsafe_allow_html=True)
                
                st.markdown("<p style='text-align:center; font-size:12px;'>🎯 DÀN 4 CHỦ LỰC (VÀO TIỀN MẠNH)</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='num-highlight'>{''.join(map(str, data['dan4']))}</div>", unsafe_allow_html=True)
                
                st.markdown("<p style='text-align:center; font-size:12px;'>🛡️ DÀN 3 LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='num-highlight' style='color:#f2cc60; text-shadow: 0 0 20px #f2cc60;'>{''.join(map(str, data['dan3']))}</div>", unsafe_allow_html=True)
                
                full_dan = "".join(map(str, data['dan4'])) + "".join(map(str, data['dan3']))
                st.text_input("📋 COPY DÀN 7 SỐ:", full_dan)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error("Hệ thống đang quá tải dữ liệu, anh bấm lại lần nữa nhé!")
        else:
            st.warning("Dán dữ liệu vào anh ơi!")

with col2:
    if st.button("🗑️ XÓA BỘ NHỚ TOOL"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

st.markdown("---")
st.markdown("<p style='text-align:center; font-size:10px; color:#444;'>Thiết kế riêng cho AIzaSyChq...RqM</p>", unsafe_allow_html=True)
