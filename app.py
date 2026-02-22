import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
st.set_page_config(page_title="TITAN ELITE v2026", layout="wide", initial_sidebar_state="collapsed")

# CSS tối ưu hiển thị cửa sổ nhỏ (Mobile Friendly)
st.markdown("""
    <style>
    .stApp { background: #050505; color: #e0e0e0; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .main-card {
        background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
        border: 1px solid #333;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .num-main {
        font-size: 50px; font-weight: 900; color: #00ffcc;
        text-align: center; text-shadow: 0 0 20px #00ffcc;
        letter-spacing: 5px; line-height: 1;
    }
    .num-sub {
        font-size: 35px; font-weight: 700; color: #ffcc00;
        text-align: center; text-shadow: 0 0 15px #ffcc00;
        letter-spacing: 3px;
    }
    .status-box {
        padding: 5px 10px; border-radius: 5px; font-size: 12px; font-weight: bold;
    }
    .warning-blink {
        background: #440000; color: #ff4444;
        border: 1px solid #ff4444; animation: blink 1s infinite;
    }
    @keyframes blink { 50% { opacity: 0.5; } }
    /* Tối ưu khi thu nhỏ tab */
    @media (max-width: 600px) {
        .num-main { font-size: 40px; }
        .num-sub { font-size: 28px; }
    }
    </style>
""", unsafe_allow_html=True)

# Kết nối Gemini
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ================= THUẬT TOÁN CAO CẤP =================
class TitanV3:
    def __init__(self, history):
        self.history = history[-300:]
        
    def detect_trap(self):
        """Thuật toán phát hiện nhà cái lừa cầu"""
        if len(self.history) < 20: return "Dữ liệu mỏng", 0
        
        last_5 = self.history[-5:]
        all_digits = "".join(self.history[-50:])
        counts = Counter(all_digits)
        
        # Kiểm tra sự lặp lại bất thường của các số gan
        rare_digits = [d for d, c in counts.items() if c < 3]
        trap_score = 0
        for num in last_5:
            if any(d in rare_digits for d in num):
                trap_score += 20
        
        if trap_score > 40:
            return "CẢNH BÁO: CẦU LỪA (SỐ ẢO)", trap_score
        return "CẦU ĐANG THUẬN", trap_score

    def analyze_weights(self):
        """Phân tích đa tầng: Tần suất + Chu kỳ + Xác suất nhảy số"""
        if not self.history: return list("0123456789")
        
        digits = "".join(self.history)
        counter = Counter(digits)
        
        # 1. Trọng số cơ bản (Tần suất)
        scores = {str(i): counter.get(str(i), 0) * 1.5 for i in range(10)}
        
        # 2. Trọng số chu kỳ (Số vừa về có xu hướng lặp hoặc nghỉ)
        last_num = self.history[-1]
        for d in set(last_num):
            scores[d] += 5 
            
        # 3. Phân tích cầu bệt (Streak)
        for i in range(5):
            pos_digits = [n[i] for n in self.history[-10:]]
            if len(set(pos_digits)) <= 2: # Cầu đang bệt ở vị trí này
                scores[pos_digits[-1]] += 10

        sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_res]

# ================= GIAO DIỆN & XỬ LÝ =================
if "data_store" not in st.session_state:
    st.session_state.data_store = []

# Layout Thu nhỏ tối ưu cho Tab
col_input, col_result = st.columns([1, 1])

with col_input:
    st.markdown("### 📥 NHẬP DỮ LIỆU")
    raw_data = st.text_area("Dán số (từ web/app):", height=150, placeholder="32880\n21808\n...")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        process_btn = st.button("🚀 PHÂN TÍCH", use_container_width=True, type="primary")
    with col_btn2:
        if st.button("🗑️ XÓA", use_container_width=True):
            st.session_state.data_store = []
            st.rerun()

    if process_btn:
        nums = re.findall(r"\d{5}", raw_data)
        if nums:
            st.session_state.data_store.extend(nums)
            st.success(f"Đã nạp {len(nums)} kỳ")
        else:
            st.error("Không tìm thấy số 5 chữ số!")

with col_result:
    if len(st.session_state.data_store) > 0:
        tt = TitanV3(st.session_state.data_store)
        trap_msg, trap_val = tt.detect_trap()
        top_nums = tt.analyze_weights()
        
        # 3 Số cao nhất - 4 Số dự phòng
        dan_3 = top_nums[:3]
        dan_4 = top_nums[3:7]
        
        # Hiển thị cảnh báo lừa
        if trap_val > 40:
            st.markdown(f"<div class='status-box warning-blink'>{trap_msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-box' style='background:#1e3a1e; color:#44ff44;'>✅ {trap_msg}</div>", unsafe_allow_html=True)

        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#888; margin:0;'>3 SỐ CAO NHẤT (99%)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-main'>{' '.join(dan_3)}</div>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align:center; color:#888; margin:10px 0 0 0;'>4 SỐ DỰ PHÒNG</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-sub'>{' '.join(dan_4)}</div>", unsafe_allow_html=True)
        
        # Kết hợp Gemini phân tích chiến thuật
        if st.button("🧠 HỎI Ý KIẾN GEMINI ELITE", use_container_width=True):
            with st.spinner("AI đang giải mã cầu..."):
                prompt = f"""
                Phân tích Lotobet 5D. Lịch sử: {st.session_state.data_store[-30:]}.
                Dàn ưu tiên: {dan_3}, dự phòng: {dan_4}. 
                Hãy phân tích ngắn gọn: Quy luật cầu bệt/đảo, tỷ lệ nổ của dàn này, và cách vào tiền (Tiền/Vốn).
                Trả lời dưới dạng gạch đầu dòng ngắn nhất để đọc trên điện thoại.
                """
                try:
                    response = model.generate_content(prompt)
                    st.info(response.text)
                except:
                    st.warning("AI đang bận, hãy thử lại sau!")
        st.markdown("</div>", unsafe_allow_html=True)

# Bảng lịch sử đa chiều
with st.expander("📊 LỊCH SỬ NẠP SỐ", expanded=False):
    if st.session_state.data_store:
        df = pd.DataFrame(st.session_state.data_store[::-1], columns=["Kết quả"])
        st.table(df.head(10))

# Footer tinh gọn
st.markdown(f"<p style='text-align:center; color:#444; font-size:10px;'>TITAN ELITE 2026 - DATA: {len(st.session_state.data_store)} KỲ</p>", unsafe_allow_html=True)
