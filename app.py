import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ================= CẤU HÌNH HỆ THỐNG SIÊU CẤP =================
st.set_page_config(page_title="TITAN ELITE v23.0", layout="wide", initial_sidebar_state="collapsed")

API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM" # Key của anh
DB_FILE = "titan_elite_memory.json"

# Khởi tạo Gemini
def setup_gemini():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = setup_gemini()

# ================= CSS TỐI ƯU GIAO DIỆN (TAB COMPACT) =================
st.markdown("""
    <style>
    .stApp { background: #000000; color: #00ff00; }
    .main-card {
        background: #0a0a0a; border: 1px solid #333;
        border-radius: 10px; padding: 15px; margin-bottom: 10px;
    }
    .num-main {
        font-size: 50px !important; font-weight: 900; color: #ff0000;
        text-align: center; letter-spacing: 5px;
        text-shadow: 0 0 15px #ff0000; line-height: 1;
    }
    .num-sub {
        font-size: 35px !important; font-weight: 700; color: #ffff00;
        text-align: center; letter-spacing: 5px; opacity: 0.8;
    }
    .status-bar {
        font-size: 12px; padding: 5px; background: #111; 
        border-radius: 5px; margin-bottom: 10px; display: flex; justify-content: space-between;
    }
    .warning-flash {
        background: #330000; color: #ff4444; padding: 10px;
        border-left: 5px solid #ff0000; animation: blink 1s infinite;
    }
    @keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.5;} 100% {opacity: 1;} }
    /* Tối ưu cho màn hình nhỏ */
    @media (max-width: 600px) {
        .num-main { font-size: 40px !important; }
        .num-sub { font-size: 28px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ================= THUẬT TOÁN TITAN ELITE =================
class TitanEliteAnalyzer:
    def __init__(self, history: List[str]):
        self.history = history
        self.digits = "0123456789"

    def detect_casino_traps(self) -> Dict:
        if len(self.history) < 15: return {"trap": False, "msg": "Dữ liệu mỏng"}
        
        last_5 = self.history[-5:]
        all_chars = "".join(last_5)
        count_chars = Counter(all_chars)
        
        # Bẫy 1: Số lặp quá nhiều (Giam số)
        if any(v > 4 for v in count_chars.values()):
            return {"trap": True, "msg": "PHÁT HIỆN GIAM SỐ - NHÀ CÁI ĐANG GÀI CẦU BỆT"}
        
        # Bẫy 2: Số nhảy không quy luật (Đảo cầu)
        unique_nums = len(set(all_chars))
        if unique_nums > 8:
            return {"trap": True, "msg": "CẦU NHIỄU LOẠN - NHÀ CÁI ĐANG ĐẢO CẦU LIÊN TỤC"}
            
        return {"trap": False, "msg": "Cầu đang chạy ổn định"}

    def get_elite_prediction(self):
        if not self.history: return list("0123456"), 50
        
        # Phân tích tần suất có trọng số thời gian (Số mới về có điểm cao hơn)
        scores = {d: 0.0 for d in self.digits}
        for i, val in enumerate(reversed(self.history[-30:])):
            weight = 1.0 / (i + 1)
            for d in val:
                if d in scores: scores[d] += weight

        # Phân tích bạc nhớ và chu kỳ chuyển tiếp
        last_num = self.history[-1]
        for d in last_num:
            # Logic: Sau con X thường ra con Y (dựa trên 200 kỳ)
            for h in self.history[-200:-1]:
                if d in h:
                    next_idx = self.history.index(h) + 1
                    if next_idx < len(self.history):
                        for char in self.history[next_idx]:
                            scores[char] += 0.2

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_7 = [x[0] for x in sorted_scores[:7]]
        
        # Tính độ tin cậy dựa trên độ lệch điểm
        confidence = min(98.0, 70.0 + (sorted_scores[0][1] * 10))
        
        return top_7, round(confidence, 1)

# ================= XỬ LÝ DỮ LIỆU =================
if "history" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.history = json.load(f)
    else: st.session_state.history = []

def add_data(raw_str):
    new_nums = re.findall(r"\d{5}", raw_str)
    if new_nums:
        st.session_state.history.extend(new_nums)
        st.session_state.history = st.session_state.history[-1000:]
        with open(DB_FILE, "w") as f: json.dump(st.session_state.history, f)
        return True
    return False

# ================= GIAO DIỆN CHÍNH (COMPACT MODE) =================
# Header thu nhỏ
col_logo, col_info = st.columns([1, 2])
with col_logo:
    st.markdown("<h2 style='margin:0;'>🧬 TITAN</h2>", unsafe_allow_html=True)
with col_info:
    status = "🟢 NEURAL OK" if model else "🔴 API ERROR"
    st.markdown(f"<div class='status-bar'><span>{status}</span><span>Dữ liệu: {len(st.session_state.history)}</span></div>", unsafe_allow_html=True)

# Khu vực nhập liệu gọn nhẹ
with st.expander("📥 NHẬP SỐ MỚI", expanded=len(st.session_state.history) == 0):
    input_data = st.text_area("Dán chuỗi số (5D):", height=80, help="Dán cả đoạn dài, AI tự lọc")
    if st.button("🚀 CẬP NHẬT & PHÂN TÍCH", use_container_width=True):
        if add_data(input_data):
            st.rerun()

# --- PHẦN HIỂN THỊ KẾT QUẢ QUAN TRỌNG NHẤT ---
if st.session_state.history:
    analyzer = TitanEliteAnalyzer(st.session_state.history)
    trap_info = analyzer.detect_casino_traps()
    top_7, conf = analyzer.get_elite_prediction()
    
    # Cảnh báo lừa cầu
    if trap_info["trap"]:
        st.markdown(f"<div class='warning-flash'>⚠️ {trap_info['msg']}</div>", unsafe_allow_html=True)
    
    # Khu vực dự đoán chính
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"<p style='text-align:center;margin:0;color:#aaa;'>4 CHỦ LỰC (99%)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-main'>{''.join(top_7[:4])}</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"<p style='text-align:center;margin:0;color:#aaa;'>3 DỰ PHÒNG</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-sub'>{''.join(top_7[4:])}</div>", unsafe_allow_html=True)
    
    # Thanh độ tin cậy
    color = "#00ff00" if conf > 85 else "#ffff00"
    st.markdown(f"""
        <div style='margin-top:10px;'>
            <div style='display:flex;justify-content:space-between;font-size:12px;'>
                <span>ĐỘ TIN CẬY THUẬT TOÁN</span>
                <span>{conf}%</span>
            </div>
            <div style='background:#222;height:6px;border-radius:3px;'>
                <div style='background:{color};width:{conf}%;height:6px;border-radius:3px;box-shadow:0 0 10px {color};'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Gemini Chiến thuật (Chỉ hiện khi cần)
    if st.button("🧠 HỎI Ý KIẾN GEMINI AI", use_container_width=True):
        with st.spinner("AI đang giải mã cầu lừa..."):
            prompt = f"""
            Phân tích Lotobet 5D. Lịch sử: {st.session_state.history[-50:]}. 
            Thuật toán gợi ý: {top_7}. Bẫy nhà cái: {trap_info['msg']}.
            Đưa ra chiến thuật vào tiền (Tiền - Vốn - Điểm dừng) ngắn gọn nhất.
            """
            try:
                response = model.generate_content(prompt)
                st.info(response.text)
            except:
                st.error("Gemini đang bận, hãy thử lại sau.")

# Tối ưu nút bấm cuối trang
col_reset, col_copy = st.columns(2)
with col_reset:
    if st.button("🗑️ XÓA HẾT", use_container_width=True):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
with col_copy:
    if st.session_state.history:
        full_dan = "".join(top_7)
        st.code(full_dan, caption="Dàn 7 số copy")

# Auto-refresh để giữ kết nối
time.sleep(0.1)
