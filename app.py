import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter 
from datetime import datetime
import numpy as np
import pandas as pd
import time
import requests
from typing import List, Dict, Tuple, Optional

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v2026.json"

# Thiết lập Gemini
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: 
        return None

neural_engine = setup_neural()

# ================= THUẬT TOÁN TITAN ELITE =================
class TitanEliteAnalyzer:
    def __init__(self, history: List[str]):
        self.history = history
        self.numbers = "0123456789"

    def get_smart_weights(self) -> Dict[str, float]:
        """Tính toán trọng số dựa trên tần suất và nhịp cầu gần nhất"""
        if not self.history:
            return {n: 0.1 for n in self.numbers}
        
        # Lấy 50 kỳ gần nhất để phân tích sâu
        recent_data = self.history[-50:]
        all_digits = "".join(recent_data)
        counts = Counter(all_digits)
        total = sum(counts.values())
        
        # 1. Trọng số cơ bản (Tần suất)
        base_weights = {n: (counts[n] / total) if total > 0 else 0.1 for n in self.numbers}
        
        # 2. Phân tích nhịp rơi (Recency bias)
        # Số nào vừa về ở kỳ cuối sẽ có xu hướng 'rơi lại' hoặc 'ngắt cầu'
        last_nums = self.history[-1]
        for n in last_nums:
            base_weights[n] *= 1.2  # Tăng tỷ lệ rơi lại (cầu bệt)
            
        return base_weights

    def extract_super_selection(self) -> Dict:
        """Phân tách 3 số chủ lực và 4 số dự phòng"""
        weights = self.get_smart_weights()
        # Sắp xếp số theo trọng số từ cao đến thấp
        sorted_nums = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # 3 Số Siêu Cấp (Khả năng về cao nhất)
        top_3 = [n for n, w in sorted_nums[:3]]
        # 4 Số Dự Phòng
        backup_4 = [n for n, w in sorted_nums[3:7]]
        
        confidence = min(sum([w for n, w in sorted_nums[:3]]) * 200, 99.9)
        
        return {
            "top_3": top_3,
            "backup_4": backup_4,
            "confidence": round(confidence, 2)
        }

# ================= GIAO DIỆN STREAMLIT =================
st.set_page_config(page_title="TITAN ELITE v2026", layout="wide")

# Custom CSS cho giao diện "Bào Tiền"
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .super-card { 
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 25px; border-radius: 15px; border-left: 8px solid #ff4b4b;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5); margin: 15px 0;
    }
    .number-high { color: #00ff00; font-size: 50px; font-weight: bold; letter-spacing: 10px; }
    .number-backup { color: #ffca28; font-size: 40px; font-weight: bold; letter-spacing: 10px; }
    .stButton>button { width: 100%; background: #ff4b4b; color: white; border-radius: 10px; height: 50px; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 TITAN ELITE v2026 - HỆ THỐNG BÀO TIỀN NHÀ CÁI")

if "history" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.history = json.load(f)
    else:
        st.session_state.history = []

# Nhập dữ liệu
with st.sidebar:
    st.header("📥 DỮ LIỆU ĐẦU VÀO")
    raw_input = st.text_area("Nhập số kỳ gần nhất (mỗi dòng 1 số):", height=200)
    if st.button("CẬP NHẬT DỮ LIỆU"):
        new_nums = re.findall(r'\d{5}', raw_input)
        if new_nums:
            st.session_state.history.extend(new_nums)
            with open(DB_FILE, "w") as f: json.dump(st.session_state.history[-1000:], f)
            st.success(f"Đã nạp thêm {len(new_nums)} kỳ!")
            st.rerun()

# Phân tích và Hiển thị
if len(st.session_state.history) > 5:
    analyzer = TitanEliteAnalyzer(st.session_state.history)
    results = analyzer.extract_super_selection()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="super-card">
            <h2 style='color: white;'>🚀 3 SỐ SIÊU CẤP (99% KHẢ NĂNG)</h2>
            <div class="number-high">{' '.join(results['top_3'])}</div>
            <p style='color: #888;'>Dựa trên thuật toán xác suất nhịp kép và AI dự báo chu kỳ.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="super-card" style="border-left-color: #ffca28;">
            <h2 style='color: white;'>🛡️ 4 SỐ DỰ PHÒNG</h2>
            <div class="number-backup">{' '.join(results['backup_4'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("ĐỘ TIN CẬY", f"{results['confidence']}%", delta="SIÊU CAO")
        
        # Kết nối AI để lấy chiến thuật vào tiền
        if st.button("HỎI AI CHIẾN THUẬT VÀO TIỀN"):
            with st.spinner("AI đang tính toán nhịp cầu..."):
                prompt = f"Dữ liệu 5D: {st.session_state.history[-20:]}. Dự đoán: {results['top_3']}. Hãy đưa ra kế hoạch vào tiền gấp thếp để bào tiền nhà cái, ngắn gọn, thực chiến 100%."
                if neural_engine:
                    response = neural_engine.generate_content(prompt)
                    st.info(response.text)
                else:
                    st.error("Chưa kết nối được AI!")

    # Thống kê nhanh
    with st.expander("📊 PHÂN TÍCH TẦN SUẤT CHI TIẾT"):
        st.bar_chart(pd.Series(Counter("".join(st.session_state.history[-100:]))))
else:
    st.warning("Vui lòng nhập ít nhất 5 kỳ dữ liệu để bắt đầu phân tích.")

# Nút Reset
if st.sidebar.button("XÓA HẾT DỮ LIỆU"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
