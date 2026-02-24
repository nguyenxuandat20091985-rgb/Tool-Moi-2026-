import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH TITAN v23.0 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v23_neural_core.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-pro') # Nâng cấp lên bản Pro để tính toán mạnh hơn
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG PHÂN TÍCH VỊ TRÍ (NEW) =================
class TitanCoreV23:
    def __init__(self, history):
        self.history = history # List các chuỗi '12345'
        self.matrix = np.array([[int(d) for d in s] for s in history]) if history else np.array([])
        self.shadow_map = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}

    def get_positional_stats(self):
        """Phân tích tần suất và nhịp cho từng vị trí trong 5 hàng"""
        if self.matrix.size == 0: return {}
        stats = {}
        labels = ['H.Vạn', 'H.Ngàn', 'H.Trăm', 'H.Chục', 'H.Đơn']
        for i in range(5):
            col = self.matrix[:, i]
            common = Counter(col).most_common(3)
            stats[labels[i]] = common
        return stats

    def detect_ai_trap(self):
        """Nhận diện bẫy nhà cái: Nếu 1 số ra liên tục > 3 kỳ ở cùng 1 vị trí"""
        if len(self.matrix) < 5: return False
        for i in range(5):
            last_4 = self.matrix[-4:, i]
            if len(set(last_4)) == 1: return True # Bệt ảo - Cực kỳ nguy hiểm
        return False

# ================= GIAO DIỆN TITAN v23 =================
st.set_page_config(page_title="TITAN v23.0 - ANTI AI KUBET", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #050a0f; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #238636;
        border-radius: 12px; padding: 25px; box-shadow: 0 0 20px rgba(35, 134, 54, 0.2);
    }
    .main-number { 
        font-size: 100px; font-weight: 800; color: #3fb950; 
        text-align: center; text-shadow: 0 0 40px #238636;
        font-family: 'Courier New', monospace;
    }
    .trap-warning { 
        background: #440505; color: #ff7b72; padding: 10px; 
        border-radius: 5px; border: 1px solid #f85149; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ TITAN v23.0 OMNI: KHẮC CHẾ AI KUBET")

# Quản lý bộ nhớ
if "history" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.history = json.load(f)
    else: st.session_state.history = []

# Input
raw_data = st.text_area("📥 NẠP DỮ LIỆU KỲ MỚI:", height=100, placeholder="Dán dãy số kết quả...")

if st.button("🚀 PHÁ MÃ NHÀ CÁI"):
    new_entries = re.findall(r"\b\d{5}\b", raw_data)
    if new_entries:
        st.session_state.history.extend(new_entries)
        st.session_state.history = st.session_state.history[-2000:] # Lưu 2000 kỳ
        with open(DB_FILE, "w") as f: json.dump(st.session_state.history, f)
        
        core = TitanCoreV23(st.session_state.history)
        is_trap = core.detect_ai_trap()
        pos_stats = core.get_positional_stats()

        # Prompt Gemini chuyên sâu hơn về giải mã 5D
        prompt = f"""
        Bạn là hệ thống TITAN v23. Phân tích sảnh 5D KU. 
        Dữ liệu 50 kỳ gần nhất: {st.session_state.history[-50:]}
        Thống kê hàng vị trí: {pos_stats}
        Cảnh báo bẫy (Trap): {'CÓ' if is_trap else 'KHÔNG'}
        Nhiệm vụ:
        1. Tìm 3 số (0-9) có xác suất xuất hiện cao nhất trong 5 hàng (chế độ 3 số 5 tinh).
        2. Dùng quy luật bóng số để bù trừ sai lệch.
        3. Nếu bẫy Trap là CÓ, hãy giảm độ tin cậy xuống dưới 50%.
        TRẢ VỀ JSON: {{"main_3": "abc", "backup": "defg", "logic": "giải mã ngắn", "safety": 95}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.result = data
            st.session_state.is_trap = is_trap
        except:
            st.error("AI đang bị nghẽn, hãy thử lại sau vài giây.")
        st.rerun()

# Hiển thị
if "result" in st.session_state:
    res = st.session_state.result
    
    if st.session_state.is_trap:
        st.markdown("<div class='trap-warning'>⚠️ PHÁT HIỆN DẤU HIỆU ĐIỀU TIẾT CỦA NHÀ CÁI - CẨN THẬN BỊ BẺ CẦU ⚠️</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #8b949e;'>💡 CHIẾN THUẬT: {res['logic']}</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #3fb950;'>💎 3 SỐ CHỦ LỰC (DÀNH CHO 3 SỐ 5 TINH)</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='text-align: center;'>🛡️ Dãy lót bảo vệ: <b>{res['backup']}</b></p>", unsafe_allow_html=True)
    st.progress(res['safety'] / 100)
    st.markdown(f"<p style='text-align: right; font-size: 12px;'>Độ an toàn hệ thống: {res['safety']}%</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê chi tiết
with st.expander("📊 Xem bảng giải mã nhịp cầu 5 hàng"):
    if st.session_state.history:
        core = TitanCoreV23(st.session_state.history)
        st.write(core.get_positional_stats())
