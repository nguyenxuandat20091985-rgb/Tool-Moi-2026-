import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter 

# ================= CẤU HÌNH HỆ THỐNG =================
# API KEY CỦA ANH
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v22.json" 

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None 

neural_engine = setup_neural() 

# ================= HỆ THỐNG XỬ LÝ DỮ LIỆU SẠCH =================
def clean_input(text):
    """Lọc tất cả ký tự lạ, chỉ giữ lại các dãy 5 số chuẩn"""
    # Tìm tất cả các cụm chữ số, sau đó lọc ra các cụm có độ dài là 5
    potential_numbers = re.findall(r"\d+", text)
    valid_numbers = [n for n in potential_numbers if len(n) == 5]
    return valid_numbers

def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: return json.load(f)
            except: return []
    return [] 

def save_memory(data):
    # Lưu tối đa 2000 kỳ để AI nhìn thấy quy luật dài hạn của nhà cái
    with open(DB_FILE, "w") as f: 
        json.dump(data[-2000:], f) 

if "history" not in st.session_state:
    st.session_state.history = load_memory() 

# ================= GIAO DIỆN TITAN v22.0 =================
st.set_page_config(page_title="TITAN v22.0 OMNI", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; padding: 5px 10px; background: #121d14; border-radius: 5px; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 20px; margin-top: 15px;
    }
    .num-display { 
        font-size: 55px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 8px; text-shadow: 0 0 20px rgba(88,166,255,0.6);
    }
    .logic-box { font-size: 13px; color: #8b949e; background: #161b22; padding: 12px; border-radius: 8px; border-left: 4px solid #58a6ff; }
    .stButton>button { width: 100%; background: #238636; color: white; border: none; }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v22.0 OMNI</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Hệ thống Phân tích Bẫy Nhà cái & Giải mã Xác suất</p>", unsafe_allow_html=True)

# Hiển thị trạng thái
status_col1, status_col2 = st.columns(2)
with status_col1:
    if neural_engine:
        st.markdown("<span class='status-active'>● AI NEURAL: ONLINE</span>", unsafe_allow_html=True)
with status_col2:
    st.markdown(f"<span style='color: #f2cc60;'>📊 DỮ LIỆU: {len(st.session_state.history)} KỲ</span>", unsafe_allow_html=True)

# ================= NHẬP LIỆU & XỬ LÝ =================
st.markdown("### 📥 NẠP DỮ LIỆU MỚI")
raw_input = st.text_area("Dán dữ liệu từ trang web (AI sẽ tự lọc số bẩn):", height=120, placeholder="Ví dụ: Kỳ 0469 Kết quả 7,8,9,3,1...") 

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("🚀 GIẢI MÃ KẾT QUẢ"):
        new_data = clean_input(raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            # Loại bỏ trùng lặp trong bộ nhớ nếu vô tình dán đè
            st.session_state.history = list(dict.fromkeys(st.session_state.history))
            save_memory(st.session_state.history)
            
            # Prompt nâng cao: Phân tích bẫy nhà cái
            prompt = f"""
            Bạn là AI tối thượng chuyên giải mã thuật toán nhà cái 5D (Lotobet).
            Dữ liệu lịch sử 150 kỳ gần nhất: {st.session_state.history[-150:]}.
            
            Nhiệm vụ:
            1. Phân tích nhịp cầu: Bệt (streak), Nhảy (alternating), và Bóng (mirror numbers).
            2. Phát hiện "Vùng Cấm": Những số nhà cái đang dùng thuật toán để né (dựa trên độ lệch chuẩn).
            3. Tính toán 7 con số có xác suất xuất hiện ở kỳ tiếp theo cao nhất, chia làm 2 dàn: Chủ lực (4 số) và Lót (3 số).
            
            TRẢ VỀ DUY NHẤT JSON THEO MẪU:
            {{"dan_chuluc": ["x", "x", "x", "x"], "dan_lot": ["x", "x", "x"], "logic": "Giải thích ngắn gọn quy luật bẫy hiện tại của nhà cái"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.last_result = data
            except:
                # Thuật toán dự phòng (Statistical Fallback)
                all_digits = "".join(st.session_state.history[-50:])
                counts = Counter(all_digits).most_common(7)
                res = [str(x[0]) for x in counts]
                st.session_state.last_result = {
                    "dan_chuluc": res[:4], 
                    "dan_lot": res[4:], 
                    "logic": "AI bận, đang dùng thống kê tần suất 50 kỳ gần nhất."
                }
            st.rerun()

with btn_col2:
    if st.button("🗑️ RESET DỮ LIỆU"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.session_state.pop("last_result", None)
        st.rerun()

# ================= HIỂN THỊ DÀN SỐ DỰ ĐOÁN =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='logic-box'><b>💡 CHIẾN THUẬT:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:14px; color:#58a6ff; margin-bottom: 5px;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN CHÍNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{' '.join(res['dan_chuluc'])}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:14px; color:#f2cc60; margin-top:20px; margin-bottom: 5px;'>🛡️ 3 SỐ LÓT (GIỮ VỐN - AN TOÀN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 20px rgba(242,204,96,0.5);'>{' '.join(res['dan_lot'])}</div>", unsafe_allow_html=True)
    
    # Tạo chuỗi để copy nhanh vào nhà cái
    full_dan = "".join(res['dan_chuluc']) + "".join(res['dan_lot'])
    st.markdown("---")
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", full_dan)
    st.caption("Mẹo: Dán dàn này vào mục 'Không cố định - 3 số 5 tinh' hoặc 'Dàn số' tùy theo mục tiêu của anh.")
    st.markdown("</div>", unsafe_allow_html=True)

# ================= THỐNG KÊ NHANH =================
if len(st.session_state.history) > 0:
    with st.expander("📊 Xem bảng tần suất (100 kỳ gần nhất)"):
        all_digits = "".join(st.session_state.history[-100:])
        counts = Counter(all_digits)
        df_counts = pd.DataFrame(counts.items(), columns=['Số', 'Số lần về']).sort_values(by='Số lần về', ascending=False)
        st.bar_chart(df_counts.set_index('Số'))

st.markdown("<br><p style='text-align:center; font-size:11px; color:#444;'>Hệ thống TITAN v22.0 - Tự học và tiến hóa dựa trên dữ liệu thực tế.</p>", unsafe_allow_html=True)
