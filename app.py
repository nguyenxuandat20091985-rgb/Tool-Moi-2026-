import streamlit as st
import re
import json
import pandas as pd
import google.generativeai as genai
import time
from collections import Counter
from pathlib import Path

# ================= CONFIG & API =================
st.set_page_config(page_title="TITAN v1500 DYNAMIC PRO", layout="wide")

# API KEY CỦA ANH
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Lỗi cấu hình API.")

DATA_FILE = "titan_dataset.json"
BACKUP_FILE = "titan_backup.json"

# ================= HỆ THỐNG LƯU TRỮ & LỌC TRÙNG =================
def load_data():
    for file_path in [DATA_FILE, BACKUP_FILE]:
        if Path(file_path).exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    return list(dict.fromkeys(data)) # Lọc trùng ngay khi load
            except: continue
    return []

def save_data(data):
    clean_data = list(dict.fromkeys(data)) # Lọc trùng trước khi lưu
    with open(DATA_FILE, "w") as f:
        json.dump(clean_data, f)
    with open(BACKUP_FILE, "w") as f:
        json.dump(clean_data, f)
    return clean_data

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

# ================= LÕI TÍNH ĐIỂM DYNAMIC (NHẢY SỐ NHANH) =================
def get_titan_score(digits_list):
    # Lấy toàn bộ lịch sử làm nền
    freq_total = Counter(digits_list)
    # Lấy 30 kỳ gần nhất để bắt nhịp hiện tại (Cực quan trọng)
    recent_30 = digits_list[-30:]
    freq_recent = Counter(recent_30)
    
    score = {str(i): 0 for i in range(10)}
    for i in score:
        # Trọng số tổng quát thấp (0.2) để không bị ì
        score[i] += freq_total.get(i, 0) * 0.2 
        # Trọng số gần đây cực cao (8.0) để bộ số nhảy theo tay nạp
        score[i] += freq_recent.get(i, 0) * 8.0
        # Thưởng điểm cho số vừa xuất hiện trong 5 kỳ cuối
        if i in digits_list[-5:]:
            score[i] += 20
            
    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score

def ask_gemini_smart(history, current_predict):
    recent = history[-20:] # Chỉ gửi 20 kỳ để AI phản hồi nhanh
    prompt = f"LotoBet: {recent}. TITAN predict: {current_predict}. Nhận xét nhịp cầu và cách đi vốn ngắn gọn."
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI đang bận soi nhịp mới, anh hãy bấm Chốt Số lần nữa."

# ================= GIAO DIỆN ĐIỆN THOẠI & WEB =================
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🚀 TITAN v1500 DYNAMIC</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 DỮ LIỆU KUBET")
    raw_input = st.text_area("Nhập các kỳ vừa quay:", height=250, placeholder="Dán kết quả vào đây...")
    btn_run = st.button("🔥 CHỐT SỐ & CẬP NHẬT", use_container_width=True)
    if st.button("Làm mới bộ nhớ"):
        st.session_state.dataset = []
        save_data([])
        st.rerun()

if btn_run and raw_input:
    # Trích xuất số
    new_nums = re.findall(r"\d{1,5}", raw_input)
    if new_nums:
        # Cập nhật và tự động lọc trùng
        st.session_state.dataset = save_data(st.session_state.dataset + new_nums)
        
        all_digits = list("".join(st.session_state.dataset))
        
        if len(all_digits) >= 10:
            ranked, scores = get_titan_score(all_digits)
            p1 = ranked[:3]
            
            # Hiển thị kết quả chính
            st.markdown(f"""
            <div style='background: #111; padding: 25px; border-radius: 20px; border: 3px solid #ff4b4b; text-align: center;'>
                <h2 style='color: white; margin:0;'>🎯 TAY TIẾP THEO</h2>
                <h1 style='color: yellow; font-size: 80px; margin: 10px 0;'>{" - ".join(p1)}</h1>
                <p style='color: #00ffcc; font-size: 20px;'>Dự phòng: {", ".join(ranked[3:6])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Biểu đồ Score
            st.subheader("📈 Biểu đồ sức mạnh số")
            st.bar_chart(pd.Series(scores))
            
            # AI Tư vấn
            st.markdown("---")
            st.subheader("🧠 CHUYÊN GIA AI PHÁN")
            with st.spinner('Đang đọc vị nhà cái...'):
                advice = ask_gemini_smart(st.session_state.dataset, p1)
                st.info(advice)
        else:
            st.warning("Hãy nhập thêm dữ liệu để bắt đầu soi.")
    else:
        st.error("Vui lòng nhập số hợp lệ.")

st.divider()
st.write(f"📊 **Hệ thống đã lưu & lọc trùng:** {len(st.session_state.dataset)} kỳ quay.")
