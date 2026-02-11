import streamlit as st
import re
import json
import pandas as pd
import google.generativeai as genai
import time
from collections import Counter
from pathlib import Path

# ================= CONFIG & API =================
st.set_page_config(page_title="TITAN v1500 PRO BACKUP", layout="wide")

API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Lỗi cấu hình API.")

DATA_FILE = "titan_dataset.json"
BACKUP_FILE = "titan_backup.json"

# ================= HÀM LƯU TRỮ CÓ BACKUP & LỌC TRÙNG =================
def load_data():
    # Thử load từ file chính, nếu lỗi thử load từ file backup
    for file_path in [DATA_FILE, BACKUP_FILE]:
        if Path(file_path).exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # LỌC TRÙNG LẬP NGAY KHI LOAD
                    return list(dict.fromkeys(data)) 
            except:
                continue
    return []

def save_data_with_backup(data):
    # 1. Lọc trùng lặp trước khi lưu
    clean_data = list(dict.fromkeys(data))
    
    # 2. Lưu vào file chính
    with open(DATA_FILE, "w") as f:
        json.dump(clean_data, f)
    
    # 3. Ghi đè vào file backup dự phòng
    with open(BACKUP_FILE, "w") as f:
        json.dump(clean_data, f)
    return clean_data

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

# ================= ENGINE & AI =================
def get_titan_score(digits_list):
    freq = Counter(digits_list)
    recent = Counter(digits_list[-30:])
    score = {str(i): 0 for i in range(10)}
    for i in score:
        score[i] += freq.get(i, 0) * 1.0
        score[i] += recent.get(i, 0) * 1.5
        if recent.get(i, 0) == 0: score[i] += 8
    return sorted(score, key=score.get, reverse=True), score

def ask_gemini_smart(history, current_predict):
    # Chỉ gửi 20 kỳ gần nhất để tránh lag khi dataset lên đến 4000+
    recent = history[-20:]
    prompt = f"LotoBet: {recent}. Titan: {current_predict}. Advice?"
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI đang bận soi dữ liệu lớn, anh hãy bấm lại nút chốt số."

# ================= GIAO DIỆN =================
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🛡️ TITAN v1500 PRO BACKUP</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 HỆ THỐNG DỮ LIỆU")
    raw_input = st.text_area("Dán kết quả mới:", height=200)
    btn_run = st.button("🔥 CHỐT SỐ & SAO LƯU", use_container_width=True)
    
    if st.button("Xóa sạch dữ liệu"):
        st.session_state.dataset = []
        save_data_with_backup([])
        st.rerun()

if btn_run and raw_input:
    new_nums = re.findall(r"\d{1,5}", raw_input)
    if new_nums:
        # Hợp nhất và tự động lọc trùng qua hàm save_data_with_backup
        combined_data = st.session_state.dataset + new_nums
        st.session_state.dataset = save_data_with_backup(combined_data)
        
        all_digits = list("".join(st.session_state.dataset))
        
        if len(all_digits) >= 10:
            ranked, scores = get_titan_score(all_digits)
            p1 = ranked[:3]
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"""
                <div style='background: #111; padding: 20px; border-radius: 15px; border: 2px solid green; text-align: center;'>
                    <h2 style='color: white;'>DỰ ĐOÁN</h2>
                    <h1 style='color: #00ffcc; font-size: 60px;'>{" - ".join(p1)}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.Series(scores))
            
            with c2:
                st.subheader("🧠 TƯ VẤN TỪ AI")
                with st.spinner('Đang kết nối Gemini...'):
                    advice = ask_gemini_smart(st.session_state.dataset, p1)
                    st.success(advice)
        else:
            st.warning("Cần thêm dữ liệu.")

st.divider()
st.info(f"✅ Đã bảo mật & lọc trùng: {len(st.session_state.dataset)} kỳ quay. (File backup đã sẵn sàng)")
