import streamlit as st
import re
import json
import pandas as pd
import google.generativeai as genai
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ================= CONFIG & API =================
st.set_page_config(page_title="TITAN v1500 FINAL", layout="wide")

# API KEY CỦA ANH
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Cấu hình API gặp vấn đề.")

DATA_FILE = "titan_dataset.json"

# ================= CORE ENGINE =================
def load_data():
    if Path(DATA_FILE).exists():
        try:
            with open(DATA_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data, f)

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

def get_titan_score(digits_list):
    freq = Counter(digits_list)
    recent = Counter(digits_list[-30:])
    score = {str(i): 0 for i in range(10)}
    for i in score:
        score[i] += freq.get(i, 0) * 1.0
        score[i] += recent.get(i, 0) * 1.5
        if recent.get(i, 0) == 0: score[i] += 8
    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score

def ask_gemini_smart(history, current_predict):
    prompt = f"LotoBet Data: {history[-15:]}. TITAN suggest: {current_predict}. Give advice."
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text
        except:
            time.sleep(1)
            continue
    return "AI đang bận, anh bấm Phân Tích lại nhé!"

# ================= GIAO DIỆN CHUẨN =================
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🛡️ TITAN v1500 FINAL</h1>", unsafe_allow_html=True)

# Thanh nhập liệu bên trái (Sidebar)
with st.sidebar:
    st.header("📥 NHẬP DỮ LIỆU")
    raw_input = st.text_area("Dán số vào đây:", height=200)
    btn_run = st.button("🔥 CHỐT SỐ AI", use_container_width=True)
    if st.button("Xóa dữ liệu"):
        st.session_state.dataset = []
        save_data([])
        st.rerun()

# Khu vực hiển thị kết quả
if btn_run and raw_input:
    new_nums = re.findall(r"\d{1,5}", raw_input)
    if new_nums:
        st.session_state.dataset += [n for n in new_nums if n not in st.session_state.dataset]
        save_data(st.session_state.dataset)
        
        all_digits = list("".join(st.session_state.dataset))
        
        if len(all_digits) >= 10:
            ranked, scores = get_titan_score(all_digits)
            p1 = ranked[:3]
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"""
                <div style='background: #1a1a1a; padding: 20px; border-radius: 15px; border: 2px solid red; text-align: center;'>
                    <h2 style='color: white;'>DỰ ĐOÁN</h2>
                    <h1 style='color: yellow; font-size: 60px;'>{" - ".join(p1)}</h1>
                </div>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.Series(scores))
            
            with c2:
                st.subheader("🧠 AI TƯ VẤN")
                # SỬA LỖI Ở DÒNG NÀY (Dùng dấu nháy đơn bao ngoài dấu nháy kép)
                with st.spinner('Đang phân tích...'):
                    advice = ask_gemini_smart(st.session_state.dataset, p1)
                    st.info(advice)
        else:
            st.warning("Nhập thêm kết quả để AI soi chuẩn hơn anh nhé!")

st.divider()
st.caption(f"Số kỳ đã lưu: {len(st.session_state.dataset)}")
