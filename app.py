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
st.set_page_config(page_title="TITAN v1500-FIX HYBRID", layout="wide")

# API KEY MỚI CỦA ANH
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # Dùng bản Flash để tốc độ nhanh hơn cho LotoBet
except:
    st.error("Lỗi cấu hình API. Vui lòng kiểm tra lại Key.")

DATA_FILE = "titan_dataset.json"

# ================= CORE ENGINE =================
def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return json.load(f)
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

# ================= AI BRAIN WITH RETRY =================
def ask_gemini_smart(history, current_predict):
    prompt = f"""
    Hệ thống soi cầu LotoBet chuyên nghiệp.
    Dữ liệu 15 kỳ gần nhất: {history[-15:]}
    TITAN đề xuất: {current_predict}
    
    Yêu cầu:
    1. Phân tích nhịp cầu (Bệt/Nhảy).
    2. Tỉ lệ nổ của {current_predict} trong 2 kỳ tới?
    3. Lời khuyên đi vốn cực ngắn gọn.
    """
    for _ in range(3): # Thử lại tối đa 3 lần nếu lag
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            time.sleep(1)
            continue
    return "⚠️ AI đang quá tải do nhiều người dùng. Anh hãy bấm 'PHÂN TÍCH' lại lần nữa nhé!"

# ================= GIAO DIỆN =================
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🚀 TITAN v1500-FIX HYBRID</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 NHẬP DỮ LIỆU")
    raw_input = st.text_area("Dán kết quả KuBet (Dòng hoặc dãy số):", height=200, placeholder="Ví dụ: 12345\n67890...")
    btn_run = st.button("🔥 PHÂN TÍCH NGAY", use_container_width=True)
    if st.button("Xóa dữ liệu cũ"):
        st.session_state.dataset = []
        save_data([])
        st.success("Đã xóa!")

if btn_run and raw_input:
    # Lọc lấy các số từ chuỗi nhập vào
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
                    <h2 style='color: white;'>🎯 KẾT QUẢ TITAN</h2>
                    <h1 style='color: yellow; font-size: 60px;'>{" - ".join(p1)}</h1>
                    <p style='color: #aaa;'>Dự phòng: {", ".join(ranked[3:6])}</p>
                </div>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.Series(scores))
            
            with c2:
                st.markdown("<div style='background: #001a1a; padding: 20px; border-radius: 15px; border: 2px solid #00ffcc;'>", unsafe_allow_html=True)
                st.subheader("🧠 CHUYÊN GIA AI PHÁN")
                with st.spinner("Đang "soi" nhà cái..."):
                    advice = ask_gemini_smart(st.session_state.dataset, p1)
                    st.write(advice)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Anh nhập thêm ít nhất 5-10 kỳ nữa để AI làm việc nhé!")
    else:
        st.error("Không tìm thấy số hợp lệ. Anh copy đúng định dạng kết quả nhé.")

st.divider()
st.caption(f"Dữ liệu đang lưu trữ: {len(st.session_state.dataset)} kỳ quay.")
