import streamlit as st
import re
import json
import os
import pandas as pd
import numpy as np
import itertools
from collections import Counter
from datetime import datetime

# ================= CẤU HÌNH =================
DB_FILE = "titan_ultra_db.json"
MAX_HISTORY = 3000

st.set_page_config(page_title="TITAN v25 ULTRA", layout="wide")

# ================= LOAD & SAVE =================
def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except:
            return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-MAX_HISTORY:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= VALIDATION =================
def clean_input(raw_text):
    numbers = re.findall(r"\b\d{5}\b", raw_text)
    return numbers

def validate_number(num):
    if not num.isdigit():
        return False
    if len(num) != 5:
        return False
    return True

# ================= CORE ENGINE =================
all_triplets = list(itertools.combinations(range(10), 3))

def build_frequency(history):
    freq = np.zeros(10)
    for num in history:
        for d in num:
            freq[int(d)] += 1
    return freq

def build_co_matrix(history):
    matrix = np.zeros((10,10))
    for num in history:
        unique_digits = list(set(num))
        for d1 in unique_digits:
            for d2 in unique_digits:
                if d1 != d2:
                    matrix[int(d1)][int(d2)] += 1
    return matrix

def score_triplets(freq, matrix):
    scores = []
    total = np.sum(freq) + 1

    for trip in all_triplets:
        freq_score = sum(freq[d] for d in trip) / total

        co_score = (
            matrix[trip[0]][trip[1]] +
            matrix[trip[0]][trip[2]] +
            matrix[trip[1]][trip[2]]
        )

        final = (freq_score * 0.6) + (co_score * 0.4)

        scores.append((trip, final))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def build_final_7digits(scores):
    top_trip = scores[0][0]
    all_digits = "".join(st.session_state.history[-100:])
    top_freq = [x[0] for x in Counter(all_digits).most_common(10)]

    support = []
    for d in top_freq:
        if int(d) not in top_trip:
            support.append(d)
        if len(support) == 4:
            break

    return "".join(map(str, top_trip)), "".join(support)

# ================= UI =================
st.markdown("""
<style>
.stApp { background:#0d1117; color:#e6edf3; }
.main-title { text-align:center; font-size:42px; font-weight:900; color:#58a6ff; }
.card { background:#161b22; padding:20px; border-radius:10px; border:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🚀 TITAN v25 ULTRA ENGINE</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    raw_input = st.text_area("📥 Dán dữ liệu 5 số (mỗi dòng 1 kỳ)", height=150)

with col2:
    st.metric("📊 Tổng kỳ", len(st.session_state.history))
    btn_add = st.button("➕ Thêm Dữ Liệu")
    btn_reset = st.button("🗑 Reset")

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.success("Đã reset sạch.")
    st.rerun()

if btn_add:
    new_nums = clean_input(raw_input)
    valid_nums = []

    for n in new_nums:
        if validate_number(n):
            valid_nums.append(n)

    if not valid_nums:
        st.error("Không có dữ liệu hợp lệ (phải đúng 5 chữ số).")
    else:
        st.session_state.history.extend(valid_nums)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)
        st.success(f"Đã thêm {len(valid_nums)} kỳ.")
        st.rerun()

# ================= ANALYSIS =================
if len(st.session_state.history) >= 10:
    st.divider()
    st.subheader("🔥 KẾT QUẢ PHÂN TÍCH")

    freq = build_frequency(st.session_state.history)
    matrix = build_co_matrix(st.session_state.history)
    scores = score_triplets(freq, matrix)

    main3, support4 = build_final_7digits(scores)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### 🔥 3 SỐ CHỦ LỰC: `{main3}`")
    st.markdown(f"### 🛡 4 SỐ LÓT: `{support4}`")

    final7 = "".join(sorted(set(main3 + support4)))
    st.text_input("📋 DÀN 7 SỐ:", final7)

    confidence = round(min(scores[0][1] * 100, 99),2)
    st.progress(int(confidence))
    st.write(f"📈 Độ mạnh mô hình: {confidence}%")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📊 Thống kê 50 kỳ gần nhất"):
        last50 = "".join(st.session_state.history[-50:])
        st.bar_chart(pd.Series(Counter(last50)).sort_index())

else:
    st.warning("Cần ít nhất 10 kỳ để phân tích chính xác.")