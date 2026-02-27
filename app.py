import streamlit as st
import re
import json
import os
import pandas as pd
import numpy as np
import itertools
from collections import Counter
import hashlib

# ================= CONFIG =================
DB_FILE = "titan_ultra_db.json"
MAX_HISTORY = 3000

st.set_page_config(page_title="TITAN v25.1 ULTRA", layout="wide")

# ================= LOAD =================
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

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = None

if "history_hash" not in st.session_state:
    st.session_state.history_hash = None

# ================= CLEAN INPUT =================
def clean_input(raw):
    return re.findall(r"\b\d{5}\b", raw)

# ================= CORE ENGINE =================
all_triplets = list(itertools.combinations(range(10), 3))

@st.cache_data(show_spinner=False)
def run_analysis(history):

    freq = np.zeros(10)
    matrix = np.zeros((10,10))

    for num in history:
        unique = set(num)
        for d in num:
            freq[int(d)] += 1
        for d1 in unique:
            for d2 in unique:
                if d1 != d2:
                    matrix[int(d1)][int(d2)] += 1

    total = np.sum(freq) + 1
    scores = []

    for trip in all_triplets:
        f_score = sum(freq[d] for d in trip) / total
        c_score = (
            matrix[trip[0]][trip[1]] +
            matrix[trip[0]][trip[2]] +
            matrix[trip[1]][trip[2]]
        )
        final = (f_score * 0.6) + (c_score * 0.4)
        scores.append((trip, final))

    scores.sort(key=lambda x: x[1], reverse=True)

    main3 = scores[0][0]

    all_digits = "".join(history[-100:])
    top_freq = [x[0] for x in Counter(all_digits).most_common(10)]

    support = []
    for d in top_freq:
        if int(d) not in main3:
            support.append(d)
        if len(support) == 4:
            break

    final7 = "".join(sorted(set("".join(map(str, main3)) + "".join(support))))

    confidence = round(min(scores[0][1] * 100, 99), 2)

    return {
        "main3": "".join(map(str, main3)),
        "support4": "".join(support),
        "final7": final7,
        "confidence": confidence
    }

# ================= UI =================
st.title("🚀 TITAN v25.1 ULTRA ENGINE")

col1, col2 = st.columns([2,1])

with col1:
    raw_input = st.text_area("📥 Dán dữ liệu (mỗi dòng 1 kỳ)", height=150)

with col2:
    st.metric("📊 Tổng kỳ", len(st.session_state.history))
    btn_add = st.button("➕ Thêm")
    btn_reset = st.button("🗑 Reset")

if btn_reset:
    st.session_state.history = []
    save_db([])
    st.session_state.analysis_cache = None
    st.success("Đã reset.")
    st.rerun()

if btn_add:
    new_nums = clean_input(raw_input)
    if not new_nums:
        st.error("Không có dữ liệu hợp lệ.")
    else:
        st.session_state.history.extend(new_nums)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)
        st.success(f"Đã thêm {len(new_nums)} kỳ.")
        st.rerun()

# ================= SMART ANALYSIS =================
if len(st.session_state.history) >= 10:

    current_hash = hashlib.md5(
        "".join(st.session_state.history).encode()
    ).hexdigest()

    if current_hash != st.session_state.history_hash:
        st.session_state.analysis_cache = run_analysis(
            st.session_state.history
        )
        st.session_state.history_hash = current_hash

    res = st.session_state.analysis_cache

    st.divider()
    st.subheader("🔥 KẾT QUẢ PHÂN TÍCH")

    st.markdown(f"### 🔥 3 SỐ CHỦ LỰC: `{res['main3']}`")
    st.markdown(f"### 🛡 4 SỐ LÓT: `{res['support4']}`")
    st.text_input("📋 DÀN 7 SỐ:", res["final7"])

    st.progress(int(res["confidence"]))
    st.write(f"📈 Độ mạnh mô hình: {res['confidence']}%")

else:
    st.warning("Cần ít nhất 10 kỳ.")