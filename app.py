import streamlit as st
import pandas as pd
import numpy as np
import itertools
import json
import os
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="5D BET ULTRA PROMAX", layout="wide")

DATA_FILE = "data.json"

# =========================
# LOAD DATA
# =========================
def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

history = load_data()

# =========================
# VALIDATION ENGINE
# =========================
def validate_input(number):
    if not number.isdigit():
        return False, "❌ Chỉ được nhập số 0-9"
    if len(number) != 5:
        return False, "❌ Phải nhập đúng 5 chữ số"
    return True, ""

# =========================
# CORE ENGINE
# =========================

# Generate all 120 combinations
all_triplets = list(itertools.combinations(range(10), 3))

def calculate_frequency(history):
    freq = np.zeros(10)
    for entry in history:
        for digit in entry["number"]:
            freq[int(digit)] += 1
    return freq

def calculate_co_occurrence(history):
    matrix = np.zeros((10, 10))
    for entry in history:
        digits = list(set(entry["number"]))
        for d1 in digits:
            for d2 in digits:
                if d1 != d2:
                    matrix[int(d1)][int(d2)] += 1
    return matrix

def score_triplets(freq, matrix):
    scores = []

    total_freq = np.sum(freq) + 1

    for triplet in all_triplets:
        f_score = sum(freq[d] for d in triplet) / total_freq

        c_score = (
            matrix[triplet[0]][triplet[1]] +
            matrix[triplet[0]][triplet[2]] +
            matrix[triplet[1]][triplet[2]]
        )

        final_score = (f_score * 0.6) + (c_score * 0.4)

        scores.append({
            "triplet": triplet,
            "score": final_score
        })

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores

# =========================
# UI
# =========================

st.title("🔥 5D BET ULTRA PROMAX ENGINE")

st.subheader("📥 Nhập Kết Quả 5 Số")

col1, col2 = st.columns(2)

with col1:
    number_input = st.text_input("Nhập 5 số (VD: 12864)", max_chars=5)

with col2:
    if st.button("➕ Thêm Kỳ Mới"):
        valid, message = validate_input(number_input)

        if not valid:
            st.error(message)
        else:
            history.append({
                "number": number_input,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_data(history)
            st.success("✅ Đã lưu kỳ mới")

# =========================
# SHOW HISTORY
# =========================
st.subheader("📜 Lịch Sử")
if history:
    df_history = pd.DataFrame(history)
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("Chưa có dữ liệu.")

# =========================
# ANALYSIS
# =========================
if len(history) >= 5:
    st.subheader("🧠 Phân Tích Engine")

    freq = calculate_frequency(history)
    matrix = calculate_co_occurrence(history)
    scores = score_triplets(freq, matrix)

    top_n = 10

    result_df = pd.DataFrame([
        {
            "Top": i+1,
            "Bộ 3 số": "".join(map(str, scores[i]["triplet"])),
            "Điểm": round(scores[i]["score"], 4)
        }
        for i in range(top_n)
    ])

    st.dataframe(result_df, use_container_width=True)

else:
    st.warning("⚠ Cần tối thiểu 5 kỳ để phân tích.")

# =========================
# ERROR SAFETY
# =========================
try:
    pass
except Exception as e:
    st.error(f"Lỗi hệ thống: {str(e)}")