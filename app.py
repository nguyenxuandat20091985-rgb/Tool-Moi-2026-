import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from itertools import combinations
from datetime import datetime

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="5D BET PRO MAX ULTRA", layout="wide")

DATA_FILE = "history_5d.json"

# ==============================
# LOAD / SAVE DATA
# ==============================
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# ==============================
# CORE ANALYSIS ENGINE
# ==============================

def digit_frequency(history, window=100):
    recent = history[-window:]
    freq = np.zeros(10)

    for item in recent:
        for d in item["digits"]:
            freq[d] += 1

    if len(recent) > 0:
        freq = freq / (len(recent) * 5)

    return freq


def co_occurrence_matrix(history, window=100):
    recent = history[-window:]
    matrix = np.zeros((10, 10))

    for item in recent:
        digits = item["digits"]
        unique = list(set(digits))
        for i in unique:
            for j in unique:
                if i != j:
                    matrix[i][j] += 1

    if len(recent) > 0:
        matrix = matrix / len(recent)

    return matrix


def score_combinations(freq, co_matrix):
    combo_scores = []

    for combo in combinations(range(10), 3):
        f_score = freq[combo[0]] + freq[combo[1]] + freq[combo[2]]

        c_score = (
            co_matrix[combo[0]][combo[1]]
            + co_matrix[combo[0]][combo[2]]
            + co_matrix[combo[1]][combo[2]]
        )

        total_score = f_score * 0.6 + c_score * 0.4

        combo_scores.append((combo, total_score))

    combo_scores.sort(key=lambda x: x[1], reverse=True)

    return combo_scores


# ==============================
# UI
# ==============================

st.title("🔥 5D BET PRO MAX ULTRA ENGINE")

history = load_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("➕ Nhập Kỳ Mới (5 số)")
    result_input = st.text_input("Nhập 5 số (vd: 12864)")

    if st.button("Lưu Kỳ"):
        if result_input.isdigit() and len(result_input) == 5:
            digits = [int(d) for d in result_input]
            history.append({
                "timestamp": str(datetime.now()),
                "digits": digits
            })
            save_data(history)
            st.success("Đã lưu thành công.")
        else:
            st.error("Phải nhập đúng 5 chữ số.")

with col2:
    st.subheader("📊 Thống Kê Hiện Tại")
    st.write(f"Tổng số kỳ đã lưu: {len(history)}")

# ==============================
# ANALYSIS SECTION
# ==============================

if len(history) >= 10:

    freq = digit_frequency(history, window=100)
    co_matrix = co_occurrence_matrix(history, window=100)
    ranked_combos = score_combinations(freq, co_matrix)

    st.markdown("---")
    st.subheader("🏆 TOP 10 Bộ 3 Số Đề Xuất")

    top10 = ranked_combos[:10]

    df_top = pd.DataFrame(
        [{
            "Bộ 3 Số": combo,
            "Điểm": round(score, 5)
        } for combo, score in top10]
    )

    st.dataframe(df_top, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Tần Suất 0-9 (100 kỳ gần nhất)")

    df_freq = pd.DataFrame({
        "Digit": range(10),
        "Frequency": np.round(freq, 4)
    })

    st.bar_chart(df_freq.set_index("Digit"))

else:
    st.warning("Cần tối thiểu 10 kỳ để phân tích.")

# ==============================
# RESET
# ==============================

st.markdown("---")
if st.button("⚠️ Reset toàn bộ dữ liệu"):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    st.success("Đã reset. Refresh lại trang.")