import streamlit as st
import pandas as pd
import numpy as np
import itertools
import json
import os
from datetime import datetime

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="5D BET ULTRA PROMAX", layout="wide")

DATA_FILE = "data.json"

# =============================
# DATA ENGINE
# =============================
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

# =============================
# VALIDATION ENGINE PRO
# =============================
def validate_input(number):
    if number is None:
        return False, "Không được để trống"

    number = number.strip()

    if len(number) != 5:
        return False, "Phải đúng 5 chữ số"

    if not number.isdigit():
        return False, "Chỉ được nhập số từ 0-9"

    return True, number

# =============================
# CORE ENGINE
# =============================
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
            "triplet": "".join(map(str, triplet)),
            "score": final_score
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores

# =============================
# UI MOBILE PRO
# =============================

st.title("🔥 5D BET ULTRA PROMAX ENGINE")

st.markdown("### ⚡ Nhập Kết Quả 5 Số (Tự động khóa ký tự sai)")

number_input = st.text_input(
    "Nhập 5 số",
    max_chars=5,
    placeholder="Ví dụ: 12864"
)

col1, col2, col3 = st.columns(3)

with col1:
    add_btn = st.button("➕ Thêm")

with col2:
    clear_btn = st.button("🗑 Xóa hết")

with col3:
    delete_last = st.button("↩ Xóa kỳ cuối")

# =============================
# ACTIONS
# =============================

if add_btn:
    valid, result = validate_input(number_input)

    if not valid:
        st.error(result)
    else:
        # kiểm tra trùng kỳ gần nhất
        if history and history[-1]["number"] == result:
            st.warning("Kỳ này đã nhập rồi.")
        else:
            history.append({
                "number": result,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            save_data(history)
            st.success("Đã lưu kỳ mới")
            st.rerun()

if clear_btn:
    history = []
    save_data(history)
    st.success("Đã xóa toàn bộ dữ liệu")
    st.rerun()

if delete_last and history:
    history.pop()
    save_data(history)
    st.success("Đã xóa kỳ cuối")
    st.rerun()

# =============================
# HISTORY DISPLAY
# =============================

st.markdown("## 📜 Lịch Sử")

if history:
    df_history = pd.DataFrame(history[::-1])
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("Chưa có dữ liệu")

# =============================
# ANALYSIS ENGINE
# =============================

if len(history) >= 5:

    st.markdown("## 🧠 Phân Tích Thông Minh")

    freq = calculate_frequency(history)
    matrix = calculate_co_occurrence(history)
    scores = score_triplets(freq, matrix)

    top_n = 12

    result_df = pd.DataFrame(scores[:top_n])
    result_df.index += 1

    st.dataframe(result_df, use_container_width=True)

    st.markdown("### 🔢 Tần Suất Digit")
    freq_df = pd.DataFrame({
        "Digit": range(10),
        "Frequency": freq.astype(int)
    })
    st.dataframe(freq_df, use_container_width=True)

else:
    st.warning("Cần ít nhất 5 kỳ để phân tích.")

# =============================
# FOOTER
# =============================

st.markdown("---")
st.caption("ULTRA ENGINE • Tốc độ cao • Lưu dữ liệu vĩnh viễn • 1 phút xử lý")