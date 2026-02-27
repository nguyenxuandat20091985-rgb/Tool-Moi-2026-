import streamlit as st
import pandas as pd
import numpy as np
import itertools
import math
import random
from collections import Counter

st.set_page_config(page_title="5D BET PROMAX ENGINE", layout="wide")

# ==============================
# CONFIG
# ==============================
WINDOW_SHORT = 30
WINDOW_MED = 60
WINDOW_LONG = 120
MONTE_CARLO_RUNS = 20000

# ==============================
# UTIL FUNCTIONS
# ==============================

def parse_history(text):
    lines = text.strip().split("\n")
    data = []
    for line in lines:
        line = line.strip()
        if len(line) == 5 and line.isdigit():
            data.append(line)
    return data


def digit_frequency(history, window):
    freq = Counter()
    for row in history[-window:]:
        for d in row:
            freq[d] += 1
    total = sum(freq.values())
    return {str(i): freq[str(i)] / total if total > 0 else 0 for i in range(10)}


def co_occurrence_matrix(history):
    matrix = np.zeros((10, 10))
    for row in history:
        digits = set(row)
        for a in digits:
            for b in digits:
                if a != b:
                    matrix[int(a)][int(b)] += 1
    return matrix


def entropy_score(freq_dict):
    entropy = 0
    for v in freq_dict.values():
        if v > 0:
            entropy -= v * math.log(v)
    return entropy


def monte_carlo_probability(combo, freq_dist):
    hits = 0
    digits = list(freq_dist.keys())
    probs = list(freq_dist.values())
    for _ in range(MONTE_CARLO_RUNS):
        sample = np.random.choice(digits, 5, p=probs)
        if all(d in sample for d in combo):
            hits += 1
    return hits / MONTE_CARLO_RUNS


def calculate_ev(prob, payout, bet):
    return (prob * payout) - ((1 - prob) * bet)


def kelly_criterion(prob, payout):
    b = payout - 1
    q = 1 - prob
    return max((b * prob - q) / b, 0)


# ==============================
# STREAMLIT UI
# ==============================

st.title("🔥 5D BET 3 SỐ 5 TINH – PROMAX ENGINE")

st.markdown("Nhập lịch sử 5 số (mỗi dòng 1 kết quả, ví dụ: 12864)")

history_input = st.text_area("LỊCH SỬ 5D", height=300)

payout = st.number_input("TỶ LỆ TRẢ THƯỞNG (VD: 50 nếu ăn 1:50)", value=50.0)
bet_amount = st.number_input("TIỀN MỖI LẦN ĐÁNH", value=100.0)

if st.button("🚀 PHÂN TÍCH PROMAX"):

    history = parse_history(history_input)

    if len(history) < 20:
        st.warning("Cần tối thiểu 20 kỳ để phân tích.")
        st.stop()

    # Frequency multi window
    freq_short = digit_frequency(history, min(WINDOW_SHORT, len(history)))
    freq_med = digit_frequency(history, min(WINDOW_MED, len(history)))
    freq_long = digit_frequency(history, min(WINDOW_LONG, len(history)))

    # Weighted frequency
    freq_final = {}
    for i in range(10):
        d = str(i)
        freq_final[d] = (
            freq_short[d] * 0.5 +
            freq_med[d] * 0.3 +
            freq_long[d] * 0.2
        )

    # Normalize
    total = sum(freq_final.values())
    for k in freq_final:
        freq_final[k] /= total

    # Co-occurrence
    matrix = co_occurrence_matrix(history)

    # Generate all 120 combinations
    combos = list(itertools.combinations([str(i) for i in range(10)], 3))

    results = []

    for combo in combos:

        # Frequency score
        freq_score = sum(freq_final[d] for d in combo)

        # Co-occurrence score
        co_score = (
            matrix[int(combo[0])][int(combo[1])] +
            matrix[int(combo[0])][int(combo[2])] +
            matrix[int(combo[1])][int(combo[2])]
        )

        # Monte Carlo probability
        prob = monte_carlo_probability(combo, freq_final)

        # EV
        ev = calculate_ev(prob, payout, bet_amount)

        # Kelly
        kelly = kelly_criterion(prob, payout)

        total_score = freq_score + (co_score * 0.0001) + (prob * 2)

        results.append({
            "combo": "".join(combo),
            "probability": prob,
            "freq_score": freq_score,
            "co_score": co_score,
            "EV": ev,
            "Kelly": kelly,
            "score": total_score
        })

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)

    st.subheader("🏆 TOP 10 BỘ 3 TỐI ƯU")
    st.dataframe(df.head(10), use_container_width=True)

    best = df.iloc[0]

    st.success(f"""
    🎯 Bộ đề xuất mạnh nhất: {best['combo']}
    📊 Xác suất mô phỏng: {round(best['probability']*100,2)}%
    💰 EV: {round(best['EV'],2)}
    📈 Kelly vốn đề xuất: {round(best['Kelly']*100,2)}% vốn
    """)

    st.markdown("---")
    st.subheader("📊 Phân phối Digit hiện tại")
    st.json(freq_final)