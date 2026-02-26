import streamlit as st
import pandas as pd
import numpy as np
import itertools
import math
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="3 TINH ENGINE PRO",
    layout="wide"
)

# ================= CORE FUNCTIONS =================

def clean_data(input_text):
    lines = input_text.strip().split("\n")
    results = []
    for line in lines:
        digits = ''.join(filter(str.isdigit, line))
        if len(digits) == 5:
            results.append(digits)
    return results


def frequency_analysis(results, window):
    recent = results[-window:]
    digits = ''.join(recent)
    counter = Counter(digits)
    total = len(digits)
    freq = {str(i): counter.get(str(i), 0)/total for i in range(10)}
    return freq


def co_occurrence_matrix(results, window):
    recent = results[-window:]
    matrix = np.zeros((10,10))
    for r in recent:
        unique_digits = set(r)
        for a in unique_digits:
            for b in unique_digits:
                if a != b:
                    matrix[int(a)][int(b)] += 1
    return matrix


def markov_model(results):
    transitions = np.zeros((10,10))
    for r in results:
        for i in range(4):
            transitions[int(r[i])][int(r[i+1])] += 1
    row_sums = transitions.sum(axis=1)
    prob = np.divide(transitions, row_sums[:, None], 
                     out=np.zeros_like(transitions), 
                     where=row_sums[:, None]!=0)
    return prob


def entropy_score(freq):
    ent = 0
    for v in freq.values():
        if v > 0:
            ent -= v * math.log(v)
    return ent


def monte_carlo_score(combo, simulations=5000):
    hit = 0
    for _ in range(simulations):
        sample = np.random.randint(0,10,5)
        if all(int(c) in sample for c in combo):
            hit += 1
    return hit/simulations


def expected_value(prob, payout, stake):
    return prob * payout - (1-prob)*stake


def kelly_fraction(prob, payout):
    b = payout - 1
    return (prob*(b+1)-1)/b if b!=0 else 0


# ================= UI =================

st.title("🔥 3 TINH ENGINE PRO - KHÔNG CỐ ĐỊNH")

input_text = st.text_area("Dán kết quả 5 số mỗi dòng:")

if st.button("🚀 PHÂN TÍCH"):

    results = clean_data(input_text)

    if len(results) < 50:
        st.error("Cần tối thiểu 50 kỳ dữ liệu.")
    else:

        freq30 = frequency_analysis(results,30)
        freq50 = frequency_analysis(results,50)
        freq100 = frequency_analysis(results,100)

        co_matrix = co_occurrence_matrix(results,100)
        markov = markov_model(results)

        entropy = entropy_score(freq100)

        combos = list(itertools.combinations("0123456789",3))
        scores = []

        for combo in combos:
            base_prob = monte_carlo_score(combo,2000)

            freq_boost = sum(freq100[d] for d in combo)/3

            co_score = sum(co_matrix[int(a)][int(b)] 
                           for a in combo for b in combo if a!=b)

            final_score = base_prob*0.5 + freq_boost*0.3 + (co_score/1000)*0.2

            scores.append({
                "combo": ''.join(combo),
                "prob_estimate": base_prob,
                "score": final_score
            })

        df = pd.DataFrame(scores)
        df = df.sort_values("score", ascending=False)

        st.subheader("🏆 TOP 10 BỘ 3 TỐI ƯU")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📊 Entropy thị trường")
        st.write(f"Entropy 100 kỳ: {round(entropy,4)}")

        st.subheader("💰 EV & Kelly Calculator")

        payout = st.number_input("Tỷ lệ trả thưởng (ví dụ 70)", value=70.0)
        stake = st.number_input("Tiền cược mỗi bộ", value=1.0)

        top_prob = df.iloc[0]["prob_estimate"]

        ev = expected_value(top_prob, payout, stake)
        kelly = kelly_fraction(top_prob, payout)

        st.write(f"Xác suất ước tính bộ mạnh nhất: {round(top_prob,4)}")
        st.write(f"Expected Value: {round(ev,4)}")
        st.write(f"Kelly fraction đề xuất: {round(kelly,4)}")

        if ev > 0:
            st.success("EV dương - Có lợi thế")
        else:
            st.error("EV âm - Nhà cái có lợi thế")

        st.subheader("📈 Co-occurrence Matrix (100 kỳ)")
        st.dataframe(pd.DataFrame(co_matrix),
                     use_container_width=True)