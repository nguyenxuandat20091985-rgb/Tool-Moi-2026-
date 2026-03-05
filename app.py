import streamlit as st
import numpy as np
import random
from collections import Counter, defaultdict

st.set_page_config(page_title="TITAN GOD ENGINE v120", layout="wide")

st.title("TITAN GOD ENGINE v120 – ULTIMATE AI")

st.write("Nhập kết quả 5 số mỗi kỳ (mỗi dòng 1 kỳ)")

raw = st.text_area(
"",
"""18015
49775
14464
23469
99531
"""
)

run = st.button("PHÂN TÍCH AI")


##################################################
# PARSE DATA
##################################################

def parse_data(text):

    rows = []

    for r in text.split("\n"):

        r = r.strip()

        if len(r) == 5 and r.isdigit():

            rows.append(r)

    return rows


##################################################
# FREQUENCY
##################################################

def digit_frequency(rows):

    digits = []

    for r in rows:
        digits += list(r)

    return Counter(digits)


##################################################
# GAP ENGINE
##################################################

def gap_analysis(rows):

    gap = {str(i):0 for i in range(10)}

    rev = rows[::-1]

    for d in gap:

        g = 0

        for r in rev:

            if d in r:
                break

            g += 1

        gap[d] = g

    return gap


##################################################
# POSITION ENGINE
##################################################

def position_analysis(rows):

    pos = [Counter() for _ in range(5)]

    for r in rows:

        for i,d in enumerate(r):

            pos[i][d] += 1

    score = {str(i):0 for i in range(10)}

    for p in pos:

        for d,c in p.items():

            score[d] += c

    return score


##################################################
# MARKOV ENGINE
##################################################

def markov_chain(rows):

    table = defaultdict(lambda: Counter())

    for r in rows:

        for i in range(len(r)-1):

            a = r[i]
            b = r[i+1]

            table[a][b] += 1

    score = {str(i):0 for i in range(10)}

    for a in table:

        total = sum(table[a].values())

        for b in table[a]:

            score[b] += table[a][b] / total

    return score


##################################################
# MONTE CARLO
##################################################

def monte_carlo(freq, sims=10000):

    digits = list(freq.keys())

    if len(digits) == 0:

        digits = [str(i) for i in range(10)]
        weights = [1]*10

    else:

        weights = list(freq.values())

    prob = np.array(weights) / sum(weights)

    counter = Counter()

    for _ in range(sims):

        draw = np.random.choice(digits,5,p=prob)

        for d in draw:

            counter[d] += 1

    return counter


##################################################
# PATTERN ENGINE
##################################################

def pattern_engine(rows):

    pattern = Counter()

    for r in rows:

        pattern.update(r)

    score = {str(i):pattern.get(str(i),0) for i in range(10)}

    return score


##################################################
# AI ENSEMBLE
##################################################

def ai_engine(rows):

    freq = digit_frequency(rows)

    gap = gap_analysis(rows)

    pos = position_analysis(rows)

    markov = markov_chain(rows)

    monte = monte_carlo(freq)

    pattern = pattern_engine(rows)

    score = {str(i):0 for i in range(10)}

    for i in range(10):

        d = str(i)

        score[d] += freq.get(d,0) * 1.2

        score[d] += gap.get(d,0) * 2.5

        score[d] += pos.get(d,0) * 1.3

        score[d] += markov.get(d,0) * 3.0

        score[d] += monte.get(d,0) * 0.01

        score[d] += pattern.get(d,0) * 0.8

        score[d] += random.random()

    sort = sorted(score.items(), key=lambda x:x[1], reverse=True)

    main = [x[0] for x in sort[:3]]

    lot = [x[0] for x in sort[3:7]]

    return main, lot, score


##################################################
# RISK ANALYSIS
##################################################

def risk(score):

    values = list(score.values())

    spread = max(values) - min(values)

    if spread < 8:
        return "LOW"

    if spread < 16:
        return "MEDIUM"

    return "HIGH"


##################################################
# BACKTEST
##################################################

def backtest(rows):

    if len(rows) < 25:

        return 0

    win = 0
    total = 0

    for i in range(20,len(rows)-1):

        train = rows[:i]

        main,_,_ = ai_engine(train)

        next_draw = rows[i]

        if any(d in next_draw for d in main):

            win += 1

        total += 1

    if total == 0:

        return 0

    return round(win/total*100,2)


##################################################
# HEATMAP
##################################################

def heatmap(freq):

    st.subheader("Digit Heatmap")

    cols = st.columns(10)

    for i in range(10):

        d = str(i)

        cols[i].metric(d,freq.get(d,0))


##################################################
# RUN
##################################################

if run:

    rows = parse_data(raw)

    if len(rows) == 0:

        st.error("Dữ liệu không hợp lệ")

    else:

        main,lot,score = ai_engine(rows)

        r = risk(score)

        acc = backtest(rows)

        freq = digit_frequency(rows)

        st.subheader("🔮 3 SỐ CHÍNH")

        c1,c2,c3 = st.columns(3)

        c1.metric("AI",main[0])
        c2.metric("AI",main[1])
        c3.metric("AI",main[2])

        st.subheader("🎯 4 SỐ LÓT")

        c4,c5,c6,c7 = st.columns(4)

        c4.metric("",lot[0])
        c5.metric("",lot[1])
        c6.metric("",lot[2])
        c7.metric("",lot[3])

        st.subheader("RISK LEVEL")

        st.write(r)

        st.subheader("BACKTEST ACCURACY")

        st.write(str(acc)+" %")

        heatmap(freq)