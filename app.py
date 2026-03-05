import streamlit as st
import pandas as pd
import re
import random
import math
from collections import Counter, defaultdict

# =========================
# AI ENGINE
# =========================

class TitanEngine:

    def predict(self, history):

        if len(history) < 3:
            return None

        # ---------- MARKOV CHAIN ----------
        markov = defaultdict(Counter)

        for i in range(len(history)-1):
            last_digit = history[i][-1]
            next_digit = history[i+1][-1]
            markov[last_digit][next_digit] += 1

        last = history[-1][-1]

        mk_res = []

        if last in markov:
            mk_res = [x[0] for x in markov[last].most_common(3)]

        # ---------- MONTE CARLO ----------
        pool = list("".join(history[-80:]))

        sim = Counter()

        random.seed(42)

        for _ in range(50000):
            sample = random.choices(pool, k=3)
            for n in sample:
                sim[n] += 1

        mc_res = [x[0] for x in sim.most_common(4)]

        # ---------- PATTERN ----------
        p_res = [
            history[-1][0],
            history[-1][2],
            history[-1][4]
        ]

        # ---------- ENSEMBLE ----------
        votes = Counter()

        for n in mk_res:
            votes[n] += 30

        for n in mc_res:
            votes[n] += 40

        for n in p_res:
            votes[n] += 20

        final = votes.most_common(7)

        main = [x[0] for x in final[:3]]
        support = [x[0] for x in final[3:7]]

        # ---------- RISK ----------
        data = "".join(history[-40:])

        counts = Counter(data)

        probs = [c/len(data) for c in counts.values()]

        entropy = -sum(p * math.log2(p) for p in probs)

        risk_score = int(max(0, min(100, (3.32 - entropy) * 160)))

        if risk_score > 60:
            risk_lvl = "HIGH"
        elif risk_score > 35:
            risk_lvl = "MEDIUM"
        else:
            risk_lvl = "LOW"

        return {
            "main": main,
            "support": support,
            "risk": risk_score,
            "risk_lvl": risk_lvl,
            "freq": counts
        }


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="TITAN GOD ENGINE",
    layout="wide"
)

# =========================
# STYLE
# =========================

st.markdown("""
<style>

.stApp{
background:#050505;
color:white;
}

.header{
font-size:32px;
font-weight:900;
text-align:center;
color:#00f2ff;
margin-top:10px;
}

.main-grid{
display:flex;
justify-content:center;
gap:20px;
margin-top:20px;
}

.card{
background:#111;
border:2px solid #00f2ff;
border-radius:15px;
width:90px;
height:120px;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
box-shadow:0 0 15px rgba(0,242,255,0.4);
}

.num{
font-size:45px;
font-weight:900;
}

.tag{
font-size:10px;
color:#00f2ff;
}

.support{
display:flex;
gap:10px;
justify-content:center;
margin-top:20px;
}

.sup{
background:#222;
padding:10px 15px;
border-radius:8px;
font-size:22px;
border:1px solid #555;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================

if "engine" not in st.session_state:
    st.session_state.engine = TitanEngine()

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None


# =========================
# UI
# =========================

st.markdown("<div class='header'>TITAN GOD ENGINE v50</div>", unsafe_allow_html=True)

raw = st.text_area(
"Nhập kết quả 5 số mỗi kỳ",
height=120,
placeholder="Ví dụ:\n12345\n67890\n45678"
)

col1,col2 = st.columns(2)

with col1:

    if st.button("PHÂN TÍCH"):

        nums = re.findall(r'\d{5}', raw)

        if nums:

            st.session_state.history = nums

            res = st.session_state.engine.predict(nums)

            st.session_state.result = res

            st.rerun()

with col2:

    if st.button("RESET"):

        st.session_state.history = []

        st.session_state.result = None

        st.rerun()

# =========================
# RESULT
# =========================

if st.session_state.result:

    res = st.session_state.result

    st.markdown("## 🔮 3 SỐ CHÍNH")

    main_html = ""

    for n in res["main"]:

        main_html += f"""
        <div class='card'>
        <div class='num'>{n}</div>
        <div class='tag'>AI</div>
        </div>
        """

    st.markdown(f"<div class='main-grid'>{main_html}</div>", unsafe_allow_html=True)

    st.markdown("## 🎯 4 SỐ LÓT")

    sup_html=""

    for n in res["support"]:

        sup_html += f"<div class='sup'>{n}</div>"

    st.markdown(f"<div class='support'>{sup_html}</div>", unsafe_allow_html=True)

    st.markdown(f"## ⚠️ RỦI RO: {res['risk']}/100 - {res['risk_lvl']}")

    st.markdown("## 📊 Digit Heatmap")

    df = pd.DataFrame(
        [{"digit":k,"freq":v} for k,v in res["freq"].items()]
    )

    st.bar_chart(df.set_index("digit"))