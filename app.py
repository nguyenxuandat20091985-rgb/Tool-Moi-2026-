import streamlit as st
import pandas as pd
import re
import random
import math
from collections import Counter, defaultdict

# ===============================
# TITAN GOD ENGINE v50
# ===============================

class TitanEngine:

    def __init__(self):
        self.weights = {
            "markov":25,
            "monte":25,
            "freq":20,
            "cycle":15,
            "pattern":15
        }

    # ---------------------------
    # Frequency
    # ---------------------------
    def frequency(self,history):
        digits="".join(history)
        return Counter(digits)

    # ---------------------------
    # Cycle detection
    # ---------------------------
    def cycle_score(self,history):
        last_seen={str(i):-1 for i in range(10)}
        for i,h in enumerate(history[::-1]):
            for d in set(h):
                if last_seen[d]==-1:
                    last_seen[d]=i

        score={}
        for d in last_seen:
            score[d]=last_seen[d]

        return score

    # ---------------------------
    # Markov
    # ---------------------------
    def markov(self,history):

        nodes=defaultdict(Counter)

        for i in range(len(history)-1):
            a=history[i][-1]
            b=history[i+1][-1]
            nodes[a][b]+=1

        last=history[-1][-1]

        return [x[0] for x in nodes[last].most_common(5)]

    # ---------------------------
    # Monte Carlo
    # ---------------------------
    def monte_carlo(self,history):

        pool=list("".join(history[-80:]))

        sim=Counter()

        for _ in range(200000):

            sample=random.choices(pool,k=3)

            for s in sample:
                sim[s]+=1

        return [x[0] for x in sim.most_common(5)]

    # ---------------------------
    # Pattern
    # ---------------------------
    def pattern(self,history):

        last=history[-1]

        return [last[0],last[2],last[4]]

    # ---------------------------
    # Entropy
    # ---------------------------
    def entropy(self,history):

        data="".join(history[-50:])

        counts=Counter(data)

        probs=[c/len(data) for c in counts.values()]

        ent=-sum(p*math.log2(p) for p in probs)

        risk=int(max(0,min(100,(3.32-ent)*160)))

        return risk

    # ---------------------------
    # Ensemble Prediction
    # ---------------------------
    def predict(self,history):

        freq=self.frequency(history)

        cycle=self.cycle_score(history)

        mk=self.markov(history)

        mc=self.monte_carlo(history)

        pt=self.pattern(history)

        votes=Counter()

        for k,v in freq.items():
            votes[k]+=v*self.weights["freq"]

        for d,c in cycle.items():
            votes[d]+=c*self.weights["cycle"]

        for n in mk:
            votes[n]+=self.weights["markov"]

        for n in mc:
            votes[n]+=self.weights["monte"]

        for n in pt:
            votes[n]+=self.weights["pattern"]

        final=votes.most_common(7)

        risk=self.entropy(history)

        return{
            "main":[x[0] for x in final[:3]],
            "support":[x[0] for x in final[3:7]],
            "risk":risk
        }


# ===============================
# UI
# ===============================

st.set_page_config(page_title="TITAN GOD ENGINE",layout="wide")

st.markdown("""
<style>

.stApp{
background:#050505;
color:white;
}

.header{
font-size:34px;
text-align:center;
font-weight:900;
color:#00f2ff;
margin-bottom:10px;
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
width:100px;
height:130px;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
box-shadow:0 0 20px rgba(0,242,255,0.4);
}

.num{
font-size:50px;
font-weight:900;
}

.tag{
font-size:10px;
color:#00f2ff;
}

.support{
display:flex;
justify-content:center;
gap:10px;
margin-top:20px;
}

.sup{
background:#222;
padding:10px 15px;
border-radius:8px;
font-size:22px;
border:1px solid #58a6ff;
}

.risk{
text-align:center;
font-size:20px;
font-weight:bold;
margin-top:25px;
}

</style>
""",unsafe_allow_html=True)

# ===============================
# SESSION
# ===============================

if "engine" not in st.session_state:
    st.session_state.engine=TitanEngine()

if "history" not in st.session_state:
    st.session_state.history=[]

if "result" not in st.session_state:
    st.session_state.result=None

# ===============================
# HEADER
# ===============================

st.markdown("<div class='header'>TITAN GOD ENGINE v50</div>",unsafe_allow_html=True)

# ===============================
# INPUT
# ===============================

with st.expander("📥 Nhập kết quả 5 số mỗi kỳ",expanded=True):

    raw=st.text_area("",height=120,placeholder="Ví dụ:\n12345\n67890\n45821")

    if st.button("PHÂN TÍCH"):

        nums=re.findall(r'\d{5}',raw)

        if nums:

            st.session_state.history=nums

            res=st.session_state.engine.predict(nums)

            st.session_state.result=res

            st.rerun()

# ===============================
# RESULT
# ===============================

if st.session_state.result:

    res=st.session_state.result

    st.subheader("🔮 3 SỐ CHÍNH")

    main_html="".join([
    f"<div class='card'><div class='num'>{n}</div><div class='tag'>AI</div></div>"
    for n in res["main"]
    ])

    st.markdown(f"<div class='main-grid'>{main_html}</div>",unsafe_allow_html=True)

    st.subheader("🎯 4 SỐ LÓT")

    sup_html="".join([
    f"<div class='sup'>{n}</div>"
    for n in res["support"]
    ])

    st.markdown(f"<div class='support'>{sup_html}</div>",unsafe_allow_html=True)

    risk=res["risk"]

    lvl="LOW"

    if risk>60:
        lvl="HIGH"
    elif risk>35:
        lvl="MEDIUM"

    st.markdown(f"<div class='risk'>RỦI RO: {risk}/100 - {lvl}</div>",unsafe_allow_html=True)

    # Heatmap

    freq=Counter("".join(st.session_state.history))

    df=pd.DataFrame({
        "Digit":list(freq.keys()),
        "Count":list(freq.values())
    }).sort_values("Digit")

    st.subheader("📊 Digit Heatmap")

    st.bar_chart(df.set_index("Digit"))

    st.write(f"Data size: {len(st.session_state.history)} kỳ")