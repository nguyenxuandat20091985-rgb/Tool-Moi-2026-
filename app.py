import streamlit as st
import pandas as pd
import re
import random
import math
from collections import Counter, defaultdict

# ================================
# TITAN GOD ENGINE v45
# ================================

class TitanGodEngine:

    def __init__(self):

        self.weights = {
            "frequency":20,
            "hotcold":15,
            "cycle":15,
            "markov":20,
            "montecarlo":20,
            "pattern":10
        }

    # ------------------------------
    # Frequency Engine
    # ------------------------------
    def frequency_engine(self,history):

        data="".join(history[-80:])
        c=Counter(data)

        return [x[0] for x in c.most_common(5)]

    # ------------------------------
    # Hot Cold Engine
    # ------------------------------

    def hotcold_engine(self,history):

        recent="".join(history[-20:])
        c=Counter(recent)

        hot=[x[0] for x in c.most_common(3)]

        all_digits=set("0123456789")
        cold=list(all_digits-set(c.keys()))

        return hot+cold[:2]

    # ------------------------------
    # Cycle Engine
    # ------------------------------

    def cycle_engine(self,history):

        digits="0123456789"
        gap={}

        for d in digits:
            for i in range(len(history)-1,-1,-1):
                if d in history[i]:
                    gap[d]=len(history)-i
                    break

        ranked=sorted(gap.items(),key=lambda x:x[1],reverse=True)

        return [x[0] for x in ranked[:5]]

    # ------------------------------
    # Markov Engine
    # ------------------------------

    def markov_engine(self,history):

        nodes=defaultdict(Counter)

        for i in range(len(history)-1):
            a=history[i][-1]
            b=history[i+1][-1]
            nodes[a][b]+=1

        last=history[-1][-1]

        return [x[0] for x in nodes[last].most_common(5)]

    # ------------------------------
    # Monte Carlo Engine
    # ------------------------------

    def montecarlo_engine(self,history):

        pool=list("".join(history[-60:]))

        sim=Counter()

        for _ in range(200000):

            sample=random.choices(pool,k=3)

            for n in sample:
                sim[n]+=1

        return [x[0] for x in sim.most_common(5)]

    # ------------------------------
    # Pattern Engine
    # ------------------------------

    def pattern_engine(self,history):

        last=history[-1]

        return list(set([
            last[0],
            last[2],
            last[4]
        ]))

    # ------------------------------
    # Momentum Engine
    # ------------------------------

    def momentum_engine(self,history):

        prev="".join(history[-40:-20])
        now="".join(history[-20:])

        c1=Counter(prev)
        c2=Counter(now)

        diff={}

        for d in "0123456789":

            diff[d]=c2[d]-c1[d]

        ranked=sorted(diff.items(),key=lambda x:x[1],reverse=True)

        return [x[0] for x in ranked[:4]]

    # ------------------------------
    # Entropy Risk
    # ------------------------------

    def risk_engine(self,history):

        data="".join(history[-40:])

        c=Counter(data)

        probs=[v/len(data) for v in c.values()]

        entropy=-sum(p*math.log2(p) for p in probs)

        score=int(max(0,min(100,(3.32-entropy)*160)))

        if score>60:
            lvl="HIGH"
        elif score>35:
            lvl="MEDIUM"
        else:
            lvl="LOW"

        return score,lvl

    # ------------------------------
    # Ensemble Voting
    # ------------------------------

    def ensemble(self,history):

        votes=Counter()

        engines={
            "frequency":self.frequency_engine(history),
            "hotcold":self.hotcold_engine(history),
            "cycle":self.cycle_engine(history),
            "markov":self.markov_engine(history),
            "montecarlo":self.montecarlo_engine(history),
            "pattern":self.pattern_engine(history),
            "momentum":self.momentum_engine(history)
        }

        for eng,res in engines.items():

            w=self.weights.get(eng,10)

            for d in res:
                votes[d]+=w

        ranked=votes.most_common(7)

        return [x[0] for x in ranked[:3]],[x[0] for x in ranked[3:7]]

    # ------------------------------
    # Backtest Engine
    # ------------------------------

    def backtest(self,history):

        if len(history)<30:
            return 0

        win=0
        total=0

        for i in range(20,len(history)-1):

            sample=history[:i]

            main,_=self.ensemble(sample)

            next_draw=history[i]

            if all(x in next_draw for x in main):
                win+=1

            total+=1

        if total==0:
            return 0

        return round((win/total)*100,2)

    # ------------------------------
    # Main Predict
    # ------------------------------

    def predict(self,history):

        main,support=self.ensemble(history)

        risk_val,risk_lvl=self.risk_engine(history)

        acc=self.backtest(history)

        return {
            "main":main,
            "support":support,
            "risk_val":risk_val,
            "risk_lvl":risk_lvl,
            "accuracy":acc
        }


# ================================
# UI TITAN
# ================================

st.set_page_config(page_title="TITAN GOD ENGINE",layout="wide")

st.title("TITAN GOD ENGINE v45")

if "engine" not in st.session_state:
    st.session_state.engine=TitanGodEngine()

if "history" not in st.session_state:
    st.session_state.history=[]

raw=st.text_area("Nhập kết quả 5 số mỗi kỳ")

if st.button("PHÂN TÍCH"):

    nums=re.findall(r"\d{5}",raw)

    if nums:

        st.session_state.history=nums

        res=st.session_state.engine.predict(nums)

        st.subheader("3 SỐ CHÍNH")

        st.write(res["main"])

        st.subheader("4 SỐ LÓT")

        st.write(res["support"])

        st.subheader("RỦI RO")

        st.write(res["risk_lvl"],res["risk_val"])

        st.subheader("BACKTEST ACCURACY")

        st.write(str(res["accuracy"])+" %")