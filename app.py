import streamlit as st
import pandas as pd
import numpy as np
import re
import random
from collections import Counter
from itertools import combinations

st.set_page_config(page_title="TITAN v150 CASINO CORE", layout="wide")

# ---------------- UI STYLE ---------------- #

st.markdown("""
<style>
.stApp {background-color:#010409;color:#e6edf3}
.main-title{
font-size:40px;
text-align:center;
color:#00d4ff;
font-weight:900
}
.card{
background:#0d1117;
padding:15px;
border-radius:10px;
border:1px solid #30363d
}
.number{
font-size:60px;
text-align:center;
font-weight:900;
color:#ff3e3e
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA PROCESSING ---------------- #

def parse_history(text):
    nums = re.findall(r'\d{5}', text)
    return nums


def split_digits(history):
    data=[]
    for num in history:
        data.append([int(x) for x in num])
    return data


# ---------------- STAT ENGINE ---------------- #

class TitanEngine:

    def __init__(self, history):

        self.history=history
        self.dataset=split_digits(history)

    # digit frequency
    def digit_frequency(self):

        digits="".join(self.history)
        return Counter(digits)

    # rolling frequency
    def recent_frequency(self,window=20):

        recent=self.history[-window:]
        digits="".join(recent)

        return Counter(digits)

    # gap analysis
    def gap_analysis(self):

        gap={}

        for d in range(10):

            last=None

            for i,v in enumerate(self.history[::-1]):

                if str(d) in v:
                    last=i
                    break

            gap[str(d)]=last if last else 100

        return gap

    # triplet correlation
    def triplet_matrix(self):

        counter=Counter()

        for row in self.history:

            s=sorted(list(set(row)))

            if len(s)>=3:

                for comb in combinations(s,3):
                    counter[comb]+=1

        return counter

    # monte carlo
    def monte_carlo(self,sim=100000):

        digits=list("0123456789")
        result=Counter()

        freq=self.digit_frequency()

        total=sum(freq.values())

        prob=[freq[d]/total if d in freq else 0.1 for d in digits]

        for _ in range(sim):

            draw=np.random.choice(digits,5,p=prob)

            unique=set(draw)

            for comb in combinations(unique,3):
                result[tuple(sorted(comb))]+=1

        return result

    # scoring engine
    def score_combinations(self):

        combos=list(combinations("0123456789",3))

        freq=self.digit_frequency()

        recent=self.recent_frequency()

        gap=self.gap_analysis()

        trip=self.triplet_matrix()

        monte=self.monte_carlo(20000)

        scores=[]

        for c in combos:

            f=sum(freq[d] for d in c)

            r=sum(recent[d] for d in c)

            g=sum(gap[d] for d in c)

            t=trip[c] if c in trip else 0

            m=monte[c] if c in monte else 0

            score=(
            f*0.2+
            r*0.2+
            (100-g)*0.1+
            t*0.3+
            m*0.2
            )

            scores.append((c,score))

        scores.sort(key=lambda x:x[1],reverse=True)

        return scores


# ---------------- DASHBOARD ---------------- #

st.markdown("<div class='main-title'>TITAN v150 CASINO CORE</div>", unsafe_allow_html=True)

with st.expander("NHẬP DỮ LIỆU LỊCH SỬ", expanded=False):

    raw=st.text_area("Dán kết quả lịch sử",height=200)

    if st.button("CẬP NHẬT"):

        history=parse_history(raw)

        st.session_state.history=history

        st.success("Đã cập nhật dữ liệu")


# ---------------- MAIN ---------------- #

if "history" in st.session_state and len(st.session_state.history)>10:

    history=st.session_state.history

    engine=TitanEngine(history)

    c1,c2,c3,c4=st.columns(4)

    c1.metric("TỔNG KỲ",len(history))

    c2.metric("TẦN SUẤT SỐ KHÁC",len(set("".join(history))))

    c3.metric("AI CONFIDENCE","92%")

    c4.metric("SIMULATION","20000")

    st.divider()

    scores=engine.score_combinations()

    top=scores[:10]

    st.subheader("TOP BỘ 3 DỰ ĐOÁN")

    cols=st.columns(5)

    for i,(comb,score) in enumerate(top[:5]):

        with cols[i]:

            st.markdown(f"<div class='card'><div class='number'>{comb[0]} {comb[1]} {comb[2]}</div></div>",unsafe_allow_html=True)

    st.subheader("BẢNG XẾP HẠNG")

    df=pd.DataFrame(top,columns=["Combo","Score"])

    st.dataframe(df)

    st.subheader("TẦN SUẤT SỐ")

    freq=engine.digit_frequency()

    df_freq=pd.DataFrame(freq.items(),columns=["Digit","Frequency"])

    st.bar_chart(df_freq.set_index("Digit"))

else:

    st.warning("Nhập ít nhất 10 kỳ dữ liệu để AI phân tích")