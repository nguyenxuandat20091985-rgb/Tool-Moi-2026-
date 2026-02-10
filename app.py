import streamlit as st
import re
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="TITAN v1300 PRO CORE",layout="wide")

DATA_FILE="titan_dataset.json"

# ================= LOAD =================
def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE,"r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE,"w") as f:
        json.dump(data,f)

if "dataset" not in st.session_state:
    st.session_state.dataset=load_data()

if "history" not in st.session_state:
    st.session_state.history=[]

# ================= UTIL =================
def extract_numbers(text):
    return re.findall(r"\d{3,5}",text)

def flatten(ds):
    d=[]
    for n in ds:
        d+=list(n)
    return d

def detect_patterns(nums):
    res=[]
    last=nums[-20:]
    for i in range(len(last)-1):
        a=last[i]
        b=last[i+1]
        if a==b:
            res.append("bệt")
        elif a[::-1]==b:
            res.append("đảo")
        elif a[-1]==b[-1]:
            res.append("lặp")
        else:
            res.append("nhảy")
    return Counter(res)

def find_duplicates(nums):
    return [k for k,v in Counter(nums).items() if v>1]

def bong_so(d):
    return {str(i):str((i+5)%10) for i in range(10)}

# ================= ENGINES =================
def engine_freq(d):
    return Counter(d)

def engine_recent(d):
    return Counter(d[-30:])

def engine_gap(d):
    last={}
    for i,x in enumerate(d):
        last[x]=i
    total=len(d)
    gap={}
    for i in range(10):
        k=str(i)
        if k in last:
            gap[k]=(total-last[k])
        else:
            gap[k]=20
    return gap

def engine_heat(d):
    c=Counter(d[-50:])
    return c

def engine_bong(d):
    b=bong_so(d)
    score={}
    for k,v in b.items():
        score[k]=d.count(v)
    return score

# ================= CORE =================
def titan_engine(dataset):

    digits=flatten(dataset)

    if len(digits)<80:
        return None

    freq=engine_freq(digits)
    recent=engine_recent(digits)
    gap=engine_gap(digits)
    heat=engine_heat(digits)
    bong=engine_bong(digits)

    score={str(i):0 for i in range(10)}

    for i in score:
        score[i]+=freq.get(i,0)*1.0
        score[i]+=recent.get(i,0)*1.5
        score[i]+=gap.get(i,0)*0.7
        score[i]+=heat.get(i,0)*1.2
        score[i]+=bong.get(i,0)*0.8

        # anti nóng
        if freq.get(i,0)>len(digits)*0.18:
            score[i]-=15

        # anti chết
        if recent.get(i,0)==0:
            score[i]+=8

    ranked=sorted(score,key=score.get,reverse=True)

    predict1=ranked[:3]
    predict2=ranked[3:6]

    eliminated=ranked[-3:]

    patterns=detect_patterns(dataset)
    dup=find_duplicates(dataset)

    return predict1,predict2,eliminated,patterns,dup,score

# ================= UI =================
st.title("☠️ TITAN v1300 PRO CORE")

manual=st.text_area("Nhập mỗi kỳ 1 dòng")

nums=extract_numbers(manual)

if st.button("🚀 CHẠY TITAN v1300"):

    new=[n for n in nums if n not in st.session_state.dataset]

    if new:
        st.session_state.dataset+=new
        save_data(st.session_state.dataset)

    result=titan_engine(st.session_state.dataset)

    if result:

        p1,p2,el,patterns,dup,score=result

        st.markdown(f"""
        <div style='border:2px solid cyan;padding:20px;border-radius:15px;background:#111;text-align:center'>
        <h2>🎯 TAY TIẾP THEO</h2>
        <h1 style='color:yellow;font-size:65px'>{" - ".join(p1)}</h1>
        <h3>🧠 TAY 2 DỰ PHÒNG: {" - ".join(p2)}</h3>
        <h3>🚫 LOẠI: {", ".join(el)}</h3>
        </div>
        """,unsafe_allow_html=True)

        st.subheader("📊 Phân loại cầu")
        st.write(patterns)

        st.subheader("♻️ Số trùng lặp")
        st.write(dup)

        st.subheader("📈 Score")
        st.write(score)

        st.session_state.history.append({
            "time":datetime.now().strftime("%H:%M:%S"),
            "predict":p1
        })

st.divider()
st.subheader("📂 Dataset size")
st.write(len(st.session_state.dataset))

st.subheader("🧠 History")
for h in st.session_state.history[-10:]:
    st.write(h)