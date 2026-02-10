import streamlit as st
import requests
import re
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="TITAN v999 FINAL CORE",layout="wide")

DATA_FILE="titan_v999_dataset.json"
WEIGHT_FILE="titan_v999_weights.json"

# ================= LOAD =================
def load_json(file,default):
    if Path(file).exists():
        with open(file,"r") as f:
            return json.load(f)
    return default

def save_json(file,data):
    with open(file,"w") as f:
        json.dump(data,f)

if "dataset" not in st.session_state:
    st.session_state.dataset=load_json(DATA_FILE,[])

if "weights" not in st.session_state:
    st.session_state.weights=load_json(WEIGHT_FILE,{
        "freq":1.0,
        "recent":1.2,
        "gap":1.1,
        "entropy":0.8,
        "momentum":1.3
    })

if "history" not in st.session_state:
    st.session_state.history=[]

# ================= UTIL =================
def extract_numbers(text):
    return re.findall(r"\d{1,5}",text)

def fetch_url(url):
    try:
        headers={"User-Agent":"Mozilla/5.0"}
        r=requests.get(url,headers=headers,timeout=10)
        return r.text
    except:
        return ""

def flatten(dataset):
    d=[]
    for n in dataset:
        d+=list(n)
    return d

def entropy(seq):
    if not seq:return 0
    c=Counter(seq)
    total=len(seq)
    e=0
    for v in c.values():
        p=v/total
        e-=p*math.log2(p)
    return e

# ================= MODELS =================
def model_freq(d):
    c=Counter(d)
    return {i:c.get(str(i),0) for i in range(10)}

def model_recent(d):
    c=Counter(d[-40:])
    return {i:c.get(str(i),0)*2 for i in range(10)}

def model_gap(d):
    last={}
    for i,x in enumerate(d):
        last[x]=i
    total=len(d)
    s={}
    for i in range(10):
        k=str(i)
        if k in last:
            s[i]=(total-last[k])/3
        else:
            s[i]=5
    return s

def model_momentum(d):
    c=Counter(d[-20:])
    return {i:c.get(str(i),0)*3 for i in range(10)}

def model_entropy(d):
    e=entropy(d[-60:])
    return {i:e for i in range(10)}

# ================= ENGINE =================
def titan_engine(dataset,weights):

    d=flatten(dataset)
    if len(d)<100:
        return [],[],{},0

    m1=model_freq(d)
    m2=model_recent(d)
    m3=model_gap(d)
    m4=model_entropy(d)
    m5=model_momentum(d)

    score={i:0 for i in range(10)}

    for i in range(10):
        score[i]+=m1[i]*weights["freq"]
        score[i]+=m2[i]*weights["recent"]
        score[i]+=m3[i]*weights["gap"]
        score[i]+=m4[i]*weights["entropy"]
        score[i]+=m5[i]*weights["momentum"]

    ranked=sorted(score,key=score.get,reverse=True)

    predict=[str(x) for x in ranked[:3]]
    eliminated=[str(x) for x in ranked[-3:]]

    reliability=backtest(dataset)

    return predict,eliminated,score,reliability

# ================= BACKTEST =================
def backtest(dataset):

    if len(dataset)<200:
        return 0

    hit=0
    total=0

    for i in range(150,len(dataset)):
        sub=dataset[:i]
        d=flatten(sub)
        c=Counter(d)
        ranked=[k for k,v in c.most_common(3)]

        if dataset[i][0] in ranked:
            hit+=1
        total+=1

    if total==0:
        return 0

    return round(hit/total*100,2)

# ================= ADAPTIVE =================
def adaptive_learn(real):

    for d in real:
        st.session_state.weights["recent"]*=1.02
        st.session_state.weights["momentum"]*=1.01
        st.session_state.weights["freq"]*=0.99

    save_json(WEIGHT_FILE,st.session_state.weights)

# ================= UI =================
st.title("☠️ TITAN v999 FINAL CORE")

mode=st.radio("Nguồn dữ liệu",["Nhập tay","Paste HTML/Text","URL"])

manual=""
paste=""
url=""

if mode=="Nhập tay":
    manual=st.text_area("Nhập mỗi kỳ 1 dòng")

elif mode=="Paste HTML/Text":
    paste=st.text_area("Paste dữ liệu")

elif mode=="URL":
    url=st.text_input("URL")

nums=[]

if mode=="Nhập tay":
    nums=extract_numbers(manual)

elif mode=="Paste HTML/Text":
    nums=extract_numbers(paste)

elif mode=="URL" and url:
    html=fetch_url(url)
    nums=extract_numbers(html)

predict=[]
eliminated=[]
score={}
reliability=0

if st.button("🚀 CHẠY TITAN v999"):

    new=[n for n in nums if n not in st.session_state.dataset]

    if new:
        st.session_state.dataset+=new
        save_json(DATA_FILE,st.session_state.dataset)

    if st.session_state.dataset:
        predict,eliminated,score,reliability=titan_engine(
            st.session_state.dataset,
            st.session_state.weights
        )

    st.session_state.history.append({
        "time":datetime.now().strftime("%H:%M:%S"),
        "predict":predict,
        "reliability":reliability,
        "size":len(st.session_state.dataset)
    })

# ================= OUTPUT =================
if predict:
    st.markdown(f"""
    <div style='border:2px solid #00ffcc;padding:20px;
    border-radius:15px;background:#111;text-align:center'>
    <h2>🎯 TITAN FINAL PREDICT</h2>
    <h1 style='color:yellow;font-size:65px'>
    {" - ".join(predict)}
    </h1>
    <h3>🚫 Loại: {", ".join(eliminated)}</h3>
    <h3>Backtest reliability: {reliability}%</h3>
    </div>
    """,unsafe_allow_html=True)

st.divider()
st.subheader("📊 Score")
st.write(score)

st.subheader("📂 Dataset size")
st.write(len(st.session_state.dataset))

st.subheader("🧠 History")
for h in st.session_state.history[-10:]:
    st.write(h)