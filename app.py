import streamlit as st
import requests
import re
import time
from datetime import datetime
from collections import Counter

st.set_page_config(page_title="AI TITAN CORE FINAL",layout="wide")

# ================= SESSION =================
for key in ["history","heatmap","timeline","dataset"]:
    if key not in st.session_state:
        st.session_state[key]=[]

if isinstance(st.session_state.heatmap,list):
    st.session_state.heatmap={}

# ================= UI =================
st.title("☠️ AI TITAN CORE – FINAL ALL IN ONE")

mode=st.radio("Nguồn dữ liệu",["Nhập tay","Paste HTML/Text","URL Auto Feed"])

manual=""
paste=""
url=""

if mode=="Nhập tay":
    manual=st.text_area("Nhập chuỗi số")

elif mode=="Paste HTML/Text":
    paste=st.text_area("Paste HTML/Text")

elif mode=="URL Auto Feed":
    url=st.text_input("Nhập URL")

poll=st.slider("Auto Refresh (giây)",0,30,0)

# ================= FUNCTIONS =================
def extract_numbers(text):
    return re.findall(r"\d+",text)

def fetch_url(url):
    try:
        headers={"User-Agent":"Mozilla/5.0"}
        r=requests.get(url,headers=headers,timeout=10)
        return r.text
    except:
        return ""

def auto_parse(html):
    nums=re.findall(r">\s*(\d{1,5})\s*<",html)
    if nums:
        return nums
    return extract_numbers(html)

def update_heatmap(nums):
    for n in nums:
        for d in n:
            st.session_state.heatmap[d]=st.session_state.heatmap.get(d,0)+1

def titan_predict(nums):

    digits=[]
    for n in nums:
        digits+=list(n)

    freq=Counter(digits)
    recent=digits[-30:]
    momentum=Counter(recent)

    score={str(i):0 for i in range(10)}

    for d in score:
        score[d]+=momentum.get(d,0)*2
        score[d]+=freq.get(d,0)

        # anti nóng giả
        if freq.get(d,0)/max(len(digits),1)>0.25:
            score[d]-=2

    ranked=sorted(score,key=score.get,reverse=True)

    return ranked[:3],ranked[-3:]

# ================= INPUT =================
nums=[]

if mode=="Nhập tay":
    nums=extract_numbers(manual)

elif mode=="Paste HTML/Text":
    nums=extract_numbers(paste)

elif mode=="URL Auto Feed":
    if url:
        html=fetch_url(url)
        nums=auto_parse(html)

# ================= RUN =================
predict=[]
loai=[]

if st.button("🚀 CHẠY TITAN ENGINE"):

    st.session_state.dataset+=nums

    update_heatmap(nums)

    if st.session_state.dataset:
        predict,loai=titan_predict(st.session_state.dataset)

    st.session_state.timeline.append(nums[-5:])

    st.session_state.history.append({
        "time":datetime.now().strftime("%H:%M:%S"),
        "predict":predict,
        "count":len(st.session_state.dataset)
    })

# ================= AUTO =================
if poll>0:
    time.sleep(poll)
    st.rerun()

# ================= RESULT =================
if predict:
    st.markdown(f"""
    <div style='border:2px solid #00ffcc;
    padding:20px;
    border-radius:15px;
    text-align:center;
    background:#111'>
    <h2>🎯 DỰ ĐOÁN TITAN</h2>
    <h1 style='color:yellow;font-size:60px'>
    {" - ".join(predict)}
    </h1>
    <p>🚫 LOẠI: {", ".join(loai)}</p>
    </div>
    """,unsafe_allow_html=True)

# ================= DASHBOARD =================
st.divider()

st.subheader("🔥 Heatmap")
st.write(st.session_state.heatmap)

st.subheader("🧠 Timeline gần")
for t in st.session_state.timeline[-10:]:
    st.write(t)

st.subheader("📊 Dashboard AI")
for item in reversed(st.session_state.history[-10:]):
    st.write(item)

st.subheader("📂 Dataset size")
st.write(len(st.session_state.dataset))