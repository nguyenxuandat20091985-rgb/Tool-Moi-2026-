import streamlit as st
import requests
import re
import time
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="AI TITAN CORE FINAL",
    layout="wide"
)

# ================= SESSION =================
for key in ["history","heatmap","timeline","dataset"]:
    if key not in st.session_state:
        st.session_state[key]=[]

if isinstance(st.session_state.heatmap,list):
    st.session_state.heatmap={}

# ================= UI =================
st.title("☠️ AI TITAN CORE – FINAL ALL IN ONE")

mode=st.radio(
    "Nguồn dữ liệu",
    ["Nhập tay","Paste HTML/Text","URL Auto Feed"]
)

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

def analyze_patterns(nums):
    patterns=[]
    last=nums[-15:]

    for i in range(len(last)-1):
        a=last[i]
        b=last[i+1]

        if a==b:
            patterns.append("bệt")
        elif a[::-1]==b:
            patterns.append("đảo")
        elif a[-1]==b[-1]:
            patterns.append("lặp đuôi")
        else:
            patterns.append("nhảy")

    return Counter(patterns)

def titan_engine(nums):

    if not nums:
        return "Không có dữ liệu"

    digits=[]
    for n in nums:
        digits+=list(n)

    freq=Counter(digits).most_common(5)

    recent=nums[-20:]
    recent_digits=[]
    for n in recent:
        recent_digits+=list(n)

    momentum=Counter(recent_digits).most_common(3)

    patterns=analyze_patterns(nums)

    return {
        "freq":freq,
        "momentum":momentum,
        "patterns":patterns
    }

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
if st.button("🚀 CHẠY TITAN ENGINE"):

    st.session_state.dataset+=nums

    update_heatmap(nums)

    result=titan_engine(st.session_state.dataset)

    st.session_state.timeline.append(nums[-5:])

    st.session_state.history.append({
        "time":datetime.now().strftime("%H:%M:%S"),
        "result":result,
        "count":len(st.session_state.dataset)
    })

# ================= AUTO =================
if poll>0:
    time.sleep(poll)
    st.rerun()

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