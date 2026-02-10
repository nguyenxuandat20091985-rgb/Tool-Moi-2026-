import streamlit as st
import collections
import requests
import time
import re
import json

st.set_page_config(page_title="AI 3-TINH ELITE v86 HYBRID",layout="centered")

# ================= CONFIG =================
GEMINI_API_KEY=st.secrets.get("GEMINI_API_KEY","")

if "live_digits" not in st.session_state:
    st.session_state.live_digits=""

# ================= UI =================
st.markdown("""
<style>
.stApp{background:#0b0f13;color:#e0e0e0}
.result{border:2px solid #00ffcc;border-radius:15px;padding:20px;background:#161b22;text-align:center;margin-top:20px}
.big{font-size:60px;color:#ffff00;font-weight:bold}
.bigbox{display:flex;justify-content:center;gap:15px}
</style>
""",unsafe_allow_html=True)

st.title("🧠 AI 3-TINH ELITE v86 HYBRID ENGINE")

# ================= INPUT MODE =================
mode=st.radio("Nguồn dữ liệu",[
"✍️ Nhập số tay",
"📋 Paste HTML/Text",
"🌐 URL Feed"
])

manual=""
paste=""
url=""

if mode=="✍️ Nhập số tay":
    manual=st.text_area("Nhập chuỗi số:",height=120)

if mode=="📋 Paste HTML/Text":
    paste=st.text_area("Dán HTML / JSON:",height=120)

if mode=="🌐 URL Feed":
    url=st.text_input("Link Feed")
    auto=st.checkbox("⚡ Auto refresh 5s")

# ================= PARSER =================
def extract_digits(text):
    return "".join(re.findall(r'\d',text))

def fetch_url(url):
    try:
        headers={"User-Agent":"Mozilla/5.0"}
        r=requests.get(url,headers=headers,timeout=8)
        return extract_digits(r.text)
    except:
        return ""

# ================= GEMINI =================
def gemini_ai(data):

    if not GEMINI_API_KEY:
        return []

    try:
        headers={"Content-Type":"application/json"}
        body={
        "contents":[
        {"parts":[
        {"text":f"choose 3 digits from 0-9 based on this data {data} return only numbers"}
        ]}
        ]}
        r=requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
        headers=headers,json=body)

        txt=r.json()["candidates"][0]["content"]["parts"][0]["text"]
        digits=[c for c in txt if c.isdigit()]
        return digits[:3]

    except:
        return []

# ================= AI CORE =================
def analyze(raw):

    nums=list(raw)
    freq=collections.Counter(nums)

    scores={str(i):0 for i in range(10)}

    for n in scores:
        scores[n]+=1-(freq.get(n,0)/max(len(nums),1))

    eliminated=sorted(scores,key=scores.get,reverse=True)[:3]
    remaining=[str(i) for i in range(10) if str(i) not in eliminated]

    local=sorted(remaining,key=lambda x:freq.get(x,0),reverse=True)[:3]
    gemini=gemini_ai(raw[-50:])

    if gemini:
        vote=collections.Counter(local+gemini)
        final=[n for n,_ in vote.most_common(3)]
    else:
        final=local

    return eliminated,remaining,final

# ================= RUN =================
if st.button("🚀 CHẠY AI v86",use_container_width=True):

    digits=""

    if mode=="✍️ Nhập số tay":
        digits=extract_digits(manual)

    if mode=="📋 Paste HTML/Text":
        digits=extract_digits(paste)

    if mode=="🌐 URL Feed":
        digits=fetch_url(url)

    if digits:
        st.session_state.live_digits+=digits

    raw=st.session_state.live_digits

    if len(raw)<10:
        st.warning("Chưa đủ dữ liệu")
    else:

        eliminated,remaining,tinh3=analyze(raw)

        big_html="".join([f"<div class='big'>{n}</div>" for n in tinh3])

        st.markdown(f"""
        <div class='result'>
        <p>🎯 DÀN 3 TINH</p>
        <div class='bigbox'>{big_html}</div>
        <p>🚫 Loại: {", ".join(eliminated)}</p>
        <p>✅ 7 số: {", ".join(remaining)}</p>
        </div>
        """,unsafe_allow_html=True)

# ================= AUTO =================
if mode=="🌐 URL Feed" and 'auto' in locals() and auto:
    time.sleep(5)
    st.rerun()

st.info("🔥 v86: Hybrid Feed + Manual Input + URL + Gemini AI")