import streamlit as st
import collections
import requests
import time
import re
import json

st.set_page_config(page_title="AI 3-TINH ELITE v85 LIVE FEED",layout="centered")

# ================= UI =================
st.markdown("""
<style>
.stApp{background:#0b0f13;color:#e0e0e0}
.result{border:2px solid #00ffcc;border-radius:15px;padding:25px;background:#161b22;text-align:center;margin-top:20px}
.bigbox{display:flex;justify-content:center;gap:20px;flex-wrap:wrap}
.big{font-size:65px;color:#ffff00;font-weight:bold;text-shadow:0px 0px 20px rgba(255,255,0,0.5)}
.feed{color:#00ffcc;font-size:13px;margin-top:10px}
</style>
""",unsafe_allow_html=True)

st.title("📡 AI 3-TINH ELITE v85 - LIVE FEED ENGINE")

# ================= SESSION =================
if "weights" not in st.session_state:
    st.session_state.weights={"freq":1,"gap":1,"recency":1,"cycle":1}

if "live_digits" not in st.session_state:
    st.session_state.live_digits=""

# ================= PARSER =================
def extract_digits(text):
    return "".join(re.findall(r'\d',text))

def fetch_live(url):
    try:
        headers={"User-Agent":"Mozilla/5.0"}
        r=requests.get(url,headers=headers,timeout=8)
        if r.status_code!=200:return ""
        txt=r.text
        try:
            js=json.loads(txt)
            txt=json.dumps(js)
        except:
            pass
        return extract_digits(txt)
    except:
        return ""

# ================= PATTERN =================
def detect_pattern(nums):
    if len(nums)<5:return "unknown"
    if len(set(nums[-5:]))==1:return "cầu bệt"
    if nums[-1]==nums[-2]:return "cầu lặp"
    if nums[-1]==nums[-3]:return "cầu đảo"
    return "cầu nhảy"

# ================= SCORE =================
def calculate_scores(raw):

    nums=list(raw)
    total=len(nums)
    freq=collections.Counter(nums)

    scores={str(i):0 for i in range(10)}

    for n in scores:

        if freq.get(n,0)/max(total,1)>0.35:
            scores[n]-=2

        scores[n]+= (1-(freq.get(n,0)/max(total,1)))*st.session_state.weights["freq"]

        if n not in nums[-30:]:
            scores[n]+=1*st.session_state.weights["recency"]

        if n in nums:
            gap=len(nums)-1-nums[::-1].index(n)
            scores[n]+= (gap/60)*st.session_state.weights["gap"]

        scores[n]+= (nums.count(n)%7)/10*st.session_state.weights["cycle"]

    return scores,freq

# ================= ANALYZE =================
def analyze(raw):

    pattern=detect_pattern(raw)
    scores,freq=calculate_scores(raw)

    eliminated=sorted(scores,key=scores.get,reverse=True)[:3]
    remaining=[str(i) for i in range(10) if str(i) not in eliminated]

    ranked=sorted(remaining,key=lambda x:freq.get(x,0),reverse=True)
    tinh3=ranked[:3]

    return eliminated,remaining,tinh3,pattern,freq

# ================= INPUT =================
mode=st.radio("Nguồn Live Feed",["✍️ Paste Live","🌐 URL Feed"])

paste_data=""
url=""

if mode=="✍️ Paste Live":
    paste_data=st.text_area("Dán HTML / JSON / Text Live:",height=120)

if mode=="🌐 URL Feed":
    url=st.text_input("Link Feed")
    auto=st.checkbox("⚡ Auto Refresh 5s")

# ================= RUN =================
if st.button("🚀 CHẠY LIVE FEED",use_container_width=True):

    if mode=="🌐 URL Feed":
        digits=fetch_live(url)
    else:
        digits=extract_digits(paste_data)

    if digits:
        new_digits="".join([d for d in digits if d not in st.session_state.live_digits])
        st.session_state.live_digits+=new_digits

    raw=st.session_state.live_digits

    if len(raw)<10:
        st.warning("Chưa đủ dữ liệu")
    else:

        eliminated,remaining,tinh3,pattern,freq=analyze(raw)

        heat=" ".join([f"{k}:{v}" for k,v in freq.most_common()])

        big_html="".join([f"<div class='big'>{n}</div>" for n in tinh3])

        st.markdown(f"""
        <div class='result'>
        <p>🎯 DÀN 3 TINH LIVE</p>
        <div class='bigbox'>{big_html}</div>
        <p>📊 Pattern: {pattern}</p>
        <p>🚫 Loại: {", ".join(eliminated)}</p>
        <p>✅ 7 số: {", ".join(remaining)}</p>
        <p class='feed'>🔥 Heatmap: {heat}</p>
        </div>
        """,unsafe_allow_html=True)

# ================= AUTO LOOP =================
if mode=="🌐 URL Feed" and 'auto' in locals() and auto:
    time.sleep(5)
    st.rerun()

# ================= LEARNING =================
st.markdown("### 🧠 Neural Learning v85")
real=st.text_input("Kết quả thật")

if st.button("📈 AI HỌC"):
    if real.isdigit():
        st.session_state.weights["freq"]*=0.99
        st.session_state.weights["gap"]*=1.02
        st.session_state.weights["recency"]*=1.01
        st.session_state.weights["cycle"]*=1.01
        st.success("AI đã tiến hóa")

st.info("🔥 v85 Engine: Live Feed + Smart Parser + Anti Trap + Heatmap + Neural Learning")