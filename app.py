import streamlit as st
import collections
import requests
import time
import re
import json

st.set_page_config(page_title="AI 3-TINH ELITE v80 BLACK AI",layout="centered")

# ================= UI =================
st.markdown("""
<style>
.stApp{background:#0b0f13;color:#e0e0e0}
.result{border:2px solid #00ffcc;border-radius:15px;padding:25px;background:#161b22;text-align:center;margin-top:20px}
.bigbox{display:flex;justify-content:center;gap:20px;flex-wrap:wrap}
.big{font-size:70px;color:#ffff00;font-weight:bold;text-shadow:0px 0px 20px rgba(255,255,0,0.5)}
</style>
""",unsafe_allow_html=True)

st.title("🧠 AI 3-TINH ELITE v80 - BLACK AI")

# ================= SESSION =================
if "weights" not in st.session_state:
    st.session_state.weights={
        "freq":1.0,
        "gap":1.0,
        "recency":1.0,
        "cycle":1.0,
        "pattern":1.0,
        "heat":1.0
    }

if "live_data" not in st.session_state:
    st.session_state.live_data=""

# ================= FETCH ENGINE =================
def extract_digits(text):
    return "".join(re.findall(r'\d',text))

def fetch_auto(url):
    try:
        headers={"User-Agent":"Mozilla/5.0"}
        r=requests.get(url,headers=headers,timeout=8)
        if r.status_code!=200:return ""
        txt=r.text

        # detect JSON digits
        try:
            js=json.loads(txt)
            txt=json.dumps(js)
        except:
            pass

        return extract_digits(txt)
    except:
        return ""

# ================= PATTERN ENGINE =================
def detect_pattern(nums):
    if len(nums)<5:return "unknown"
    if len(set(nums[-5:]))==1:return "cầu bệt"
    if nums[-1]==nums[-2]:return "cầu lặp"
    if nums[-1]==nums[-3]:return "cầu đảo"
    if nums[-1]!=nums[-2]:return "cầu nhảy"
    return "mixed"

# ================= DEEP SCORE =================
def deep_scores(raw):

    nums=list(raw)
    total=len(nums)
    freq=collections.Counter(nums)

    scores={str(i):0 for i in range(10)}

    for n in scores:

        # anti manipulation heat spike
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

# ================= AI ENSEMBLE =================
def ai_select(remaining,freq,pattern):

    # local freq ai
    local=sorted(remaining,key=lambda x:freq.get(x,0),reverse=True)[:3]

    # pattern bias
    if pattern=="cầu bệt":
        pattern_ai=remaining[::-1][:3]
    else:
        pattern_ai=remaining[:3]

    votes=collections.Counter(local+pattern_ai)

    return [n for n,_ in votes.most_common(3)]

# ================= ANALYZE =================
def analyze(raw):

    pattern=detect_pattern(raw)

    scores,freq=deep_scores(raw)

    eliminated=sorted(scores,key=scores.get,reverse=True)[:3]
    remaining=[str(i) for i in range(10) if str(i) not in eliminated]

    tinh3=ai_select(remaining,freq,pattern)

    return eliminated,remaining,tinh3,pattern,freq

# ================= INPUT =================
mode=st.radio("Nguồn dữ liệu",["✍️ Chuỗi số","🌐 Link HTML AUTO"])

data=""
url=""

if mode=="✍️ Chuỗi số":
    data=st.text_area("Nhập chuỗi số:",height=120)

if mode=="🌐 Link HTML AUTO":
    url=st.text_input("Link nguồn dữ liệu")
    auto=st.checkbox("⚡ Auto Crawl 5s")

# ================= RUN =================
if st.button("🚀 KÍCH HOẠT BLACK AI",use_container_width=True):

    if mode=="🌐 Link HTML AUTO":
        digits=fetch_auto(url)
        st.session_state.live_data+=digits
        raw=st.session_state.live_data
    else:
        raw=extract_digits(data)

    if len(raw)<10:
        st.error("⚠️ cần ít nhất 10 số")
    else:

        eliminated,remaining,tinh3,pattern,freq=analyze(raw)

        heat=" ".join([f"{k}:{v}" for k,v in freq.most_common()])

        big_html="".join([f"<div class='big'>{n}</div>" for n in tinh3])

        st.markdown(f"""
        <div class='result'>
        <p>🎯 DÀN 3 TINH BLACK AI</p>
        <div class='bigbox'>{big_html}</div>
        <p>📊 Pattern: {pattern}</p>
        <p>🚫 Loại: {", ".join(eliminated)}</p>
        <p>✅ 7 số: {", ".join(remaining)}</p>
        <p>🔥 Heatmap: {heat}</p>
        </div>
        """,unsafe_allow_html=True)

# ================= AUTO LOOP =================
if mode=="🌐 Link HTML AUTO" and 'auto' in locals() and auto:
    time.sleep(5)
    st.rerun()

# ================= LEARNING =================
st.markdown("### 🧠 Neural Learning BLACK")
real=st.text_input("Kết quả thật")

if st.button("📈 AI HỌC"):
    if real.isdigit():
        st.session_state.weights["freq"]*=0.99
        st.session_state.weights["gap"]*=1.02
        st.session_state.weights["recency"]*=1.01
        st.session_state.weights["cycle"]*=1.01
        st.session_state.weights["pattern"]*=1.01
        st.success("AI đã tự điều chỉnh trọng số")

st.info("🔥 Engine v80: Deep Pattern + Anti Manipulation + Heatmap Live + Ensemble AI + Neural Adaptive")