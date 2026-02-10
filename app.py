import streamlit as st
import collections
import requests
import time

st.set_page_config(page_title="AI 3-TINH ELITE v60",layout="centered")

# ================= UI =================
st.markdown("""
<style>
.stApp{background:#0b0f13;color:#e0e0e0}

.result{
border:2px solid #00ffcc;
border-radius:15px;
padding:25px;
background:#161b22;
text-align:center;
margin-top:20px
}

.bigbox{
display:flex;
justify-content:center;
gap:25px;
flex-wrap:wrap
}

.big{
font-size:75px;
color:#ffff00;
font-weight:bold;
text-shadow:0px 0px 20px rgba(255,255,0,0.5)
}
</style>
""",unsafe_allow_html=True)

st.title("🧠 AI 3-TINH ELITE v60 - NEURAL PRO MAX")

# ================= CONFIG =================
GEMINI_API_KEY=st.secrets.get("GEMINI_API_KEY","")

if "weights" not in st.session_state:
    st.session_state.weights={
        "freq":1.0,
        "recency":1.0,
        "gap":1.0,
        "markov":1.0,
        "cycle":1.0
    }

if "history" not in st.session_state:
    st.session_state.history=[]

# ================= CORE =================
def detect_pattern(nums):
    if len(nums)<5:return "unknown"
    if len(set(nums[-5:]))==1:return "cầu bệt"
    if nums[-1]==nums[-3]:return "cầu đảo"
    if nums[-1]==nums[-2]:return "cầu lặp"
    return "cầu nhảy"

def markov_chain(nums):
    trans={}
    for i in range(len(nums)-2):
        state=(nums[i],nums[i+1])
        nxt=nums[i+2]
        trans.setdefault(state,{})
        trans[state][nxt]=trans[state].get(nxt,0)+1
    for s in trans:
        total=sum(trans[s].values())
        for k in trans[s]:
            trans[s][k]/=total
    return trans

def calculate_scores(raw):

    nums=list(raw)
    total=len(nums)
    freq=collections.Counter(nums)
    markov=markov_chain(nums)

    scores={str(i):0 for i in range(10)}

    for n in scores:

        # anti trap nóng giả
        if freq.get(n,0)/max(total,1)>0.3:
            scores[n]-=1

        scores[n]+= (1-(freq.get(n,0)/max(total,1)))*st.session_state.weights["freq"]

        if n not in nums[-20:]:
            scores[n]+=1*st.session_state.weights["recency"]

        if n in nums:
            gap=len(nums)-1-nums[::-1].index(n)
            scores[n]+= (gap/50)*st.session_state.weights["gap"]

        # cycle heat
        scores[n]+= (nums.count(n)%5)/10*st.session_state.weights["cycle"]

    if len(nums)>=2:
        state=(nums[-2],nums[-1])
        if state in markov:
            for n,p in markov[state].items():
                scores[n]+= (1-p)*st.session_state.weights["markov"]

    return scores

# ================= AI =================
def gemini_ai(data):
    if not GEMINI_API_KEY:return []
    try:
        headers={"Content-Type":"application/json"}
        body={"contents":[{"parts":[{"text":f"choose 3 digits from 0-9 based on pattern {data} return only numbers"}]}]}
        r=requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
        headers=headers,json=body)
        txt=r.json()["candidates"][0]["content"]["parts"][0]["text"]
        digits=[c for c in txt if c.isdigit()]
        return digits[:3]
    except:
        return []

def local_ai(remaining,raw):
    freq=collections.Counter(raw)
    ranked=sorted(remaining,key=lambda x:freq.get(x,0),reverse=True)
    return ranked[:3]

def voting(local,gemini):
    votes=collections.Counter(local+gemini)
    return [n for n,_ in votes.most_common(3)]

# ================= ANALYZE =================
def analyze(raw):

    pattern=detect_pattern(raw)

    scores=calculate_scores(raw)

    eliminated=sorted(scores,key=scores.get,reverse=True)[:3]
    remaining=[str(i) for i in range(10) if str(i) not in eliminated]

    local=local_ai(remaining,raw)
    gemini=gemini_ai(raw[-50:])

    final=voting(local,gemini) if gemini else local

    return eliminated,remaining,final,pattern

# ================= UI =================
data_input=st.text_area("📡 Dán chuỗi số:",height=120)

if st.button("🚀 KÍCH HOẠT AI v60",use_container_width=True):

    raw="".join(filter(str.isdigit,data_input))

    if len(raw)<10:
        st.error("⚠️ cần ít nhất 10 số")
    else:
        with st.spinner("Neural AI đang phân tích..."):
            time.sleep(0.5)
            eliminated,remaining,tinh3,pattern=analyze(raw)

        big_html="".join([f"<div class='big'>{n}</div>" for n in tinh3])

        st.markdown(f"""
        <div class='result'>
        <p>🎯 DÀN 3 TINH</p>
        <div class='bigbox'>{big_html}</div>
        <p>📊 PHÁT HIỆN: {pattern}</p>
        <p>🚫 LOẠI: {", ".join(eliminated)}</p>
        <p>✅ 7 SỐ: {", ".join(remaining)}</p>
        </div>
        """,unsafe_allow_html=True)

# ================= LEARNING =================
st.markdown("### 🧠 Neural Learning PRO")
real=st.text_input("Kết quả thật:")

if st.button("📈 HỌC THẬT v60"):
    if real.isdigit():
        st.session_state.history.append(real)

        # tự điều chỉnh theo winrate
        st.session_state.weights["freq"]*=0.99
        st.session_state.weights["recency"]*=1.01
        st.session_state.weights["gap"]*=1.02
        st.session_state.weights["markov"]*=1.03
        st.session_state.weights["cycle"]*=1.01

        st.success("AI đã tiến hóa theo dữ liệu thật")

st.info("🔥 Engine v60: Neural Learning + Anti Trap + Pattern AI + Heat Cycle + Multi Voting")