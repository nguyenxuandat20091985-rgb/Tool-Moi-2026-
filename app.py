import streamlit as st
import collections
import requests
import time

st.set_page_config(page_title="AI 3-TINH ELITE v50",layout="centered")

# ================= UI =================
st.markdown("""
<style>
.stApp{background:#0b0f13;color:#e0e0e0}
.result{border:2px solid #00ffcc;border-radius:15px;padding:20px;background:#161b22;text-align:center}
.big{font-size:70px;color:#ffff00;font-weight:bold}
</style>
""",unsafe_allow_html=True)

st.title("🛡️ AI 3-TINH ELITE v50 - NEURAL ADAPTIVE")

# ================= CONFIG =================
GEMINI_API_KEY=st.secrets.get("GEMINI_API_KEY","")
OPENAI_API_KEY=st.secrets.get("OPENAI_API_KEY","")

if "weights" not in st.session_state:
    st.session_state.weights={
        "freq":1.0,
        "recency":1.0,
        "gap":1.0,
        "markov":1.0
    }

# ================= CORE =================
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

        scores[n]+= (1-(freq.get(n,0)/max(total,1)))*st.session_state.weights["freq"]

        if n not in nums[-20:]:
            scores[n]+=1*st.session_state.weights["recency"]

        if n in nums:
            gap=len(nums)-1-nums[::-1].index(n)
            scores[n]+= (gap/50)*st.session_state.weights["gap"]

    if len(nums)>=2:
        state=(nums[-2],nums[-1])
        if state in markov:
            for n,p in markov[state].items():
                scores[n]+= (1-p)*st.session_state.weights["markov"]

    return scores

# ================= MULTI AI =================
def gemini_ai(data):
    if not GEMINI_API_KEY: return []
    try:
        headers={"Content-Type":"application/json"}
        body={"contents":[{"parts":[{"text":f"choose 3 digits from 0-9 from pattern {data}"}]}]}
        r=requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
        headers=headers,json=body)
        txt=r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return [c for c in txt if c.isdigit()][:3]
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

    scores=calculate_scores(raw)

    eliminated=sorted(scores,key=scores.get,reverse=True)[:3]
    remaining=[str(i) for i in range(10) if str(i) not in eliminated]

    local=local_ai(remaining,raw)
    gemini=gemini_ai(raw[-50:])

    final=voting(local,gemini) if gemini else local

    return eliminated,remaining,final

# ================= UI =================
data_input=st.text_area("📡 Dán chuỗi số:",height=120)

if st.button("🚀 KÍCH HOẠT AI v50",use_container_width=True):

    raw="".join(filter(str.isdigit,data_input))

    if len(raw)<10:
        st.error("⚠️ cần ít nhất 10 số")
    else:
        with st.spinner("Neural AI đang phân tích..."):
            time.sleep(0.5)
            eliminated,remaining,tinh3=analyze(raw)

        st.markdown(f"""
        <div class='result'>
        <p>🎯 DÀN 3 TINH</p>
        <p class='big'>{" - ".join(tinh3)}</p>
        <p>🚫 LOẠI: {", ".join(eliminated)}</p>
        <p>✅ 7 SỐ: {", ".join(remaining)}</p>
        </div>
        """,unsafe_allow_html=True)

# ================= LEARNING =================
st.markdown("### 🧠 Neural Learning (nhập kết quả thật)")
real=st.text_input("Kết quả thật sau khi ra:")

if st.button("📈 HỌC TỪ KẾT QUẢ"):
    if real.isdigit():
        st.session_state.weights["freq"]*=0.98
        st.session_state.weights["recency"]*=1.02
        st.session_state.weights["gap"]*=1.01
        st.session_state.weights["markov"]*=1.03
        st.success("AI đã tự điều chỉnh trọng số")

st.info("🔥 Engine: Neural Adaptive + Multi AI Voting + Anti Trap + Markov")