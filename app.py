import streamlit as st
import collections
import requests
import time
import re

st.set_page_config(page_title="AI 3-TINH ELITE v86.1",layout="centered")

# ================= SESSION =================
for key in ["manual","paste","url","live_digits"]:
    if key not in st.session_state:
        st.session_state[key]=""

# ================= UI =================
st.markdown("""
<style>
.stApp{background:#0b0f13;color:#e0e0e0}
.result{border:2px solid #00ffcc;border-radius:15px;padding:20px;background:#161b22;text-align:center;margin-top:20px}
.big{font-size:60px;color:#ffff00;font-weight:bold}
.bigbox{display:flex;justify-content:center;gap:15px}
</style>
""",unsafe_allow_html=True)

st.title("🧠 AI 3-TINH ELITE v86.1 FIX INPUT")

# ================= INPUT TABS =================
tab1,tab2,tab3=st.tabs([
"✍️ Nhập số tay",
"📋 Paste HTML/Text",
"🌐 URL Feed"
])

with tab1:
    st.session_state.manual=st.text_area(
    "Nhập chuỗi số:",
    value=st.session_state.manual,
    height=120)

with tab2:
    st.session_state.paste=st.text_area(
    "Dán HTML / JSON:",
    value=st.session_state.paste,
    height=120)

with tab3:
    st.session_state.url=st.text_input(
    "Link Feed",
    value=st.session_state.url)

    auto=st.checkbox("⚡ Auto refresh 5s")

# ================= TOOL =================
def extract_digits(text):
    return "".join(re.findall(r'\d',text))

def fetch_url(url):
    try:
        headers={"User-Agent":"Mozilla/5.0"}
        r=requests.get(url,headers=headers,timeout=8)
        return extract_digits(r.text)
    except:
        return ""

# ================= AI CORE =================
def analyze(raw):

    nums=list(raw)
    freq=collections.Counter(nums)

    scores={str(i):0 for i in range(10)}

    for n in scores:
        scores[n]+=1-(freq.get(n,0)/max(len(nums),1))

    eliminated=sorted(scores,key=scores.get,reverse=True)[:3]
    remaining=[str(i) for i in range(10) if str(i) not in eliminated]

    final=sorted(remaining,key=lambda x:freq.get(x,0),reverse=True)[:3]

    return eliminated,remaining,final

# ================= RUN =================
if st.button("🚀 CHẠY AI",use_container_width=True):

    digits=""

    if st.session_state.manual:
        digits+=extract_digits(st.session_state.manual)

    if st.session_state.paste:
        digits+=extract_digits(st.session_state.paste)

    if st.session_state.url:
        digits+=fetch_url(st.session_state.url)

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
if "auto" in locals() and auto and st.session_state.url:
    time.sleep(5)
    st.rerun()

st.info("🔥 v86.1 FIX: Tabs Input + Session Buffer + No Reset")