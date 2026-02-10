import streamlit as st
import collections
import requests
import time
import re
import json
import threading
import websocket

st.set_page_config(page_title="AI 3-TINH ELITE v95 AUTO DETECTOR",layout="centered")

# ================= SESSION =================
for k in ["manual","paste","url","socket","live","html"]:
    if k not in st.session_state:
        st.session_state[k]=""

# ================= UI =================
st.title("🔥 AI 3-TINH ELITE v95 - AUTO DATA DETECTOR")

tab1,tab2,tab3,tab4=st.tabs([
"✍️ Nhập tay",
"📋 Paste HTML",
"🌐 URL AUTO SCAN",
"⚡ WebSocket"
])

with tab1:
    st.session_state.manual=st.text_area("Chuỗi số:",value=st.session_state.manual)

with tab2:
    st.session_state.paste=st.text_area("HTML/Text:",value=st.session_state.paste,height=150)

with tab3:
    st.session_state.url=st.text_input("URL trang hoặc API:",value=st.session_state.url)
    auto=st.checkbox("Auto refresh 5s")

with tab4:
    st.session_state.socket=st.text_input("WebSocket wss://",value=st.session_state.socket)

# ================= TOOL =================
def digits(t):
    return "".join(re.findall(r'\d',str(t)))

# auto scan api trong html
def find_api(html):
    links=re.findall(r'https?://[^\s"\']+',html)
    return [l for l in links if "api" in l or "json" in l or "result" in l]

# fetch html
def fetch_html(url):
    try:
        r=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        return r.text
    except:
        return ""

# fetch json
def fetch_json(url):
    try:
        r=requests.get(url,timeout=10)
        return digits(json.dumps(r.json()))
    except:
        return ""

# websocket
def socket_worker(url):
    def on_message(ws,msg):
        st.session_state.live+=digits(msg)
    ws=websocket.WebSocketApp(url,on_message=on_message)
    ws.run_forever()

# ================= AI =================
def analyze(raw):

    nums=list(raw)
    freq=collections.Counter(nums)

    score={str(i):0 for i in range(10)}

    for n in score:
        score[n]+=1-(freq.get(n,0)/max(len(nums),1))

    eliminated=sorted(score,key=score.get,reverse=True)[:3]
    remaining=[str(i) for i in range(10) if str(i) not in eliminated]
    final=sorted(remaining,key=lambda x:freq.get(x,0),reverse=True)[:3]

    return eliminated,remaining,final

# ================= SOCKET START =================
if st.button("🔌 START SOCKET"):
    if st.session_state.socket.startswith("ws"):
        threading.Thread(target=socket_worker,
        args=(st.session_state.socket,),
        daemon=True).start()
        st.success("Socket running")

# ================= RUN =================
if st.button("🚀 CHẠY AI v95",use_container_width=True):

    raw=""
    raw+=digits(st.session_state.manual)
    raw+=digits(st.session_state.paste)

    # URL AUTO
    if st.session_state.url:

        html=fetch_html(st.session_state.url)
        raw+=digits(html)

        apis=find_api(html)

        for a in apis[:3]:
            raw+=fetch_json(a)

    raw+=st.session_state.live

    if len(raw)<10:
        st.warning("Chưa đủ dữ liệu")
    else:
        eliminated,remaining,tinh3=analyze(raw)

        big="".join([f"<h1 style='color:yellow'>{n}</h1>" for n in tinh3])

        st.markdown(f"""
        <div style='border:2px solid #00ffcc;padding:20px;border-radius:15px'>
        <h3>🎯 DÀN 3 TINH</h3>
        {big}
        <p>🚫 Loại: {", ".join(eliminated)}</p>
        <p>✅ 7 số: {", ".join(remaining)}</p>
        </div>
        """,unsafe_allow_html=True)

# ================= AUTO REFRESH =================
if "auto" in locals() and auto:
    time.sleep(5)
    st.rerun()

st.info("🔥 v95 Engine: Auto HTML Scan + API Detect + WebSocket Feed + Hybrid Input")