import streamlit as st
import collections
import requests
import time
import re
import json
import websocket
import threading

st.set_page_config(page_title="AI 3-TINH ELITE v90 REALTIME",layout="centered")

# ================= SESSION =================
for key in ["manual","paste","url","socket","live"]:
    if key not in st.session_state:
        st.session_state[key]=""

# ================= UI =================
st.title("🔥 AI 3-TINH ELITE v90 REALTIME ENGINE")

tab1,tab2,tab3,tab4=st.tabs([
"✍️ Nhập tay",
"📋 Paste HTML/Text",
"🌐 API JSON",
"⚡ WebSocket"
])

with tab1:
    st.session_state.manual=st.text_area("Nhập số:",value=st.session_state.manual)

with tab2:
    st.session_state.paste=st.text_area("Dán HTML:",value=st.session_state.paste)

with tab3:
    st.session_state.url=st.text_input("Link JSON API:",value=st.session_state.url)
    auto=st.checkbox("Auto refresh 5s")

with tab4:
    st.session_state.socket=st.text_input("Link WebSocket (wss://...)",
    value=st.session_state.socket)

# ================= PARSER =================
def extract_digits(text):
    return "".join(re.findall(r'\d',str(text)))

# ================= FETCH JSON =================
def fetch_json(url):
    try:
        r=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        data=r.json()
        return extract_digits(json.dumps(data))
    except:
        return ""

# ================= SOCKET =================
def socket_worker(url):
    def on_message(ws,msg):
        st.session_state.live+=extract_digits(msg)
    def on_error(ws,e):pass
    def on_close(ws,a,b):pass
    def on_open(ws):pass

    ws=websocket.WebSocketApp(url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close)
    ws.run_forever()

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

# ================= RUN SOCKET =================
if st.button("🔌 START SOCKET"):
    if st.session_state.socket.startswith("ws"):
        threading.Thread(target=socket_worker,
        args=(st.session_state.socket,),
        daemon=True).start()
        st.success("Socket started")

# ================= RUN AI =================
if st.button("🚀 CHẠY AI v90",use_container_width=True):

    raw=""

    raw+=extract_digits(st.session_state.manual)
    raw+=extract_digits(st.session_state.paste)

    if st.session_state.url:
        raw+=fetch_json(st.session_state.url)

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

# ================= AUTO =================
if "auto" in locals() and auto:
    time.sleep(5)
    st.rerun()

st.info("🔥 v90 Engine: JSON API + WebSocket + Hybrid Feed + Realtime Buffer")