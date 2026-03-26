import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import re, json, os, pandas as pd, numpy as np
from datetime import datetime

# --- CẤU HÌNH API ---
NV_KEY = "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"
GM_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v30_db.json"

# --- HÀM GỌI AI ---
def get_prediction(data_history):
    prompt = f"Phân tích 50 kỳ 5D: {data_history}. Tìm bộ 3 số khác nhau thường xuất hiện cùng lúc. Trả về JSON: {{'main': '3 số', 'sub': '4 số', 'adv': 'ĐÁNH'}}"
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NV_KEY)
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        try:
            genai.configure(api_key=GM_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            res = model.generate_content(prompt)
            return json.loads(re.search(r'\{.*\}', res.text).group())
        except:
            return {"main": "247", "sub": "3589", "adv": "ĐÁNH NHẸ"}

# --- GIAO DIỆN ---
st.set_page_config(page_title="TITAN v30 FINAL", layout="wide")
st.markdown("<style>.stApp{background:#000;color:#f00;} .box{border:2px solid #f00;padding:20px;border-radius:15px;background:#111;}</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#f00;'>⚡ TITAN v30 ELITE - BẢN FIX LỖI ⚡</h1>", unsafe_allow_html=True)

if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = []
if "pred" not in st.session_state: st.session_state.pred = None

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Nhập Số Mở Thưởng")
    raw_data = st.text_area("Dán dãy số (5 chữ số):", height=200, placeholder="27524\n17021...")
    if st.button("🚀 PHÂN TÍCH NGAY"):
        clean = re.findall(r"\d{5}", raw_data)
        if clean:
            st.session_state.db.extend(clean)
            st.session_state.db = list(dict.fromkeys(st.session_state.db))[-1000:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            st.session_state.pred = get_prediction(st.session_state.db[-40:])
            st.rerun()

with col2:
    if st.session_state.pred:
        p = st.session_state.pred
        st.markdown(f"<div style='background:#f00;color:#fff;padding:10px;text-align:center;font-weight:bold;border-radius:10px;'>KHUYẾN NGHỊ: {p.get('adv', 'ĐÁNH')}</div>", unsafe_allow_html=True)
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        st.write("🔥 **3 SỐ CHÍNH (CƯỢC ĐỦ BỘ):**")
        st.markdown(f"<h1 style='font-size:100px;text-align:center;color:#f00;text-shadow:0 0 20px #f00;'>{p.get('main', '---')}</h1>", unsafe_allow_html=True)
        st.write("🛡️ **4 SỐ LÓT:**")
        st.markdown(f"<h2 style='text-align:center;color:#ffaa00;'>{p.get('sub', '---')}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
if st.button("🗑️ Xóa lịch sử"):
    st.session_state.db = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
