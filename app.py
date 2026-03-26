import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import re, json, os, pandas as pd
from collections import Counter

# --- CẤU HÌNH ---
NV_KEY = "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"
GM_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v28_recovery.json"

# Bộ quy tắc "Cặp số đi cùng" xương máu của anh Đạt
PAIR_RULES = ["178", "034", "458", "578", "019", "679", "235", "456", "124", "245", 
              "247", "248", "246", "340", "349", "348", "015", "236", "028", "026", 
              "047", "046", "056", "136", "138", "378"]

def get_ai_prediction(data_history):
    prompt = f"Phân tích 50 kỳ 5D: {data_history}. Dựa theo bộ số {PAIR_RULES}, hãy loại 7 số và giữ 3 số nổ cao nhất. Trả về JSON: {{'main': '3 số', 'sub': '4 số', 'advice': 'VÀO TIỀN/DỪNG', 'conf': 95, 'target': 'Vào nhẹ/Vào mạnh'}}"
    
    # 1. Thử NVIDIA
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NV_KEY)
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        # 2. Dự phòng Gemini
        try:
            genai.configure(api_key=GM_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            res = model.generate_content(prompt)
            return json.loads(re.search(r'\{.*\}', res.text).group())
        except:
            return None

# --- GIAO DIỆN ---
st.set_page_config(page_title="TITAN V28 RECOVERY", layout="wide")
st.markdown("<style>.stApp{background:#000; color:#0f0;} .main-card{border:2px solid #0f0; padding:20px; border-radius:15px; background:#050505;}</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#0f0;'>☣️ TITAN V28 - CHẾ ĐỘ HỒI VỐN ☣️</h1>", unsafe_allow_html=True)

if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = []

col_in, col_res = st.columns([1, 1.2])

with col_in:
    st.subheader("📡 Nhập dữ liệu mới")
    raw_data = st.text_area("Dán dãy số 5 chữ số:", height=200)
    if st.button("🚀 PHÂN TÍCH NHỊP CẦU"):
        clean = re.findall(r"\d{5}", raw_data)
        if clean:
            st.session_state.db.extend(clean)
            st.session_state.db = list(dict.fromkeys(st.session_state.db))[-2000:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            
            pred = get_ai_prediction(st.session_state.db[-40:])
            if pred:
                st.session_state.last_pred = pred
                st.rerun()

with col_res:
    if "last_pred" in st.session_state:
        p = st.session_state.last_pred
        color = "#00ff00" if p['advice'] == "VÀO TIỀN" else "#ff0000"
        
        st.markdown(f"<div style='background:{color}; color:#000; padding:10px; border-radius:10px; text-align:center; font-weight:bold;'>{p['advice']} - {p['target']} ({p['conf']}%)</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.write("🔥 **3 SỐ CHỦ LỰC (LOẠI 7 CHỌN 3):**")
        st.markdown(f"<h1 style='font-size:100px; text-align:center; color:#0f0;'>{p['main']}</h1>", unsafe_allow_html=True)
        
        st.write("🛡️ **4 SỐ LÓT (GIỮ VỐN):**")
        st.markdown(f"<h3 style='text-align:center; color:#00d4ff;'>{p['sub']}</h3>", unsafe_allow_html=True)
        
        dan_7 = "".join(sorted(set(p['main'] + p['sub'])))
        st.text_input("📋 DÀN 7 SỐ KUBET:", dan_7)
        st.markdown("</div>", unsafe_allow_html=True)

if st.button("🗑️ Reset Dữ liệu"):
    st.session_state.db = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
