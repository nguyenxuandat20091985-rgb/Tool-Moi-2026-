"""
🚀 TITAN V27 - COMPACT VERSION
Version: 4.0.1-FIXED
"""
import streamlit as st
import pandas as pd
from collections import Counter
from openai import OpenAI
import google.generativeai as genai
import json
import re
import itertools

# ================= ⚙️ CONFIG =================
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

PAIR_RULES = ["178", "034", "458", "578", "019", "679", "235", "456", "124", "245", 
              "247", "248", "246", "340", "349", "348", "015", "236", "028", "026", 
              "047", "046", "056", "136", "138", "378"]

# ================= 🎨 CSS COMPACT =================
st.set_page_config(page_title="TITAN V27", page_icon="🎲", layout="centered")

st.markdown("""
<style>
    .main {padding: 1rem;}
    .stTextArea label {display: none;}
    .stButton>button {width: 100%; background: linear-gradient(135deg, #76b900, #5a9e00); color: #000; font-weight: 700; border: none; border-radius: 10px; padding: 15px; font-size: 18px;}
    .result-box {background: #0d1117; border: 2px solid #76b900; border-radius: 15px; padding: 15px; margin: 10px 0;}
    .number-display {font-size: 48px; font-weight: bold; color: #76b900; text-align: center; font-family: monospace; letter-spacing: 5px;}
    .pair-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0;}
    .pair-item {background: rgba(0,212,255,0.1); border-left: 3px solid #00d4ff; padding: 8px; text-align: center; border-radius: 5px; font-family: monospace; font-size: 16px; font-weight: bold;}
    .status-bar {background: #76b900; color: #000; padding: 12px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 16px; margin: 10px 0;}
    .status-stop {background: #ff4444; color: #fff;}
    .metric-box {background: rgba(118,185,0,0.1); border-radius: 8px; padding: 10px; text-align: center; margin: 5px 0;}
    .metric-value {font-size: 24px; font-weight: bold; color: #76b900;}
    .metric-label {font-size: 12px; color: #888;}
    .tabs .stTabsContent {padding: 10px;}
    @media (max-width: 600px) {
        .number-display {font-size: 36px;}
        .pair-grid {grid-template-columns: repeat(3, 1fr);}
    }
</style>
""", unsafe_allow_html=True)

# ================= 🔧 FUNCTIONS =================
def extract_numbers(text):
    return re.findall(r"\b\d{5}\b", text) if text else []

def generate_combos(numbers_7):
    digits = sorted(list(set(numbers_7)))[:7]
    if len(digits) < 7:
        remaining = [str(i) for i in range(10) if str(i) not in digits]
        digits = sorted(digits + remaining[:7-len(digits)])
    pairs = ["".join(p) for p in itertools.combinations(digits, 2)]
    triples = ["".join(t) for t in itertools.combinations(digits, 3)]
    return {"base_7": "".join(digits), "pairs": pairs, "triples": triples}

def calc_scores(db, last_draw):
    if not db:
        return {str(i): 0 for i in range(10)}
    all_digits = "".join(db[-30:])
    scores = {str(i): all_digits.count(str(i)) * 2 for i in range(10)}
    if last_draw:
        for digit in set(last_draw):
            scores[digit] += 30
    return scores

def predict(db):
    if not db or len(db) < 5:
        return {"base_7": "0123456", "main_3": "012", "pairs_sample": ["01","23","45"], 
                "triples_sample": ["012","234","456"], "adv": "DỪNG", "logic": "Chưa đủ dữ liệu", "conf": 50}
    
    try:
        scores = calc_scores(db, db[-1])
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_7 = [x[0] for x in sorted_scores[:7]]
        base_7 = "".join(sorted(top_7))
        combos = generate_combos(base_7)
        
        return {
            "base_7": base_7,
            "main_3": combos["triples"][0] if combos["triples"] else "012",
            "pairs_sample": combos["pairs"][:12],
            "triples_sample": combos["triples"][:10],
            "adv": "ĐÁNH",
            "logic": f"Phân tích {len(db)} kỳ",
            "conf": min(90, 60 + len(db)//5)
        }
    except:
        return {"base_7": "0123456", "main_3": "012", "pairs_sample": ["01","23","45"],
                "triples_sample": ["012","234","456"], "adv": "DỪNG", "logic": "Lỗi", "conf": 50}

# ================= 🧠 INIT =================
if "db" not in st.session_state:
    st.session_state.db = []
if "pred" not in st.session_state:
    st.session_state.pred = None

# ================= 🖥️ UI =================
st.markdown('<h1 style="text-align:center;color:#76b900;margin:10px 0;">🎲 TITAN V27</h1>', unsafe_allow_html=True)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-box"><div class="metric-value">{len(st.session_state.db)}</div><div class="metric-label">Tổng kỳ</div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size:18px;">{last}</div><div class="metric-label">Kỳ cuối</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-box"><div class="metric-value">🎯</div><div class="metric-label">Sẵn sàng</div></div>', unsafe_allow_html=True)

# Input
raw_input = st.text_area("📡 Dán dữ liệu (mỗi dòng 1 số 5 chữ số):", height=80, 
                        placeholder="16923\n51475\n31410\n...")

# Button
if st.button("⚡ CHỐT SỐ AI", type="primary"):
    clean_data = extract_numbers(raw_input)
    
    # ✅ LINE 121 - PHẢI ĐẦY ĐỦ: if clean_data:
    if clean_data:
        st.session_state.db.extend(clean_data)
        st.session_state.pred = predict(st.session_state.db)
        st.rerun()
    else:
        st.error("❌ Không tìm thấy số 5 chữ số!")

# Results
if st.session_state.pred:
    p = st.session_state.pred
    is_go = p.get("adv", "").upper() == "ĐÁNH"
    status_class = "" if is_go else "status-stop"
    status_icon = "🔥" if is_go else "⏸️"
    status_text = "KHUYÊN ĐÁNH" if is_go else "NÊN DỪNG"
    
    st.markdown(f'<div class="status-bar {status_class}">{status_icon} {status_text} | Tin cậy: {p.get("conf",0)}%</div>', unsafe_allow_html=True)
    
    # Base 7 numbers - PROMINENT
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#888;margin-bottom:5px;">🎲 DÀN 7 SỐ</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="number-display">{p.get("base_7", "0123456")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["🎯 2 SỐ", "🎯 3 SỐ"])
    
    with tab1:
        pairs = p.get("pairs_sample", [])
        if not pairs and "base_7" in p:
            combos = generate_combos(p["base_7"])
            pairs = combos["pairs"][:12]
        
        st.markdown('<div class="pair-grid">', unsafe_allow_html=True)
        for pair in pairs[:12]:
            st.markdown(f'<div class="pair-item">{pair}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        triples = p.get("triples_sample", [])
        if not triples and "base_7" in p:
            combos = generate_combos(p["base_7"])
            triples = combos["triples"][:10]
        
        st.markdown('<div class="pair-grid">', unsafe_allow_html=True)
        for triple in triples[:10]:
            st.markdown(f'<div class="pair-item" style="font-size:18px;">{triple}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main number
    if p.get("main_3"):
        st.markdown(f'<div class="result-box" style="text-align:center;"><div style="color:#888;font-size:14px;">⭐ SỐ CHỦ LỰC</div><div class="number-display" style="font-size:36px;">{p["main_3"]}</div></div>', unsafe_allow_html=True)
    
    # AI Logic
    st.markdown(f'<div style="background:rgba(0,212,255,0.1);padding:10px;border-radius:8px;margin:10px 0;"><strong style="color:#00d4ff;">🧠 AI:</strong> {p.get("logic", "")}</div>', unsafe_allow_html=True)
    
    # Copy button
    all_numbers = f"Dàn 7 số: {p.get('base_7', '')}\nSố chủ lực: {p.get('main_3', '')}\n2 số: {', '.join(p.get('pairs_sample', [])[:8])}"
    st.download_button("📋 SAO CHÉP KẾT QUẢ", all_numbers, file_name="titan_v27_result.txt", mime="text/plain", use_container_width=True)

# Quick actions
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🗑️ Xóa dữ liệu", use_container_width=True):
        st.session_state.db = []
        st.session_state.pred = None
        st.rerun()
with col_b:
    if st.button("📊 Thống kê", use_container_width=True):
        st.session_state.show_stats = not st.session_state.get("show_stats", False)

# Stats
if st.session_state.get("show_stats", False) and st.session_state.db:
    with st.expander("📊 Thống kê tần suất", expanded=True):
        freq = Counter("".join(st.session_state.db[-50:]))
        df = pd.DataFrame([{"Số": str(i), "Tần suất": freq.get(str(i), 0)} for i in range(10)])
        st.bar_chart(df.set_index("Số"), color="#00d4ff")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔥 Nóng**")
            for _, row in df.nlargest(3, "Tần suất").iterrows():
                st.write(f"`{row['Số']}`: {row['Tần suất']}")
        with c2:
            st.markdown("**❄️ Lạnh**")
            for _, row in df.nsmallest(3, "Tần suất").iterrows():
                st.write(f"`{row['Số']}`: {row['Tần suất']}")

# Footer
st.markdown('<div style="text-align:center;color:#666;font-size:12px;margin-top:20px;">TITAN V27 v4.0.1-FIXED</div>', unsafe_allow_html=True)